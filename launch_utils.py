import logging
from typing import List,Dict
import copy
import os
import numpy as np
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation as R
import cv2
import copy

import matplotlib.pyplot as plt
import torch

from rlbench.backend.observation import Observation
from rlbench.backend.const import TABLE_COORD
from rlbench.demo import Demo
from yarr.envs.env import Env
from yarr.replay_buffer.prioritized_replay_buffer import \
    PrioritizedReplayBuffer, ObservationElement
from yarr.replay_buffer.replay_buffer import ReplayElement, ReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer

from arm import demo_loading_utils, utils
from arm.ota.custom_rlbench_env import OtaCustomRLBenchEnv
from arm.ota.preprocess_agent import PreprocessAgent
from arm.ota.networks import Qattention3DNet
from arm.ota.qattention_agent import QAttentionAgent
from arm.ota.qattention_stack_agent import QAttentionStackAgent
from arm.ota.const import SAVE_OBS_KEYS,SAVE_OBS_ELEMENT_KEYS
from arm.ota.voxel_grid import VoxelGrid
from arm.ota.aux_task.aux_reward import AuxReward

REWARD_SCALE = 100.0


def keypoint_discovery(demo: Demo, stopping_delta=0.1) -> List[List[int]]:

    episode_keypoints = []
    transition_keypoints = []
    prev_transition_index = 0
    prev_stage = 'viewpoint'
    demo_length = len(demo)
    for step, obs in enumerate(demo):
        stage = obs.stage
        current_transition_index = obs.transition_index
        if stage is None and current_transition_index==0:
            continue
        # 
        if current_transition_index == prev_transition_index and stage is not None:

            if stage != prev_stage:
                transition_keypoints.append(step-1)
                prev_stage = stage
                continue
        elif prev_transition_index !=0 :

            transition_keypoints.append(step-1)
            assert len(transition_keypoints) == 2
            episode_keypoints.append(transition_keypoints)
            transition_keypoints = []
            prev_stage = 'viewpoint'
            prev_transition_index = current_transition_index
        else:
            prev_stage = 'viewpoint'
            prev_transition_index = current_transition_index
                
    logging.debug('Found %d keypoints.' % len(episode_keypoints),
                  episode_keypoints)
    return episode_keypoints

def create_replay(batch_size: int, timesteps: int, prioritisation: bool,
                  save_dir: str, cameras: list, env: Env,
                  viewpoint_agent_bounds: list,viewpoint_resolution: list,
                  voxel_sizes, replay_size=1e5):

    trans_indicies_size = 3 
    rot_and_grip_indicies_size = (3 + 1)
    rot_vision_indicies_size = 3
    observation_elements = env.observation_elements
    

    viewpoint_categories_xyz_axes = np.array(
        (np.array(viewpoint_agent_bounds)[3:] - 
         np.array(viewpoint_agent_bounds)[:3])//np.array(viewpoint_resolution)).astype(int)
    viewpoint_axes_movable = viewpoint_categories_xyz_axes>0
    viewpoint_categories_num = int(viewpoint_categories_xyz_axes.sum())
    # gripper tip state + gripper trans + gripper quat + cam vp class 
    low_dim_state_size = 3 + 3 + 4 + int(viewpoint_axes_movable.sum()) + int(env._time_in_state)
    
    print('==========Env observations==========')
    [print(el.name,el.shape) for el in observation_elements]
    print('============================================')
    

    save_obs_elm = []
    for el in observation_elements:
        if el.name in SAVE_OBS_KEYS:
            for depth in range(len(voxel_sizes)):
                copy_el = copy.deepcopy(el)

                if copy_el.name == 'low_dim_state':
                    copy_el.shape = (low_dim_state_size,)
                copy_el.name = copy_el.name + '_layer_{}'.format(depth)
                save_obs_elm.append(copy_el)
    
    
    # 10
    #for cname in cameras:
    #    save_obs_elm.append(
    #        ObservationElement('%s_pixel_coord' % cname, (2,), np.int32))
    
    # 14
    for depth in range(len(voxel_sizes)):
        if depth == len(voxel_sizes)-1:

            rot_len = rot_and_grip_indicies_size 
        else:

            rot_len = int(viewpoint_axes_movable.sum())
        
        save_obs_elm.extend([

            ReplayElement('translation_idxs' + '_layer_{}'.format(depth), (trans_indicies_size,),np.int32),

            ReplayElement('rot_grip_action_indicies' + '_layer_{}'.format(depth), (rot_len,),np.int32) 
        ])

    for depth in range(len(voxel_sizes)):
        save_obs_elm.append(
            ReplayElement('attention_output_layer_%d' % depth, (3,), np.float32)
        )
        save_obs_elm.append(
            ReplayElement('attention_input_layer_%d' % depth, (3,), np.float32)
        )
        # 
        save_obs_elm.append(ReplayElement('viewpoint_coordinate_layer_%d' % depth, (3,), np.float32))
        
    save_obs_elm.append(ReplayElement('reach_reward', (1,), np.float32))
    save_obs_elm.append(ReplayElement('focus_reward', (1,), np.float32))
        


    extra_replay_elements = [ReplayElement('demo', (), bool),]

    replay_class = UniformReplayBuffer
    if prioritisation:
        replay_class = PrioritizedReplayBuffer
    replay_buffer = replay_class(
        save_dir=save_dir,
        batch_size=batch_size,
        timesteps=timesteps,
        replay_capacity=int(replay_size),

        action_shape=(8,),
        action_dtype=np.float32,
        reward_shape=(),
        reward_dtype=np.float32,
        update_horizon=1,

        observation_elements=save_obs_elm,
        extra_replay_elements=extra_replay_elements
    )
    print('==========All observation elements==========')
    [print(el.name,el.shape) for el in save_obs_elm]
    print('==========Extra   replay  elements==========')
    [print(el.name,el.shape) for el in extra_replay_elements]
    print('============================================')
    return replay_buffer


def _get_action(
        obs_dict_tp0: Dict,
        obs_dict_tp0_1: Dict,
        obs_dict_tp1: Dict,
        
        rlbench_scene_bounds: List[float],   # AKA: DEPTH0_BOUNDS
        viewpoint_agent_bounds: List[float], 
        viewpoint_env_bounds: List[float], 

        viewpoint_resolution: List[float], 
        voxel_sizes: List[int],
        bounds_offset: List[float],
        rotation_resolution: int,
        crop_augmentation: bool,
        viewpoint_align:bool,
        env:OtaCustomRLBenchEnv,
        device="cpu"):
    

    viewpoint_categories_xyz_axes = np.array(
        (np.array(viewpoint_agent_bounds)[3:] - 
         np.array(viewpoint_agent_bounds)[:3])//np.array(viewpoint_resolution)).astype(int)
    viewpoint_axes_movable = viewpoint_categories_xyz_axes>0
    

    gp_world_quat_tp0 = utils.normalize_quaternion(obs_dict_tp0["gripper_pose"][3:])
    gp_world_quat_tp0_1 = utils.normalize_quaternion(obs_dict_tp0_1["gripper_pose"][3:])
    gp_world_quat_tp1 = utils.normalize_quaternion(obs_dict_tp1["gripper_pose"][3:])
    

    if gp_world_quat_tp0[-1] < 0:
        gp_world_quat_tp0 = -gp_world_quat_tp0
    if gp_world_quat_tp0_1[-1] < 0:
        gp_world_quat_tp0_1 = -gp_world_quat_tp0_1
    if gp_world_quat_tp1[-1] < 0:
        gp_world_quat_tp1 = -gp_world_quat_tp1
        

    pc_world_tp0 = obs_dict_tp0["active_point_cloud"] 
    pc_world_tp0_1 = obs_dict_tp0_1["active_point_cloud"] 
    pc_world_tp1 = obs_dict_tp1["active_point_cloud"] 
    # timestep encode
    time_state_tp0 = obs_dict_tp0["low_dim_state"][[-1]] if env._time_in_state else np.array([])
    time_state_tp0_1 = obs_dict_tp0_1["low_dim_state"][[-1]] if env._time_in_state else np.array([])
    time_state_tp1 = obs_dict_tp1["low_dim_state"][[-1]] if env._time_in_state else np.array([])

    gp_tip_state_tp0 = np.concatenate([obs_dict_tp0["low_dim_state"][[0]],obs_dict_tp0["low_dim_state"][8:10]])
    gp_tip_state_tp0_1 = np.concatenate([obs_dict_tp0_1["low_dim_state"][[0]],obs_dict_tp0_1["low_dim_state"][8:10]])
    gp_tip_state_tp1 = np.concatenate([obs_dict_tp1["low_dim_state"][[0]],obs_dict_tp1["low_dim_state"][8:10]])

    gp_world_pos_tp0 = obs_dict_tp0["gripper_pose"][:3]
    gp_world_pos_tp0_1 = obs_dict_tp0_1["gripper_pose"][:3]
    gp_world_pos_tp1 = obs_dict_tp1["gripper_pose"][:3]

    vp_world_pos_tp0 = obs_dict_tp0["active_cam_pose"][:3]
    vp_world_pos_tp0_1 = obs_dict_tp0_1["active_cam_pose"][:3]
    vp_world_pos_tp1 = obs_dict_tp1["active_cam_pose"][:3]   

    disc_vp_world_spher_tp0,vp_world_spher_tp0 = utils.world_cart_to_disc_world_spher(
        world_cartesion_position=vp_world_pos_tp0,
        bounds=viewpoint_agent_bounds,
        spher_res=viewpoint_resolution,
        )
    disc_vp_world_spher_tp0_1,vp_world_spher_tp0_1 = utils.world_cart_to_disc_world_spher(
        world_cartesion_position=vp_world_pos_tp0_1,
        bounds=viewpoint_agent_bounds,
        spher_res=viewpoint_resolution,
        )
    disc_vp_world_spher_tp1,vp_world_spher_tp1 = utils.world_cart_to_disc_world_spher(
        world_cartesion_position=vp_world_pos_tp1,
        bounds=viewpoint_agent_bounds,
        spher_res=viewpoint_resolution,
        )

    norm_vp_world_spher_tp0 = utils.viewpoint_normlize(vp_world_spher_tp0,viewpoint_env_bounds)
    norm_vp_world_spher_tp0_1 = utils.viewpoint_normlize(vp_world_spher_tp0_1,viewpoint_env_bounds)
    norm_vp_world_spher_tp1 = utils.viewpoint_normlize(vp_world_spher_tp1,viewpoint_env_bounds)
    # 3+3+4+2
    low_dim_state_world_tp0 = np.concatenate([gp_tip_state_tp0,gp_world_pos_tp0,gp_world_quat_tp0,
                                        norm_vp_world_spher_tp0[viewpoint_axes_movable],time_state_tp0],axis=-1)
    low_dim_state_world_tp0_1 = np.concatenate([gp_tip_state_tp0_1,gp_world_pos_tp0_1,gp_world_quat_tp0_1,
                                            norm_vp_world_spher_tp0_1[viewpoint_axes_movable],time_state_tp0_1],axis=-1)
    low_dim_state_world_tp1 = np.concatenate([gp_tip_state_tp1,gp_world_pos_tp1,gp_world_quat_tp1,
                                            norm_vp_world_spher_tp1[viewpoint_axes_movable],time_state_tp1],axis=-1)

    # world action 
    disc_gp_eluer_goal = utils.quaternion_to_discrete_euler(gp_world_quat_tp1, rotation_resolution)
    disc_vp_spher_goal = disc_vp_world_spher_tp0_1
    attention_output_tp1 = gp_world_pos_tp1

    if crop_augmentation:
        shift = bounds_offset[0] * 0.65
        rand_shift = np.random.uniform(-shift, shift, size=(3,))

        attention_output_tp0_1 = attention_output_tp1  + rand_shift
    else:
        attention_output_tp0_1 = attention_output_tp1
    
    attention_input_tp0_1 = attention_output_tp0_1
    
    attention_world_pos = np.copy(attention_output_tp0_1)
    
    vp_spher_goal = vp_world_spher_tp0_1

    # observation
    low_dim_states = [low_dim_state_world_tp0,low_dim_state_world_tp0_1,low_dim_state_world_tp1]
    pointclouds = [pc_world_tp0,pc_world_tp0_1,pc_world_tp1]
    
    if viewpoint_align:

        wtl_rot_tp0 = utils.world_to_local_rotation(world_viewpoint_position=vp_world_pos_tp0)
        wtl_rot_tp0_1 = utils.world_to_local_rotation(world_viewpoint_position=vp_world_pos_tp0_1)
        wtl_rot_tp1 = utils.world_to_local_rotation(world_viewpoint_position=vp_world_pos_tp1)
        
        gp_local_pos_tp0,gp_local_quat_tp0,gp_local_euler_tp0 = utils.world_pose_to_local_pose(world_to_local_rotation=wtl_rot_tp0,
                                                                               world_point_position=gp_world_pos_tp0,
                                                                               world_point_quat=gp_world_quat_tp0)
        
        gp_local_pos_tp0_1,gp_local_quat_tp0_1,gp_local_euler_tp0_1 = utils.world_pose_to_local_pose(world_to_local_rotation=wtl_rot_tp0_1,
                                                                               world_point_position=gp_world_pos_tp0_1,
                                                                               world_point_quat=gp_world_quat_tp0_1)
        
        gp_local_pos_tp1,gp_local_quat_tp1,gp_local_euler_tp1 = utils.world_pose_to_local_pose(world_to_local_rotation=wtl_rot_tp1,
                                                                               world_point_position=gp_world_pos_tp1,
                                                                               world_point_quat=gp_world_quat_tp1)
        #utils.show_pc(pc_world_tp0.reshape(3,-1),
        #         gripper_pose=np.concatenate([gp_world_pos_tp0,gp_world_quat_tp0],-1),
        #         color=obs_dict_tp0['active_rgb'].reshape(3,-1))

        
        

        gp_local_pos_goal_tp0,_,_ = utils.world_pose_to_local_pose(world_to_local_rotation=wtl_rot_tp0,
                                                                            world_point_position=gp_world_pos_tp1,
                                                                            world_point_quat=gp_world_quat_tp1)
        gp_local_pos_goal_tp0_1,gp_local_quat_goal_tp0_1,_ = utils.world_pose_to_local_pose(world_to_local_rotation=wtl_rot_tp0_1,
                                                                            world_point_position=gp_world_pos_tp1,
                                                                            world_point_quat=gp_world_quat_tp1)
        
        attention_output_local_tp0_1,_,_ = utils.world_pose_to_local_pose(world_to_local_rotation=wtl_rot_tp0,
                                                                            world_point_position=attention_output_tp0_1,
                                                                            world_point_quat=gp_world_quat_tp1) # placeholder
        
        attention_input_local_tp0_1,_,_ = utils.world_pose_to_local_pose(world_to_local_rotation=wtl_rot_tp0_1,
                                                                            world_point_position=attention_input_tp0_1,
                                                                            world_point_quat=gp_world_quat_tp1) # placeholder
        
        
        
        #max_indices = (np.array(viewpoint_agent_bounds[3:])-np.array(viewpoint_agent_bounds[:3]))//np.array(viewpoint_resolution)
        
        vp_local_pos_goal_tp0_1,_,_ = utils.world_pose_to_local_pose(world_to_local_rotation=wtl_rot_tp0,
                                                                     world_point_position=vp_world_pos_tp0_1,
                                                                     world_point_quat=np.array([0,0,0,1]))
        disc_vp_local_spher_goal_tp0_1,vp_local_spher_tp0_1 = utils.world_cart_to_disc_world_spher(
                                                                    world_cartesion_position=vp_local_pos_goal_tp0_1,
                                                                    bounds=viewpoint_agent_bounds,
                                                                    spher_res=viewpoint_resolution,
                                                                    )
        

            
                

        disc_gp_eluer_goal = utils.quaternion_to_discrete_euler(gp_local_quat_goal_tp0_1, rotation_resolution)
        disc_vp_spher_goal = disc_vp_local_spher_goal_tp0_1
        vp_spher_goal = vp_local_spher_tp0_1
        attention_output_tp0_1 = np.clip(attention_output_local_tp0_1,rlbench_scene_bounds[:3],rlbench_scene_bounds[3:])
        attention_output_tp1 = gp_local_pos_goal_tp0_1
        attention_input_tp0_1 = attention_input_local_tp0_1
        
        #assert gp_local_pos_tp1 == gp_local_pos_goal_tp0_1

        

        
        low_dim_state_local_tp0 = np.concatenate([gp_tip_state_tp0,gp_local_pos_tp0,gp_local_quat_tp0,
                                            norm_vp_world_spher_tp0[viewpoint_axes_movable],time_state_tp0],axis=-1)
        low_dim_state_local_tp0_1 = np.concatenate([gp_tip_state_tp0_1,gp_local_pos_tp0_1,gp_local_quat_tp0_1,
                                             norm_vp_world_spher_tp0_1[viewpoint_axes_movable],time_state_tp0_1],axis=-1)
        low_dim_state_local_tp1 = np.concatenate([gp_tip_state_tp1,gp_local_pos_tp1,gp_local_quat_tp1,
                                             norm_vp_world_spher_tp1[viewpoint_axes_movable],time_state_tp1],axis=-1)
        
        pc_local_tp0 = utils.world_pc_to_local_pc(world_viewpoint_position=vp_world_pos_tp0,world_points=pc_world_tp0)
        #utils.show_pc(pc_local_tp0.reshape(3,-1),
        #         gripper_pose=np.concatenate([gp_local_pos_tp0,gp_local_quat_tp0],-1),
        #         color=obs_dict_tp0['active_rgb'].reshape(3,-1))
        pc_local_tp0_1 = utils.world_pc_to_local_pc(world_viewpoint_position=vp_world_pos_tp0_1,world_points=pc_world_tp0_1)
        pc_local_tp1 = utils.world_pc_to_local_pc(world_viewpoint_position=vp_world_pos_tp1,world_points=pc_world_tp1)
        low_dim_states = [low_dim_state_local_tp0,low_dim_state_local_tp0_1,low_dim_state_local_tp1]
        pointclouds = [pc_local_tp0,pc_local_tp0_1,pc_local_tp1]
    

    

    assert len(bounds_offset) == len(voxel_sizes) -1
    bounds = np.array(rlbench_scene_bounds)
    attention_and_gp_pos_indicies, attention_outputs =[], []
    for depth, vox_size in enumerate(voxel_sizes):

        if depth > 0:
                
            bounds = np.concatenate(
                [attention_input_tp0_1 - bounds_offset[depth - 1],
                attention_input_tp0_1 + bounds_offset[depth - 1]])
            
                

        if depth < len(voxel_sizes)-1:

            index = utils.point_to_voxel_index(
                attention_output_tp0_1, vox_size, bounds)
        else: 

            index = utils.point_to_voxel_index(
                attention_output_tp1, vox_size, bounds)
            
        if sum(index<0) > 0:
            print(depth,index,attention_output_tp0_1,attention_output_tp1,bounds)
        
        
        attention_and_gp_pos_indicies.append(index)

        res = (bounds[3:] - bounds[:3]) / vox_size
        disc_coordinate = bounds[:3] + res * index + res/2
        attention_outputs.append(disc_coordinate)



    
    vp_and_gp_rot_indicies = [disc_vp_spher_goal[viewpoint_axes_movable].astype(np.int32).tolist(),
                              disc_gp_eluer_goal.tolist() + [int(obs_dict_tp1["low_dim_state"][0])]]
    viewpoints_spher = [vp_spher_goal,np.zeros_like(vp_spher_goal)]
    attention_inputs = [np.zeros_like(attention_input_tp0_1), attention_input_tp0_1]
    
    
    return (attention_and_gp_pos_indicies,  

            vp_and_gp_rot_indicies, 

            np.concatenate([obs_dict_tp1["gripper_pose"], np.array([obs_dict_tp1["low_dim_state"][0]])]), 

            attention_inputs,

            attention_outputs,

            viewpoints_spher,

            low_dim_states,
            #  
            pointclouds,

            attention_world_pos) # 

def _get_demo_viewpoints(demo: Demo,
                         env: OtaCustomRLBenchEnv,
                         episode_keypoints: List[List[int]],
                         viewpoint_agent_bounds:List[float],
                         viewpoint_resolution:List[float],
                         ):

    viewpoints_list = []
    
    for k, keypoints in enumerate(episode_keypoints):
        obs_tp0_1 = demo[keypoints[0]]
        obs_tp0_1_dict = env.extract_obs(obs_tp0_1, t=k, )
        vp_world_pos_tp0_1 = obs_tp0_1_dict["active_cam_pose"][:3]
        disc_vp_world_spher_tp0_1,_ = utils.world_cart_to_disc_world_spher(
            world_cartesion_position=vp_world_pos_tp0_1,
            bounds=viewpoint_agent_bounds,
            spher_res=viewpoint_resolution,
            )
        
        disc_vp_spher  = np.array(viewpoint_agent_bounds[:3]) + np.array(viewpoint_resolution) * disc_vp_world_spher_tp0_1 + np.array(viewpoint_resolution)/2
        #viewpoint_agent_bounds[0] = viewpoint_agent_bounds[3] = 1.27
        disc_vp_spher = np.clip(disc_vp_spher,viewpoint_agent_bounds[:3],viewpoint_agent_bounds[3:])
        _,_,_,vp_world_cart_pose_goal = utils.local_spher_to_world_pose(local_viewpoint_spher_coord=disc_vp_spher)
        
        
        viewpoints_list.append(vp_world_cart_pose_goal[:3])
        
    #viewpoints = np.array(viewpoints_list)
    
    return viewpoints_list
    
    

def _get_demo_entropy(
        demo: Demo,
        env: OtaCustomRLBenchEnv,
        episode_keypoints: List[List[int]],
        cameras: List[str],
        aux_reward:AuxReward):

    prev_action = None
    name = cameras[0]
    
    emtropy_sum = 0
    occupy_sum = 0
    interaction_step = 0
    obs_tp0 = demo[0]
    last_gripper_status = obs_tp0.gripper_open
    for k, keypoints in enumerate(episode_keypoints):
        obs_tp0_1 = demo[keypoints[0]]
        obs_tp1 = demo[keypoints[1]]
        obs_tp0_dict = env.extract_obs(obs_tp0, t=k, )
        obs_tp0_1_dict = env.extract_obs(obs_tp0_1, t=k, )
        obs_tp1_dict = env.extract_obs(obs_tp1, t=k, )
        
        action = obs_tp1_dict["gripper_pose"][:3]
        gripper_status = obs_tp1.gripper_open
        information_gain,roi_entropy_tp0,roi_entropy_tp0_1,occ_ratio_tp0_1 = aux_reward.update_grid(target_point=action[:3],
                                            extrinsics_tp0=obs_tp0_dict['%s_camera_extrinsics' % name],
                                            extrinsics_tp0_1=obs_tp0_1_dict['%s_camera_extrinsics' % name],
                                            depth_tp0=obs_tp0_dict['%s_depth' % name] ,
                                            depth_tp0_1=obs_tp0_1_dict['%s_depth' % name] ,
                                            pc_tp0=obs_tp0_dict['%s_point_cloud' % name] ,
                                            pc_tp0_1=obs_tp0_1_dict['%s_point_cloud' % name] ,
                                            roi_size=0.25)
        #print(roi_entropy_tp0)
        if  gripper_status != last_gripper_status:
            interaction_step += 1
            emtropy_sum += roi_entropy_tp0_1 
            occupy_sum += occ_ratio_tp0_1
            
            
        obs_tp0 = obs_tp1  # Set the next obs
        last_gripper_status = gripper_status
    #entropy_mean = emtropy_sum/len(episode_keypoints)
    return emtropy_sum,occupy_sum,interaction_step

def _add_keypoints_to_replay(
        replay: ReplayBuffer,
        obs_tp0: Observation,
        tp0_step_index: int,
        raw_tp0_1_indices:List,
        #global_obs: Dict,
        demo: Demo,
        env: OtaCustomRLBenchEnv,
        episode_keypoints: List[List[int]],
        cameras: List[str],
        rlbench_scene_bounds: List[float],   # AKA: DEPTH0_BOUNDS
        viewpoint_agent_bounds: List[float], 
        viewpoint_env_bounds: List[float], 
        viewpoint_resolution: List[float], 
        voxel_sizes: List[int],
        bounds_offset: List[float],
        rotation_resolution: int,
        crop_augmentation: bool,
        viewpoint_align:bool,
        aux_reward:AuxReward,
        reach_reward:float=0.02,
        device="cpu"):

    obs_dict_tp0 = env.extract_obs(obs_tp0,)


    
    key = []
    
    for keypoints_index, keypoints in enumerate(episode_keypoints):
        raw_tp0_1_step_index = keypoints[0]
        raw_tp1_step_index = keypoints[1]
        
        tp0_1_step_index = raw_tp0_1_step_index
        tp1_step_index = raw_tp1_step_index
        
        key.append([tp0_step_index,tp0_1_step_index,tp1_step_index])
        
        final_obs = {}

        raw_obs_dict_tp0_1 = env.extract_obs(demo[raw_tp0_1_indices[keypoints_index]])
        obs_dict_tp0_1 = env.extract_obs(demo[tp0_1_step_index])
        obs_dict_tp1 = env.extract_obs(demo[tp1_step_index])
        



        attention_and_gp_pos_indicies, vp_and_gp_rot_indicies, action,attention_inputs, attention_outputs,\
        viewpoints_spher, new_low_dim_states, new_pointclouds, attention_world_pos= _get_action(obs_dict_tp0,obs_dict_tp0_1,obs_dict_tp1, rlbench_scene_bounds,
            viewpoint_agent_bounds,viewpoint_env_bounds,viewpoint_resolution, voxel_sizes, bounds_offset,
            rotation_resolution, crop_augmentation,viewpoint_align,env,device)
        

        terminal = (keypoints_index == len(episode_keypoints) - 1)
        reward = float(terminal) * REWARD_SCALE if terminal else 0        
        

        obs_dict_tp0["active_point_cloud"] = new_pointclouds[0]
        obs_dict_tp0_1["active_point_cloud"] = new_pointclouds[1]

        obs_dict_tp0["low_dim_state"] = new_low_dim_states[0]
        obs_dict_tp0_1["low_dim_state"] = new_low_dim_states[1]
        #utils.show_pc(obs_dict_tp0_1["active_point_cloud"].reshape(3,-1),color=obs_dict_tp0_1['active_rgb'].reshape(3,-1))
        obs_dict_tp0["attention_input"] = attention_inputs[0]
        obs_dict_tp0_1["attention_input"] = attention_inputs[1]
            

        for depth,obs in enumerate([obs_dict_tp0,obs_dict_tp0_1]):  
            for k,v in obs.items():
                if k in SAVE_OBS_KEYS:
                    final_obs[k+'_layer_{}'.format(depth)] = v          

        obs_element = [attention_outputs,viewpoints_spher,vp_and_gp_rot_indicies,attention_and_gp_pos_indicies]
        for index, k in enumerate(SAVE_OBS_ELEMENT_KEYS):
            for depth in range(len(voxel_sizes)):
                final_obs[k+'_layer_{}'.format(depth)] = obs_element[index][depth]
        
        information_gain,roi_entropy_tp0,roi_entropy_tp0_1,occ_ratio_tp0_1 = aux_reward.update_grid(target_point=attention_world_pos,
                                            extrinsics_tp0=obs_dict_tp0["active_camera_extrinsics"],
                                            extrinsics_tp0_1=raw_obs_dict_tp0_1["active_camera_extrinsics"],
                                            depth_tp0=obs_dict_tp0["active_depth"],
                                            depth_tp0_1=raw_obs_dict_tp0_1["active_depth"],
                                            pc_tp0=obs_dict_tp0["active_point_cloud"],
                                            pc_tp0_1=obs_dict_tp0_1["active_point_cloud"],
                                            roi_size=0.25)
        


        focus_reward = information_gain

        if occ_ratio_tp0_1>0.0: #attention_occupy_tp0_1:
            _reach_reward =  reach_reward
        else:
            _reach_reward =  0.0
        
        #_reach_reward += focus_reward
            
        final_obs["reach_reward"] =  np.array([_reach_reward])
        final_obs["focus_reward"] =  np.array([focus_reward])

        
        # 13
        others = {'demo': True}


        others.update(final_obs)
        timeout = False

        replay.add(action, reward, terminal, timeout, **others)
        # Set the next obs
        obs_dict_tp0 = obs_dict_tp1  
        tp0_step_index = tp1_step_index


    obs_dict_tp1.pop('active_world_to_cam', None)

    obs_dict_tp1["active_point_cloud"] = new_pointclouds[2]
    obs_dict_tp1["low_dim_state"] = new_low_dim_states[2]
    


    save_obs_tp1 = {}
    for k,v in obs_dict_tp1.items():
        if k in SAVE_OBS_KEYS:

            for depth in range(len(voxel_sizes)):
                save_obs_tp1[k+'_layer_{}'.format(depth)] = v

    final_obs.update(save_obs_tp1)
    replay.add_final(**final_obs)


def plot_and_save_voxel(step,obs,env):

    obs_dict = env.extract_obs(obs,)

    viewpoint = obs_dict["active_cam_pose"][:3]
    fixationpoint = np.array(TABLE_COORD)
    wpc = obs_dict["active_point_cloud"] 
    active_rgb = obs_dict["active_rgb"] 

    world_bounds = np.array([-0.3, -0.5, 0.6, 0.7, 0.5, 1.6])

    local_bounds=np.array([-0.5, -0.5, -0.05, 0.5, 0.5, 0.95])


    vp_spher = utils.world_cart_to_local_spher(world_cartesion_position=viewpoint,
                                                    world_fixation_position=fixationpoint,
                                                    world_to_local_rotation=None)
        
    cp = utils.world_pc_to_local_pc(world_viewpoint_position=viewpoint,
                                world_fixation_position=fixationpoint,
                                world_points=wpc)
    
    #cp_flat = np.reshape(cp,[3,-1])
    cp_voxel_obj = VoxelGrid(coord_bounds=local_bounds,
                         voxel_size=100,
                         device="cpu",
                         batch_size=1,
                         feature_size=3,
                         max_num_coords=20000
                         )
    wp_voxel_obj = VoxelGrid(coord_bounds=world_bounds,
                         voxel_size=100,
                         device="cpu",
                         batch_size=1,
                         feature_size=3,
                         max_num_coords=20000
                         )
    wp_flat = torch.tensor(wpc[None]).permute(0, 2, 3, 1).reshape(1, -1, 3).type(torch.float32)
    cp_flat = torch.tensor(cp[None]).permute(0, 2, 3, 1).reshape(1, -1, 3).type(torch.float32)
    active_rgb = torch.tensor(active_rgb[None]).permute(0, 2, 3, 1).reshape(1, -1, 3)/255
    #cp
    wp_voxel = wp_voxel_obj.coords_to_bounding_voxel_grid(coords=wp_flat,
                                                          coord_features=active_rgb,
                                                          )

    cp_voxel = cp_voxel_obj.coords_to_bounding_voxel_grid(coords=cp_flat,
                                                          coord_features=active_rgb,
                                                          )
    #voxel_grid.permute(0, 4, 1, 2, 3)
    cp_voxel = cp_voxel.numpy()[0].transpose((3,0,1,2))
    wp_voxel = wp_voxel.numpy()[0].transpose((3,0,1,2))

    
    cp_image = utils.visualise_voxel(voxel_grid=cp_voxel,
                                        show_bb=False,
                                        show=False)
    
    wp_image= utils.visualise_voxel(voxel_grid=wp_voxel,
                                        show_bb=False,
                                        show=False)
    
    suc = cv2.imwrite("/data/open_drawer/image/move_{}.png".format(step), cp_image)
    suc = cv2.imwrite("/data/open_drawer/image/static_{}.png".format(step), wp_image)
   
    #print(suc)
    





def fill_replay(replay: ReplayBuffer,
                task: str,
                env: OtaCustomRLBenchEnv,
                num_demos: int,
                viewpoint_augmentation:bool,
                demo_augmentation: bool,
                demo_augmentation_every_n: int,
                cameras: List[str],
                rlbench_scene_bounds: List[float],  # AKA: DEPTH0_BOUNDS
                viewpoint_agent_bounds: List[float], 
                viewpoint_env_bounds: List[float],
                viewpoint_resolution: List[float], 

                voxel_sizes: List[int],
                bounds_offset: List[float],
                rotation_resolution: int,
                crop_augmentation: bool,
                viewpoint_align:bool,
                logdir:str,
                reach_reward:float = 0.02,
                device="cpu"):

    logging.info('Filling replay with demos...')
    
    aux_reward = AuxReward(scene_bound=rlbench_scene_bounds,voxel_size=50,device=device)
    demo_keypoint_num = 0
    demos_entropy = 0
    demos_occupy = 0
    demos_viewpoint_list = []
    
    for d_idx in range(num_demos):

        demo = env.env.get_demos(
            task, 1, variation_number=0, random_selection=False,
            from_episode_number=d_idx)[0]

        episode_keypoints = keypoint_discovery(demo)
        raw_tp0_1_indices = [keypoints[0] for keypoints in episode_keypoints]
        
        
        clip_demo = copy.deepcopy(demo)

        clip_observations = []
        raw_indince = list(range(len(demo)))
        clip_indice = []
        [clip_observations.extend(demo._observations[start+1:end+1]) for start, end in episode_keypoints]
        [clip_indice.extend(raw_indince[start+1:end+1]) for start, end in episode_keypoints]
        clip_demo._observations = clip_observations
        #[print(step,obs.stage) for step,obs in enumerate(clip_observations)]
        clip_keypoints = demo_loading_utils.keypoint_discovery(clip_demo)

        clip_tp1_keypoints =  [clip_indice[keypoint] for keypoint in clip_keypoints]
        episode_keypoints = [[keypoints[0],tp0_1]  for keypoints,tp0_1 in  zip(episode_keypoints,clip_tp1_keypoints)]
        #assert  sum([(pair[0] > pair[1]) for pair in episode_keypoints]) == 0


        entropy, occupy, step = _get_demo_entropy(demo,env,episode_keypoints,cameras,aux_reward)
        demos_entropy += entropy
        demos_occupy += occupy
        demo_keypoint_num += step
        demos_viewpoint_list.extend(_get_demo_viewpoints(demo,env,episode_keypoints,viewpoint_agent_bounds,viewpoint_resolution))


        for tp0_index in range(len(demo) - 1):
            if not demo_augmentation and tp0_index > 0 :
                break
            if not viewpoint_augmentation and tp0_index != episode_keypoints[0][0]:
                continue

            # If our starting point is past one of the keypoints, then remove it
            while len(episode_keypoints) > 0 and tp0_index >= episode_keypoints[0][0]:

                episode_keypoints = episode_keypoints[1:]
            if len(episode_keypoints) == 0:
                break
            
            if  viewpoint_augmentation:

                if tp0_index % demo_augmentation_every_n != 0  and  tp0_index != episode_keypoints[0][0]:
                    continue

                if tp0_index >= episode_keypoints[0][0] and tp0_index < episode_keypoints[0][1]:
                    continue
                
            obs_tp0 = demo[tp0_index]
            

            augmented_episode_keypoints = copy.deepcopy(episode_keypoints)
            for tp0_1_index in range(episode_keypoints[0][0],episode_keypoints[0][1]):
                # 
                if not demo_augmentation and tp0_1_index > episode_keypoints[0][0]:
                    break

                if tp0_1_index % demo_augmentation_every_n != 0 and tp0_1_index != episode_keypoints[0][0]:
                    continue
                #print(tp0_1_index)
                augmented_episode_keypoints[0][0] = tp0_1_index
                _add_keypoints_to_replay(
                    replay, obs_tp0, tp0_index, raw_tp0_1_indices,demo, env, augmented_episode_keypoints, cameras,
                    rlbench_scene_bounds, viewpoint_agent_bounds,viewpoint_env_bounds,viewpoint_resolution,
                    voxel_sizes, bounds_offset,
                    rotation_resolution, crop_augmentation,viewpoint_align,aux_reward,reach_reward,device)
    
    
    #demos_viewpoints = np.array(demos_viewpoint_list)

    #base_path  = "/home/ubuntu/code/ARM/tools/viewpoints/"
    #task_name =  logdir.split("/")[4]
    #demo_viewpoints_file = os.path.join(base_path,task_name,"demo_viewpoints.csv")
    #demo_viewpoints_file = os.path.join(os.path.dirname(logdir),"demo_viewpoints.csv")
    #np.savetxt(demo_viewpoints_file, demos_viewpoints, delimiter=",")
    
    entropy_mean = demos_entropy/demo_keypoint_num
    occupy_mean = demos_occupy/demo_keypoint_num
    print(task,cameras[0],"entropy",entropy_mean, "occupy",occupy_mean)
    
    
    logging.info('Replay filled with {} initial transitions.'.format(replay.add_count.item()))


def get_viewpoint_bounds(cfg: DictConfig):
    if cfg.method.name == "OTA" or cfg.method.name == "BCA":
        if cfg.method.floating_cam:
            return    cfg.method.floating_viewpoint_agent_bounds,cfg.method.floating_viewpoint_env_bounds,cfg.method.floating_viewpoint_resolution
        else:
            return    cfg.method.arm_viewpoint_agent_bounds,cfg.method.arm_viewpoint_env_bounds,cfg.method.arm_viewpoint_resolution
    else:
            return    [1.3,20,-135, 1.3,60,135] , [1.3,20,-135, 1.3,60,135], [0.05,10.0,10.0]
        



def create_agent(cfg: DictConfig, env,viewpoint_agent_bounds,
                  viewpoint_env_bounds,viewpoint_resolution,
                  depth_0bounds=None, cam_resolution=None):

    VOXEL_FEATS = 3
    LATENT_SIZE = 64

    depth_0bounds = depth_0bounds or [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]
    cam_resolution = cam_resolution or [128, 128]

    include_prev_layer = False


    num_rotation_classes = int(360. // cfg.method.rotation_resolution)

    

    viewpoint_categories_xyz_axes = np.array(
        (np.array(viewpoint_agent_bounds)[3:] - 
         np.array(viewpoint_agent_bounds)[:3])//np.array(viewpoint_resolution)).astype(int)
    viewpoint_axes_movable = viewpoint_categories_xyz_axes>0
    viewpoint_categories_num = int(viewpoint_categories_xyz_axes.sum())
    
    # gripper tip state + gripper trans + gripper quat + cam vp class 
    low_dim_state_size = 3 + 3 + 4 + int(viewpoint_axes_movable.sum()) + int(env._time_in_state)

    # 
    qattention_agents = []
    
    for depth, vox_size in enumerate(cfg.method.voxel_sizes):

        last = depth == len(cfg.method.voxel_sizes) - 1

        unet3d = Qattention3DNet(
            in_channels=VOXEL_FEATS + 3 + 1 + 3,
            out_channels=1, 
            voxel_size=vox_size, 
            out_dense=((num_rotation_classes * 3) + 2) if last else viewpoint_categories_num,
            kernels=LATENT_SIZE,
            norm=None if 'None' in cfg.method.norm else cfg.method.norm,
            dense_feats=128, 
            activation=cfg.method.activation,
            low_dim_size=low_dim_state_size,
            include_prev_layer=include_prev_layer and depth > 0,
            use_vae= cfg.method.use_vae and depth == 0)


        qattention_agent = QAttentionAgent(
            layer=depth,
            layer_num=len(cfg.method.voxel_sizes),
            coordinate_bounds=depth_0bounds,
            viewpoint_agent_bounds=viewpoint_agent_bounds,
            viewpoint_resolution=viewpoint_resolution,
            unet3d=unet3d,
            camera_names=cfg.rlbench.cameras,
            voxel_size=vox_size,
            bounds_offset=cfg.method.bounds_offset[depth - 1] if depth > 0 else None,
            image_crop_size=cfg.method.image_crop_size,
            tau=cfg.method.tau,
            lr=0.0 if (cfg.method.random_viewpoint and depth == 0) else cfg.method.lr,
            lambda_trans_qreg=cfg.method.lambda_trans_qreg,
            lambda_rot_qreg=cfg.method.lambda_rot_qreg,
            include_low_dim_state=True,
            image_resolution=cam_resolution,
            batch_size=cfg.replay.batch_size,
            voxel_feature_size=3,
            exploration_strategy=cfg.method.exploration_strategy,
            lambda_weight_l2=cfg.method.lambda_weight_l2,
            num_rotation_classes=num_rotation_classes,
            rotation_resolution=cfg.method.rotation_resolution,
            grad_clip=0.01,
            gamma=0.99
        )

        qattention_agents.append(qattention_agent)

    rotation_agent = QAttentionStackAgent(
        qattention_agents=qattention_agents,
        rotation_resolution=cfg.method.rotation_resolution,
        viewpoint_agent_bounds=viewpoint_agent_bounds,
        viewpoint_env_bounds=viewpoint_env_bounds,
        viewpoint_resolution=viewpoint_resolution,

        camera_names=cfg.rlbench.cameras,
    )

    preprocess_agent = PreprocessAgent(pose_agent=rotation_agent)
    return preprocess_agent
