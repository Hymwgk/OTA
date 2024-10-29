from multiprocessing import Value
import copy
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt

from rlbench.backend.const import TABLE_COORD

from yarr.agents.agent import Agent
from yarr.envs.env import Env
from yarr.utils.transition import ReplayTransition

from arm import utils
from arm.ota.const import SAVE_OBS_KEYS,SAVE_OBS_ELEMENT_KEYS
from arm.ota.voxel_grid import VoxelGrid
from arm.ota.aux_task.aux_reward import AuxReward


class OtaRolloutGenerator(object):
    
    def __init__(self,
                 scene_bounds,
                 viewpoint_agent_bounds=None,
                 viewpoint_resolution=None,
                 viewpoint_env_bounds=None,
                 viewpoint_align=True,
                 reach_reward = 0.02,
                 rollout=False,
                 device="cpu"):
        self._scene_bounds = np.array(scene_bounds)
        self._device = device
        self._init_global_env_obs = None
        self._viewpoint_agent_bounds = np.array(viewpoint_agent_bounds)
        self._viewpoint_env_bounds=np.array(viewpoint_env_bounds)
        self._viewpoint_align = viewpoint_align
        self._reach_reward = reach_reward
        self._rollout = rollout

        viewpoint_categories_xyz_axes= np.array(
            (np.array(self._viewpoint_agent_bounds)[3:] - 
            np.array(self._viewpoint_agent_bounds)[:3])//np.array(viewpoint_resolution)).astype(int)
        self._viewpoint_axes_movable = viewpoint_categories_xyz_axes>0
        self._viewpoint_categories_num = int(self._viewpoint_axes_movable.sum())
        self._init_obs_dict_tp0 = None

        # self._scene_voxels = VoxelGrid(coord_bounds=scene_bounds,
        #                     voxel_size= 100,
        #                     device=self._device,
        #                     batch_size=1,
        #                     feature_size=0,
        #                     max_num_coords=1000000,
        #                     )
        
        self._aux_reward = AuxReward(scene_bound=scene_bounds,
                                       voxel_size=50,
                                       device=self._device)

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    def act_and_execute(self,step_signal: Value, env: Env, agent: Agent,
                  timesteps: int, eval: bool,obs_dict_tp0:dict, final:bool=False,viewpoint_align:bool=True):
        
        nbv_input_obs_dict,nbp_input_obs_dict = {},{}
        for obs_key in SAVE_OBS_KEYS:
            nbv_input_obs_dict[obs_key],nbp_input_obs_dict[obs_key] = [],[]
            

        for step_index in range(timesteps):
            gp_world_pos_tp0 = obs_dict_tp0["gripper_pose"][step_index][:3]
            gp_world_quat_tp0 = utils.normalize_quaternion(obs_dict_tp0["gripper_pose"][step_index][3:])
            vp_world_cart_pos_tp0 = obs_dict_tp0["active_cam_pose"][step_index][:3]
            time_state_tp0 = obs_dict_tp0["low_dim_state"][step_index][[-1]] if env._time_in_state else np.array([])
            gp_tip_state_tp0 = np.concatenate([obs_dict_tp0["low_dim_state"][step_index][[0]],obs_dict_tp0["low_dim_state"][step_index][8:10]])
            pc_world_tp0 = obs_dict_tp0["active_point_cloud"][step_index]
            #utils.show_pc(pc_world_tp0.reshape(3,-1),
            #              gripper_pose=np.concatenate([gp_world_pos_tp0,gp_world_quat_tp0],-1),
            #              color=obs_dict_tp0['active_rgb'][0].reshape(3,-1))
            
            if gp_world_quat_tp0[-1] < 0:
                gp_world_quat_tp0 = -gp_world_quat_tp0

            _,vp_world_spher_tp0 = utils.world_cart_to_disc_world_spher(
                world_cartesion_position=vp_world_cart_pos_tp0)
            
            
            norm_vp_world_spher_tp0 = utils.viewpoint_normlize(vp_world_spher_tp0,self._viewpoint_env_bounds)
            low_dim_state_world_tp0 = np.concatenate([gp_tip_state_tp0,
                                                gp_world_pos_tp0,
                                                gp_world_quat_tp0,
                                                norm_vp_world_spher_tp0[self._viewpoint_axes_movable],
                                                time_state_tp0],axis=-1)
            
            
            pc_tp0 = pc_world_tp0
            low_dim_state_tp0 = low_dim_state_world_tp0
            vp_local_cart_pos_tp0 = vp_world_cart_pos_tp0
            
            if viewpoint_align:

                wtl_rot_tp0 = utils.world_to_local_rotation(world_viewpoint_position=vp_world_cart_pos_tp0)
                # 
                gp_local_pos_tp0,gp_local_quat_tp0,gp_local_euler_tp0 = utils.world_pose_to_local_pose(world_to_local_rotation=wtl_rot_tp0,
                                                                                    world_point_position=gp_world_pos_tp0,
                                                                                    world_point_quat=gp_world_quat_tp0)
                # 
                vp_local_cart_pos_tp0,_,_ = utils.world_pose_to_local_pose(world_to_local_rotation=wtl_rot_tp0,
                                                                                    world_point_position=vp_world_cart_pos_tp0,
                                                                                    world_point_quat=gp_world_quat_tp0)
                                

                low_dim_state_local_tp0 = np.concatenate([gp_tip_state_tp0,
                                                    gp_local_pos_tp0,
                                                    gp_local_quat_tp0,
                                                    norm_vp_world_spher_tp0[self._viewpoint_axes_movable],
                                                    time_state_tp0],axis=-1)
                #utils.show_pc(pc_world_tp0.reshape(3,-1),
                #            gripper_pose=np.concatenate([gp_world_pos_tp0,gp_world_quat_tp0],-1),
                #            color=obs_dict_tp0['active_rgb'][0].reshape(3,-1))


                pc_local_tp0 = utils.world_pc_to_local_pc(world_viewpoint_position=vp_world_cart_pos_tp0,
                                            world_points=pc_world_tp0)

                #utils.show_pc(pc_local_tp0.reshape(3,-1),
                #              gripper_pose=np.concatenate([gp_local_pos_tp0,gp_local_quat_tp0],-1),
                #              color=obs_dict_tp0['active_rgb'][0].reshape(3,-1))

                pc_tp0 = pc_local_tp0
                low_dim_state_tp0 = low_dim_state_local_tp0
            
            

            nbv_input_obs_dict["active_point_cloud"].append(pc_tp0.astype(
                obs_dict_tp0["active_point_cloud"][step_index].dtype))

            nbv_input_obs_dict["low_dim_state"].append(low_dim_state_tp0.astype(
                obs_dict_tp0["low_dim_state"][step_index].dtype))

            nbv_input_obs_dict["active_rgb"].append(obs_dict_tp0["active_rgb"][step_index])

            nbv_input_obs_dict["attention_input"].append(np.zeros((3,),dtype=np.float32))               
                  

        prepped_data = {k:torch.tensor(np.array(v)[None], device=self._device) 
                        for k, v in nbv_input_obs_dict.items()}
        # nbv agent 
        nbv_output =agent.act(step=step_signal.value, 
                              observation=prepped_data, 
                              deterministic=eval)


        nbv_agent_obs_elems = {k: np.array(v) for k, v in nbv_output.observation_elements.items()}
        nbv_out_save = {k+'_layer_0':v for k,v in nbv_agent_obs_elems.items() if k in SAVE_OBS_ELEMENT_KEYS}
        #print(nbv_agent_obs_elems['rot_grip_action_indicies'][0])

        
        vp_world_spher_goal = vp_local_spher_goal = nbv_output.action[:3]
        
        #  inverse-align
        if viewpoint_align:
            vp_world_spher_goal = np.copy(vp_world_spher_tp0)
            vp_world_spher_goal[1] = vp_local_spher_goal[1]
            vp_world_spher_goal[2] += vp_local_spher_goal[2]
        

        vp_world_spher_goal = np.clip(vp_world_spher_goal,self._viewpoint_env_bounds[:3],self._viewpoint_env_bounds[3:])

        _,_,_,vp_world_cart_pose_goal = utils.local_spher_to_world_pose(local_viewpoint_spher_coord=vp_world_spher_goal)
        

        attention_local_pos_goal =  nbv_agent_obs_elems["attention_output"]
        # add fake orientation
        attention_local_pose_goal = np.concatenate([attention_local_pos_goal , np.array([0,0,0,1])],-1)

        #utils.show_pc(pc_local_tp0.reshape(3,-1),
        #              gripper_pose=attention_local_pose_goal,
        #              color=obs_dict_tp0['active_rgb'][0].reshape(3,-1))
        

        attention_world_pos_goal,_ = utils.local_pose_to_world_pose(local_point_position=attention_local_pose_goal[:3],
                                                                local_point_quat=attention_local_pose_goal[3:],
                                                                world_viewpoint_position=vp_world_cart_pos_tp0)
        attention_world_pos_goal = np.clip(attention_world_pos_goal,self._scene_bounds[:3],self._scene_bounds[3:])
        # =========================================
        if eval:
            init_vp = self._init_obs_dict_tp0["active_cam_pose"][0]
            if self._rollout:
                init_vp_obs_dict = self._init_obs_dict_tp0
            else:
                init_vp_transition = env.step(init_vp,'vision',final)
                init_vp_obs_dict = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in init_vp_transition.observation.items()}
            
        
                    
        # =========================================

        tp0_1_transition = env.step(vp_world_cart_pose_goal,'vision',final,eval)
        obs_dict_tp0_1 = tp0_1_transition.observation
        obs_dict_tp0_1['attention_input'] =  attention_world_pos_goal
        env.show_point(attention_world_pos_goal+TABLE_COORD)

        obs_dict_tp0_1 = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs_dict_tp0_1.items()}

        for step_index in range(timesteps):
            gp_world_pos_tp0_1 = obs_dict_tp0_1["gripper_pose"][step_index][:3]
            gp_world_quat_tp0_1 = utils.normalize_quaternion(obs_dict_tp0_1["gripper_pose"][step_index][3:])
            vp_world_cart_pos_tp0_1 = obs_dict_tp0_1["active_cam_pose"][step_index][:3]
            time_state_tp0_1 = obs_dict_tp0_1["low_dim_state"][step_index][[-1]] if env._time_in_state else np.array([])
            gp_tip_state_tp0_1 = obs_dict_tp0_1["low_dim_state"][step_index][:3]
            pc_world_tp0_1 = obs_dict_tp0_1["active_point_cloud"][step_index]
            attention_world_pos_tp0_1 = obs_dict_tp0_1['attention_input'][step_index]
            
            
            if gp_world_quat_tp0_1[-1] < 0:
                gp_world_quat_tp0_1 = -gp_world_quat_tp0_1

            _,vp_world_spher_tp0_1 = utils.world_cart_to_disc_world_spher(
                world_cartesion_position=vp_world_cart_pos_tp0_1)


            norm_vp_world_spher_tp0_1 = utils.viewpoint_normlize(vp_world_spher_tp0_1,self._viewpoint_env_bounds)
            low_dim_state_world_tp0_1 = np.concatenate([gp_tip_state_tp0_1,
                                                gp_world_pos_tp0_1,
                                                gp_world_quat_tp0_1,
                                                norm_vp_world_spher_tp0_1[self._viewpoint_axes_movable],
                                                time_state_tp0_1],axis=-1)
            pc_tp0_1 = pc_world_tp0_1
            low_dim_state_tp0_1 = low_dim_state_world_tp0_1
            attention_pos_tp0_1 = attention_world_pos_tp0_1

            if viewpoint_align:
            
                wtl_rot_tp0_1 = utils.world_to_local_rotation(world_viewpoint_position=vp_world_cart_pos_tp0_1)

                gp_local_pos_tp0_1,gp_local_quat_tp0_1,gp_local_euler_tp0_1 = utils.world_pose_to_local_pose(world_to_local_rotation=wtl_rot_tp0_1,
                                                                                    world_point_position=gp_world_pos_tp0_1,
                                                                                    world_point_quat=gp_world_quat_tp0_1)

                low_dim_state_local_tp0_1 = np.concatenate([gp_tip_state_tp0_1,
                                                    gp_local_pos_tp0_1, 
                                                    gp_local_quat_tp0_1,
                                                    norm_vp_world_spher_tp0_1[self._viewpoint_axes_movable],
                                                    time_state_tp0_1],axis=-1)
                #utils.show_pc(pc_world_tp0_1.reshape(3,-1),
                #            gripper_pose=np.concatenate([gp_world_pos_tp0_1,gp_world_quat_tp0_1],-1),
                #            color=obs_dict_tp0_1['active_rgb'][0].reshape(3,-1))


                pc_local_tp0_1 = utils.world_pc_to_local_pc(world_viewpoint_position=vp_world_cart_pos_tp0_1,
                                            world_points=pc_world_tp0_1)

                #utils.show_pc(pc_local_tp0_1.reshape(3,-1),
                #              gripper_pose=np.concatenate([gp_local_pos_tp0_1,gp_local_quat_tp0_1],-1),
                #              color=obs_dict_tp0_1['active_rgb'][0].reshape(3,-1))



                attention_local_pos_tp0_1,_,_ = utils.world_pose_to_local_pose(world_to_local_rotation=wtl_rot_tp0_1,
                                                                world_point_position=attention_world_pos_tp0_1,
                                                                world_point_quat=np.array([0,0,0,1]))
                #utils.show_pc(pc_local_tp0_1.reshape(3,-1),
                #            gripper_pose=attention_local_pos_tp0_1,
                #            color=obs_dict_tp0_1['active_rgb'][0].reshape(3,-1))

                pc_tp0_1 = pc_local_tp0_1
                low_dim_state_tp0_1 = low_dim_state_local_tp0_1
                attention_pos_tp0_1 = attention_local_pos_tp0_1

            

            nbp_input_obs_dict["active_point_cloud"].append(pc_tp0_1.astype(np.float32))
            nbp_input_obs_dict["low_dim_state"].append(low_dim_state_tp0_1.astype(np.float32))
            nbp_input_obs_dict["active_rgb"].append(obs_dict_tp0_1["active_rgb"][step_index])
            

            nbp_input_obs_dict["attention_input"].append(attention_pos_tp0_1.astype(np.float32))
            
        # [B,T, ...]  
        prepped_data = {k:torch.tensor(np.array(v)[None], device=self._device) 
                        for k, v in nbp_input_obs_dict.items()}
        # nbp agent 
        nbp_output = agent.act(step=step_signal.value,
                               observation=prepped_data,
                               deterministic=eval)
        
        #utils.show_pc(pc_local_tp0_1.reshape(3,-1),
        #              gripper_pose= nbp_output.action,
        #              color=obs_dict_tp0_1['active_rgb'][0].reshape(3,-1))



        nbp_agent_obs_elems   = {k: np.array(v) for k, v in  nbp_output.observation_elements.items()}
        extra_replay_elements = {k: np.array(v) for k, v in  nbp_output.replay_elements.items()}


        #attention_occupy_tp0_1 = nbp_agent_obs_elems["prev_layer_voxel_grid"][-1].reshape(-1)==1
        neigborhood = utils.get_neighborhood_indices(index=[7,7,7],m=1,max_shape=[16,16,16])
        attention_occupy_tp0_1 = utils.check_occupancy(nbp_agent_obs_elems["prev_layer_voxel_grid"],
                                                       neigborhood)


        # if attention_occupy_tp0_1:
        #     print("Here")
        

        nbp_out_save = {k+'_layer_1':v for k,v in nbp_agent_obs_elems.items() if k in SAVE_OBS_ELEMENT_KEYS}


        nbp_action = nbp_output.action
        if viewpoint_align:
            nbp_action = utils.local_gripper_action_to_world_action(local_gripper_action=nbp_output.action,
                                                                    world_viewpoint_position=vp_world_cart_pos_tp0_1)

            #utils.show_pc(pc_world_tp0_1.reshape(3,-1),
            #            gripper_pose=nbp_action,
            #            color=obs_dict_tp0_1['active_rgb'][0].reshape(3,-1))


        tp1_transition = env.step(nbp_action,'worker',final,eval)
        #print(np.linalg.norm(nbp_action[:3]-attention_world_pos_goal[:3]))
        
        terminal = tp0_1_transition.terminal or tp1_transition.terminal

        nbv_in_save= {k+'_layer_0':v[0] for k,v in nbv_input_obs_dict.items() if k in SAVE_OBS_KEYS} 
        nbp_in_save = {k+'_layer_1':v[0] for k,v in nbp_input_obs_dict.items() if k in SAVE_OBS_KEYS}

        obs_and_replay_elems = {}
        obs_and_replay_elems.update(nbv_in_save)
        obs_and_replay_elems.update(nbv_out_save)
        obs_and_replay_elems.update(nbp_in_save)
        obs_and_replay_elems.update(nbp_out_save)
        obs_and_replay_elems.update(extra_replay_elements)
        
        
        
        
        ################################

        goal_position_interable = env.check_interaction(tp1_transition.observation["gripper_pose"][:3]+np.array(TABLE_COORD)) if eval else None
        
        information_gain,roi_entropy_tp0,roi_entropy_tp0_1, occ_ratio_tp0_1 = self._aux_reward.update_grid(target_point=attention_world_pos_goal if not eval else nbp_action[:3],
                                            extrinsics_tp0=obs_dict_tp0["active_camera_extrinsics"][0],
                                            extrinsics_tp0_1=obs_dict_tp0_1["active_camera_extrinsics"][0],
                                            depth_tp0=obs_dict_tp0["active_depth"][0],
                                            depth_tp0_1=obs_dict_tp0_1["active_depth"][0],
                                            pc_tp0=obs_dict_tp0["active_point_cloud"][0],
                                            pc_tp0_1=obs_dict_tp0_1["active_point_cloud"][0],
                                            roi_size=0.25)
        if eval:
            init_information_gain,roi_entropy_init,_, _ = self._aux_reward.update_grid(target_point=attention_world_pos_goal if not eval else nbp_action[:3],
                                                extrinsics_tp0=init_vp_obs_dict["active_camera_extrinsics"][0],
                                                extrinsics_tp0_1=obs_dict_tp0_1["active_camera_extrinsics"][0],
                                                depth_tp0=init_vp_obs_dict["active_depth"][0],
                                                depth_tp0_1=obs_dict_tp0_1["active_depth"][0],
                                                pc_tp0=init_vp_obs_dict["active_point_cloud"][0],
                                                pc_tp0_1=obs_dict_tp0_1["active_point_cloud"][0],
                                                roi_size=0.25)
            tp1_transition.info["init_information_gain"] = init_information_gain/roi_entropy_init
            

        gripper_action_result = tp1_transition.observation["gripper_pose"][:3]
        gripper_pos_goal = attention_world_pos_goal  # attention_world_pos_goal         nbp_action[:3]
        gripper_attention_distance = np.linalg.norm(gripper_pos_goal - gripper_action_result)
        reach_reward = -self._reach_reward
        roi_reachable = gripper_attention_distance <= 0.22
        roi_non_empty = occ_ratio_tp0_1>0.0
        if roi_reachable:
            if roi_non_empty:
                reach_reward = self._reach_reward
            else:
                reach_reward = 0 #self._reach_reward/2
        
        focus_reward = information_gain
        
        #reach_reward += focus_reward    
        
        obs_and_replay_elems.update({"reach_reward":np.array([reach_reward])})
        obs_and_replay_elems.update({"focus_reward":np.array([focus_reward])})

        # print(" information_gain:{} ,tp0_1_attention_occupy:{}  ,final:{},  terminal:{},  gripper_dis:{},  reach:{} ,focus:{}".
        #       format(round(information_gain,5),attention_occupy_tp0_1,final,terminal, np.round(gripper_attention_distance,3),reach_reward,focus_reward))
        
        tp1_transition.info["roi_entropy"] = roi_entropy_tp0_1
        tp1_transition.info["roi_reachable"] = float(roi_reachable)
        tp1_transition.info["roi_non_empty"] = float(roi_non_empty)
        tp1_transition.info["goal_position_interable"] = goal_position_interable
        tp1_transition.info["information_gain"] = information_gain/(roi_entropy_tp0 + (1e-10))
        tp1_transition.info["viewpoint_tp0_1"] =  vp_world_cart_pose_goal[:3]
        
        
        
        # s,a,s',done
        return obs_and_replay_elems, nbp_output.action, tp1_transition, terminal
         

    def generator(self, step_signal: Value, env: Env, agent: Agent,
                  episode_length: int, timesteps: int, eval: bool):

        obs = env.reset()
        agent.reset()

        obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}
        self._init_obs_dict_tp0 = copy.deepcopy(obs_history)
        
        for step in range(episode_length):
            # s,a,s',terminal
            obs_and_replay_elems, tp0_1_action, tp1_transition, terminal = \
                self.act_and_execute(step_signal, env, agent, timesteps, eval,obs_history,False,self._viewpoint_align)            
            
            timeout = False
            if step == episode_length - 1:
                # If last transition, and not terminal, then we timed out
                timeout = not terminal
                if timeout:
                    terminal = True
                    if "needs_reset" in tp1_transition.info:
                        tp1_transition.info["needs_reset"] = True
            

            for k in obs_history.keys():
                obs_history[k].append(tp1_transition.observation[k])
                obs_history[k].pop(0)

            tp1_transition.info["active_task_id"] = env.active_task_id

            replay_transition = ReplayTransition(
                obs_and_replay_elems, tp0_1_action, tp1_transition.reward,
                terminal, timeout, summaries=tp1_transition.summaries,
                info=tp1_transition.info)


            if terminal or timeout:
                # If the agent gives us observations then we need to call act
                # one last time (i.e. acting in the terminal state).
                #debug_obs = copy.deepcopy(obs_and_replay_elems)
                
                obs_and_replay_elems, _, _, _ = self.act_and_execute(step_signal, env, agent,
                                                                     timesteps, eval,obs_history,final=True,viewpoint_align=self._viewpoint_align)
                #obs_tp1 = dict(tp1_transition.observation)           
                obs_and_replay_elems.pop('demo',None)

                replay_transition.final_observation = obs_and_replay_elems 


            yield replay_transition


            if tp1_transition.info.get("needs_reset", tp1_transition.terminal):
                return
