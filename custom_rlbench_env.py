from typing import Type, List,Union,Dict

import numpy as np
import copy 
from scipy.spatial.transform import Rotation as R
from rlbench import ObservationConfig, ActionMode
from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.observation import Observation
from rlbench.backend.task import Task
from rlbench.backend.const import *

from yarr.agents.agent import ActResult, VideoSummary,ScalarSummary
from yarr.envs.rlbench_env import RLBenchEnv
from yarr.utils.observation_type import ObservationElement
from yarr.utils.transition import Transition

from pyrep.const import RenderMode
from pyrep.errors import IKError, ConfigurationPathError
from pyrep.objects import VisionSensor, Dummy, ProximitySensor

from arm import utils


RECORD_EVERY = 20


class OtaCustomRLBenchEnv(RLBenchEnv):

    def __init__(self,
                 task_class: Type[Task],
                 observation_config: ObservationConfig,
                 viewpoint_env_bounds:list,
                 viewpoint_agent_bounds:list,
                 viewpoint_resolution:list,
                 action_mode: ActionMode,
                 episode_length: int,
                 dataset_root: str = '',
                 channels_last: bool = False,
                 reward_scale=100.0,
                 headless: bool = True,
                 floating_cam:bool = False,
                 robot_setup: str = 'panda,ur5_blind',
                 time_in_state: bool = False,
                 low_dim_size: int = None,
                 ):
        super(OtaCustomRLBenchEnv, self).__init__(
            task_class, observation_config, action_mode, dataset_root,robot_setup,
            channels_last, headless=headless,floating_cam=floating_cam)
        self._observation_config = observation_config
        self._robot_setup = robot_setup
        self._reward_scale = reward_scale
        self._episode_index = 0
        self._record_current_episode = False
        self._record_cam = None
        self._previous_obs, self._previous_obs_dict = None, None
        self._recorded_images = []
        self._episode_length = episode_length
        self._time_in_state = time_in_state
        self._i = 0
        self._record_success = True
        self._low_dim_size = low_dim_size
        self._floating_cam = floating_cam
        self._viewpoint_resolution = viewpoint_resolution
        self._viewpoint_env_bounds = viewpoint_env_bounds
        self._viewpoint_agent_bounds = viewpoint_agent_bounds
        self._viewpoint_categories_xyz_axes = np.array(
                (np.array(viewpoint_agent_bounds)[3:] - 
                np.array(viewpoint_agent_bounds)[:3])//np.array(viewpoint_resolution)).astype(int)


        #self._floating_cam_placeholder = Dummy('floating_cam_placeholder')

    def show_virtual_cam(self,pose:np.ndarray):
        self._task._scene.show_virtual_cam(pose)


    @property
    def observation_elements(self) -> List[ObservationElement]:
        obs_elems = super(OtaCustomRLBenchEnv, self).observation_elements
        for oe in obs_elems:
            if oe.name == 'low_dim_state':
                robot_setup = list(filter(None,self._robot_setup.lower().split(',')))
                #dual_arm = int(len(robot_setup)==2)
                #oe.shape = (oe.shape[0] - 7 * 2 + dual_arm*7 + int(self._time_in_state),)  # remove pose and joint velocities as they will not be included
                oe.shape = (oe.shape[0] - 7 * 2  + int(self._time_in_state),)  # remove pose and joint velocities as they will not be included
                self.low_dim_state_len = oe.shape[0]
                if self._low_dim_size is not None:
                    self.low_dim_state_len = self._low_dim_size + int(self._time_in_state)
                    self._low_dim_size = self._low_dim_size + int(self._time_in_state)
                    
        obs_elems.append(ObservationElement('gripper_pose', (7,), np.float32))
        return obs_elems

    def extract_obs(self, obs: Observation, t=None, prev_action=None):
        obs.joint_velocities = None
        grip_mat = obs.gripper_matrix
        grip_pose = obs.gripper_pose
        joint_pos = obs.joint_positions
        # obs.gripper_pose = None
        obs.gripper_matrix = None
        obs.active_camera_matrix = None
        obs.joint_positions = None
        if obs.gripper_joint_positions is not None:
            obs.gripper_joint_positions = np.clip(
                obs.gripper_joint_positions, 0., 0.04)

        obs_dict = super(OtaCustomRLBenchEnv, self).extract_obs(obs)

        if self._time_in_state:
            time = (1. - ((self._i if t is None else t) / float(
                self._episode_length - 1))) * 2. - 1.
            obs_dict['low_dim_state'] = np.concatenate(
                [obs_dict['low_dim_state'], [time]]).astype(np.float32)

        obs.gripper_matrix = grip_mat
        # obs.gripper_pose = grip_pose
        obs.joint_positions = joint_pos

        obs_dict['gripper_pose'] = grip_pose
        if obs.active_cam_pose is not None:
            obs_dict['active_cam_pose'] = obs.active_cam_pose
        

        obs_dict = self.reset_raw_obs_world_frame(obs_dict,TABLE_COORD)

        return obs_dict

    def launch(self):
        super(OtaCustomRLBenchEnv, self).launch()
        self._interact_sensor = ProximitySensor("interaction_sensor")
        self._task._scene.register_step_callback(self._my_callback)
        self._task_interactable_objs = self._task._task.get_interactable_objects()
        self._debug_point = Dummy.create(size=0.05)
        if self.eval:
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            cam_base = Dummy('cam_cinematic_base')
            cam_base.rotate([0, 0, np.pi * 0.75])
            self._record_cam = VisionSensor.create([320, 180])
            self._record_cam.set_explicit_handling(True)
            self._record_cam.set_pose(cam_placeholder.get_pose())
            self._record_cam.set_render_mode(RenderMode.OPENGL)
    
    def check_interaction(self,target_pos:Union[list, np.ndarray])->bool:
        if isinstance(target_pos, np.ndarray):
            assert target_pos.shape[0] == 3        # "The numpy array must have a shape of (3,)."
        elif isinstance(target_pos, list):
            assert len(target_pos) == 3           # "The list must have exactly 3 elements."
        else:
            raise TypeError("target_pos must be either a numpy array or a list.")            

        self._interact_sensor.set_position(target_pos)
        interaction_flag = any(self._interact_sensor.is_detected(obj) for obj in self._task_interactable_objs)
        return interaction_flag
        
        

    def reset(self) -> dict:
        self._i = 0
        self._previous_obs_dict = super(OtaCustomRLBenchEnv, self).reset()
        self._record_current_episode = (
                self.eval and self._episode_index % RECORD_EVERY == 0)
        self._episode_index += 1
        self._recorded_images.clear()
        if self._observation_config.active_camera.rgb:
            self._previous_obs_dict = self._reset_cam_pose(random=False)
        
        return self._previous_obs_dict
    
    def _reset_cam_pose(self,random=False):

        if random:
            init_viewpoint = np.random.uniform(self._viewpoint_env_bounds[:3],self._viewpoint_env_bounds[3:])
        else:
            init_r = (self._viewpoint_env_bounds[0]+self._viewpoint_env_bounds[3])/2
            init_theta = (self._viewpoint_env_bounds[1]+self._viewpoint_env_bounds[4])/2
            init_phi = 0.0
            init_viewpoint = np.array([init_r,init_theta,init_phi])
        

        _,_,_,init_viewpoint_pose = utils.local_spher_to_world_pose(init_viewpoint)
        

        if self._floating_cam:
            transition = self.step_vision(init_viewpoint_pose)

        else:
            transition = self.step_vision(init_viewpoint_pose)
        
        assert not transition.terminal
        init_obs = transition.observation
        
        return init_obs
            
    

    def register_callback(self, func):
        self._task._scene.register_step_callback(func)

    def _my_callback(self):
        if self._record_current_episode:
            self._record_cam.handle_explicitly()
            cap = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
            self._recorded_images.append(cap)

    def _append_final_frame(self, success: bool):
        self._record_cam.handle_explicitly()
        img = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
        self._recorded_images.append(img)
        final_frames = np.zeros((10, ) + img.shape[:2] + (3,), dtype=np.uint8)
        # Green/red for success/failure
        final_frames[:, :, :, 1 if success else 0] = 255
        self._recorded_images.extend(list(final_frames))
        
    def show_point(self,pose:np.ndarray):
        shape = pose.shape[-1]
        if shape == 7:
            self._debug_point.set_pose(pose=pose)
        elif shape == 8:
            self._debug_point.set_pose(pose[:7])
        elif shape == 6:
            self._debug_point.set_position(pose[:3])
            self._debug_point.set_orientation(pose[3:])
        elif shape == 3:
            self._debug_point.set_position(pose[:3])
        else:
            raise TypeError('pose error')
        self._task._scene.step()
        #self._debug_point.scale_object()
        #pass
        


    def step(self,act_result:Union[np.ndarray,ActResult],role:str="worker",final:bool=False,eval:bool=False)->  Transition:
        assert role in ["worker","vision"]
        action = act_result.action if type(act_result) == ActResult else  act_result 

        if role=="worker":
            transition = self.step_worker(gripper_pose=action,final=final)
        elif role=="vision":
            transition = self.step_vision(viewpoint=action,final=final,eval=eval)
    
        return transition

    def step_worker(self, gripper_pose: np.ndarray,final:bool=False) -> Transition:
        gripper_pose = self.reset_raw_action_world_frame(gripper_pose,TABLE_COORD)
        action = {'vision_arm':None,'worker_arm':gripper_pose}
        self.show_point(gripper_pose)
        success = False
        obs = self._previous_obs_dict  # in case action fails.
        terminal = False
        reward = 0.0

        if not final:
            try:

                obs, reward, terminal = self._task.step(action)
                if reward >= 1:
                    success = True
                    reward *= self._reward_scale
                else:
                    reward = 0.0

                obs = self.extract_obs(obs)
                self._previous_obs_dict = obs

            except (IKError, ConfigurationPathError, InvalidActionError) as e:
                terminal = True
                reward = 0.0

        summaries = []
        self._i += 1
        if ((terminal or self._i == self._episode_length) and self._record_current_episode):
            self._append_final_frame(success)
            vid = np.array(self._recorded_images).transpose((0, 3, 1, 2))
            summaries.append(VideoSummary(
                'episode_rollout_' + ('success' if success else 'fail'),
                vid, fps=30))
            self._record_success = True
                        
            
        return Transition(obs, reward, terminal, summaries=summaries)


    def step_vision(self, viewpoint: np.ndarray,final:bool=False,eval:bool=False) -> Transition:
        viewpoint = self.reset_raw_action_world_frame(viewpoint,TABLE_COORD)
        action = {'vision_arm':viewpoint,'worker_arm':None}
        success = False
        obs = self._previous_obs_dict  # in case action fails.
        terminal = False

        if not final:
            try:
                if not self._rlbench_env._floating_cam:

                    obs, reward, terminal = self._task.step(action)
                else:
                    obs, reward, terminal = self._task.step_active_cam(action,eval)
                    self.show_virtual_cam(viewpoint)
                    
                if reward >= 1:
                    success = True
                    reward *= self._reward_scale
                else:
                    reward = 0.0

                obs = self.extract_obs(obs)
                self._previous_obs_dict = obs

            except (IKError, ConfigurationPathError, InvalidActionError) as e:
                terminal = True
                reward = 0.0

        summaries = []
        reward = 0
        return Transition(obs, reward, terminal, summaries=summaries)

    def reset_to_demo(self, i):
        d, = self._task.get_demos(
            1, live_demos=False, random_selection=False, from_episode_number=i)
        self._task.reset_to_demo(d)


    def reset_raw_obs_world_frame(self,observations:dict,trans:any):
        new_observations = copy.deepcopy(observations)
        for key in new_observations.keys():
            if "pose" in key:
                new_observations[key][:3] -= np.array(trans)
            elif "point_cloud" in key:
                new_observations[key] -= np.array(trans)[:,None,None]
            elif "camera_extrinsics" in key:
                new_observations[key][:3,3] -= np.array(trans)
            elif "low_dim_state" in key:
                new_observations[key][1:4] -= np.array(trans)
                
                

        return new_observations


    def reset_raw_action_world_frame(self,action:np.ndarray,trans:any):
        new_action = copy.deepcopy(action)
        new_action[:3] += np.array(trans)
        return new_action