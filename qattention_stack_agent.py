
from typing import List

import torch
from yarr.agents.agent import Agent, ActResult, Summary

import numpy as np
from rlbench.backend.const import TABLE_COORD

from arm import utils
from arm.ota.qattention_agent import QAttentionAgent

NAME = 'QAttentionStackAgent'
GAMMA = 0.99
NSTEP = 1
REPLAY_ALPHA = 0.7
REPLAY_BETA = 0.5


class QAttentionStackAgent(Agent):

    def __init__(self,
                 qattention_agents: List[QAttentionAgent],
                 rotation_resolution: float,
                 viewpoint_agent_bounds: List[float],
                 viewpoint_env_bounds: List[float],
                 viewpoint_resolution: List[float],
                 camera_names: List[str],
                 rotation_prediction_depth: int = 0,
                 ):
        super(QAttentionStackAgent, self).__init__()
        self._qattention_agents = qattention_agents
        self._rotation_resolution = rotation_resolution
        self._camera_names = camera_names
        self._rotation_prediction_depth = rotation_prediction_depth
        # r,theta,phi
        self._viewpoint_agent_bounds = np.array(viewpoint_agent_bounds)
        self._viewpoint_resolution = np.array(viewpoint_resolution)
        self._viewpoint_env_bounds = np.array(viewpoint_env_bounds)

    def build(self, training: bool, device=None) -> None:
        self._device = device
        if self._device is None:
            self._device = torch.device('cpu')
        for qa in self._qattention_agents:
            qa.build(training, device)
            
        self._act_depth = 0
        #self._observation_elements = {}
        self._translation_results = []
        self._rot_grip_results = [] 
        self._infos = {}
        

    def update(self, step: int, replay_sample: dict) -> dict:
        priorities = 0
        for qa in self._qattention_agents:
            update_dict = qa.update(step, replay_sample)
            priorities += update_dict['priority']
            replay_sample.update(update_dict)
            
        return {
            'priority': (priorities) ** REPLAY_ALPHA,
        }
        
    def reset(self):
        self._act_depth = 0
        #self._observation_elements = {}
        self._translation_results = []
        self._infos = {}
        
    def act(self, step: int, observation: dict,
            deterministic=False) -> ActResult:

        observation_elements = {}
        infos = {}
        #translation_results = []
        
        act_results = self._qattention_agents[self._act_depth].\
            act(step, observation, deterministic)
            

        attention_output = act_results.observation_elements['attention_output'].cpu().numpy()
        observation_elements['attention_output'] = attention_output[0]
        
        viewpoint_coordinate = act_results.observation_elements['viewpoint_coordinate'].cpu().numpy()
        observation_elements['viewpoint_coordinate'] = viewpoint_coordinate[0]
        

        translation_idxs, rot_grip_idxs = act_results.action
        observation_elements['translation_idxs'] = translation_idxs[0].cpu().numpy()



        observation_elements['prev_layer_voxel_grid'] \
            = act_results.observation_elements['prev_layer_voxel_grid'][0].cpu().numpy()
        

        infos.update(act_results.info)
        
            
        rgai = rot_grip_idxs[0].cpu().numpy() if rot_grip_idxs is not None else np.array([],dtype=np.int32)

        observation_elements['rot_grip_action_indicies'] = rgai
        

        if self._act_depth < len(self._qattention_agents)-1:

            reference_fixation = np.array([0,0,0])
            continuous_action = np.concatenate([observation_elements['viewpoint_coordinate'],
                                                reference_fixation],axis=-1)
        else:
            # 
            continuous_action = np.concatenate([
                observation_elements['attention_output'],
                utils.discrete_euler_to_quaternion(rgai[:-1], self._rotation_resolution),
                rgai[-1:]])


        if self._act_depth < len(self._qattention_agents)-1:
            self._act_depth += 1
        else:
            self._act_depth = 0
            
        return ActResult(

            continuous_action,

            observation_elements=observation_elements,
            info=self._infos
        )
        
        

    def update_summaries(self) -> List[Summary]:
        summaries = []
        for qa in self._qattention_agents:
            summaries.extend(qa.update_summaries())
        return summaries

    def act_summaries(self) -> List[Summary]:
        s = []
        for qa in self._qattention_agents:
            s.extend(qa.act_summaries())
        return s

    def load_weights(self, savedir: str):
        for qa in self._qattention_agents:
            qa.load_weights(savedir)

    def save_weights(self, savedir: str):
        for qa in self._qattention_agents:
            qa.save_weights(savedir)
