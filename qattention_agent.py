import copy
import logging
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from yarr.agents.agent import Agent, ActResult, ScalarSummary, \
    HistogramSummary, ImageSummary, Summary

from arm import utils
from arm.utils import visualise_voxel, stack_on_channel
from arm.ota.voxel_grid import VoxelGrid

NAME = 'QAttentionAgent'
REPLAY_BETA = 1.0


class QFunction(nn.Module):

    def __init__(self,
                 layer: int,
                 layer_num:int,
                 unet_3d: nn.Module,
                 voxel_grid: VoxelGrid,
                 global_voxel_grid: VoxelGrid,
                 bounds_offset: float,
                 rotation_resolution: float,
                 viewpoint_agent_bounds: list,
                 viewpoint_resolution: list,
                 device):
        super(QFunction, self).__init__()
        self._layer = layer
        self._layer_num = layer_num
        self._rotation_resolution = rotation_resolution
        self._voxel_grid = voxel_grid
        self._global_voxel_grid = global_voxel_grid
        self._bounds_offset = bounds_offset
        self._qnet = copy.deepcopy(unet_3d)
        self._qnet._dev = device
        self._device = device

        viewpoint_categories_xyz_axes = np.array((np.array(viewpoint_agent_bounds)[3:] - np.array(viewpoint_agent_bounds)[:3])//np.array(viewpoint_resolution)).astype(int)

        self._viewpoint_categories_num = int(viewpoint_categories_xyz_axes.sum())
        # 
        self._viewpoint_axis_class_breakpoints = [viewpoint_categories_xyz_axes[0],viewpoint_categories_xyz_axes[0]+viewpoint_categories_xyz_axes[1]]
        
        

        self._qnet.build()

    def _argmax_3d(self, tensor_orig):
        b, c, d, h, w = tensor_orig.shape  # c will be one
        idxs = tensor_orig.view(b, c, -1).argmax(-1)
        indices = torch.cat([((idxs // h) // d), (idxs // h) % w, idxs % w], 1)
        return indices

    def choose_highest_action(self, q_trans, q_rot_grip):
        coords = self._argmax_3d(q_trans)
        rot_and_grip_indicies = None
        

        if q_rot_grip is not None and q_rot_grip.shape[-1]==self._viewpoint_categories_num:
            
            #q_rot = torch.stack(torch.split(q_rot_grip,self._viewpoint_axis_class_breakpoints,dim=1), dim=1)
            axis_0 = q_rot_grip[:, 0:self._viewpoint_axis_class_breakpoints[0]]
            axis_1 = q_rot_grip[:, self._viewpoint_axis_class_breakpoints[0]:self._viewpoint_axis_class_breakpoints[1]]
            axis_2 = q_rot_grip[:, self._viewpoint_axis_class_breakpoints[1]:]
            
            #B,_ = q_rot_grip.shape
            #zeros = torch.zeros([B,1]).to(self._device)
                        
            rot_and_grip_indicies = torch.as_tensor(torch.cat(
                [axis_0.argmax(-1, keepdim=True) if axis_0.shape[-1]!=0 else axis_0,
                 axis_1.argmax(-1, keepdim=True) if axis_1.shape[-1]!=0 else axis_1,
                 axis_2.argmax(-1, keepdim=True) if axis_2.shape[-1]!=0 else axis_2,], -1),dtype=torch.int64)
            
        
        
        elif q_rot_grip is not None and q_rot_grip.shape[-1]%3==2:
            q_rot = torch.stack(torch.split(
                q_rot_grip[:, :-2],
                int(360 // self._rotation_resolution),
                dim=1), dim=1)
            rot_and_grip_indicies = torch.cat(
                [q_rot[:, 0:1].argmax(-1),
                 q_rot[:, 1:2].argmax(-1),
                 q_rot[:, 2:3].argmax(-1),
                 q_rot_grip[:, -2:].argmax(-1, keepdim=True)], -1)
            
        elif q_rot_grip is not None and q_rot_grip.shape[-1]%3==0:
            q_rot = torch.stack(torch.split(
                q_rot_grip,
                int(360 // self._rotation_resolution),
                dim=1), dim=1)
            rot_and_grip_indicies = torch.cat(
                [q_rot[:, 0:1].argmax(-1),
                 q_rot[:, 1:2].argmax(-1),
                 q_rot[:, 2:3].argmax(-1)], -1)
            
        # [b,3] 
        return coords, rot_and_grip_indicies

    def forward(self, x, proprio, pcd,
                bounds=None, latent=None,recon=False):
        # x will be list of list (list of [rgb, pcd])  i.e.   [[rgb,pcd]]
        b = x[0][0].shape[0]
        # [B,w*h,3]
        pcd_flat = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(b, -1, 3) for p in pcd], 1) 

        image_features = [x[0][0]] #[xx[0] for xx in x]
        feat_size = image_features[0].shape[1]
        # [B,w*h,3] 
        flat_imag_features = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(b, -1, feat_size) for p in
             image_features], 1)

        # [b,d,h,w,c]
        voxel_grid = self._voxel_grid.coords_to_bounding_voxel_grid(
            pcd_flat, coord_features=flat_imag_features, coord_bounds=bounds)

        # Swap to channels fist [b,c,d,h,w]
        voxel_grid = voxel_grid.permute(0, 4, 1, 2, 3).detach()

        q_trans, rot_and_grip_q = self._qnet(voxel_grid, proprio[0], latent,recon=recon)
        
        return q_trans, rot_and_grip_q, voxel_grid

    def latents(self):
        return self._qnet.latent_dict


class QAttentionAgent(Agent):

    def __init__(self,
                 layer: int,
                 layer_num:int,
                 coordinate_bounds: list,
                 viewpoint_agent_bounds: list,
                 viewpoint_resolution: list,
                 unet3d: nn.Module,
                 camera_names: list,
                 batch_size: int,
                 voxel_size: int,
                 bounds_offset: float,
                 voxel_feature_size: int,
                 image_crop_size: int,
                 exploration_strategy: str,
                 num_rotation_classes: int,
                 rotation_resolution: float,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 nstep: int = 1,
                 lr: float = 0.0001,
                 lambda_trans_qreg: float = 1e-6,
                 lambda_rot_qreg: float = 1e-6,
                 grad_clip: float = 20.,
                 include_low_dim_state: bool = False,
                 image_resolution: list = None,
                 lambda_weight_l2: float = 0.0,
                 use_aux_reward: bool = True,
                 ):
        self._layer = layer
        self._layer_num = layer_num
        self._lambda_trans_qreg = lambda_trans_qreg
        self._lambda_rot_qreg = lambda_rot_qreg
        self._coordinate_bounds = coordinate_bounds
        self._unet3d = unet3d
        self._voxel_feature_size = voxel_feature_size
        self._bounds_offset = bounds_offset
        self._image_crop_size = image_crop_size
        self._tau = tau
        self._gamma = gamma
        self._nstep = nstep
        self._lr = lr
        self._grad_clip = grad_clip
        self._include_low_dim_state = include_low_dim_state
        self._image_resolution = image_resolution or [128, 128]
        self._voxel_size = voxel_size
        self._camera_names = camera_names
        self._num_cameras = len(camera_names)
        self._batch_size = batch_size
        self._exploration_strategy = exploration_strategy
        self._lambda_weight_l2 = lambda_weight_l2
        self._use_aux_reward = use_aux_reward

        self._num_rotation_classes = num_rotation_classes
        self._rotation_resolution = rotation_resolution

        self._name = NAME + '_layer' + str(self._layer)
        # r,theta,phi
        self._viewpoint_agent_bounds = viewpoint_agent_bounds
        self._viewpoint_resolution = viewpoint_resolution

        self._viewpoint_categories_xyz_axes = np.array((np.array(viewpoint_agent_bounds)[3:] - np.array(viewpoint_agent_bounds)[:3])//np.array(viewpoint_resolution)).astype(int)

        self._viewpoint_axis_class_breakpoints = [self._viewpoint_categories_xyz_axes[0],self._viewpoint_categories_xyz_axes[0]+self._viewpoint_categories_xyz_axes[1]]
        # 
        self._viewpoint_axes_movable = self._viewpoint_categories_xyz_axes>0
                
                
        

    def build(self, training: bool, device: torch.device = None):
        if device is None:
            device = torch.device('cpu')

        vox_grid = VoxelGrid(
            coord_bounds=self._coordinate_bounds,
            voxel_size=self._voxel_size,
            device=device,
            batch_size=self._batch_size if training else 1,
            feature_size=self._voxel_feature_size,
            max_num_coords=np.prod(self._image_resolution) * self._num_cameras,
        )
        global_vox_grid = VoxelGrid(
            coord_bounds=self._coordinate_bounds,
            voxel_size=self._voxel_size,
            device=device,
            batch_size=self._batch_size if training else 1,
            feature_size=self._voxel_feature_size,
            max_num_coords=np.prod(self._image_resolution) * self._num_cameras,
        )
        self._vox_grid = vox_grid
        self._global_voxel_grid = global_vox_grid

        self._q = QFunction(self._layer,self._layer_num,
                            self._unet3d, vox_grid,
                            self._global_voxel_grid,
                            self._bounds_offset,
                            self._rotation_resolution,
                            self._viewpoint_agent_bounds,
                            self._viewpoint_resolution,
                            device).to(device).train(training)
        self._q_target = None
        if training:
            self._q_target = QFunction(self._layer,self._layer_num,
                                       self._unet3d, vox_grid,
                                       self._global_voxel_grid,
                                       self._bounds_offset,
                                       self._rotation_resolution,
                                       self._viewpoint_agent_bounds,
                                       self._viewpoint_resolution,
                                       device).to(device).train(False)
            for param in self._q_target.parameters():
                param.requires_grad = False
            utils.soft_updates(self._q, self._q_target, 1.0)
            self._optimizer = torch.optim.Adam(
                self._q.parameters(), lr=self._lr,
                weight_decay=self._lambda_weight_l2)

            logging.info('# Q Params: %d' % sum(
                p.numel() for p in self._q.parameters() if p.requires_grad))
        else:
            for param in self._q.parameters():
                param.requires_grad = False

        grid_for_crop = torch.arange(
            0, self._image_crop_size, device=device).unsqueeze(0).repeat(
            self._image_crop_size, 1).unsqueeze(-1)
        self._grid_for_crop = torch.cat([grid_for_crop.transpose(1, 0),
                                         grid_for_crop], dim=2).unsqueeze(0)

        self._coordinate_bounds = torch.tensor(self._coordinate_bounds,
                                               device=device).unsqueeze(0)
        self._viewpoint_agent_bounds = torch.tensor(self._viewpoint_agent_bounds,
                                               device=device).unsqueeze(0)

        self._viewpoint_resolution = torch.tensor(self._viewpoint_resolution,
                                               device=device)
        self._device = device

    def _extract_crop(self, pixel_action, observation):
        # Pixel action will now be (B, 2)
        observation = stack_on_channel(observation)
        h = observation.shape[-1]
        top_left_corner = torch.clamp(
            pixel_action - self._image_crop_size // 2, 0,
            h - self._image_crop_size)
        grid = self._grid_for_crop + top_left_corner.unsqueeze(1)
        grid = ((grid / float(h)) * 2.0) - 1.0  # between -1 and 1
        # Used for cropping the images across a batch
        # swap fro y x, to x, y
        grid = torch.cat((grid[:, :, :, 1:2], grid[:, :, :, 0:1]), dim=-1)
        crop = F.grid_sample(observation, grid, mode='nearest',
                             align_corners=True)
        return crop
    
    def _apply_mask(self, obs, l, fill_value=0):
        """
        Apply a random lxl square mask to a batch of images using broadcasting and torch.masked_fill.
        Args:
        images (torch.Tensor): Tensor of shape [B, 3, w, h]
        l (int): Side length of the square mask
        fill_value (int, optional): Value to fill the masked area. Defaults to 0.

        Returns:
        torch.Tensor: Images with applied masks
        """
        _, c, w, h = obs.shape
        # Initialize the mask with True
        mask = torch.ones(c, w, h, dtype=torch.bool).to(self._device)

        # Generate random top-left corner for the mask in the first image
        x_start = torch.randint(0, w - l + 1, (1,)).item()
        y_start = torch.randint(0, h - l + 1, (1,)).item()

        # Set the mask area to False
        mask[:, x_start:x_start + l, y_start:y_start + l] = False

        # Apply the mask using torch.masked_fill
        masked_obs = torch.masked_fill(obs, ~mask, fill_value)

        return masked_obs

    def _preprocess_inputs(self, replay_sample):

        obs, obs_tp1 = [], []
        pcds, pcds_tp1 = [], []
        self._crop_summary, self._crop_summary_tp1 = [], []
        for n in self._camera_names:
            if self._layer > 0 and 'active' not in n:
                
                pc_t = replay_sample['%s_pixel_coord' % n]
                pc_tp1 = replay_sample['%s_pixel_coord_tp1' % n]

                rgb = self._extract_crop(pc_t, replay_sample['{}_rgb_layer_{}'.format(n,self._layer)])
                rgb_tp1 = self._extract_crop(pc_tp1,replay_sample['{}_rgb_layer_{}_tp1'.format(n,self._layer)])
                pcd = self._extract_crop(pc_t,replay_sample['{}_point_cloud_layer_{}'.format(n,self._layer)])
                pcd_tp1 = self._extract_crop(pc_tp1, replay_sample['{}_point_cloud_layer_{}_tp1'.format(n,self._layer)])
                self._crop_summary.append((n, rgb))
                self._crop_summary_tp1.append(('%s_tp1' % n, rgb_tp1))
            else:
                # rgb = self._apply_mask(stack_on_channel(replay_sample['{}_rgb_layer_{}'.format(n,self._layer)]),l=20)
                # rgb_tp1 = self._apply_mask(stack_on_channel(replay_sample['{}_rgb_layer_{}_tp1'.format(n,self._layer)]),l=20)
                # # [B,3,w,h]
                # pcd = self._apply_mask(stack_on_channel(replay_sample['{}_point_cloud_layer_{}'.format(n,self._layer)]),l=20)
                # pcd_tp1 = self._apply_mask(stack_on_channel(replay_sample['{}_point_cloud_layer_{}_tp1'.format(n,self._layer)]),l=20)


                rgb = stack_on_channel(replay_sample['{}_rgb_layer_{}'.format(n,self._layer)])
                rgb_tp1 = stack_on_channel(replay_sample['{}_rgb_layer_{}_tp1'.format(n,self._layer)])
                # [B,3,w,h]
                pcd = stack_on_channel(replay_sample['{}_point_cloud_layer_{}'.format(n,self._layer)])
                pcd_tp1 = stack_on_channel(replay_sample['{}_point_cloud_layer_{}_tp1'.format(n,self._layer)])

                #pc_t = replay_sample['%s_pixel_coord' % n]
                #pc_tp1 = replay_sample['%s_pixel_coord_tp1' % n]
                #crop_rgb = self._extract_crop(pc_t, replay_sample['{}_rgb_layer_{}'.format(n,self._layer)])
                #crop_rgb_tp1 = self._extract_crop(pc_tp1,replay_sample['{}_rgb_layer_{}_tp1'.format(n,self._layer)])
                #self._crop_summary.append((n, crop_rgb))
                #self._crop_summary_tp1.append(('%s_tp1' % n, crop_rgb_tp1))
                
            obs.append([rgb, pcd])
            obs_tp1.append([rgb_tp1, pcd_tp1])
            pcds.append(pcd)
            pcds_tp1.append(pcd_tp1)
        return obs, obs_tp1, pcds, pcds_tp1

    def _act_preprocess_inputs(self, observation):
        obs, pcds = [], []
        for n in self._camera_names:
            if self._layer > 0  and 'active' not in n:
                pc_t = observation['%s_pixel_coord' % n]
                rgb = self._extract_crop(pc_t, observation['%s_rgb' % n])
                pcd = self._extract_crop(pc_t, observation['%s_point_cloud' % n])
            else:
                rgb = stack_on_channel(observation['%s_rgb' % n])
                pcd = stack_on_channel(observation['%s_point_cloud' % n])           
            obs.append([rgb, pcd])
            pcds.append(pcd)
        return obs, pcds

    def _get_value_from_voxel_index(self, q, voxel_idx):
        b, c, d, h, w = q.shape
        q_flat = q.view(b, c, d * h * w)
        flat_indicies = (voxel_idx[:, 0] * d * h + voxel_idx[:, 1] * h + voxel_idx[:, 2])[:, None].long()
        highest_idxs = flat_indicies.unsqueeze(-1).repeat(1, c, 1)
        chosen_voxel_values = q_flat.gather(2, highest_idxs)[..., 0]  # (B, trans + rot + grip)
        return chosen_voxel_values

    def _get_value_from_rot_and_grip(self, rot_grip_q, rot_and_grip_idx):

        if self._layer==self._layer_num-1:
            q_rot = torch.stack(torch.split(
                rot_grip_q[:, :-2], int(360 // self._rotation_resolution),
                dim=1), dim=1)  # B, 3, 72
            q_grip = rot_grip_q[:, -2:]
            rot_and_grip_values = torch.cat(
                [q_rot[:, 0].gather(1, rot_and_grip_idx[:, 0:1]),
                q_rot[:, 1].gather(1, rot_and_grip_idx[:, 1:2]),
                q_rot[:, 2].gather(1, rot_and_grip_idx[:, 2:3]),
                q_grip.gather(1, rot_and_grip_idx[:, 3:4])], -1)

        else:
            axis_0 = rot_grip_q[:, 0:self._viewpoint_axis_class_breakpoints[0]]
            axis_1 = rot_grip_q[:, self._viewpoint_axis_class_breakpoints[0]:self._viewpoint_axis_class_breakpoints[1]]
            axis_2 = rot_grip_q[:, self._viewpoint_axis_class_breakpoints[1]:]
            

            rot_and_grip_values = torch.cat(
                [axis_0.gather(dim=-1,index=rot_and_grip_idx[:,[int(self._viewpoint_axes_movable[0])]])  if self._viewpoint_axes_movable[0] else axis_0,
                 axis_1.gather(dim=-1,index=rot_and_grip_idx[:,[int(sum(self._viewpoint_axes_movable[:1]))]])  if self._viewpoint_axes_movable[1] else axis_1,
                 axis_2.gather(dim=-1,index=rot_and_grip_idx[:,[int(sum(self._viewpoint_axes_movable[:2]))]])  if self._viewpoint_axes_movable[2] else axis_2], -1)
            
        return rot_and_grip_values

    def update(self, step: int, replay_sample: dict) -> dict:


        obs, obs_tp1, pcd, pcd_tp1 = self._preprocess_inputs(replay_sample)
        proprio = proprio_tp1 = None
        if self._include_low_dim_state:
            proprio = [stack_on_channel(replay_sample['low_dim_state_layer_{}'.format(self._layer)])]
            proprio_tp1 = [stack_on_channel(replay_sample['low_dim_state_layer_{}_tp1'.format(self._layer)])]
        
        if self._layer==0:

            for i in range(1):
                if not self._q._qnet._use_vae:
                    break
                occupancy, vae_loss,_ = self._q(obs, proprio, pcd,recon=True)
                self._optimizer.zero_grad()

                vae_loss.backward()

                self._optimizer.step()
        

        action_trans = replay_sample['translation_idxs_layer_{}'.format(self._layer)][:,-1]
        # 
        action_rot_grip = replay_sample['rot_grip_action_indicies_layer_{}'.format(self._layer)][:, -1].long()

        reward = replay_sample['reward'] * 0.01

        if self._use_aux_reward:
            reach_reward = replay_sample['reach_reward'].squeeze(1).squeeze(1)
            focus_reward = replay_sample['focus_reward'].squeeze(1).squeeze(1)
            reward += reach_reward
            if self._layer==0:
                reward += focus_reward
        
        #reward = torch.where(reward >= 0, reward, torch.zeros_like(reward))

        bounds = bounds_tp1 = self._coordinate_bounds
        if self._layer > 0:
            # [B,3]
            cp = replay_sample['attention_input_layer_{}'.format(self._layer)][:, -1]
            cp_tp1 = replay_sample['attention_input_layer_{}_tp1'.format(self._layer)][:, -1]
            # 
            bounds = torch.cat(
                [cp - self._bounds_offset, cp + self._bounds_offset], dim=1)
            bounds_tp1 = torch.cat(
                [cp_tp1 - self._bounds_offset, cp_tp1 + self._bounds_offset],
                dim=1)


        terminal = replay_sample['terminal'].float()
        # q [b,1,d,w,h]   None    [b,c,d,w,h]
        q, q_rot_grip, voxel_grid = self._q(
            obs, proprio, pcd, bounds,
            replay_sample.get('prev_layer_voxel_grid', None))
        # [b,3]  None 
        coords, rot_and_grip_indicies = self._q.choose_highest_action(q, q_rot_grip)

        with_rot_and_grip = rot_and_grip_indicies is not None

        with torch.no_grad():

            q_tp1_targ, q_rot_grip_tp1_targ, _ = self._q_target(
                obs_tp1, proprio_tp1, pcd_tp1, bounds_tp1,
                replay_sample.get('prev_layer_voxel_grid_tp1', None))

            q_tp1, q_rot_grip_tp1, voxel_grid_tp1 = self._q(
                obs_tp1, proprio_tp1, pcd_tp1, bounds_tp1,
                replay_sample.get('prev_layer_voxel_grid_tp1', None))

            coords_tp1, rot_and_grip_indicies_tp1 = self._q.choose_highest_action(q_tp1, q_rot_grip_tp1)
            # Q_target(s',argmax_a Q(s',a))
            q_tp1_at_voxel_idx = self._get_value_from_voxel_index(q_tp1_targ, coords_tp1)
            
            if with_rot_and_grip:
                # 
                target_q_tp1_rot_grip = self._get_value_from_rot_and_grip(q_rot_grip_tp1_targ, rot_and_grip_indicies_tp1)  # (B, 4)

                q_tp1_at_voxel_idx = torch.cat([q_tp1_at_voxel_idx, target_q_tp1_rot_grip], 1).mean(1, keepdim=True)
            # r+𝛾*Q_target(s',argmax_a Q(s',a))
            q_target = (reward.unsqueeze(1) + (self._gamma ** self._nstep) * (1 - terminal.unsqueeze(1)) * q_tp1_at_voxel_idx).detach()
            q_target = torch.maximum(q_target, torch.zeros_like(q_target))

        qreg_loss = F.l1_loss(q, torch.zeros_like(q), reduction='none')
        qreg_loss = qreg_loss.mean(-1).mean(-1).mean(-1).mean(-1) * self._lambda_trans_qreg

        chosen_trans_q1 = self._get_value_from_voxel_index(q, action_trans)

        q_delta = F.smooth_l1_loss(chosen_trans_q1[:, :1], q_target[:, :1], reduction='none')

        if with_rot_and_grip:
            target_q_rot_grip = self._get_value_from_rot_and_grip(q_rot_grip, action_rot_grip)  # 
            q_target = q_target.repeat((1, target_q_rot_grip.shape[-1]))
            q_delta = torch.cat([F.smooth_l1_loss(target_q_rot_grip, q_target, reduction='none'), q_delta], -1)

            qreg_loss += F.l1_loss(q_rot_grip, torch.zeros_like(q_rot_grip), reduction='none').mean(1) * self._lambda_rot_qreg
        # PER
        loss_weights = utils.loss_weights(replay_sample, REPLAY_BETA)
        combined_delta = q_delta.mean(1)
        # 
        total_loss = ((combined_delta + qreg_loss) * loss_weights).mean()

        self._optimizer.zero_grad()

        total_loss.backward()
        if self._grad_clip is not None:
            nn.utils.clip_grad_value_(self._q.parameters(), self._grad_clip)

        self._optimizer.step()

        self._summaries = {
            'q/mean_qattention': q.mean(),
            'q/max_qattention': chosen_trans_q1.max(1)[0].mean(),
            'losses/total_loss': total_loss, 
            'losses/qreg': qreg_loss.mean(),  
            
            'batch_reward/total':reward.mean(),

        }
        if self._use_aux_reward:
            self._summaries.update({'batch_reward/aux_reach':reach_reward.mean(),})
            if self._layer==0:
                self._summaries.update({'batch_reward/aux_focus':focus_reward.mean(),})

        if self._layer==0:
            if self._q._qnet._use_vae:
                self._summaries.update({'losses/vae_loss':vae_loss})
            
        if with_rot_and_grip:
            if self._layer == 0:

                self._summaries.update({
                    'q/mean_q_rotation': q_rot_grip.mean(),
                    'q/max_q_rotation': target_q_rot_grip[:, :3].mean(),
                    'losses/bellman_rotation': q_delta[:, :-1].mean(),
                    'losses/bellman_qattention': q_delta[:, -1].mean(),
                })    
            else:     

                self._summaries.update({
                    'q/mean_q_rotation': q_rot_grip.mean(),
                    'q/max_q_rotation': target_q_rot_grip[:, :3].mean(),
                    'losses/bellman_rotation': q_delta[:, :-2].mean(),
                    'losses/bellman_gripper': q_delta[:, -2].mean(),
                    'losses/bellman_qattention': q_delta[:, -1].mean(),
                })
        else:
            self._summaries.update({
                'losses/bellman_qattention': q_delta.mean(),
            })

        self._vis_voxel_grid = voxel_grid[0]
        self._vis_translation_qvalue = q[0]
        self._vis_max_coordinate = coords[0]

        utils.soft_updates(self._q, self._q_target, self._tau)

        priority = (combined_delta + 1e-10).sqrt()
        priority /= priority.max()
        prev_priority = replay_sample.get('priority', 0)

        return {
            'priority': priority + prev_priority,
            'prev_layer_voxel_grid': voxel_grid,
            'prev_layer_voxel_grid_tp1': voxel_grid_tp1,
        }
    def update_epsilon(self,step):
        max_epsilon = 2
        min_epsilon = 0
        max_steps = 40000
        ratio = 5 
        epsilon = np.round(0 + (max_epsilon - min_epsilon) * np.exp(-((step - max_steps/3) ** 2) / (2 * max_steps/ratio ** 2)))
        return torch.tensor(epsilon).to(self._device)
    
    def act(self, step: int, observation: dict,
            deterministic=False) -> ActResult:
        #deterministic = True  # TODO: Don't explicitly explore.
        bounds = self._coordinate_bounds
        
        viewpoint_agent_bounds = self._viewpoint_agent_bounds  

        if self._layer > 0:
            # [1,3]
            cp = stack_on_channel(observation['attention_input'])
            # [1,3]
            bounds = torch.cat(
                [cp - self._bounds_offset, cp + self._bounds_offset], dim=1)
            
            
        # [1,3]
        res = (bounds[:, 3:] - bounds[:, :3]) / self._voxel_size
        if self._layer < self._layer_num - 1:

            max_rot_index = ((viewpoint_agent_bounds[:,3:] - viewpoint_agent_bounds[:,:3])//self._viewpoint_resolution).int() - 1

        else:

            max_rot_index = torch.tensor((360 // self._rotation_resolution))[None].int().to(self._device) - 1
        proprio = None
        if self._include_low_dim_state:
            # [1,n]                              [1,1,n]
            proprio = [stack_on_channel(observation['low_dim_state'])]
        # list[[1,3,w,h],[1,3,w*h]]
        obs, pcd = self._act_preprocess_inputs(observation)

        # q[b,1,d,w,h] None  vox_grid[b,10,d,w,h]   
        q, q_rot_grip, vox_grid = self._q(obs, proprio, pcd, bounds,
                              observation.get('prev_layer_voxel_grid', None))
        # [b,3] b=1    [b,4] b=1
        coords, rot_and_grip_indicies = self._q.choose_highest_action(q, q_rot_grip)

        # [b,4] b=1
        rot_grip_action = rot_and_grip_indicies
        if (not deterministic) and self._exploration_strategy == 'gaussian' :
            epsilon = self.update_epsilon(step)
            trans_noise = torch.round(torch.normal(0.0, epsilon, size=(1, 3)).to(self._device))
            coords = torch.clamp(coords + trans_noise, 0,
                                 self._voxel_size - 1)

            rg_noise = torch.round(torch.normal(0.0, 1, size=(1, 3)).to(self._device))
            if False and rot_grip_action is not None and self._layer == self._layer_num - 1:

                explore_rot = torch.clamp(rot_and_grip_indicies[:, :3] + rg_noise, 
                                          torch.tensor(0).to(self._device), max_rot_index)

                grip = rot_and_grip_indicies[:, 3:]
                # For now, randomly swap gripper 20% of time
                if np.random.random() < 0.2 and grip.shape[-1] > 0:
                    grip = torch.randint(0, 2, size=(1, 1)).to(self._device)
                # For now, randomly swap gripper 20% of time
                rot_grip_action = torch.cat([explore_rot, grip], -1)

        # [1,3] indices
        coords = coords.int()

        # [1,3] 
        attention_output = bounds[:, :3] + res * coords + res / 2
        
        
        # viewpoint coordinate  
        if self._layer < self._layer_num - 1:

            complate_vp = torch.zeros([1,3]).int().to(self._device)
            if rot_and_grip_indicies is not None:
                viewpoint_indices = rot_and_grip_indicies.int()
                complate_vp[:,self._viewpoint_axes_movable] = viewpoint_indices[:,:]

            viewpoint_coordinate = viewpoint_agent_bounds[:,:3] + \
                self._viewpoint_resolution * complate_vp[0] + self._viewpoint_resolution/2
        else:
            # nbp agent 
            viewpoint_coordinate = torch.tensor([0,0,0],dtype=torch.float32)[None].to(self._device) 

        viewpoint_coordinate = torch.clip(viewpoint_coordinate,viewpoint_agent_bounds[:,:3],viewpoint_agent_bounds[:,3:])
        
        observation_elements = {
            'attention_output': attention_output, # [1,3] 
            'viewpoint_coordinate':viewpoint_coordinate,
            'prev_layer_voxel_grid': vox_grid, # [b,c,d,h,w] b=1
        }
            
        
        info = {
            'voxel_grid_depth%d' % self._layer: vox_grid,
            'q_depth%d' % self._layer: q, # [b,1,d,h,w] b=1
            'voxel_idx_depth%d' % self._layer: coords # [b,3] b=1
        }
        self._act_voxel_grid = vox_grid[0]
        self._act_max_coordinate = coords[0]
        self._act_qvalues = q[0]
        return ActResult((coords, rot_grip_action),
                         observation_elements=observation_elements,
                         info=info)

    def update_summaries(self) -> List[Summary]:
        summaries = []
        try:
            summaries.append(
                ImageSummary('%s/update_qattention' % self._name,
                             transforms.ToTensor()(visualise_voxel(
                                 self._vis_voxel_grid.detach().cpu().numpy(),
                                 self._vis_translation_qvalue.detach().cpu().numpy(),
                                 self._vis_max_coordinate.detach().cpu().numpy())))
            )
        except Exception as e:
            print('Failed to visualize Q-attention.')

        for n, v in self._summaries.items():
            summaries.append(ScalarSummary('%s/%s' % (self._name, n), v))

        for (name, crop) in (self._crop_summary + self._crop_summary_tp1):
            crops = (torch.cat(torch.split(crop, 3, dim=1), dim=3) + 1.0) / 2.0
            summaries.extend([
                ImageSummary('%s/crops/%s' % (self._name, name), crops)])

        for tag, param in self._q.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad.abs() <= 1.0).all()
                summaries.append(
                    HistogramSummary('%s/gradient/%s' % (self._name, tag),
                                    param.grad))
                summaries.append(
                    HistogramSummary('%s/weight/%s' % (self._name, tag),
                                    param.data))

        for name, t in self._q.latents().items():
            summaries.append(
                HistogramSummary('%s/activations/%s' % (self._name, name), t))

        return summaries

    def act_summaries(self) -> List[Summary]:
        return [
            ImageSummary('%s/act_Qattention' % self._name,
                         transforms.ToTensor()(visualise_voxel(
                             self._act_voxel_grid.cpu().numpy(),
                             self._act_qvalues.cpu().numpy(),
                             self._act_max_coordinate.cpu().numpy())))]

    def load_weights(self, savedir: str):
        self._q.load_state_dict(
            torch.load(os.path.join(savedir, '%s.pt' % self._name),
                       map_location=self._device))

    def save_weights(self, savedir: str):
        torch.save(
            self._q.state_dict(), os.path.join(savedir, '%s.pt' % self._name))
