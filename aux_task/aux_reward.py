import torch
import numpy as np
import math
from arm import utils
#from arm.ota.voxel_grid import VoxelGrid
from arm.ota.aux_task.voxel_grid import VoxelGrid
from rlbench.backend.const import TABLE_COORD


class AuxReward():

    def __init__(self,
                 scene_bound,
                 world_pc_tp0=None, #[B,N,C]
                 world_pc_tp0_1=None,
                 voxel_size=100,
                 batch_size = 1,
                 max_num_coords = 1000000,
                 feature_size=0,
                 device="cpu",) -> None:
        
        self._pc_tp0 = world_pc_tp0
        self._pc_tp0_1 = world_pc_tp0_1
        self._feature_size = feature_size
        self._scene_bound = torch.tensor(scene_bound,device=device)
        self._voxel_size = voxel_size
        self._device = device
        self._index_max_bounds = torch.tensor([voxel_size]*3, dtype=torch.int,device=device)-1
        self._index_min_bounds = torch.tensor([0]*3, dtype=torch.int,device=device)

        
        # self._grid_tp0 = VoxelGrid(coord_bounds=scene_bound,
        #                     voxel_size= voxel_size,
        #                     device=device,
        #                     batch_size=batch_size,
        #                     feature_size=feature_size,
        #                     max_num_coords=max_num_coords,
        #                     )
        # self._grid_tp0_1 = VoxelGrid(coord_bounds=scene_bound,
        #                     voxel_size= voxel_size,
        #                     device=device,
        #                     batch_size=batch_size,
        #                     feature_size=feature_size,
        #                     max_num_coords=max_num_coords,
        #                     )
        grid_center = (self._scene_bound[3:] + self._scene_bound[:3])/2
        self._grid_tp0 = VoxelGrid(grid_size=torch.tensor([1.0,1.0,1.0], dtype=torch.float,device=device),
                                   voxel_size=torch.tensor([0.02], dtype=torch.float,device=device),
                                   grid_center=grid_center,width=128,height=128,fx=-110.85124795,fy=-110.85124795,
                                   cx=64,cy=64,z_far=3.5,z_near=0.1,device=device)
        self._grid_tp0_1 = VoxelGrid(grid_size=torch.tensor([1.0,1.0,1.0], dtype=torch.float,device=device),
                                   voxel_size=torch.tensor([0.02], dtype=torch.float,device=device),
                                   grid_center=grid_center,width=128,height=128,fx=-110.85124795,fy=-110.85124795,
                                   cx=64,cy=64,z_far=3.5,z_near=0.1,device=device)

        # 
        if (self._pc_tp0 is not None)  and  (self._pc_tp0_1 is not None):
            self.update_grid(world_pc_tp0,world_pc_tp0_1)
        
    def update_grid(self,
                    target_point,
                    extrinsics_tp0,
                    extrinsics_tp0_1,
                    depth_tp0,
                    depth_tp0_1,
                    pc_tp0,
                    pc_tp0_1,
                    roi_size = 0.15,
                    ):

        C,W,H = depth_tp0.shape
        assert  C == 1 

        self._extrinsics_tp0 = torch.tensor(extrinsics_tp0, dtype=torch.float,device=self._device).unsqueeze(0) 
        self._extrinsics_tp0_1 = torch.tensor(extrinsics_tp0_1, dtype=torch.float,device=self._device).unsqueeze(0) 
        
        self._depth_tp0 = torch.tensor(depth_tp0, dtype=torch.float,device=self._device)# [b,w,h]
        self._depth_tp0_1 = torch.tensor(depth_tp0_1, dtype=torch.float,device=self._device)# [b,w,h]
        
        self._pc_tp0 = torch.tensor(np.transpose(pc_tp0,(1,2,0)).reshape(-1,3), dtype=torch.float,device=self._device).unsqueeze(0) # [b,w*h,3]
        self._pc_tp0_1 = torch.tensor(np.transpose(pc_tp0_1,(1,2,0)).reshape(-1,3), dtype=torch.float,device=self._device).unsqueeze(0) # [b,w*h,3]
        
        target_point = torch.tensor(target_point, dtype=torch.float,device=self._device) 
            

        #self._grid_tp0.set_target_roi(target_params=)
        

        roi_entropy_tp0,occ_ratio_tp0 = self._grid_tp0.compute_ROI_information(depth_image=self._depth_tp0,
                                                     transforms=self._extrinsics_tp0,
                                                     target_params=target_point,
                                                     point_cloud = self._pc_tp0,
                                                     roi_size=roi_size) 
        roi_entropy_tp0_1,occ_ratio_tp0_1 = self._grid_tp0_1.compute_ROI_information(depth_image=self._depth_tp0_1,
                                                     transforms=self._extrinsics_tp0_1,
                                                     target_params=target_point,
                                                     point_cloud = self._pc_tp0_1,
                                                     roi_size=roi_size) 
        
        information_gain = roi_entropy_tp0 - roi_entropy_tp0_1
        
        #print(occ_ratio_tp0.cpu().item(),occ_ratio_tp0_1.cpu().item())
        
        #print(information_gain.cpu().item())
        
        return (information_gain.cpu().item(),
                roi_entropy_tp0.cpu().item(),
                roi_entropy_tp0_1.cpu().item(),
                occ_ratio_tp0_1.cpu().item())
        
        

        
    def _point_to_pixel_index(self,
                            point: torch.Tensor,  # [n, 3]
                            extrinsics: torch.Tensor,  # [4, 4]
                            intrinsics: torch.Tensor):  # [4, 4]

        point = torch.cat([point, torch.ones(point.shape[0], 1, device=point.device)], dim=1)  # [n, 4]
        

        world_to_cam = torch.linalg.inv(extrinsics)  # [4, 4]
        

        point_in_cam_frame = torch.matmul(world_to_cam, point.T).T  # [n, 4]
        

        px, py, pz = point_in_cam_frame[:, 0], point_in_cam_frame[:, 1], point_in_cam_frame[:, 2]
        

        px = intrinsics[0, 2] - intrinsics[0, 0] * (px / pz) + intrinsics[0, 2]
        py = intrinsics[1, 2] - intrinsics[1, 1] * (py / pz) + intrinsics[1, 2]
        

        px = px.long()
        py = py.long()
        
        return px, py

        
        
    def _point_to_voxel_index(self,
                              point: torch.Tensor, 
                              voxel_size: torch.Tensor, 
                              coord_bounds: torch.Tensor,
                              ):          
        #point = torch.tensor(point, dtype=torch.float,device=device) 
        
        bb_mins = coord_bounds[0:3]
        bb_maxs = coord_bounds[3:]
        dims_m_one = torch.tensor([voxel_size] * 3,device=self._device,dtype=torch.int32) - 1
        bb_ranges = bb_maxs - bb_mins
        res = bb_ranges / (torch.tensor([voxel_size] * 3,device=self._device) + 1e-12) 
        voxel_indicy = torch.min(torch.floor((point - bb_mins) / (res + 1e-12)).type(torch.int32), 
                                  dims_m_one)
        return voxel_indicy

    def _get_neighborhood_indices(self,center_idx, zone_size):

        zone_voxel_size = zone_size / (1 / self._voxel_size)
        extens_m = math.ceil((zone_voxel_size - 1) / 2) 
        #print(center_idx,self._index_max_bounds,extens_m)

        ranges = [torch.arange(max(0, i - extens_m), min(s, i + extens_m + 1)) for i, s in zip(center_idx, self._index_max_bounds)]
        
        grid = torch.cartesian_prod(*ranges).to(self._device)

        #grid = torch.clamp(grid, min=index_min_bounds, max=index_max_bounds) 

        # [n,3]
        return grid 


    def check_occupancy(self, voxel_tensor, indices):
        # voxel_tensor [w,h,l,c]     indices [n,3]
        indices = indices.T
        occupancy_status = voxel_tensor[indices[0], indices[1], indices[2],-1] == 1
        return occupancy_status.any()

    def calculate_entropy(self, voxel_tensor):
        p_occ = (voxel_tensor[..., -1] > 0).float().mean()
        if p_occ == 0 or p_occ == 1:
            return torch.tensor(0.0, device=self._device)
        return -p_occ * torch.log2(p_occ) - (1 - p_occ) * torch.log2(1 - p_occ)
    
    
    def attention_zone_occupancy(self,attention_world_pos,zone_size:float,show_tp0:bool=False,show_tp0_1:bool=False):

        _attention_world_pos = torch.tensor(attention_world_pos, dtype=torch.float,device=self._device) 

        _attention_world_pos_idx = self._point_to_voxel_index(point=_attention_world_pos,
                                                             voxel_size=self._voxel_size,
                                                             coord_bounds=self._scene_bound,
                                                             )
        
        

        _neighborhood_indices = self._get_neighborhood_indices(_attention_world_pos_idx,zone_size)
        
        _neighborhood_x_indices = _neighborhood_indices[:, 0]
        _neighborhood_y_indices = _neighborhood_indices[:, 1]
        _neighborhood_z_indices = _neighborhood_indices[:, 2]
        
        # [w,h,l,c] 
        attention_zone_tp0 =  self.voxel_tp0[_neighborhood_x_indices,_neighborhood_y_indices,_neighborhood_z_indices,:]
        attention_zone_tp0_1 =  self.voxel_tp0_1[_neighborhood_x_indices,_neighborhood_y_indices,_neighborhood_z_indices,:]
        # [w,h,l,3]  
        attention_zone_voxel_centers_tp0 =  self.grid_centers_tp0[_neighborhood_x_indices,
                                                                  _neighborhood_y_indices,
                                                                  _neighborhood_z_indices,:]
        
        attention_zone_voxel_centers_tp0_1 =  self.grid_centers_tp0_1[_neighborhood_x_indices,
                                                                      _neighborhood_y_indices,
                                                                      _neighborhood_z_indices,:]
        

        attention_zone_padding_tp0 = torch.zeros_like(self.voxel_tp0)
        attention_zone_padding_tp0_1 = torch.zeros_like(self.voxel_tp0_1)
        attention_zone_padding_tp0[_neighborhood_x_indices, _neighborhood_y_indices, _neighborhood_z_indices, :] = attention_zone_tp0
        attention_zone_padding_tp0_1[_neighborhood_x_indices, _neighborhood_y_indices, _neighborhood_z_indices, :] = attention_zone_tp0_1
        
        

        attention_zone_occupancy_statues_tp0,attention_zone_occupancy_statues_tp0_1 = \
            attention_zone_tp0[...,-1]==1,attention_zone_tp0_1[...,-1]==1
            
        attention_zone_occupancy_statues_change = attention_zone_tp0_1[...,-1] - attention_zone_tp0[...,-1]
        
        voxel_lost = torch.sum(attention_zone_occupancy_statues_change== -1)
        voxel_new = torch.sum(attention_zone_occupancy_statues_change== 1)
        
        voxel_gain = voxel_new - voxel_lost
        
        total_voxel = attention_zone_tp0.shape[0]
        
        #print(voxel_gain.item(),total_voxel,round(voxel_gain.item()/total_voxel,5) )
        
        if voxel_gain.item() >= 20:
            ig_flag = 1
        elif voxel_gain.item()<= -20:
            ig_flag = -1
        else:
            ig_flag = 0
            
        

        occupancy_flag_tp0,occupancy_flag_tp0_1 = \
            attention_zone_occupancy_statues_tp0.any(),attention_zone_occupancy_statues_tp0_1.any()




        # entropy_tp0 = self.calculate_entropy(attention_zone_tp0)
        # entropy_tp0_1 = self.calculate_entropy(attention_zone_tp0_1)
        # entropy_gain = entropy_tp0 - entropy_tp0_1
        
        if show_tp0:
            show_voxel_tp0 = utils.visualise_voxel(voxel_grid=self.voxel_tp0.cpu().numpy().transpose(3,0,1,2),
                                              show_bb=True,
                                              show=True,
                                              highlight_coordinate=_neighborhood_indices.cpu().numpy().T,
                                              highlight_alpha=0.5)
                
            show_zone_tp0 = utils.visualise_voxel(voxel_grid=attention_zone_padding_tp0.cpu().numpy().transpose(3,0,1,2),
                                              show_bb=True,
                                              show=True,
                                              highlight_alpha=0.5)
        if show_tp0_1:
            show_voxel_tp0_1 = utils.visualise_voxel(voxel_grid=self.voxel_tp0_1.cpu().numpy().transpose(3,0,1,2),
                                            show_bb=True,
                                            show=True,
                                            highlight_coordinate=_neighborhood_indices.cpu().numpy().T,
                                            highlight_alpha=0.5)

            show_zone_tp0_1 = utils.visualise_voxel(voxel_grid=attention_zone_padding_tp0_1.cpu().numpy().transpose(3,0,1,2),
                                            show_bb=True,
                                            show=True,
                                            highlight_alpha=0.5)
            
        return occupancy_flag_tp0,occupancy_flag_tp0_1,ig_flag
    
    def ray_voxel_intersection(self,
                               voxels, # [w,h,l,c]
                               voxel_centers, # [w,h,l,3]
                               pointA, # start  [3,]
                               pointB, # end  [3,]
                               pointB_idx, # [3]
                               voxel_real_size=0.03 # 
                               ):

        # [n,3]
        voxel_centers = voxel_centers.reshape(-1,3) 
        # [n,]
        voxel_occupy_statue = voxels[...,-1].reshape(-1)
        
                

        direction = pointB - pointA
        direction = direction / torch.norm(direction)  
        length = torch.norm(pointB - pointA)
        

        vectors = voxel_centers - pointA
        

        projections = torch.matmul(vectors, direction)
        

        cross_product = torch.cross(direction.repeat(vectors.shape[0], 1), vectors)
        distances = torch.norm(cross_product, dim=1) / torch.norm(direction)
        

        valid_projections = (projections >= 0) & (projections <= length) & (distances <= voxel_real_size)
        

        valid_indices  = valid_projections.nonzero().squeeze()
        occupied_voxels = voxels[...,-1].reshape(-1)[valid_indices ]
        occupied = occupied_voxels == 1
        
        
    def attention_zone_ig(self,attention_world_pos, # [3,]
                          viewpoint_world_tp0,  # [3,]
                          viewpoint_world_tp0_1,  # [3,]
                          zone_size:float,
                          show_tp0:bool=False,
                          show_tp0_1:bool=False):
        #print(attention_world_pos)

        _viewpoint_world_tp0 = torch.tensor(viewpoint_world_tp0, dtype=torch.float,device=self._device) 
        _viewpoint_world_tp0_1 = torch.tensor(viewpoint_world_tp0_1, dtype=torch.float,device=self._device) 
        _attention_world_pos = torch.tensor(attention_world_pos, dtype=torch.float,device=self._device) 
        
        # [d, h, w, c=3] 
        _voxel_centers_tp0 = self._grid_tp0.get_voxel_centers()[0]
        _voxel_centers_tp0_1 = self._grid_tp0_1.get_voxel_centers()[0]
        
        
        

        _attention_world_pos_idx = self._point_to_voxel_index(point=_attention_world_pos,
                                                             voxel_size=self._voxel_size,
                                                             coord_bounds=self._scene_bound,
                                                             )
        
        

        _neighborhood_indices = self._get_neighborhood_indices(_attention_world_pos_idx,zone_size)
        
        _neighborhood_x_indices = _neighborhood_indices[:, 0]
        _neighborhood_y_indices = _neighborhood_indices[:, 1]
        _neighborhood_z_indices = _neighborhood_indices[:, 2]
        
        

        
        # [w,h,l,c] 
        attention_zone_tp0 =  self.voxel_tp0[_neighborhood_x_indices,_neighborhood_y_indices,_neighborhood_z_indices,:]
        attention_zone_tp0_1 =  self.voxel_tp0_1[_neighborhood_x_indices,_neighborhood_y_indices,_neighborhood_z_indices,:]
        

        attention_zone_occupancy_statues_tp0,attention_zone_occupancy_statues_tp0_1 = \
            attention_zone_tp0[...,-1]==1,attention_zone_tp0_1[...,-1]==1
            
            
        attention_zone_occupancy_statues_change = attention_zone_occupancy_statues_tp0_1 - attention_zone_occupancy_statues_tp0
        
        #ig = attention_zone_occupancy_statues_change.sum().cpu
        
