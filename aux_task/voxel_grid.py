"""
Author: Akshay K. Burusa
Maintainer: Akshay K. Burusa
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from arm.ota.aux_task.raysampler import RaySampler
from arm.ota.aux_task.torch_utils import look_at_rotation, transform_from_rotation_translation


class VoxelGrid:
    """
    3D representation to store occupancy information and other features (e.g. semantics) over
    multiple viewpoints
    """

    def __init__(
        self,
        grid_size: torch.tensor,
        voxel_size: torch.tensor,
        grid_center: torch.tensor,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        z_near: float = 0.05,
        z_far: float = 0.6,
        roi_size: float = 0.15,
        target_params: torch.tensor = None,
        num_pts_per_ray: int = 128,
        num_features: int = 2,
        eps: torch.float32 = 1e-7,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        """
        Constructor
        :param grid_size: size of the voxel grid
        :param voxel_size: size of each voxel
        :param grid_center: center of the voxel grid
        :param width: image width
        :param height: image height
        :param fx: focal length along x-axis
        :param fy: focal length along y-axis
        :param cx: principal point along x-axis
        :param cy: principal point along y-axis
        :param z_near: near clipping plane
        :param z_far: far clipping plane
        :param num_pts_per_ray: number of points sampled along each ray
        :param eps: epsilon value for numerical stability
        :param device: device to use for computation
        """
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.grid_center = grid_center
        self.width = width
        self.height = height
        self.num_pts_per_ray = num_pts_per_ray
        self.num_features = num_features
        self.eps = eps
        self.device = device

        self.voxel_dims = (grid_size / voxel_size).long()
        self.origin = grid_center - grid_size / 2.0
        self.min_bound = self.origin
        self.max_bound = self.origin + grid_size

        # 4D voxel grid
        self.empty_voxel_grid = torch.zeros(
            (
                self.voxel_dims[0],
                self.voxel_dims[1],
                self.voxel_dims[2],
                num_features,  # ROIs, occ_prob,  
            ),
            dtype=torch.float32,
            device=self.device,
        )
        # Initialize occupancy probability as 0.5
        self.empty_voxel_grid[..., 1] = 0.5  
        
        self.voxel_grid = None
        # Define regions of interest around the target
        #self.set_target_roi(target_params,roi_size)

        # Occupancy and semantic information along a ray
        # Occupancy probabilities along the ray is initialized to 0.2, which gives a log odds of -1.4

        ray_occ = -1.99 * torch.ones(
            (self.num_pts_per_ray, 1),
            dtype=torch.float32,
            device=self.device,
        )

        self.ray_occ = ray_occ.unsqueeze(0).repeat(
            width * height, 1, 1
        )  # (W x H, num_pts_per_ray, 1) 
        
        # Ray sampler
        self.ray_sampler = RaySampler(
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            z_near=z_near,
            z_far=z_far,
            device=device,
        )
        
        self.t_vals = torch.linspace(
            0.0,
            1.0,
            self.num_pts_per_ray,
            dtype=torch.float32,
            device=self.device,
        )



    def insert_depth(
        self,
        target_params:torch.tensor,
        depth_image: torch.tensor,
        transforms: torch.tensor,
        point_cloud: torch.tensor,
    ) -> None:
        """
        Insert a point cloud into the voxel grid
        :param depth_image: depth image from the current viewpoint (W x H)
        :param position: position of the current viewpoint (3,)
        :param orientation: orientation of the current viewpoint (4,)
        :return: None
        """
        # Convert depth image to point cloud
        (
            ray_origins, 
            ray_targets,
            ray_directions,  
            points_mask, # [b,w*h]
        ) = self.ray_sampler.ray_origins_directions(point_cloud=point_cloud,
            depth_image=depth_image, transforms=transforms
        )
        self.transforms = transforms
        

        ray_points = (
            ray_directions[:, :, None, :] * self.t_vals[None, :, None]
            + ray_origins[:, :, None, :]
        ).reshape(-1, 3)


        grid_coords = torch.div(ray_points - self.origin, self.voxel_size, rounding_mode="floor")

        valid_indices = self.get_valid_indices(grid_coords, self.voxel_dims)

        gx, gy, gz = grid_coords[valid_indices].to(torch.long).unbind(-1)

        if self.voxel_grid is None:
            self.voxel_grid = self.empty_voxel_grid.clone()
        # Get the log odds of the occupancy and semantic probabilities

        log_odds = torch.log(
            torch.div(
                self.voxel_grid[gx, gy, gz, 1], 1.0 - self.voxel_grid[gx, gy, gz, 1]
            )
        )
        # Update the log odds of the occupancy probabilities  

        ray_occ = self.ray_occ.clone() 

        ray_occ[:, -2:, :] = points_mask.permute(1, 0).repeat(1, 2).unsqueeze(-1)

        log_odds  += ray_occ.view(-1, 1)[valid_indices, -1]

        odds = torch.exp(log_odds)
        self.voxel_grid[gx, gy, gz, 1] = torch.div(odds, 1.0 + odds)
        self.voxel_grid[..., 1] = torch.clamp(
            self.voxel_grid[..., 1], self.eps, 1.0 - self.eps
        )
        # Check the values within the target bounds and count the number of updated voxels

        if self.target_bounds is not None:
            target_voxels = self.voxel_grid[
                self.target_bounds[0] : self.target_bounds[3],
                self.target_bounds[1] : self.target_bounds[4],
                self.target_bounds[2] : self.target_bounds[5],
                1,
            ]

            coverage = torch.sum((target_voxels != 0.5)) / target_voxels.numel() * 100
            occ_ratio = torch.sum((target_voxels > 0.7)) / target_voxels.numel()
            target_voxels_entropy = self.entropy(target_voxels) 

            mean_entropy = target_voxels_entropy.mean()
            return coverage,occ_ratio,mean_entropy
        else:
            return None,None, None
        
    def compute_ROI_information(
        self,
        depth_image: torch.tensor,
        transforms: torch.tensor,
        target_params: torch.tensor,
        point_cloud: torch.tensor,
        roi_size: float,
    ) -> None:
        self.voxel_grid = None

        self.set_target_roi(target_params,roi_size)

        coverage,occ_ratio,roi_entropy = self.insert_depth(target_params,depth_image,transforms,point_cloud)
        #
        #roi_indices = torch.nonzero(self.voxel_grid[...,0]==1)
        #roi_entrop = self.entropy(self.voxel_grid[roi_indices][...,1])
        # 
        return roi_entropy,occ_ratio
        


    def compute_gain(
        self,
        camera_params: torch.tensor=None,
        target_params: torch.tensor=None,
    ) -> torch.tensor:
        """
        Compute the gain for a given set of parameters
        :param camera_params: camera parameters
        :param target_params: target parameters
        :param current_params: current parameters
        :return: total gain for the viewpoint defined by the parameters
        """
        if camera_params is not None and target_params is not None:

            quat = look_at_rotation(camera_params, target_params)
            transforms = transform_from_rotation_translation(
                quat[None, :], camera_params[None, :]
            )
        else:
            transforms = self.transforms
        # Compute point cloud by ray-tracing along ray origins and directions

        ray_origins, ray_targets,ray_directions, _ = self.ray_sampler.ray_origins_directions(
            transforms=transforms
        )
        # 
        t_vals = self.t_vals.clone()

        ray_points = (
            ray_directions[:, :, None, :] * t_vals[None, :, None]
            + ray_origins[:, :, None, :]).view(-1, 3)
        

        ray_points_nor = self.normalize_3d_coordinate(ray_points)
        ray_points_nor = ray_points_nor.view(1, -1, 1, 1, 3).repeat(2, 1, 1, 1, 1)
        # Sample the occupancy probabilities and semantic confidences along each ray
        grid = self.voxel_grid[None, ..., 1:3].permute(4, 0, 1, 2, 3)

        occ_sem_confs = F.grid_sample(grid, ray_points_nor, align_corners=True)
        occ_sem_confs = occ_sem_confs.view(2, -1, self.num_pts_per_ray)
        occ_sem_confs = occ_sem_confs.clamp(self.eps, 1.0 - self.eps)
        # Compute the entropy of the semantic confidences along each ray

        opacities = torch.sigmoid(1e7 * (occ_sem_confs[0, ...] - 0.51))
        transmittance = self.shifted_cumprod(1.0 - opacities)
        ray_gains = transmittance * self.entropy(occ_sem_confs[1, ...])
        # Create a gain image for visualization
        gain_image = ray_gains.view(-1, self.num_pts_per_ray).sum(1)
        gain_image = gain_image.view(self.height, self.width)
        gain_image = gain_image - gain_image.min()
        # gain_image = gain_image / gain_image.max()
        gain_image = gain_image / 32.0
        gain_image = gain_image.detach().cpu().numpy()
        gain_image = plt.cm.viridis(gain_image)[..., :3]
        # Compute the semantic gain
        semantic_gain = torch.log(torch.mean(ray_gains) + self.eps)
        loss = -semantic_gain
        return loss, gain_image

    def entropy(self, probs: torch.tensor) -> torch.tensor:
        """
        Compute the entropy of a set of probabilities
        :param probs: tensor of probabilities
        :return: tensor of entropies
        """
        probs_inv = 1.0 - probs
        gains = -(probs * torch.log2(probs)) - (probs_inv * torch.log2(probs_inv))
        return gains

    def set_target_roi(self, target_params: torch.tensor,roi_size: float) -> None:
        # Define regions of interest around the target
        
        extens_m = torch.ceil((roi_size / self.voxel_size) / 2).int()
        if self.voxel_grid is None:
            self.voxel_grid = self.empty_voxel_grid.clone()
        
        self.target_bounds = None
        if target_params is None:
            return
        target_coords = torch.div(
            target_params - self.origin, self.voxel_size, rounding_mode="floor"
        ).to(torch.long)
        x_min = torch.clamp(target_coords[0] - extens_m, 0, self.voxel_dims[0])
        x_max = torch.clamp(target_coords[0] + extens_m, 0, self.voxel_dims[0])
        y_min = torch.clamp(target_coords[1] - extens_m, 0, self.voxel_dims[1])
        y_max = torch.clamp(target_coords[1] + extens_m, 0, self.voxel_dims[1])
        z_min = torch.clamp(target_coords[2] - extens_m, 0, self.voxel_dims[2])
        z_max = torch.clamp(target_coords[2] + extens_m, 0, self.voxel_dims[2])
        self.voxel_grid[x_min:x_max, y_min:y_max, z_min:z_max, 0] = 1 
        self.target_bounds = torch.tensor(
            [x_min, y_min, z_min, x_max, y_max, z_max], device=self.device
        )
        # occ_prob 
        self.voxel_grid[..., 1] = torch.clamp(
            self.voxel_grid[..., 1], self.eps, 1.0 - self.eps
        )

    def get_valid_indices(
        self, grid_coords: torch.tensor, dims: torch.tensor
    ) -> torch.tensor:
        """
        Get the indices of the grid coordinates that are within the grid bounds
        :param grid_coords: tensor of grid coordinates
        :param dims: tensor of grid dimensions
        :return: tensor of valid indices
        """
        valid_indices = (
            (grid_coords[:, 0] >= 0)
            & (grid_coords[:, 0] < dims[0])
            & (grid_coords[:, 1] >= 0)
            & (grid_coords[:, 1] < dims[1])
            & (grid_coords[:, 2] >= 0)
            & (grid_coords[:, 2] < dims[2])
        )
        return valid_indices

    def normalize_3d_coordinate(self, points):
        """
        Normalize a tensor of 3D points to the range [-1, 1] along each axis.
        :param points: tensor of 3D points of shape (N, 3)
        :return: tensor of normalized 3D points of shape (N, 3)
        """
        # Compute the range of values for each dimension
        x_min, y_min, z_min = self.min_bound
        x_max, y_max, z_max = self.max_bound
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        # Normalize the points to the range [-1, 1]
        n_points = points.clone()
        n_points_out = torch.zeros_like(n_points)
        n_points_out[..., 0] = 2.0 * (n_points[..., 2] - z_min) / z_range - 1.0
        n_points_out[..., 1] = 2.0 * (n_points[..., 1] - y_min) / y_range - 1.0
        n_points_out[..., 2] = 2.0 * (n_points[..., 0] - x_min) / x_range - 1.0
        return n_points_out

    def shifted_cumprod(self, x: torch.tensor, shift: int = 1) -> torch.tensor:
        """
        Computes `torch.cumprod(x, dim=-1)` and prepends `shift` number of ones and removes
        `shift` trailing elements to/from the last dimension of the result
        :param x: tensor of shape (N, ..., C)
        :param shift: number of elements to prepend/remove
        :return: tensor of shape (N, ..., C)
        """
        x_cumprod = torch.cumprod(x, dim=-1)
        x_cumprod_shift = torch.cat(
            [torch.ones_like(x_cumprod[..., :shift]), x_cumprod[..., :-shift]], dim=-1
        )
        return x_cumprod_shift

    def get_occupied_points(self):
        """
        Returns the coordinates of the occupied points in the grid
        :return: tensor of shape (N, 3) containing the coordinates of the occupied points
        """

        grid_coords = torch.nonzero(self.voxel_grid[..., 1] > 0.5)

        points = grid_coords * self.voxel_size + self.origin
        return points
