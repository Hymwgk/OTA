"""
Author: Akshay K. Burusa
Maintainer: Akshay K. Burusa
"""

import torch
import torch.nn as nn
import numpy as np
import open3d as o3d

from typing import Optional, Tuple
from scipy.spatial.transform import Rotation as scipy_r

#from  conversions import T_from_rot_trans_np, T_from_rot_trans
from arm.ota.aux_task.conversions import T_from_rot_trans_np, T_from_rot_trans


class RaySampler:
    def __init__(
        self,
        width: int, 
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        z_near: float,
        z_far: float,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
    ):
        """
        Constructor
        :param width: image width
        :param height: image height
        :param fx: focal length along x-axis
        :param fy: focal length along y-axis
        :param cx: principal point along x-axis
        :param cy: principal point along y-axis
        :param z_near: near clipping plane
        :param z_far: far clipping plane
        :param device: device to use for computation
        """
        self.width = width
        self.height = height
        self.z_near = z_near
        self.z_far = z_far
        self.intrinsic = torch.tensor(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ],
            dtype=torch.float32,
            device=device,
        )
        self.device = device

        self.generate_camera_coords()
        # Transformation from optical frame to camera frame
        r = scipy_r.from_euler("xyz", [-np.pi / 2, 0.0, -np.pi / 2])
        #r = scipy_r.from_euler("xyz", [0.0, 0.0, -np.pi / 2])
        self.T_oc = T_from_rot_trans_np(r.as_matrix(), np.zeros((1, 3)))
        self.T_oc = torch.as_tensor(self.T_oc, dtype=torch.float32, device=self.device)

    def generate_camera_coords(self):
        """
        Generate camera coordinates, which will be used for ray sampling
        """
        # Create a mesh grid of (u, v) coordinates
        u, v = torch.meshgrid(
            [
                torch.arange(0.0, self.width, device=self.device, dtype=torch.float32),
                torch.arange(0.0, self.height, device=self.device, dtype=torch.float32),
            ],
            indexing="xy",
        )
        u, v = u + 0.5, v + 0.5
        # Convert the pixel coordinates to homogeneous coordinates
        pixel_coords = torch.stack((u, v, torch.ones_like(u)), dim=-1) # [w,h,3]
        # Transform the pixel coordinates to camera coordinates  
        self.camera_coords = pixel_coords.view(-1, 3) @ torch.inverse(
            self.intrinsic
        ).t().type(torch.float32)

    def ray_origins_directions(
        self,
        point_cloud: torch.tensor,
        transforms: torch.tensor,
        depth_image: Optional[torch.tensor] = None,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Compute the origins and directions for all rays
        :param depth_image: depth image (batch_size x width x height)
        :param transforms: transformation matrices (batch_size x 4 x 4)   
        :return: ray origins and directions
        """
        # 
        batch_size = transforms.shape[0]
        # [z_near, b, w*h]
        min_depths = self.z_near * torch.ones(
            (batch_size, self.width * self.height),
            dtype=torch.float32,
            device=self.device,
        )
        # If depth image is provided, use it to compute the max depth
        # Otherwise, use the far clipping plane
        if depth_image is not None:
            depth_image[torch.isnan(depth_image)] = self.z_far
            max_depths = depth_image.view(1, -1)*self.z_far # [1,w*h]
        else:
            max_depths = self.z_far * torch.ones(
                (batch_size, self.width * self.height),
                dtype=torch.float32,
                device=self.device,
                )
        # Create a mask that is log odds 0.9 if the depth is less than far and log odds of 0.4 otherwise
        # 
        points_mask = torch.where(max_depths < self.z_far, 2.2, -1.99)
        # Transform the camera coordinates to world coordinates
        camera_coords = self.camera_coords.clone()  # [w*h,3]
        
        ray_origins = (camera_coords * min_depths.unsqueeze(-1)).view(batch_size, -1, 3)
        ray_targets = (camera_coords * max_depths.unsqueeze(-1)).view(batch_size, -1, 3)

        ray_origins = self.transform_points(ray_origins, transforms)
        ray_targets = point_cloud#self.transform_points(ray_targets, transforms)
        # Compute the ray directions
        ray_directions = ray_targets - ray_origins
        # [b,w*h,3]    [b,w*h,3]     [b,w*h]
        return ray_origins,ray_targets, ray_directions, points_mask

    def transform_points(
        self,
        points: torch.tensor,
        transforms: torch.tensor,  #wtc
    ) -> torch.tensor:
        """
        Transform a point cloud from 'camera_frame' to 'world_frame'
        :param points: point cloud
        :param transforms: transformation matrices
        """
        #points[..., 1] += 0.024  # TODO: remove this hack, only for gazebo
        #T_oc = self.T_oc.clone()
        T_cws = transforms.clone().to(torch.float32)
        #T_ows = T_cws @ T_oc # [b,4,4]
        points_h = nn.functional.pad(points, (0, 1), "constant", 1.0) # [b,n,4]
        points_w = points_h @ T_cws.permute(0, 2, 1) # [b,n,4]
        return points_w[:, :, :3]       


if __name__ == "__main__":
    # Create a ray sampler
    sampler = RaySampler(
        width=128,
        height=128,
        fx=-110.85124795,
        fy=-110.85124795,
        cx=64,
        cy=64,
        z_near=0.1,
        z_far=3.5,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Camera coordinates:", sampler.camera_coords.shape)

    # Transformation matrix
    T = [[[-3.57627869e-06, -7.65771389e-01, -6.43112957e-01, 8.35809946e-01],
       [-1.00000000e+00, -2.38418579e-05,  3.37362289e-05, -1.97086483e-05],
       [-4.11272049e-05,  6.43112957e-01, -7.65771389e-01, 9.95957587e-01],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]]
    transforms = torch.tensor(T, dtype=torch.float32, device=sampler.device)
    depth_image = torch.ones((1, 128, 128), dtype=torch.float32, device=sampler.device)
    depth_image[:,10:20,10:30] = 3.0

    # Compute the ray origins and directions
    # [b,w*h,3]  [b,w*h,3]     [b,w*h]
    ray_origins,ray_targets, ray_directions, points_mask = sampler.ray_origins_directions(
        transforms, depth_image
    )
    t_vals = torch.linspace(0.0, 1.0, 128, dtype=torch.float32, device=device)
    # [b,w*h,t_vals,3]
    ray_points = (
        ray_directions[:, :, None, :] * t_vals[None, :, None]
        + ray_origins[:, :, None, :]
    ).view(-1, 3)

    # # Visualize the camera coordinates in Open3D
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        ray_points.detach().cpu().numpy().astype(np.float64)
    )
    o3d.visualization.draw_geometries([origin_frame, pcd])
