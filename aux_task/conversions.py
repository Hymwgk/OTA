"""
Author: Akshay K. Burusa
Maintainer: Akshay K. Burusa
"""

import torch
import numpy as np


def T_from_rot_trans_np(rot, trans):
    """
    Convert rotation matrix and translation vector to a 4x4 transformation matrice.
    :param rot: numpy array of rotation matrices (3, 3)
    :param trans: numpy array of translation vectors (3)
    :return: numpy array of transformation matrices (4, 4)
    """
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3:] = trans.T
    return T


def T_from_rot_trans(
    quaternions: torch.Tensor, translations: torch.Tensor
) -> torch.Tensor:
    """
    Convert rotations given as quaternions and translation vectors to 4x4 transformation matrices.
    :param quaternions: tensor of quaternions (..., 4) ordered (w, x, y, z)
    :param translations: tensor of translation vectors (..., 3)
    :return: tensor of transformation matrices (..., 4, 4)
    """
    matrices = (
        torch.eye(4, device=quaternions.device)
        .unsqueeze(0)
        .repeat(quaternions.shape[0], 1, 1)
    )
    matrices[:, :3, :3] = quaternion_to_matrix(quaternions)
    matrices[:, :3, 3] = translations
    return matrices


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.
    :param quaternions: tensor of quaternions (..., 4) ordered (w, x, y, z)
    :return: tensor of rotation matrices (..., 3, 3)
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))
