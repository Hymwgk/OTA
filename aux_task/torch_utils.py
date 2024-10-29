import torch


def look_at_rotation(
    eye: torch.tensor,
    target: torch.tensor,
    ref: torch.tensor = torch.tensor([1.0, 0.0, 0.0]),
    up: torch.tensor = torch.tensor([0.0, 0.0, -1.0]),
) -> torch.tensor:
    """
    Compute the quaternion rotation to look at a target from a given eye position
    :param eye: eye position
    :param target: target position
    :param ref: reference vector
    :return: quaternion rotation
    """
    dir = target - eye
    dir = dir / torch.norm(dir)
    ref = ref.to(dir.device).to(dir.dtype)
    up = up.to(dir.device).to(dir.dtype)
    one = torch.ones(1, device=dir.device, dtype=dir.dtype)
    # Calculate quaternion between reference vector and direction vector
    vec1 = torch.cross(ref, dir)
    w1 = one + torch.dot(ref, dir)
    quat1 = torch.cat((w1, vec1))
    quat1 = quat1 / torch.norm(quat1)
    return quat1


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


def transform_from_rotation_translation(
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
