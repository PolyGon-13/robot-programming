import math

import torch

L1 = 0.143
JOINT2_OFFSET = (0.143, 0.0, 0.052)
JOINT1_HEIGHT = 0.128
LINK2_HEIGHT = JOINT1_HEIGHT + JOINT2_OFFSET[2]

R_LINK2_OPTICAL = torch.tensor([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def rot_z(yaw: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(yaw), torch.sin(yaw)
    z, o = torch.zeros_like(yaw), torch.ones_like(yaw)
    return torch.stack((c, -s, z, s, c, z, z, z, o), dim=-1).reshape(*yaw.shape, 3, 3)


def link2_pose(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q1, q2 = q[..., 0], q[..., 1]
    pos = torch.stack((L1 * torch.cos(q1), L1 * torch.sin(q1), torch.full_like(q1, LINK2_HEIGHT)), dim=-1)
    return pos, rot_z(q1 + q2)


def camera_pose(q: torch.Tensor, mount_pos: torch.Tensor, mount_yaw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    p2, r2 = link2_pose(q)
    r_l2c = rot_z(mount_yaw) @ R_LINK2_OPTICAL.to(q.device)
    pos = p2 + (r2 @ mount_pos.unsqueeze(-1)).squeeze(-1)
    return pos, r2 @ r_l2c


def marker_rotation(normal_yaw: torch.Tensor) -> torch.Tensor:
    n = torch.stack((torch.cos(normal_yaw), torch.sin(normal_yaw), torch.zeros_like(normal_yaw)), dim=-1)
    up = torch.zeros_like(n)
    up[..., 2] = 1.0
    x = torch.cross(up, n, dim=-1)
    return torch.stack((x, up, n), dim=-1)


def marker_in_camera(
    cam_pos: torch.Tensor, cam_rot: torch.Tensor, marker_pos: torch.Tensor, marker_rot: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rt = cam_rot.transpose(-1, -2)
    p = (rt @ (marker_pos - cam_pos).unsqueeze(-1)).squeeze(-1)
    return p, rt @ marker_rot


def tracking_errors(p_cam: torch.Tensor, r_cam: torch.Tensor, d_goal: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    e_theta = torch.atan2(p_cam[..., 0], p_cam[..., 2])
    e_d = p_cam[..., 2] - d_goal
    n = r_cam[..., :, 2]
    e_psi = torch.atan2(n[..., 0], -n[..., 2])
    return e_theta, e_d, e_psi


def visible(
    p_cam: torch.Tensor, r_cam: torch.Tensor, tan_half_h: float, tan_half_v: float, z_range: tuple[float, float],
    max_view_angle: float, half_size: float = 0.0,
) -> torch.Tensor:
    z = p_cam[..., 2]
    in_fov = (p_cam[..., 0].abs() + half_size < z * tan_half_h) & (p_cam[..., 1].abs() + half_size < z * tan_half_v)
    facing = -r_cam[..., 2, 2] > math.cos(max_view_angle)
    return (z > z_range[0]) & (z < z_range[1]) & in_fov & facing
