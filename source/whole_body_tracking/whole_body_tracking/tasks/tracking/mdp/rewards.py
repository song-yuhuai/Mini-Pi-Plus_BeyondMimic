from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import matrix_from_quat, quat_error_magnitude

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward


def feet_slide_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 10.0,
    speed_deadband: float = 0.05,
) -> torch.Tensor:
    """Penalize horizontal foot motion while the corresponding foot is in contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
        > contact_threshold
    )

    asset: Articulation = env.scene[asset_cfg.name]
    body_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2].norm(dim=-1)
    sliding_speed = torch.clamp(body_vel_xy - speed_deadband, min=0.0)
    return torch.sum(sliding_speed * contacts, dim=1)


def feet_contact_switch_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize frequent contact-state toggles to reduce chattering and tiny shuffle steps."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0

    prev_attr = f"_contact_switch_prev_{sensor_cfg.name}"
    ep_len_attr = f"_contact_switch_prev_ep_len_{sensor_cfg.name}"

    if not hasattr(env, prev_attr):
        setattr(env, prev_attr, torch.zeros_like(in_contact, dtype=torch.bool))
    if not hasattr(env, ep_len_attr):
        setattr(env, ep_len_attr, torch.zeros_like(env.episode_length_buf))

    prev_contact = getattr(env, prev_attr)
    prev_ep_len = getattr(env, ep_len_attr)

    reset_envs = env.episode_length_buf <= prev_ep_len
    if torch.any(reset_envs):
        prev_contact[reset_envs] = in_contact[reset_envs]

    switches = (in_contact ^ prev_contact).float().sum(dim=1)

    prev_contact.copy_(in_contact)
    prev_ep_len.copy_(env.episode_length_buf)
    return switches


def feet_distance_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    soft_threshold: float = 0.18,
    hard_threshold: float = 0.16,
    hard_scale: float = 8.0,
    use_xy: bool = True,
) -> torch.Tensor:
    """Penalize feet getting too close, with a sharper increase below the hard threshold."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]
    if body_pos.shape[1] != 2:
        raise ValueError("feet_distance_penalty expects exactly two body names in asset_cfg.body_names.")

    pos_dim = 2 if use_xy else 3
    feet_delta = body_pos[:, 0, :pos_dim] - body_pos[:, 1, :pos_dim]
    feet_distance = torch.norm(feet_delta, dim=-1)

    soft_gap = torch.clamp(soft_threshold - feet_distance, min=0.0)
    hard_gap = torch.clamp(hard_threshold - feet_distance, min=0.0)

    return soft_gap.square() + hard_scale * hard_gap.square()


def joint_pos_target_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target: float | list[float] = 0.0,
) -> torch.Tensor:
    """L1 penalty of selected joint positions to a target position."""
    asset: Articulation = env.scene[asset_cfg.name]

    if asset_cfg.joint_ids == slice(None):
        joint_pos = asset.data.joint_pos
    else:
        joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]

    target_tensor = torch.as_tensor(target, dtype=joint_pos.dtype, device=joint_pos.device)
    if target_tensor.ndim == 0:
        target_tensor = target_tensor.expand(joint_pos.shape[-1])

    error = torch.abs(joint_pos - target_tensor)
    return torch.sum(error, dim=-1)


def motion_joint_position_error_exp_windowed(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    last_n_steps: int = 100,
    ramp: bool = True,
) -> torch.Tensor:
    """Track reference joint positions, emphasizing the terminal phase of the motion.

    The reward is computed over all joints exposed by the motion command. It is
    linearly ramped up over the final ``last_n_steps`` frames so the policy gets
    stronger supervision near the clip end without a hard reward discontinuity.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.square(command.joint_pos - command.robot_joint_pos)
    reward = torch.exp(-error.mean(dim=-1) / std**2)

    if last_n_steps <= 0:
        return reward

    window_start = max(command.phase_end_count - last_n_steps + 1, command.phase_start_count)
    if ramp:
        denom = max(command.phase_end_count - window_start + 1, 1)
        phase_scale = (command.time_steps - window_start + 1).float() / float(denom)
        phase_scale = torch.clamp(phase_scale, min=0.0, max=1.0)
    else:
        phase_scale = (command.time_steps >= window_start).float()
    return reward * phase_scale


def action_acc_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """L2 penalty on action second-order difference (discrete acceleration / jerk proxy).

    Uses persistent per-env buffers on the env instance:
    - previous action rate: (a_t - a_{t-1})
    - previous episode length snapshot to detect resets
    """
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    action_rate = action - prev_action

    # Lazily initialize persistent buffers.
    if not hasattr(env, "_action_acc_prev_rate"):
        env._action_acc_prev_rate = torch.zeros_like(action_rate)
    if not hasattr(env, "_action_acc_prev_ep_len"):
        env._action_acc_prev_ep_len = torch.zeros_like(env.episode_length_buf)

    # Reset buffer on newly reset envs (episode length decreases / restarts).
    reset_envs = env.episode_length_buf <= env._action_acc_prev_ep_len
    if torch.any(reset_envs):
        env._action_acc_prev_rate[reset_envs] = 0.0

    action_acc = action_rate - env._action_acc_prev_rate
    env._action_acc_prev_rate.copy_(action_rate.detach())
    env._action_acc_prev_ep_len.copy_(env.episode_length_buf)

    return torch.sum(torch.square(action_acc), dim=1)


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """L2 penalty on joint acceleration computed from joint velocity differences."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel if asset_cfg.joint_ids == slice(None) else asset.data.joint_vel[:, asset_cfg.joint_ids]

    # Lazily initialize persistent buffers.
    if not hasattr(env, "_joint_acc_prev_vel"):
        env._joint_acc_prev_vel = torch.zeros_like(joint_vel)
    if not hasattr(env, "_joint_acc_prev_ep_len"):
        env._joint_acc_prev_ep_len = torch.zeros_like(env.episode_length_buf)

    # Reset buffer on newly reset envs (episode length decreases / restarts).
    reset_envs = env.episode_length_buf <= env._joint_acc_prev_ep_len
    if torch.any(reset_envs):
        env._joint_acc_prev_vel[reset_envs] = joint_vel[reset_envs]

    joint_acc = joint_vel - env._joint_acc_prev_vel
    env._joint_acc_prev_vel.copy_(joint_vel.detach())
    env._joint_acc_prev_ep_len.copy_(env.episode_length_buf)

    return torch.sum(torch.square(joint_acc), dim=1)


def action_jerk_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """L2 penalty on action third-order difference (discrete jerk)."""
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    action_rate = action - prev_action

    if not hasattr(env, "_action_jerk_prev_rate"):
        env._action_jerk_prev_rate = torch.zeros_like(action_rate)
    if not hasattr(env, "_action_jerk_prev_acc"):
        env._action_jerk_prev_acc = torch.zeros_like(action_rate)
    if not hasattr(env, "_action_jerk_prev_ep_len"):
        env._action_jerk_prev_ep_len = torch.zeros_like(env.episode_length_buf)

    reset_envs = env.episode_length_buf <= env._action_jerk_prev_ep_len
    if torch.any(reset_envs):
        env._action_jerk_prev_rate[reset_envs] = 0.0
        env._action_jerk_prev_acc[reset_envs] = 0.0

    action_acc = action_rate - env._action_jerk_prev_rate
    action_jerk = action_acc - env._action_jerk_prev_acc

    env._action_jerk_prev_rate.copy_(action_rate.detach())
    env._action_jerk_prev_acc.copy_(action_acc.detach())
    env._action_jerk_prev_ep_len.copy_(env.episode_length_buf)

    return torch.sum(torch.square(action_jerk), dim=1)


def joint_jerk_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """L2 penalty on joint jerk computed from joint acceleration differences."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel if asset_cfg.joint_ids == slice(None) else asset.data.joint_vel[:, asset_cfg.joint_ids]

    if not hasattr(env, "_joint_jerk_prev_vel"):
        env._joint_jerk_prev_vel = torch.zeros_like(joint_vel)
    if not hasattr(env, "_joint_jerk_prev_acc"):
        env._joint_jerk_prev_acc = torch.zeros_like(joint_vel)
    if not hasattr(env, "_joint_jerk_prev_ep_len"):
        env._joint_jerk_prev_ep_len = torch.zeros_like(env.episode_length_buf)

    reset_envs = env.episode_length_buf <= env._joint_jerk_prev_ep_len
    if torch.any(reset_envs):
        env._joint_jerk_prev_vel[reset_envs] = joint_vel[reset_envs]
        env._joint_jerk_prev_acc[reset_envs] = 0.0

    joint_acc = joint_vel - env._joint_jerk_prev_vel
    joint_jerk = joint_acc - env._joint_jerk_prev_acc

    env._joint_jerk_prev_vel.copy_(joint_vel.detach())
    env._joint_jerk_prev_acc.copy_(joint_acc.detach())
    env._joint_jerk_prev_ep_len.copy_(env.episode_length_buf)

    return torch.sum(torch.square(joint_jerk), dim=1)


def _point_to_segment_distance_sq_2d(points: torch.Tensor, seg_a: torch.Tensor, seg_b: torch.Tensor) -> torch.Tensor:
    """Batched squared distance from 2D points to 2D segments.

    Args:
        points: [N, 2]
        seg_a: [N, 2]
        seg_b: [N, 2]
    Returns:
        distance_sq: [N]
    """
    ab = seg_b - seg_a
    ap = points - seg_a
    denom = torch.sum(ab * ab, dim=-1).clamp_min(1.0e-8)
    t = torch.sum(ap * ab, dim=-1) / denom
    t = torch.clamp(t, 0.0, 1.0)
    proj = seg_a + t.unsqueeze(-1) * ab
    return torch.sum((points - proj) ** 2, dim=-1)


def _segments_intersect_2d(a0: torch.Tensor, a1: torch.Tensor, b0: torch.Tensor, b1: torch.Tensor) -> torch.Tensor:
    """Batched segment intersection test in 2D (non-collinear case + tolerant boundaries)."""
    r = a1 - a0
    s = b1 - b0
    w = b0 - a0

    def cross2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]

    denom = cross2(r, s)
    numer_t = cross2(w, s)
    numer_u = cross2(w, r)

    eps = 1.0e-8
    non_parallel = torch.abs(denom) > eps
    t = torch.zeros_like(denom)
    u = torch.zeros_like(denom)
    t[non_parallel] = numer_t[non_parallel] / denom[non_parallel]
    u[non_parallel] = numer_u[non_parallel] / denom[non_parallel]

    return non_parallel & (t >= -eps) & (t <= 1.0 + eps) & (u >= -eps) & (u <= 1.0 + eps)


def feet_capsule_overlap_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    foot_length: float,
    foot_width: float,
    foot_center_offset_xy: tuple[float, float] = (0.0, 0.0),
    safety_margin: float = 0.0,
    penetration_in_cm: bool = True,
) -> torch.Tensor:
    """Penalty based on overlap between two yaw-oriented foot capsules on XY plane.

    Each foot is modeled as a 2D capsule (stadium / runway shape):
    - capsule radius = effective_width / 2
    - center segment length = max(effective_length - effective_width, 0)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice) or len(body_ids) != 2:
        raise ValueError("feet_capsule_overlap_penalty expects exactly two body names in asset_cfg.body_names.")

    feet_pos_w = asset.data.body_pos_w[:, body_ids, :2]  # [N, 2, 2]
    feet_quat_w = asset.data.body_quat_w[:, body_ids]  # [N, 2, 4]

    # Build world-frame XY axes from each foot local frame (yaw-aware, robust to roll/pitch).
    feet_rot = matrix_from_quat(feet_quat_w.reshape(-1, 4)).reshape(-1, 2, 3, 3)  # [N, 2, 3, 3]
    feet_x_axis_xy = feet_rot[:, :, :2, 0]  # local +X projected to XY, [N, 2, 2]
    feet_y_axis_xy = feet_rot[:, :, :2, 1]  # local +Y projected to XY, [N, 2, 2]
    feet_x_axis_xy = feet_x_axis_xy / feet_x_axis_xy.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    feet_y_axis_xy = feet_y_axis_xy / feet_y_axis_xy.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)

    offset_local = torch.tensor(foot_center_offset_xy, device=feet_pos_w.device, dtype=feet_pos_w.dtype)
    feet_center_w = feet_pos_w + feet_x_axis_xy * offset_local[0] + feet_y_axis_xy * offset_local[1]

    length_eff = max(foot_length + safety_margin, 0.0)
    width_eff = max(foot_width + safety_margin, 0.0)
    radius = 0.5 * width_eff
    seg_half = 0.5 * max(length_eff - width_eff, 0.0)

    c1 = feet_center_w[:, 0]
    c2 = feet_center_w[:, 1]
    u1 = feet_x_axis_xy[:, 0]
    u2 = feet_x_axis_xy[:, 1]

    # Capsule center-line segment endpoints.
    a0 = c1 - seg_half * u1
    a1 = c1 + seg_half * u1
    b0 = c2 - seg_half * u2
    b1 = c2 + seg_half * u2

    # Segment-segment minimum distance in 2D via endpoint-to-segment distances + intersection test.
    d2_candidates = torch.stack(
        (
            _point_to_segment_distance_sq_2d(a0, b0, b1),
            _point_to_segment_distance_sq_2d(a1, b0, b1),
            _point_to_segment_distance_sq_2d(b0, a0, a1),
            _point_to_segment_distance_sq_2d(b1, a0, a1),
        ),
        dim=1,
    )
    seg_dist_sq = torch.min(d2_candidates, dim=1).values
    seg_dist = torch.sqrt(seg_dist_sq.clamp_min(0.0))
    intersect_mask = _segments_intersect_2d(a0, a1, b0, b1)
    seg_dist = torch.where(intersect_mask, torch.zeros_like(seg_dist), seg_dist)

    penetration = torch.clamp(2.0 * radius - seg_dist, min=0.0)
    if penetration_in_cm:
        penetration = penetration * 100.0
    return penetration * penetration


def cog_tracking_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    feet_body_names: tuple[str, str] | list[str],
    sigma: float = 0.2,
) -> torch.Tensor:
    """Reward the XY projection of the center of gravity staying near the midpoint between the feet."""
    asset: Articulation = env.scene[asset_cfg.name]

    if not hasattr(env, "_cog_tracking_body_masses"):
        env._cog_tracking_body_masses = {}

    mass = env._cog_tracking_body_masses.get(asset_cfg.name)
    if mass is None:
        mass = asset.root_physx_view.get_masses().clone().to(device=asset.data.body_pos_w.device, dtype=asset.data.body_pos_w.dtype)
        if mass.ndim == 1:
            mass = mass.unsqueeze(0)
        env._cog_tracking_body_masses[asset_cfg.name] = mass

    body_pos_w = asset.data.body_pos_w
    if mass.shape[0] != body_pos_w.shape[0]:
        mass = mass[0].unsqueeze(0).expand(body_pos_w.shape[0], -1)

    total_mass = torch.sum(mass, dim=1, keepdim=True).clamp_min(1.0e-8)
    cog_xy = torch.sum(mass.unsqueeze(-1) * body_pos_w, dim=1)[:, :2] / total_mass

    foot_body_ids = asset.find_bodies(list(feet_body_names), preserve_order=True)[0]
    if len(foot_body_ids) != 2:
        raise ValueError("cog_tracking_reward expects exactly two foot body names.")

    feet_mid_xy = asset.data.body_pos_w[:, foot_body_ids, :2].mean(dim=1)
    dist = torch.norm(cog_xy - feet_mid_xy, dim=1)
    return torch.exp(-(dist**2) / (sigma**2))
