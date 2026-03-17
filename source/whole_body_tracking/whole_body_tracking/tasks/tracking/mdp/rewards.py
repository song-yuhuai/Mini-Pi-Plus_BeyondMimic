from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

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
