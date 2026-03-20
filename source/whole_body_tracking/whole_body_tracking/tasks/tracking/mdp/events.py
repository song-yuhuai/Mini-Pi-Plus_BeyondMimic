from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal, Callable

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op, push_by_setting_velocity
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _get_curriculum_alpha(env: ManagerBasedEnv, start_step: int, end_step: int) -> float:
    """Return a normalized curriculum progress in ``[0, 1]`` based on the global env step."""
    step = getattr(env, "common_step_counter", 0)
    if isinstance(step, torch.Tensor):
        step = int(step.item())
    else:
        step = int(step)

    if end_step <= start_step:
        return 1.0
    if step <= start_step:
        return 0.0
    if step >= end_step:
        return 1.0
    return float(step - start_step) / float(end_step - start_step)


def _lerp_scalar(start: float, end: float, alpha: float) -> float:
    return (1.0 - alpha) * start + alpha * end


def _lerp_range(
    start_range: tuple[float, float], target_range: tuple[float, float], alpha: float
) -> tuple[float, float]:
    return (
        _lerp_scalar(start_range[0], target_range[0], alpha),
        _lerp_scalar(start_range[1], target_range[1], alpha),
    )


def _lerp_range_dict(
    start_ranges: dict[str, tuple[float, float]],
    target_ranges: dict[str, tuple[float, float]],
    alpha: float,
) -> dict[str, tuple[float, float]]:
    keys = target_ranges.keys() | start_ranges.keys()
    return {
        key: _lerp_range(start_ranges.get(key, (0.0, 0.0)), target_ranges.get(key, (0.0, 0.0)), alpha)
        for key in keys
    }


def _set_term_noise(term, magnitude: float) -> None:
    if term is None or getattr(term, "noise", None) is None:
        return
    term.noise.n_min = -magnitude
    term.noise.n_max = magnitude


def _get_motion_command(env: ManagerBasedEnv, command_name: str = "motion"):
    return env.command_manager.get_term(command_name)


def _performance_is_good(
    env: ManagerBasedEnv,
    command_name: str = "motion",
    min_anchor_good_ratio: float = 0.75,
    max_body_pos_error: float = 0.18,
    max_ori_error: float = 0.22,
) -> bool:
    """Check whether the current batch performance is good enough to unlock more difficulty."""
    motion_command = _get_motion_command(env, command_name)
    anchor_good_ratio = motion_command.anchor_conditions_good.float().mean().item()
    body_pos_error = motion_command.metrics["error_body_pos"].mean().item()
    ori_error = motion_command.metrics["ori_error"].mean().item()
    return (
        anchor_good_ratio >= min_anchor_good_ratio
        and body_pos_error <= max_body_pos_error
        and ori_error <= max_ori_error
    )


def _update_hybrid_curriculum_state(
    env: ManagerBasedEnv,
    state_key: str,
    target_alpha: float,
    performance_gate: bool,
    max_increase_per_step: float,
) -> float:
    """Keep a monotonic curriculum state that advances only when performance is good."""
    if not hasattr(env, "_curriculum_state"):
        env._curriculum_state = {}

    current_alpha = float(env._curriculum_state.get(state_key, 0.0))
    if performance_gate:
        current_alpha = min(target_alpha, current_alpha + max_increase_per_step)
    env._curriculum_state[state_key] = max(current_alpha, 0.0)
    return env._curriculum_state[state_key]


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)


def conditional_push_by_setting_velocity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    velocity_range: dict[str, tuple[float, float]],
    condition_func: Callable[[ManagerBasedEnv, torch.Tensor], torch.Tensor] | None = None,
    condition_params: dict | None = None,
):
    """有条件地推动机器人，只对满足特定条件的环境执行推动。
    
    这个函数是 `push_by_setting_velocity` 的扩展版本，允许根据自定义条件
    选择性地对环境执行推动操作。
    
    Args:
        env: 环境实例
        env_ids: 要考虑推动的环境ID。如果为None，则考虑所有环境
        velocity_range: 推动速度范围字典，包含线速度和角速度的范围
        condition_func: 条件判断函数，接收(env, env_ids)参数，返回布尔张量
        condition_params: 传递给条件函数的额外参数
        
    Returns:
        None
        
    Examples:
        # 示例1: 只推动机器人高度低于某个阈值的环境
        def height_condition(env, env_ids):
            robot = env.scene["robot"]
            base_pos = robot.data.root_pos_w[env_ids, 2]  # Z坐标
            return base_pos < 0.8  # 高度低于0.8m
            
        # 示例2: 只推动机器人速度低于某个阈值的环境  
        def velocity_condition(env, env_ids):
            robot = env.scene["robot"]
            base_vel = torch.norm(robot.data.root_lin_vel_w[env_ids, :2], dim=1)  # XY平面速度
            return base_vel < 0.5  # 速度低于0.5m/s
            
        # 示例3: 只推动跟踪误差大于某个阈值的环境
        def tracking_error_condition(env, env_ids):
            command_manager = env.command_manager
            motion_command = command_manager.get_command("motion")
            # 计算跟踪误差（这里需要根据具体实现调整）
            error = compute_tracking_error(env, env_ids, motion_command)
            return error > 0.3  # 误差大于0.3
    """
    # 如果没有指定环境ID，则考虑所有环境
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    
    # 如果没有条件函数，则对所有指定环境执行推动
    if condition_func is None:
        push_env_ids = env_ids
    else:
        # 应用条件函数筛选环境
        if condition_params is None:
            condition_params = {}
        
        # 调用条件函数获取满足条件的环境掩码
        condition_mask = condition_func(env, env_ids, **condition_params)
        
        # 筛选出满足条件的环境ID
        push_env_ids = env_ids[condition_mask]
    
    # 如果有满足条件的环境，则执行推动
    if len(push_env_ids) > 0:
        push_by_setting_velocity(env, push_env_ids, velocity_range)


# 预定义的条件函数示例

def height_based_condition(
    env: ManagerBasedEnv, 
    env_ids: torch.Tensor, 
    height_threshold: float = 0.8,
    below_threshold: bool = True
) -> torch.Tensor:
    """基于机器人高度的条件函数
    
    Args:
        env: 环境实例
        env_ids: 环境ID
        height_threshold: 高度阈值
        below_threshold: True表示高度低于阈值时推动，False表示高度高于阈值时推动
        
    Returns:
        满足条件的环境掩码
    """
    robot = env.scene["robot"]
    base_height = robot.data.root_pos_w[env_ids, 2]
    
    if below_threshold:
        return base_height < height_threshold
    else:
        return base_height > height_threshold


def velocity_based_condition(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_threshold: float = 0.5,
    below_threshold: bool = True
) -> torch.Tensor:
    """基于机器人速度的条件函数
    
    Args:
        env: 环境实例
        env_ids: 环境ID
        velocity_threshold: 速度阈值
        below_threshold: True表示速度低于阈值时推动，False表示速度高于阈值时推动
        
    Returns:
        满足条件的环境掩码
    """
    robot = env.scene["robot"]
    base_vel_magnitude = torch.norm(robot.data.root_lin_vel_w[env_ids, :2], dim=1)
    
    if below_threshold:
        return base_vel_magnitude < velocity_threshold
    else:
        return base_vel_magnitude > velocity_threshold

def ori_error_condition(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    command_name: str = "motion",
    error_threshold: float = 0.3,
    above_threshold: bool = True
) -> torch.Tensor:
    """基于方向误差的条件函数
    
    Args:
        env: 环境实例
        env_ids: 环境ID
        command_name: 运动命令名称
        error_threshold: 误差阈值
        above_threshold: True表示误差大于阈值时推动，False表示误差小于阈值时推动
        
    Returns:
        满足条件的环境掩码
    """
    # 获取运动命令
    command_manager = env.command_manager
    motion_command = command_manager.get_term(command_name)
    
    # 获取机器人当前状态
    ori_error = motion_command.ori_error[env_ids]
    
    if above_threshold:
        return ori_error > error_threshold
    else:
        return ori_error < error_threshold


def tracking_error_condition(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    command_name: str = "motion",
    error_threshold: float = 0.3,
    above_threshold: bool = True
) -> torch.Tensor:
    """基于跟踪误差的条件函数
    
    Args:
        env: 环境实例
        env_ids: 环境ID
        command_name: 运动命令名称
        error_threshold: 误差阈值
        above_threshold: True表示误差大于阈值时推动，False表示误差小于阈值时推动
        
    Returns:
        满足条件的环境掩码
    """
    # 获取运动命令
    command_manager = env.command_manager
    motion_command = command_manager.get_command(command_name)
    
    # 获取机器人当前状态
    robot = env.scene["robot"]
    current_pos = robot.data.root_pos_w[env_ids]
    
    # 获取目标位置（这里简化为锚点位置，实际可能需要更复杂的计算）
    target_pos = motion_command.anchor_pos_w[env_ids]
    
    # 计算位置误差
    pos_error = torch.norm(current_pos - target_pos, dim=1)
    
    if above_threshold:
        return pos_error > error_threshold
    else:
        return pos_error < error_threshold


def random_condition(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    probability: float = 0.5
) -> torch.Tensor:
    """随机条件函数，以指定概率选择环境
    
    Args:
        env: 环境实例
        env_ids: 环境ID
        probability: 选择概率
        
    Returns:
        满足条件的环境掩码
    """
    return torch.rand(len(env_ids), device=env.device) < probability


def update_force_curriculum(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    command_name: str = "motion",
    update_interval: int = 10,
    force_reduction_rate: float = 10.0,
    min_force: float = 0.0,
    max_force: float = 500.0,
    standing_base_force: float = 50.0,
    ori_threshold: float = 0.3,
) -> None:
    """力课程学习函数 - 根据机器人表现动态调整辅助力
    
    这个函数作为课程学习项被调用，根据机器人的跟踪表现动态调整辅助力：
    - 对于表现良好的非站立环境，减少辅助力
    - 对于表现不佳的环境，保持或增加辅助力
    
    Args:
        env: 环境实例
        env_ids: 环境ID（通常为所有环境）
        command_name: 运动命令名称
        update_interval: 更新间隔（步数）
        force_reduction_rate: 力减少率
        min_force: 最小辅助力
        max_force: 最大辅助力
        standing_base_force: 站立环境基础力
        ori_threshold: 方向误差阈值
    """
    # 获取运动命令
    command_manager = env.command_manager
    motion_command = command_manager.get_term(command_name)
    
    # 检查是否需要更新（基于步数间隔）
    # if not hasattr(motion_command, '_force_curriculum_counter'):
    #     motion_command._force_curriculum_counter = 0
    
    # motion_command._force_curriculum_counter += 1
    
    # # 只在指定间隔更新
    # if motion_command._force_curriculum_counter % update_interval != 0:
    #     return
    
    # 调用 MotionCommand 的 update_force_curriculum 方法
    motion_command.update_force_curriculum(env_ids)


def update_push_curriculum(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    event_name: str = "push_robot",
    command_name: str = "motion",
    start_step: int = 0,
    end_step: int = 10000,
    start_probability: float = 0.1,
    target_probability: float = 0.35,
    start_velocity_range: dict[str, tuple[float, float]] | None = None,
    target_velocity_range: dict[str, tuple[float, float]] | None = None,
    min_anchor_good_ratio: float = 0.75,
    max_body_pos_error: float = 0.18,
    max_ori_error: float = 0.22,
    max_increase_per_step: float = 0.01,
) -> None:
    """Ramp push disturbance with a time cap and a performance gate."""
    del env_ids
    event_term = getattr(env.cfg.events, event_name, None)
    if event_term is None:
        return

    time_alpha = _get_curriculum_alpha(env, start_step, end_step)
    performance_gate = _performance_is_good(
        env,
        command_name=command_name,
        min_anchor_good_ratio=min_anchor_good_ratio,
        max_body_pos_error=max_body_pos_error,
        max_ori_error=max_ori_error,
    )
    alpha = _update_hybrid_curriculum_state(
        env,
        state_key=f"{event_name}_curriculum_alpha",
        target_alpha=time_alpha,
        performance_gate=performance_gate,
        max_increase_per_step=max_increase_per_step,
    )
    target_velocity_range = target_velocity_range or {}
    if start_velocity_range is None:
        start_velocity_range = {key: (0.0, 0.0) for key in target_velocity_range}

    event_term.params["velocity_range"] = _lerp_range_dict(start_velocity_range, target_velocity_range, alpha)
    condition_params = event_term.params.setdefault("condition_params", {})
    condition_params["probability"] = _lerp_scalar(start_probability, target_probability, alpha)


def update_actuator_delay_curriculum(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_name: str = "robot",
    command_name: str = "motion",
    start_step: int = 0,
    end_step: int = 10000,
    start_delay: int = 0,
    target_delay: int = 2,
    min_anchor_good_ratio: float = 0.8,
    max_body_pos_error: float = 0.16,
    max_ori_error: float = 0.18,
    max_increase_per_step: float = 0.01,
) -> None:
    """Ramp actuator delay with a time cap and a performance gate."""
    del env_ids
    asset_cfg = getattr(env.cfg.scene, asset_name, None)
    if asset_cfg is None:
        return

    time_alpha = _get_curriculum_alpha(env, start_step, end_step)
    performance_gate = _performance_is_good(
        env,
        command_name=command_name,
        min_anchor_good_ratio=min_anchor_good_ratio,
        max_body_pos_error=max_body_pos_error,
        max_ori_error=max_ori_error,
    )
    alpha = _update_hybrid_curriculum_state(
        env,
        state_key=f"{asset_name}_delay_curriculum_alpha",
        target_alpha=time_alpha,
        performance_gate=performance_gate,
        max_increase_per_step=max_increase_per_step,
    )
    max_delay = int(round(_lerp_scalar(start_delay, target_delay, alpha)))
    for actuator_cfg in asset_cfg.actuators.values():
        actuator_cfg.min_delay = 0
        actuator_cfg.max_delay = max_delay


def update_observation_noise_curriculum(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    group_name: str = "policy",
    command_name: str = "motion",
    start_step: int = 0,
    end_step: int = 10000,
    start_noise: dict[str, float] | None = None,
    target_noise: dict[str, float] | None = None,
    min_anchor_good_ratio: float = 0.82,
    max_body_pos_error: float = 0.14,
    max_ori_error: float = 0.16,
    max_increase_per_step: float = 0.01,
) -> None:
    """Ramp observation noise with a time cap and a performance gate."""
    del env_ids
    obs_group = getattr(env.cfg.observations, group_name, None)
    if obs_group is None:
        return

    time_alpha = _get_curriculum_alpha(env, start_step, end_step)
    performance_gate = _performance_is_good(
        env,
        command_name=command_name,
        min_anchor_good_ratio=min_anchor_good_ratio,
        max_body_pos_error=max_body_pos_error,
        max_ori_error=max_ori_error,
    )
    alpha = _update_hybrid_curriculum_state(
        env,
        state_key=f"{group_name}_noise_curriculum_alpha",
        target_alpha=time_alpha,
        performance_gate=performance_gate,
        max_increase_per_step=max_increase_per_step,
    )
    target_noise = target_noise or {}
    start_noise = start_noise or {key: 0.0 for key in target_noise}

    for term_name, target_magnitude in target_noise.items():
        term_cfg = getattr(obs_group, term_name, None)
        if term_cfg is None:
            continue
        magnitude = _lerp_scalar(start_noise.get(term_name, 0.0), target_magnitude, alpha)
        _set_term_noise(term_cfg, magnitude)
   
