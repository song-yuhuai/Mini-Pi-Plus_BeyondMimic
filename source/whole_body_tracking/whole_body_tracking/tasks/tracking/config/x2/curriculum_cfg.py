from isaaclab.managers import CurriculumTermCfg as CurrTerm

import whole_body_tracking.tasks.tracking.mdp as mdp


def apply_x2_robust_curriculum(env_cfg) -> None:
    """Attach curriculum terms for the X2 robust flat tracking setup."""
    env_cfg.curriculum.push_disturbance = CurrTerm(
        func=mdp.update_push_curriculum,
        params={
            "event_name": "push_robot",
            "start_step": 2_000,
            "end_step": 40_000,
            "start_probability": 0.10,
            "target_probability": 0.35,
            "min_anchor_good_ratio": 0.72,
            "max_body_pos_error": 0.20,
            "max_ori_error": 0.24,
            "max_increase_per_step": 0.01,
            "start_velocity_range": {
                "x": (-0.08, 0.08),
                "y": (-0.08, 0.08),
                "z": (-0.03, 0.03),
                "roll": (-0.10, 0.10),
                "pitch": (-0.10, 0.10),
                "yaw": (-0.12, 0.12),
            },
            "target_velocity_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
                "z": (-0.1, 0.1),
                "roll": (-0.35, 0.35),
                "pitch": (-0.35, 0.35),
                "yaw": (-0.45, 0.45),
            },
        },
    )
    env_cfg.curriculum.actuator_delay = CurrTerm(
        func=mdp.update_actuator_delay_curriculum,
        params={
            "asset_name": "robot",
            "start_step": 5_000,
            "end_step": 45_000,
            "start_delay": 0,
            "target_delay": 2,
            "min_anchor_good_ratio": 0.78,
            "max_body_pos_error": 0.17,
            "max_ori_error": 0.20,
            "max_increase_per_step": 0.01,
        },
    )
    env_cfg.curriculum.observation_noise = CurrTerm(
        func=mdp.update_observation_noise_curriculum,
        params={
            "group_name": "policy",
            "start_step": 1_000,
            "end_step": 30_000,
            "min_anchor_good_ratio": 0.82,
            "max_body_pos_error": 0.14,
            "max_ori_error": 0.16,
            "max_increase_per_step": 0.01,
            "start_noise": {
                "motion_anchor_pos_b": 0.01,
                "base_ang_vel": 0.03,
                "joint_pos": 0.003,
                "joint_vel": 0.12,
                "actions": 0.005,
            },
            "target_noise": {
                "motion_anchor_pos_b": 0.03,
                "base_ang_vel": 0.08,
                "joint_pos": 0.01,
                "joint_vel": 0.4,
                "actions": 0.02,
            },
        },
    )


def disable_x2_curriculum(env_cfg) -> None:
    """Disable optional curriculum terms for play or eval-only configs."""
    env_cfg.curriculum.push_disturbance = None
    env_cfg.curriculum.actuator_delay = None
    env_cfg.curriculum.observation_noise = None
