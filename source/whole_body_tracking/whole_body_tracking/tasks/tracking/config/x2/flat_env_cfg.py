from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.robots.x2 import X2_ACTION_SCALE, X2_CFG
from whole_body_tracking.tasks.tracking.config.x2.curriculum_cfg import (
    apply_x2_robust_curriculum,
    disable_x2_curriculum,
)
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class X2BaseEnvCfg(TrackingEnvCfg):
    """Flat-ground X2 tracking baseline."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = X2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.stair_step = None
        self.scene.terrain.terrain_type = "plane"

        self.actions.joint_pos.scale = X2_ACTION_SCALE
        self.actions.joint_pos.joint_names = [".*"]
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}

        self.commands.motion.joint_names = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_pitch_joint",
            "waist_roll_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_yaw_joint",
            "left_wrist_pitch_joint",
            "left_wrist_roll_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_yaw_joint",
            "right_wrist_pitch_joint",
            "right_wrist_roll_joint",
        ]
        self.commands.motion.anchor_body_name = "pelvis"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "waist_yaw_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]
        self.commands.motion.pose_range = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.commands.motion.velocity_range = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.commands.motion.joint_position_range = (0.0, 0.0)
        self.commands.motion.phase_start_count = 0
        self.commands.motion.phase_end_count = -1

        self.observations.policy.motion_anchor_pos_b = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        self.observations.policy.motion_anchor_ori_b = None
        self.observations.policy.base_lin_vel = None
        self.observations.policy.enable_corruption = False
        self.observations.policy.command.clip = (-100.0, 100.0)
        self.observations.policy.command.scale = 1.0
        self.observations.policy.base_ang_vel.clip = (-100.0, 100.0)
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.clip = (-100.0, 100.0)
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.clip = (-100.0, 100.0)
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.actions.clip = (-100.0, 100.0)
        self.observations.policy.actions.scale = 1.0

        self.viewer.eye = (3.2, 2.5, 2.2)
        self.viewer.lookat = (0.8, 0.0, 0.9)
        self.viewer.origin_type = "world"
        self.viewer.asset_name = None
        self.scene.contact_forces.debug_vis = False

        self.rewards.motion_body_pos.weight = 1.6
        self.rewards.motion_body_pos.params["std"] = 0.12
        self.rewards.action_rate_l2.weight = -0.20
        self.rewards.action_acc_l2.weight = -1e-2
        self.rewards.joint_acc_l2.weight = -7e-3
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=[
                r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_elbow_link$)(?!right_elbow_link$).+$"
            ],
        )
        self.rewards.joint_pos_target = RewTerm(
            func=mdp.joint_pos_target_l1,
            weight=-1.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_wrist_yaw_joint",
                        "left_wrist_pitch_joint",
                        "left_wrist_roll_joint",
                        "right_wrist_yaw_joint",
                        "right_wrist_pitch_joint",
                        "right_wrist_roll_joint",
                    ],
                ),
                "target": 0.0,
            },
        )
        self.rewards.feet_capsule_overlap = RewTerm(
            func=mdp.feet_capsule_overlap_penalty,
            weight=-0.05,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                ),
                "foot_length": 0.22,
                "foot_width": 0.130,
                "foot_center_offset_xy": (0.037, 0.0),
                "safety_margin": 0.04,
                "penetration_in_cm": True,
            },
        )
        self.rewards.feet_distance = RewTerm(
            func=mdp.feet_distance_penalty,
            weight=-80.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                ),
                "soft_threshold": 0.18,
                "hard_threshold": 0.16,
                "hard_scale": 50.0,
                "use_xy": True,
            },
        )
        self.rewards.cog_tracking = RewTerm(
            func=mdp.cog_tracking_reward,
            weight=0.25,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "feet_body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
                "sigma": 0.2,
            },
        )
        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide_penalty,
            weight=-0.08,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                ),
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                ),
                "contact_threshold": 10.0,
                "speed_deadband": 0.05,
            },
        )
        self.rewards.feet_contact_switch = RewTerm(
            func=mdp.feet_contact_switch_penalty,
            weight=-0.02,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                ),
            },
        )

        self.terminations.ee_body_pos.params["body_names"] = [
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_elbow_link",
            "right_elbow_link",
        ]

        self.events.base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
                "com_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            },
        )
        self.events.add_joint_default_pos = None


@configclass
class X2RobustEnvCfg(X2BaseEnvCfg):
    """Flat-ground X2 tracking robust variant."""

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.enable_corruption = True
        self.observations.policy.motion_anchor_pos_b.noise = Unoise(n_min=-0.01, n_max=0.01)
        self.observations.policy.base_ang_vel.noise = Unoise(n_min=-0.03, n_max=0.03)
        self.observations.policy.joint_pos.noise = Unoise(n_min=-0.003, n_max=0.003)
        self.observations.policy.joint_vel.noise = Unoise(n_min=-0.12, n_max=0.12)
        self.observations.policy.actions.noise = Unoise(n_min=-0.005, n_max=0.005)

        self.commands.motion.pose_range = {
            "x": (-0.02, 0.02),
            "y": (-0.03, 0.03),
            "z": (-0.005, 0.005),
            "roll": (-0.06, 0.06),
            "pitch": (-0.04, 0.04),
            "yaw": (-0.08, 0.08),
        }
        self.commands.motion.velocity_range = {
            "x": (-0.2, 0.2),
            "y": (-0.2, 0.2),
            "z": (-0.1, 0.1),
            "roll": (-0.25, 0.25),
            "pitch": (-0.2, 0.2),
            "yaw": (-0.3, 0.3),
        }
        self.commands.motion.joint_position_range = (-0.05, 0.05)

        for actuator_cfg in self.scene.robot.actuators.values():
            actuator_cfg.min_delay = 0
            actuator_cfg.max_delay = 2

        self.events.physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.6, 1.4),
                "dynamic_friction_range": (0.6, 1.2),
                "restitution_range": (0.0, 0.2),
                "num_buckets": 64,
            },
        )
        self.events.base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
                "com_range": {"x": (-0.01, 0.01), "y": (-0.015, 0.015), "z": (-0.01, 0.01)},
            },
        )
        self.events.push_robot = EventTerm(
            func=mdp.conditional_push_by_setting_velocity,
            mode="interval",
            interval_range_s=(2.0, 4.0),
            params={
                "velocity_range": {
                    "x": (-0.08, 0.08),
                    "y": (-0.08, 0.08),
                    "z": (-0.03, 0.03),
                    "roll": (-0.10, 0.10),
                    "pitch": (-0.10, 0.10),
                    "yaw": (-0.12, 0.12),
                },
                "condition_func": mdp.random_condition,
                "condition_params": {"probability": 0.10},
            },
        )
        self.events.add_joint_default_pos = EventTerm(
            func=mdp.randomize_joint_default_pos,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "pos_distribution_params": (-0.005, 0.005),
                "operation": "add",
            },
        )

        self.rewards.action_rate_l2.weight = -0.18
        self.rewards.joint_acc_l2.weight = -5e-3
        self.rewards.motion_body_lin_vel.weight = 0.8
        self.rewards.motion_body_ang_vel.weight = 0.8
        self.terminations.anchor_pos.params["threshold"] = 0.30
        self.terminations.ee_body_pos.params["threshold"] = 0.30
        apply_x2_robust_curriculum(self)


@configclass
class X2SimpleRobustEnvCfg(X2BaseEnvCfg):
    """Flat-ground X2 tracking with light-weight sim-to-real randomization."""

    def __post_init__(self):
        super().__post_init__()

        # Keep the base setup intact and only add mild noise to the states that
        # are likely to drift on hardware.
        self.observations.policy.enable_corruption = True
        self.observations.policy.motion_anchor_pos_b.noise = Unoise(n_min=-0.002, n_max=0.002)
        self.observations.policy.base_ang_vel.noise = Unoise(n_min=-0.01, n_max=0.01)
        self.observations.policy.joint_pos.noise = Unoise(n_min=-0.001, n_max=0.001)
        self.observations.policy.joint_vel.noise = Unoise(n_min=-0.04, n_max=0.04)

        # Keep contact variation centered near the default ground while still
        # covering slightly slicker and slightly grippier surfaces.
        self.events.physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.3, 1.6),
                "dynamic_friction_range": (0.3, 1.2),
                "restitution_range": (0.0, 0.5),
                "num_buckets": 64,
            },
        )
        self.events.base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
                "com_range": {"x": (-0.002, 0.002), "y": (-0.002, 0.002), "z": (-0.002, 0.002)},
            },
        )
        self.events.add_joint_default_pos = EventTerm(
            func=mdp.randomize_joint_default_pos,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "pos_distribution_params": (-0.001, 0.001),
                "operation": "add",
            },
        )
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.0, 3.0),
            params={
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.2, 0.2),
                    "roll": (-0.52, 0.52),
                    "pitch": (-0.52, 0.52),
                    "yaw": (-0.78, 0.78),
                },
            },
        )

        # Leave command randomization, delay, and curriculum off so the training
        # distribution stays close to the base run that already transfers well.
        self.commands.motion.pose_range = {key: (0.0, 0.0) for key in self.commands.motion.pose_range}
        self.commands.motion.velocity_range = {key: (0.0, 0.0) for key in self.commands.motion.velocity_range}
        self.commands.motion.joint_position_range = (0.0, 0.0)
        disable_x2_curriculum(self)


@configclass
class X2BasePlayEnvCfg(X2BaseEnvCfg):
    """Play-only X2 flat config with deterministic resets from the first motion frame."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.phase_start_count = 0
        self.commands.motion.phase_end_count = -1
        self.commands.motion.fixed_phase_reset = True

        self.terminations.anchor_pos = None
        self.terminations.anchor_ori = None
        self.terminations.ee_body_pos = None

        self.events.physics_material = None
        self.events.push_robot = None
        disable_x2_curriculum(self)


@configclass
class X2RobustPlayEnvCfg(X2RobustEnvCfg):
    """Play-only X2 robust flat config with deterministic phase resets."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.phase_start_count = 0
        self.commands.motion.phase_end_count = -1
        self.commands.motion.fixed_phase_reset = True

        self.terminations.anchor_pos = None
        self.terminations.anchor_ori = None
        self.terminations.ee_body_pos = None
        disable_x2_curriculum(self)
