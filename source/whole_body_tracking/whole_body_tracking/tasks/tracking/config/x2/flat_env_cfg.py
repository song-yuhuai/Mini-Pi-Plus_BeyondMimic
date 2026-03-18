import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.robots.x2 import X2_ACTION_SCALE, X2_CFG
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class X2StairEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = X2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = X2_ACTION_SCALE
        # Keep full-joint control and lock motion-joint ordering to avoid NPZ/robot index mismatches.
        self.actions.joint_pos.joint_names = [".*"]
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
        # Deterministic resets: no random spawn offsets/rotations/velocities/joint jitters.
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

        # BMIMIC-compatible actor observation layout:
        # command + projected_gravity + base_ang_vel + joint_pos + joint_vel + actions
        self.observations.policy.motion_anchor_pos_b = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        self.observations.policy.motion_anchor_ori_b = None
        self.observations.policy.base_lin_vel = None
        self.observations.policy.enable_corruption = False

        # Match X2 BMIMIC scaling and clipping conventions.
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

        # Keep policy action clipping consistent with BMIMIC action_clip.
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}

        # Mimic phase window used by X2 get-up configs.
        self.commands.motion.phase_start_count = 0
        self.commands.motion.phase_end_count = -1

        # Single stair block for one-step-up motions.
        # Height = 0.15m, top surface at z = 0.15.
        self.scene.stair_step = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/StairStep",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.60, -0.4, 0.075)),
            spawn=sim_utils.CuboidCfg(
                size=(0.40, 1.20, 0.15),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.02,
                    rest_offset=0.0,
                ),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.2,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )

        # Camera setup.
        self.viewer.eye = (3.2, 2.5, 2.2)
        self.viewer.lookat = (0.8, 0.0, 0.9)
        self.viewer.origin_type = "world"
        self.viewer.asset_name = None
        self.scene.contact_forces.debug_vis = False

        # Stair task tuning.
        self.rewards.motion_body_pos.weight = 1.6
        self.rewards.motion_body_pos.params["std"] = 0.12
        self.terminations.ee_body_pos.params["body_names"] = [
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_elbow_link",
            "right_elbow_link",
        ]
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
class X2StairRobustEnvCfg(X2StairEnvCfg):
    """BFM-inspired robust variant on top of fixed one-stair task."""

    def __post_init__(self):
        super().__post_init__()

        # Keep reset location deterministic; add only mild dynamics randomization.
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

        # Conditional pushes for disturbance recovery, instead of pushing every env each time.
        self.events.push_robot = EventTerm(
            func=mdp.conditional_push_by_setting_velocity,
            mode="interval",
            interval_range_s=(2.0, 4.0),
            params={
                "velocity_range": {
                    "x": (-0.3, 0.3),
                    "y": (-0.3, 0.3),
                    "z": (-0.1, 0.1),
                    "roll": (-0.35, 0.35),
                    "pitch": (-0.35, 0.35),
                    "yaw": (-0.45, 0.45),
                },
                "condition_func": mdp.random_condition,
                "condition_params": {"probability": 0.35},
            },
        )

        # Slightly more recovery-friendly shaping.
        self.rewards.action_rate_l2.weight = -5e-2
        self.rewards.action_acc_l2.weight = -1e-2
        self.rewards.joint_acc_l2.weight = -2e-4
        self.rewards.motion_body_lin_vel.weight = 0.8
        self.rewards.motion_body_ang_vel.weight = 0.8
        self.terminations.anchor_pos.params["threshold"] = 0.30
        self.terminations.ee_body_pos.params["threshold"] = 0.30


@configclass
class X2FlatEnvCfg(X2StairEnvCfg):
    """Flat-ground X2 tracking variant with no stair/chassis obstacle."""

    def __post_init__(self):
        super().__post_init__()

        # Keep the same tracking task configuration, but remove the stair obstacle.
        self.scene.stair_step = None

        # Explicitly enforce flat terrain.
        self.scene.terrain.terrain_type = "plane"

        # Increase action smoothness penalty for flat-task deployment stability.
        self.rewards.action_rate_l2.weight = -0.15
        self.rewards.action_acc_l2.weight = -1e-2
        self.rewards.joint_acc_l2.weight = -8e-4
        self.rewards.feet_min_distance = RewTerm(
            func=mdp.feet_min_distance_penalty,
            weight=-2.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                ),
                "min_distance": 0.16,
                "use_xy_distance": True,
            },
        )

@configclass
class X2FlatRobustEnvCfg(X2StairRobustEnvCfg):
    """Flat-ground X2 tracking variant with no stair/chassis obstacle."""

    def __post_init__(self):
        super().__post_init__()

        # Keep the same tracking task configuration, but remove the stair obstacle.
        self.scene.stair_step = None

        # Explicitly enforce flat terrain.
        self.scene.terrain.terrain_type = "plane"

        # Increase action smoothness penalty for flat-task deployment stability.
        self.rewards.action_rate_l2.weight = -0.15
        self.rewards.action_acc_l2.weight = -4e-2
        self.rewards.joint_acc_l2.weight = -8e-4
        self.rewards.feet_min_distance = RewTerm(
            func=mdp.feet_min_distance_penalty,
            weight=-2.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                ),
                "min_distance": 0.12,
                "use_xy_distance": True,
            },
        )

@configclass
class X2FlatPlayEnvCfg(X2FlatEnvCfg):
    """Play-only X2 flat config with deterministic resets from the first motion frame."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.phase_start_count = 0
        self.commands.motion.phase_end_count = -1
        self.commands.motion.fixed_phase_reset = True

        # Keep replay stable: disable early-failure terminations during play.
        self.terminations.anchor_pos = None
        self.terminations.anchor_ori = None
        self.terminations.ee_body_pos = None

        # Disable domain randomization/disturbances during play.
        self.events.physics_material = None
        self.events.push_robot = None
