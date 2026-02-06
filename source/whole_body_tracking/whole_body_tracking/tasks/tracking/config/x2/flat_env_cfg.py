from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
import whole_body_tracking.tasks.tracking.mdp as mdp

from whole_body_tracking.robots.x2 import (
    X2_ACTION_SCALE,
    X2_ANCHOR_BODY_NAME,
    X2_CFG,
    X2_MOTION_BODY_NAMES,
)
from whole_body_tracking.tasks.tracking.config.x2.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class X2FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = X2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = X2_ACTION_SCALE
        self.commands.motion.anchor_body_name = X2_ANCHOR_BODY_NAME
        self.commands.motion.body_names = X2_MOTION_BODY_NAMES

        # 相机设置：自由视角，不跟随机器人
        self.viewer.eye = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 1.0)
        self.viewer.origin_type = "world"
        self.viewer.asset_name = None

        # 关闭调试可视化显示
        self.scene.contact_forces.debug_vis = False

        # TODO: Update reward/termination tuning once X2 dynamics are validated.
        self.rewards.motion_body_pos.weight = 1.5
        self.rewards.motion_body_pos.params["std"] = 0.15

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

        self.events.base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=X2_ANCHOR_BODY_NAME),
                "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
            },
        )


@configclass
class X2FlatWoEnvCfg(X2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None
        self.events.physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.5, 3),
                "dynamic_friction_range": (0.5, 3),
                "restitution_range": (0.0, 0.5),
                "num_buckets": 64,
            },
        )
