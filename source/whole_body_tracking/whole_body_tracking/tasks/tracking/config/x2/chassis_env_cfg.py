import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from whole_body_tracking.tasks.tracking.config.x2.curriculum_cfg import disable_x2_curriculum
from whole_body_tracking.tasks.tracking.config.x2.flat_env_cfg import X2BaseEnvCfg


@configclass
class X2ChassisEnvCfg(X2BaseEnvCfg):
    """X2 tracking config with a fixed chassis block in the scene."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.chassis_block = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/ChassisBlock",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.6, 0.0, 0.09)),
            spawn=sim_utils.CuboidCfg(
                size=(0.6, 0.6, 0.18),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.25, 0.25)),
            ),
        )


@configclass
class X2ChassisPlayEnvCfg(X2ChassisEnvCfg):
    """Play-only X2 chassis config with deterministic resets from the first motion frame."""

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
