import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from whole_body_tracking.tasks.tracking.config.x2.curriculum_cfg import disable_x2_curriculum
from whole_body_tracking.tasks.tracking.config.x2.chassis_constants import (
    X2_CHASSIS_COLOR,
    X2_CHASSIS_POSITION,
    X2_CHASSIS_PRIM_PATH,
    X2_CHASSIS_SIZE,
)
from whole_body_tracking.tasks.tracking.config.x2.flat_env_cfg import (
    X2BaseEnvCfg,
    X2BasePlayEnvCfg,
    X2RobustEnvCfg,
    X2RobustPlayEnvCfg,
    X2SimpleRobustEnvCfg,
)


def _add_chassis_block(scene) -> None:
    scene.chassis_block = AssetBaseCfg(
        prim_path=X2_CHASSIS_PRIM_PATH,
        init_state=AssetBaseCfg.InitialStateCfg(pos=X2_CHASSIS_POSITION),
        spawn=sim_utils.CuboidCfg(
            size=X2_CHASSIS_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=X2_CHASSIS_COLOR),
        ),
    )


@configclass
class X2ChassisEnvCfg(X2BaseEnvCfg):
    """X2 tracking config with a fixed chassis block in the scene."""

    def __post_init__(self):
        super().__post_init__()
        _add_chassis_block(self.scene)


@configclass
class X2ChassisRobustEnvCfg(X2RobustEnvCfg):
    """X2 robust tracking config with a fixed chassis block in the scene."""

    def __post_init__(self):
        super().__post_init__()
        _add_chassis_block(self.scene)


@configclass
class X2ChassisSimpleRobustEnvCfg(X2SimpleRobustEnvCfg):
    """X2 simple-robust tracking config with a fixed chassis block in the scene."""

    def __post_init__(self):
        super().__post_init__()
        _add_chassis_block(self.scene)


@configclass
class X2ChassisPlayEnvCfg(X2BasePlayEnvCfg):
    """Play-only X2 chassis config with deterministic resets from the first motion frame."""

    def __post_init__(self):
        super().__post_init__()
        _add_chassis_block(self.scene)


@configclass
class X2ChassisRobustPlayEnvCfg(X2RobustPlayEnvCfg):
    """Play-only X2 robust chassis config with deterministic phase resets."""

    def __post_init__(self):
        super().__post_init__()
        _add_chassis_block(self.scene)
        disable_x2_curriculum(self)
