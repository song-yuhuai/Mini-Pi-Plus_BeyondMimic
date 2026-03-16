import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

X2_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/x2/x2_ultra_simple_collision.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.68),
        joint_pos={
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            ".*_shoulder_pitch_joint": 0.2,
            "left_shoulder_roll_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            ".*_elbow_joint": -0.3,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "N7520-22.5": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_roll_joint", ".*_hip_yaw_joint", "waist_yaw_joint"],
            effort_limit_sim=120,
            velocity_limit_sim=12.0,
            stiffness={
                ".*_hip_roll_joint": 100.0,
                ".*_hip_yaw_joint": 100.0,
                "waist_yaw_joint": 40.1792,
            },
            damping={
                ".*_hip_roll_joint": 4.0,
                ".*_hip_yaw_joint": 4.0,
                "waist_yaw_joint": 2.5579,
            },
            armature=0.01,
        ),
        "N7520-27": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_joint", ".*_knee_joint"],
            effort_limit_sim=120,
            velocity_limit_sim=12.0,
            stiffness={
                ".*_hip_pitch_joint": 120.0,
                ".*_knee_joint": 150.0,
            },
            damping={
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
            },
            armature=0.01,
        ),
        "N5047": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
                "waist_pitch_joint",
                "waist_roll_joint",
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 36.0,
                ".*_ankle_roll_joint": 24.0,
                "waist_pitch_joint": 48.0,
                "waist_roll_joint": 48.0,
                ".*_shoulder_pitch_joint": 36.0,
                ".*_shoulder_roll_joint": 36.0,
                ".*_shoulder_yaw_joint": 24.0,
                ".*_elbow_joint": 24.0,
                ".*_wrist_yaw_joint": 24.0,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 13.0,
                ".*_ankle_roll_joint": 15.0,
                "waist_pitch_joint": 13.0,
                "waist_roll_joint": 13.0,
                ".*_shoulder_pitch_joint": 13.0,
                ".*_shoulder_roll_joint": 13.0,
                ".*_shoulder_yaw_joint": 15.0,
                ".*_elbow_joint": 15.0,
                ".*_wrist_yaw_joint": 15.0,
            },
            stiffness={
                ".*_ankle_pitch_joint": 40.0,
                ".*_ankle_roll_joint": 40.0,
                "waist_pitch_joint": 200.0,
                "waist_roll_joint": 200.0,
                ".*_shoulder_pitch_joint": 50.0,
                ".*_shoulder_roll_joint": 50.0,
                ".*_shoulder_yaw_joint": 50.0,
                ".*_elbow_joint": 50.0,
                ".*_wrist_yaw_joint": 20.0,
            },
            damping={
                ".*_ankle_pitch_joint": 2.0,
                ".*_ankle_roll_joint": 2.0,
                "waist_pitch_joint": 2.0,
                "waist_roll_joint": 2.0,
                ".*_shoulder_pitch_joint": 3.0,
                ".*_shoulder_roll_joint": 3.0,
                ".*_shoulder_yaw_joint": 3.0,
                ".*_elbow_joint": 3.0,
                ".*_wrist_yaw_joint": 2.0,
            },
            armature=0.01,
        ),
        "N24-small": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_wrist_pitch_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim={
                ".*_wrist_pitch_joint": 4.8,
                ".*_wrist_roll_joint": 4.8,
            },
            velocity_limit_sim={
                ".*_wrist_pitch_joint": 4.2,
                ".*_wrist_roll_joint": 4.2,
            },
            stiffness={
                ".*_wrist_pitch_joint": 20.0,
                ".*_wrist_roll_joint": 20.0,
            },
            damping={
                ".*_wrist_pitch_joint": 2.0,
                ".*_wrist_roll_joint": 2.0,
            },
            armature=0.002,
        ),
    },
)

# X2_ACTION_SCALE = {}
# for a in X2_CFG.actuators.values():
#     e = a.effort_limit_sim
#     s = a.stiffness
#     names = a.joint_names_expr
#     if not isinstance(e, dict):
#         e = {n: e for n in names}
#     if not isinstance(s, dict):
#         s = {n: s for n in names}
#     for n in names:
#         if n in e and n in s and s[n]:
#             X2_ACTION_SCALE[n] = 0.25 * e[n] / s[n]

X2_ACTION_SCALE = {}
for a in X2_CFG.actuators.values():
    for n in a.joint_names_expr:
        X2_ACTION_SCALE[n] = 0.25



