import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

# NOTE: Update the joint lists and limits below once the official X2 joint order
# and actuator specs are confirmed.
X2_JOINT_NAMES = [
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
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
]

X2_DEFAULT_JOINT_POS = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]

X2_ANCHOR_BODY_NAME = "pelvis"
X2_MOTION_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
]

ARMATURE_4438 = 0.03
ARMATURE_5047 = 0.03

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_4438 = 30
STIFFNESS_5047 = 80

DAMPING_4438 = 0.6
DAMPING_5047 = 1.1


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
        pos=(0.0, 0.0, 0.351),
        joint_pos={
            ".*_hip_pitch_joint": 0.0,
            ".*_knee_joint": 0.0,
            ".*_ankle_pitch_joint": 0.0,
            ".*_elbow_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 118.0,
                ".*_hip_roll_joint": 118.0,
                ".*_hip_pitch_joint": 118.0,
                ".*_knee_joint": 118.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 12.0,
                ".*_hip_roll_joint": 12.0,
                ".*_hip_pitch_joint": 12.0,
                ".*_knee_joint": 12.0,
            },
            stiffness={
                ".*_hip_pitch_joint": 150,
                ".*_hip_roll_joint": 150,
                ".*_hip_yaw_joint": 150,
                ".*_knee_joint": 180,
            },
            damping={
                ".*_hip_pitch_joint": 4,
                ".*_hip_roll_joint": 4,
                ".*_hip_yaw_joint": 4,
                ".*_knee_joint": 6,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_5047,
                ".*_hip_roll_joint": ARMATURE_5047,
                ".*_hip_yaw_joint": ARMATURE_5047,
                ".*_knee_joint": ARMATURE_5047,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=36.0,
            velocity_limit_sim=13.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=150,
            damping=4,
            armature=ARMATURE_5047,
        ),
        "waist": ImplicitActuatorCfg(
            effort_limit_sim=50,
            velocity_limit_sim=13.0,
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            stiffness=300,
            damping=6,
            armature=0.06,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            effort_limit_sim=118,
            velocity_limit_sim=12.0,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=300,
            damping=6,
            armature=0.06,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 36.0,
                ".*_shoulder_roll_joint": 36.0,
                ".*_shoulder_yaw_joint": 24.0,
                ".*_elbow_joint": 24.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 13.0,
                ".*_shoulder_roll_joint": 13.0,
                ".*_shoulder_yaw_joint": 13.0,
                ".*_elbow_joint": 13.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": 80,
                ".*_shoulder_roll_joint": 80,
                ".*_shoulder_yaw_joint": 80,
                ".*_elbow_joint": 80,
            },
            damping={
                ".*_shoulder_pitch_joint": 2,
                ".*_shoulder_roll_joint": 2,
                ".*_shoulder_yaw_joint": 2,
                ".*_elbow_joint": 2,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_4438,
                ".*_shoulder_roll_joint": ARMATURE_4438,
                ".*_shoulder_yaw_joint": ARMATURE_4438,
                ".*_elbow_joint": ARMATURE_4438,
            },
        ),
    },
)

X2_ACTION_SCALE = {}
for a in X2_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            X2_ACTION_SCALE[n] = 0.25 * e[n] / s[n]