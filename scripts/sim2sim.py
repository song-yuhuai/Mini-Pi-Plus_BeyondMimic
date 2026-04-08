"""Unified sim2sim script supporting multiple robot configurations.

Usage Examples:
    # For HI robot:
    python sim2sim.py --robot hi --motion_file source/motion/hightorque/hi/npz/dance1_subject2.npz \
    --xml_path /path/to/hi.xml --policy_path /path/to/hi_policy.onnx --save_json
    
    # For PI Plus robot:
    python sim2sim.py --robot pi_plus --motion_file source/motion/hightorque/pi_plus/npz/dance1_subject2.npz \
    --xml_path /path/to/pi_plus.xml --policy_path /path/to/pi_plus_policy.onnx --save_json
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnx
import onnxruntime
import torch
from scipy.spatial.transform import Rotation as R

# Simulation parameters
simulation_duration = 300.0
simulation_dt = 0.002
control_decimation = 10

# Robot configurations
ROBOT_CONFIGS = {
    
    "hi": {
        "num_actions": 23,
        "num_obs": 124,
        "reference_body": "base_link",
        "default_xml": None,  # Must be provided
        "joint_names": [
            "l_hip_pitch_joint",
            "l_hip_roll_joint",
            "l_hip_thigh_joint",
            "l_hip_calf_joint",
            "l_ankle_pitch_joint",
            "l_ankle_roll_joint",
            "r_hip_pitch_joint",
            "r_hip_roll_joint",
            "r_hip_thigh_joint",
            "r_hip_calf_joint",
            "r_ankle_pitch_joint",
            "r_ankle_roll_joint",
            "waist_yaw_joint",
            "l_shoulder_pitch_joint",
            "l_shoulder_roll_joint",
            "l_upper_arm_joint",
            "l_elbow_joint",
            "l_wrist_joint",
            "r_shoulder_pitch_joint",
            "r_shoulder_roll_joint",
            "r_upper_arm_joint",
            "r_elbow_joint",
            "r_wrist_joint",
        ],
        "motion_body_index": 0,
        "observation_structure": {
            "command": 46,
            "projected_gravity_b": 3,
            "base_ang_vel": 3,
            "joint_pos": 23,
            "joint_vel": 23,
            "actions": 23,
        },
    },
    "pi_plus": {
        "num_actions": 22,
        "num_obs": 119,
        "reference_body": "base_link",
        "default_xml": None,  # Must be provided
        "joint_names": [
            "l_hip_pitch_joint",
            "l_hip_roll_joint",
            "l_thigh_joint",
            "l_calf_joint",
            "l_ankle_pitch_joint",
            "l_ankle_roll_joint",
            "l_shoulder_pitch_joint",
            "l_shoulder_roll_joint",
            "l_upper_arm_joint",
            "l_elbow_joint",
            "l_wrist_joint",
            "r_hip_pitch_joint",
            "r_hip_roll_joint",
            "r_thigh_joint",
            "r_calf_joint",
            "r_ankle_pitch_joint",
            "r_ankle_roll_joint",
            "r_shoulder_pitch_joint",
            "r_shoulder_roll_joint",
            "r_upper_arm_joint",
            "r_elbow_joint",
            "r_wrist_joint",
        ],
        "motion_body_index": 0,
        "observation_structure": {
            "command": 44,
            "motion_ref_ori_b": 6,
            "base_ang_vel": 3,
            "joint_pos": 22,
            "joint_vel": 22,
            "actions": 22,
        },
    },
    "gp02_v2": {
        "num_actions": 22,
        "num_obs": 119,
        "reference_body": "pelvis",
        "default_xml": None,
        "joint_names": [
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
            "waist_roll_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
        ],
        "motion_body_index": 0,
        "observation_structure": {
            "command": 44,
            "motion_ref_ori_b": 6,
            "base_ang_vel": 3,
            "joint_pos": 22,
            "joint_vel": 22,
            "actions": 22,
        },
    },
    "x2": {
        "num_actions": 29,
        "num_obs": 151,
        "reference_body": "pelvis",
        "default_xml": None,
        "landing_debug_body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
        "landing_debug_floor_geom": "floor",
        "landing_debug_enter_force_threshold": 150.0,
        "landing_debug_exit_force_threshold": 20.0,
        "landing_debug_min_air_frames": 20,
        "joint_names": [
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
        ],
        "motion_body_index": 0,
        "observation_structure": {
            "command": 58,
            "projected_gravity_b": 3,
            "base_ang_vel": 3,
            "joint_pos": 29,
            "joint_vel": 29,
            "actions": 29,
        },
        "obs_scales": {
            "command": 1.0,
            "projected_gravity_b": 1.0,
            "base_ang_vel": 0.25,
            "joint_pos": 1.0,
            "joint_vel": 0.05,
            "actions": 1.0,
        },
        "obs_clip": 100.0,
    },
}


def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as quaternions to rotation matrices."""
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def get_obs(data):
    """Extracts an observation from the mujoco data structure"""
    qpos = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor("body-orientation").data[[0, 1, 2, 3]].astype(np.double)

    # MuJoCo framequat is [w, x, y, z], while SciPy expects [x, y, z, w].
    quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.double)
    r = R.from_quat(quat_xyzw)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)
    omega = data.sensor("body-angular-velocity").data.astype(np.double)
    gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)
    state_tau = data.qfrc_actuator.astype(np.double) - data.qfrc_bias.astype(np.double)

    return (qpos, dq, quat, v, omega, gvec, state_tau)


def quat_rotate_inverse_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a vector by the inverse of a quaternion along the last dimension of q and v (NumPy version)."""
    q_w = q[..., 0]
    q_vec = q[..., 1:]
    
    a = v * np.expand_dims(2.0 * q_w**2 - 1.0, axis=-1)
    b = np.cross(q_vec, v, axis=-1) * np.expand_dims(q_w, axis=-1) * 2.0
    
    if q_vec.ndim == 2:
        dot_product = np.sum(q_vec * v, axis=-1, keepdims=True)
        c = q_vec * dot_product * 2.0
    else:
        dot_product = np.expand_dims(np.einsum('...i,...i->...', q_vec, v), axis=-1)
        c = q_vec * dot_product * 2.0
    
    return a - b + c


def subtract_frame_transforms_mujoco(pos_a, quat_a, pos_b, quat_b):
    """Calculate relative transformation from frame A to frame B (MuJoCo version)."""
    rotm_a = np.zeros(9)
    mujoco.mju_quat2Mat(rotm_a, quat_a)
    rotm_a = rotm_a.reshape(3, 3)
    
    rel_pos = rotm_a.T @ (pos_b - pos_a)
    rel_quat = quaternion_multiply(quaternion_conjugate(quat_a), quat_b)
    rel_quat = rel_quat / np.linalg.norm(rel_quat)
    
    return rel_pos, rel_quat


def quaternion_conjugate(q):
    """Quaternion conjugate: [w, x, y, z] -> [w, -x, -y, -z]"""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_multiply(q1, q2):
    """Quaternion multiplication: q1 ⊗ q2"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return np.array([w, x, y, z])


def quat_mul_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions together."""
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return np.stack([w, x, y, z], axis=-1).reshape(shape)


def quat_conjugate_np(q: np.ndarray) -> np.ndarray:
    """Computes the conjugate of a quaternion."""
    shape = q.shape
    q = q.reshape(-1, 4)
    return np.concatenate((q[..., 0:1], -q[..., 1:]), axis=-1).reshape(shape)


def quat_inv_np(q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Computes the inverse of a quaternion."""
    return quat_conjugate_np(q) / np.clip(np.sum(q**2, axis=-1, keepdims=True), a_min=eps, a_max=None)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def get_body_linear_velocity_z(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> float:
    """Return body linear velocity z in world frame."""
    body_vel = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, body_vel, 0)
    # MuJoCo returns spatial velocity as [angular(3), linear(3)].
    return float(body_vel[5])


def build_body_geom_id_set(model: mujoco.MjModel, body_name: str) -> set[int]:
    """Collect geom ids directly attached to a MuJoCo body."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise ValueError(f"Body {body_name} not found in model")

    geom_adr = int(model.body_geomadr[body_id])
    geom_num = int(model.body_geomnum[body_id])
    return set(range(geom_adr, geom_adr + geom_num))


def body_in_contact_with_geom(data: mujoco.MjData, body_geom_ids: set[int], target_geom_id: int) -> bool:
    """Check whether any geom from a body is in contact with the target geom."""
    for i in range(data.ncon):
        contact = data.contact[i]
        if (
            (contact.geom1 in body_geom_ids and contact.geom2 == target_geom_id)
            or (contact.geom2 in body_geom_ids and contact.geom1 == target_geom_id)
        ):
            return True
    return False


def body_contact_normal_force_with_geom(
    model: mujoco.MjModel, data: mujoco.MjData, body_geom_ids: set[int], target_geom_id: int
) -> float:
    """Sum normal contact forces between a body's geoms and a target geom."""
    total_normal_force = 0.0
    contact_force = np.zeros(6, dtype=np.float64)
    for i in range(data.ncon):
        contact = data.contact[i]
        if (
            (contact.geom1 in body_geom_ids and contact.geom2 == target_geom_id)
            or (contact.geom2 in body_geom_ids and contact.geom1 == target_geom_id)
        ):
            mujoco.mj_contactForce(model, data, i, contact_force)
            total_normal_force += max(float(contact_force[0]), 0.0)
    return total_normal_force

def create_observation_hi_pi(obs, offset, motioninput, motion_ref_ori_b, omega, qpos_seq, qvel_seq, action_buffer, joint_pos_array_seq, num_actions):
    """Create observation for HI and PI Plus robots."""
    cmd_size = len(motioninput)
    obs[offset:offset + cmd_size] = motioninput
    offset += cmd_size
    obs[offset:offset + 6] = motion_ref_ori_b
    offset += 6
    obs[offset:offset + 3] = omega
    offset += 3
    obs[offset:offset + num_actions] = qpos_seq - joint_pos_array_seq
    offset += num_actions
    obs[offset:offset + num_actions] = qvel_seq
    offset += num_actions   
    obs[offset:offset + num_actions] = action_buffer
    return obs


def create_observation_projected_gravity(
    obs,
    offset,
    motioninput,
    projected_gravity_b,
    omega,
    qpos_seq,
    qvel_seq,
    action_buffer,
    joint_pos_array_seq,
    num_actions,
    obs_scales,
    obs_clip,
):
    """Create BMIMIC-style observation with projected gravity."""
    def _scaled_clipped(values, scale):
        return np.clip(np.asarray(values) * scale, -obs_clip, obs_clip)

    cmd_size = len(motioninput)
    obs[offset:offset + cmd_size] = _scaled_clipped(motioninput, obs_scales["command"])
    offset += cmd_size
    obs[offset:offset + 3] = _scaled_clipped(projected_gravity_b, obs_scales["projected_gravity_b"])
    offset += 3
    obs[offset:offset + 3] = _scaled_clipped(omega, obs_scales["base_ang_vel"])
    offset += 3
    obs[offset:offset + num_actions] = _scaled_clipped(qpos_seq - joint_pos_array_seq, obs_scales["joint_pos"])
    offset += num_actions
    obs[offset:offset + num_actions] = _scaled_clipped(qvel_seq, obs_scales["joint_vel"])
    offset += num_actions
    obs[offset:offset + num_actions] = _scaled_clipped(action_buffer, obs_scales["actions"])
    return obs


def run_simulation(
    robot_type: str,
    motion_file: str | None,
    xml_path: str,
    policy_path: str,
    save_json: bool = False,
    loop: bool = False,
    render_every: int = 10,
    plot_root_xy: bool = True,
    root_xy_plot_path: str = "outputs/root_xy_trajectory.png",
    trajectory_frame: str = "root_initial",
    flip_left_right: bool = True,
    embedded_motion_num_frames: int | None = None,
):
    """Run the sim2sim simulation."""
    def _save_root_xy_plot(reason: str = "final"):
        if not plot_root_xy:
            return
        if len(root_xy_traj) < 2:
            print("[WARN]: Not enough root XY points to plot trajectory.")
            return
        try:
            import matplotlib.pyplot as plt

            root_xy = np.asarray(root_xy_traj, dtype=np.float64)
            out_path = Path(root_xy_plot_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            plt.figure(figsize=(7, 7))
            if trajectory_frame == "root_initial":
                # root_xy stores [forward, left]; draw as x=left/right, y=forward/backward.
                lateral_sign = -1.0 if flip_left_right else 1.0
                plot_x = lateral_sign * root_xy[:, 1]
                plot_y = root_xy[:, 0]
                plt.plot(plot_x, plot_y, linewidth=1.5, label="root_xy")
                plt.scatter(plot_x[0], plot_y[0], c="green", s=40, label="start")
                plt.scatter(plot_x[-1], plot_y[-1], c="red", s=40, label="end")
                plt.xlabel("left/right (m)")
                plt.ylabel("forward/backward (m)")
                plt.title(f"Root Trajectory in Initial Root Frame ({robot_type})")
            else:
                plot_x = root_xy[:, 0]
                plot_y = root_xy[:, 1]
                plt.plot(plot_x, plot_y, linewidth=1.5, label="root_xy")
                plt.scatter(plot_x[0], plot_y[0], c="green", s=40, label="start")
                plt.scatter(plot_x[-1], plot_y[-1], c="red", s=40, label="end")
                plt.xlabel("x (m)")
                plt.ylabel("y (m)")
                plt.title(f"Root XY Trajectory in World Frame ({robot_type})")
            plt.axis("equal")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            plt.close()
            print(f"[INFO]: Root XY trajectory saved to: {out_path} (reason: {reason})")
        except Exception as err:
            print(f"[WARN]: Failed to plot root XY trajectory: {err}")

    config = ROBOT_CONFIGS[robot_type]
    print(f"[INFO]: Using robot configuration: {robot_type}")
    print(f"[INFO]: Actions: {config['num_actions']}, Observations: {config['num_obs']}")
    
    # Load motion data from NPZ if provided; otherwise use ONNX auxiliary outputs.
    use_external_motion = motion_file is not None
    motionpos = None
    motionquat = None
    motioninputpos = None
    motioninputvel = None
    num_frames = 1
    if use_external_motion:
        motion = np.load(motion_file)
        motionpos = motion["body_pos_w"]
        motionquat = motion["body_quat_w"]
        motioninputpos = motion["joint_pos"]
        motioninputvel = motion["joint_vel"]
        # number of frames available across all sequences
        num_frames = min(motioninputpos.shape[0], motioninputvel.shape[0], motionpos.shape[0], motionquat.shape[0])
    else:
        print("[INFO]: No --motion_file provided, using motion signals from ONNX outputs.")

    # Save motion data to JSON if requested and available
    if save_json and use_external_motion:
        motion_dict = {
            "body_pos_w": motionpos.tolist(),
            "body_quat_w": motionquat.tolist(),
            "joint_pos": motioninputpos.tolist(),
            "joint_vel": motioninputvel.tolist()
        }
        # Convert npz path to json path: npz/file.npz -> json/file.json
        import os
        motion_dir = os.path.dirname(motion_file)
        motion_basename = os.path.basename(motion_file)
        
        # Replace 'npz' directory with 'json' directory
        if motion_dir.endswith('/npz') or motion_dir.endswith('\\npz'):
            json_dir = motion_dir[:-3] + 'json'  # Replace last 3 characters 'npz' with 'json'
        else:
            json_dir = motion_dir  # If not in npz directory, use same directory
            
        # Create json directory if it doesn't exist
        os.makedirs(json_dir, exist_ok=True)
        
        # Create json filename
        json_basename = motion_basename.replace('.npz', '.json')
        json_filename = os.path.join(json_dir, json_basename)
        with open(json_filename, 'w') as f:
            json.dump(motion_dict, f, indent=2)
        print(f"[INFO]: Motion data saved to: {json_filename}")
    elif save_json:
        print("[WARN]: --save_json is ignored without --motion_file.")
    
    # Load ONNX model and extract metadata
    model = onnx.load(policy_path)
    metadata_map = {}
    joint_seq = None
    joint_pos_array_seq = None
    stiffness_array_seq = None
    damping_array_seq = None
    action_scale = None
    
    for prop in model.metadata_props:
        metadata_map[prop.key] = prop.value
        if prop.key == "joint_names":
            joint_seq = prop.value.split(",")
        elif prop.key == "default_joint_pos":   
            joint_pos_array_seq = np.array([float(x) for x in prop.value.split(",")])
        elif prop.key == "joint_stiffness":
            stiffness_array_seq = np.array([float(x) for x in prop.value.split(",")])
        elif prop.key == "joint_damping":
            damping_array_seq = np.array([float(x) for x in prop.value.split(",")])
        elif prop.key == "action_scale":
            action_scale = np.array([float(x) for x in prop.value.split(",")])
        print(f"{prop.key}: {prop.value}")

    inferred_embedded_frames = None
    if not use_external_motion:
        if embedded_motion_num_frames is not None and embedded_motion_num_frames > 0:
            inferred_embedded_frames = int(embedded_motion_num_frames)
        else:
            for key in ("time_step_total", "motion_num_frames", "num_frames", "phase_step_count"):
                if key in metadata_map:
                    try:
                        parsed = int(float(metadata_map[key]))
                        if parsed > 0:
                            inferred_embedded_frames = parsed
                            break
                    except ValueError:
                        continue
        if inferred_embedded_frames is not None:
            num_frames = inferred_embedded_frames
            print(f"[INFO]: Embedded motion frames inferred: {num_frames}")
        else:
            print("[WARN]: Unable to infer embedded motion length from ONNX metadata.")
            print("[WARN]: Pass --embedded_motion_num_frames N to enable no-external-motion loop/end control.")

    # Safe index helper for both external and embedded sequences.
    def frame_idx(t):
        if use_external_motion:
            if loop and num_frames > 0:
                return t % num_frames
            return t if t < num_frames else num_frames - 1
        if inferred_embedded_frames is not None and inferred_embedded_frames > 0:
            if loop:
                return t % inferred_embedded_frames
            return t if t < inferred_embedded_frames else inferred_embedded_frames - 1
        return t
    
    # Remap to XML joint order
    joint_xml = config["joint_names"]
    joint_pos_array = np.array([joint_pos_array_seq[joint_seq.index(joint)] for joint in joint_xml])
    stiffness_array = np.array([stiffness_array_seq[joint_seq.index(joint)] for joint in joint_xml])
    damping_array = np.array([damping_array_seq[joint_seq.index(joint)] for joint in joint_xml])
    
    print("stiffness_array", stiffness_array)
    print("damping_array", damping_array)
    print("action_scale", action_scale)
    
    # Initialize variables
    num_actions = config["num_actions"]
    num_obs = config["num_obs"]
    action = np.zeros(num_actions, dtype=np.float32)
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0
    
    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    
    # Load policy
    policy = onnxruntime.InferenceSession(policy_path)
    policy_output_names = [out.name for out in policy.get_outputs()]
    has_embedded_motion_outputs = {"joint_pos", "joint_vel", "body_quat_w"}.issubset(set(policy_output_names))
    if not use_external_motion and not has_embedded_motion_outputs:
        raise ValueError("Policy ONNX is missing joint_pos/joint_vel/body_quat_w outputs. Provide --motion_file instead.")
    
    action_buffer = np.zeros((num_actions,), dtype=np.float32)
    timestep = 0 
    if use_external_motion:
        motioninput = np.concatenate((motioninputpos[frame_idx(timestep), :], motioninputvel[frame_idx(timestep), :]), axis=0)
        motionquatcurrent = motionquat[frame_idx(timestep), config["motion_body_index"], :]
    else:
        motioninput = np.concatenate((joint_pos_array_seq.copy(), np.zeros_like(joint_pos_array_seq)), axis=0)
        motionquatcurrent = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    
    target_dof_pos = joint_pos_array.copy()
    if robot_type == "hi":
        d.qpos[2] = 0.68
    
    # Set reference body
    body_name = config["reference_body"]
    body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise ValueError(f"Body {body_name} not found in model")

    render_every = max(1, int(render_every))
    root_xy_traj = []
    root_origin_xy = None
    forward_axis_w_xy = None
    left_axis_w_xy = None
    landing_debug_states = {}
    has_announced_motion_end = False
    first_cycle_plot_saved = False
    if plot_root_xy and trajectory_frame == "root_initial":
        print("[INFO]: Trajectory frame is root_initial. 'Forward' is reference body local +X at the first sample.")
        if flip_left_right:
            print("[INFO]: Left-right axis is flipped for plotting.")
    if loop and not use_external_motion and inferred_embedded_frames is None:
        print("[WARN]: --loop needs embedded motion length. Pass --embedded_motion_num_frames to enable it.")

    landing_debug_body_names = config.get("landing_debug_body_names", [])
    if landing_debug_body_names:
        floor_geom_name = config.get("landing_debug_floor_geom", "floor")
        landing_debug_enter_force_threshold = float(config.get("landing_debug_enter_force_threshold", 0.0))
        landing_debug_exit_force_threshold = float(
            config.get("landing_debug_exit_force_threshold", landing_debug_enter_force_threshold)
        )
        landing_debug_min_air_frames = int(config.get("landing_debug_min_air_frames", 0))
        floor_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, floor_geom_name)
        if floor_geom_id == -1:
            raise ValueError(f"Geom {floor_geom_name} not found in model")

        for body_name in landing_debug_body_names:
            body_id_dbg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id_dbg == -1:
                raise ValueError(f"Landing debug body {body_name} not found in model")
            landing_debug_states[body_name] = {
                "body_id": body_id_dbg,
                "geom_ids": build_body_geom_id_set(m, body_name),
                "is_in_contact": False,
                "air_frames": landing_debug_min_air_frames,
                "pre_contact_vz_history": [],
            }
        print(
            "[INFO]: Landing debug enabled for bodies: "
            + ", ".join(landing_debug_body_names)
            + (
                f" (enter>{landing_debug_enter_force_threshold:.1f}N, "
                f"exit<{landing_debug_exit_force_threshold:.1f}N, "
                f"min_air_frames={landing_debug_min_air_frames}, "
                "printing pre-touchdown 3 physics-frame z velocities)."
            )
        )

    try:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            start = time.time()
            while viewer.is_running() and time.time() - start < simulation_duration:
                step_start = time.time()

                mujoco.mj_step(m, d)
                qpos, dq, quat, v, omega, gvec, state_tau = get_obs(d)
                tau = pd_control(target_dof_pos, d.qpos[7:], stiffness_array, np.zeros_like(damping_array), d.qvel[6:], damping_array)

                d.ctrl[:] = tau
                counter += 1
                if landing_debug_states:
                    for body_name, debug_state in landing_debug_states.items():
                        current_vz = get_body_linear_velocity_z(m, d, debug_state["body_id"])
                        vz_history = debug_state["pre_contact_vz_history"]
                        vz_history.append(current_vz)
                        if len(vz_history) > 3:
                            vz_history.pop(0)

                        normal_force = body_contact_normal_force_with_geom(m, d, debug_state["geom_ids"], floor_geom_id)
                        was_in_contact = debug_state["is_in_contact"]
                        if was_in_contact:
                            is_in_contact = normal_force > landing_debug_exit_force_threshold
                        else:
                            is_in_contact = normal_force > landing_debug_enter_force_threshold

                        if is_in_contact:
                            touchdown = (not was_in_contact) and (debug_state["air_frames"] >= landing_debug_min_air_frames)
                            debug_state["air_frames"] = 0
                        else:
                            touchdown = False
                            debug_state["air_frames"] += 1

                        if touchdown:
                            vz_values = ", ".join(f"{vz:+.4f}" for vz in vz_history)
                            print(
                                f"[LANDING] step={counter:06d} body={body_name} "
                                f"touchdown_force={normal_force:.2f}N "
                                f"pre_touchdown_vz=[{vz_values}]"
                            )

                        debug_state["is_in_contact"] = is_in_contact
                if plot_root_xy:
                    pos_xy = np.array([float(d.qpos[0]), float(d.qpos[1])], dtype=np.float64)
                    if trajectory_frame == "root_initial":
                        if root_origin_xy is None:
                            root_origin_xy = pos_xy.copy()
                            body_rot_w = d.xmat[body_id].reshape(3, 3)
                            # Local +X of reference body is treated as robot forward.
                            fxy = body_rot_w[:2, 0].copy()
                            norm_f = np.linalg.norm(fxy)
                            if norm_f < 1e-8:
                                fxy = np.array([1.0, 0.0], dtype=np.float64)
                            else:
                                fxy = fxy / norm_f
                            forward_axis_w_xy = fxy
                            left_axis_w_xy = np.array([-fxy[1], fxy[0]], dtype=np.float64)
                        delta_xy = pos_xy - root_origin_xy
                        forward_disp = float(np.dot(delta_xy, forward_axis_w_xy))
                        left_disp = float(np.dot(delta_xy, left_axis_w_xy))
                        root_xy_traj.append([forward_disp, left_disp])
                    else:
                        root_xy_traj.append([pos_xy[0], pos_xy[1]])
            
                if counter % control_decimation == 0:
                    # Update motion data
                    idx = frame_idx(timestep)
                    if use_external_motion:
                        motioninput = np.concatenate((motioninputpos[idx, :], motioninputvel[idx, :]), axis=0)
                        motionquatcurrent = motionquat[idx, config["motion_body_index"], :]
                
                    # Create observations based on robot type
                    offset = 0
                    if robot_type in ["hi", "pi_plus", "gp02_v2", "x2"]:
                        qpos_xml = d.qpos[7:7 + num_actions]
                        qpos_seq = np.array([qpos_xml[joint_xml.index(joint)] for joint in joint_seq])
                        qvel_xml = d.qvel[6:6 + num_actions]
                        qvel_seq = np.array([qvel_xml[joint_xml.index(joint)] for joint in joint_seq])
                        if robot_type == "x2":
                            obs = create_observation_projected_gravity(
                                obs,
                                offset,
                                motioninput,
                                gvec,
                                omega,
                                qpos_seq,
                                qvel_seq,
                                action_buffer,
                                joint_pos_array_seq,
                                num_actions,
                                config["obs_scales"],
                                config["obs_clip"],
                            )
                        else:
                            q01 = quat
                            q02 = motionquatcurrent
                            q10 = quat_inv_np(q01)
                            if q02 is not None:
                                q12 = quat_mul_np(q10, q02)
                            else:
                                q12 = q10
                            mat = matrix_from_quat(torch.from_numpy(q12))
                            motion_ref_ori_b = mat[..., :2].reshape(6)
                            obs = create_observation_hi_pi(
                                obs, offset, motioninput, motion_ref_ori_b, omega, qpos_seq, qvel_seq, action_buffer, joint_pos_array_seq, num_actions
                            )
                
                    # Run policy inference
                    obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                    output_values = policy.run(None, {
                        'obs': obs_tensor.numpy(),
                        'time_step': np.array([frame_idx(timestep)], dtype=np.float32).reshape(1, 1)
                    })
                    output_map = {name: value for name, value in zip(policy_output_names, output_values)}
                    action = output_map["actions"]
                
                    action = np.asarray(action).reshape(-1)
                    action_buffer = action.copy()
                    target_dof_pos = action * action_scale + joint_pos_array_seq
                    target_dof_pos = target_dof_pos.reshape(-1,)
                    target_dof_pos = np.array([target_dof_pos[joint_seq.index(joint)] for joint in joint_xml])

                    if not use_external_motion:
                        joint_pos_out = np.asarray(output_map["joint_pos"]).reshape(-1)
                        joint_vel_out = np.asarray(output_map["joint_vel"]).reshape(-1)
                        motioninput = np.concatenate((joint_pos_out[:num_actions], joint_vel_out[:num_actions]), axis=0)
                        body_quat_out = np.asarray(output_map["body_quat_w"])
                        if body_quat_out.ndim == 3:
                            body_quat_out = body_quat_out[0]
                        if body_quat_out.ndim == 2 and body_quat_out.shape[0] > config["motion_body_index"]:
                            motionquatcurrent = body_quat_out[config["motion_body_index"], :4]
                        else:
                            motionquatcurrent = body_quat_out.reshape(-1)[:4]
                
                    # Advance time step.
                    # `--loop` controls only whether external motion wraps around.
                    # It should not freeze time progression when loop is disabled.
                    if use_external_motion:
                        if loop:
                            if num_frames > 0 and timestep + 1 >= num_frames and not first_cycle_plot_saved:
                                _save_root_xy_plot(reason="first_cycle_complete")
                                first_cycle_plot_saved = True
                            timestep = (timestep + 1) % max(1, num_frames)
                        elif timestep + 1 < num_frames:
                            timestep += 1
                        elif not has_announced_motion_end:
                            if not first_cycle_plot_saved:
                                _save_root_xy_plot(reason="first_cycle_complete")
                                first_cycle_plot_saved = True
                            has_announced_motion_end = True
                            print("[INFO]: Motion reached the final frame, exiting simulation (set --loop to keep running).")
                            break
                    else:
                        # For embedded motion outputs, mirror external-motion loop/end behavior when frame count is known.
                        if inferred_embedded_frames is not None and inferred_embedded_frames > 0:
                            if loop:
                                if timestep + 1 >= inferred_embedded_frames and not first_cycle_plot_saved:
                                    _save_root_xy_plot(reason="first_cycle_complete")
                                    first_cycle_plot_saved = True
                                timestep = (timestep + 1) % inferred_embedded_frames
                            elif timestep + 1 < inferred_embedded_frames:
                                timestep += 1
                            elif not has_announced_motion_end:
                                if not first_cycle_plot_saved:
                                    _save_root_xy_plot(reason="first_cycle_complete")
                                    first_cycle_plot_saved = True
                                has_announced_motion_end = True
                                print("[INFO]: Embedded motion reached the final frame, exiting simulation (set --loop to keep running).")
                                break
                        else:
                            timestep += 1

                if counter % render_every == 0:
                    viewer.sync()

                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    except KeyboardInterrupt:
        print("[INFO]: KeyboardInterrupt received, stopping simulation and saving outputs.")
    finally:
        if not first_cycle_plot_saved:
            _save_root_xy_plot(reason="final")


def main():
    parser = argparse.ArgumentParser(description="Unified sim2sim script for multiple robots.")
    parser.add_argument(
        "--robot",
        type=str,
        choices=list(ROBOT_CONFIGS.keys()),
        required=True,
        help="Robot type: " + ", ".join(list(ROBOT_CONFIGS.keys())),
    )
    parser.add_argument(
        "--motion_file",
        type=str,
        default=None,
        help="Path to the motion NPZ file. Optional when motion signals are embedded in ONNX outputs.",
    )
    parser.add_argument("--xml_path", type=str, required=True,
                        help="Path to the robot XML file")
    parser.add_argument("--policy_path", type=str, required=True,
                        help="Path to the ONNX policy file")
    parser.add_argument("--save_json", action="store_true",
                        help="Save motion data to JSON file")
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=False,
                        help="Loop motion/policy when reaching the end of sequence")
    parser.add_argument(
        "--render_every",
        type=int,
        default=10,
        help="Viewer sync interval in physics steps. Larger value means lower rendering frequency.",
    )
    parser.add_argument(
        "--plot_root_xy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot root x-y trajectory and save as an image after simulation ends.",
    )
    parser.add_argument(
        "--root_xy_plot_path",
        type=str,
        default="outputs/root_xy_trajectory.png",
        help="Output path for root x-y trajectory plot.",
    )
    parser.add_argument(
        "--trajectory_frame",
        type=str,
        choices=["world", "root_initial"],
        default="root_initial",
        help="Frame for plotted trajectory: world (x/y) or root_initial (forward/left from initial root heading).",
    )
    parser.add_argument(
        "--flip_left_right",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Flip left-right direction in root_initial trajectory plot.",
    )
    parser.add_argument(
        "--embedded_motion_num_frames",
        type=int,
        default=None,
        help="Frame count for embedded motion (no --motion_file). Enables loop/end control and first-cycle auto-save.",
    )
    
    args = parser.parse_args()
    
    print(f"[INFO]: Robot: {args.robot}")
    print(f"[INFO]: Motion file: {args.motion_file if args.motion_file else 'embedded in ONNX'}")
    print(f"[INFO]: XML path: {args.xml_path}")
    print(f"[INFO]: Policy path: {args.policy_path}")
    
    run_simulation(
        args.robot,
        args.motion_file,
        args.xml_path,
        args.policy_path,
        args.save_json,
        args.loop,
        args.render_every,
        args.plot_root_xy,
        args.root_xy_plot_path,
        args.trajectory_frame,
        args.flip_left_right,
        args.embedded_motion_num_frames,
    )


if __name__ == "__main__":
    main()
