#!/usr/bin/env python3
"""Print per-joint angle ranges from a motion NPZ file.

Example:
    python3 scripts/print_joint_angle_ranges.py \
        --motion_file source/motion/x2/npz/guangboticao_win5.npz \
        --robot x2 \
        --degrees
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


ROBOT_JOINT_NAMES: dict[str, list[str]] = {
    "x2": [
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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print per-joint angle ranges from a motion NPZ file.")
    parser.add_argument("--motion_file", type=Path, required=True, help="Path to the motion npz file.")
    parser.add_argument("--robot", choices=sorted(ROBOT_JOINT_NAMES.keys()), default="x2", help="Robot joint order.")
    parser.add_argument("--degrees", action="store_true", help="Convert the output from radians to degrees.")
    parser.add_argument(
        "--companion_csv",
        type=Path,
        default=None,
        help="Optional CSV to cross-check joint-column ordering. If omitted, auto-searches a same-stem csv.",
    )
    return parser.parse_args()


def find_companion_csv(motion_file: Path) -> Path | None:
    candidate = motion_file.with_suffix(".csv")
    if candidate.is_file():
        return candidate

    if motion_file.parent.name == "npz":
        sibling = motion_file.parent.parent / "csv" / f"{motion_file.stem}.csv"
        if sibling.is_file():
            return sibling
    return None


def load_joint_data_from_csv(csv_path: Path, expected_dof: int) -> np.ndarray | None:
    try:
        data = np.loadtxt(csv_path, delimiter=",")
    except Exception:
        return None

    if data.ndim != 2 or data.shape[1] < 7 + expected_dof:
        return None
    return np.asarray(data[:, 7 : 7 + expected_dof], dtype=np.float64)


def mismatch_score(reference: np.ndarray, candidate: np.ndarray) -> float:
    compare_len = min(len(reference), len(candidate), 512)
    if compare_len <= 0:
        return float("inf")

    ref = reference[:compare_len]
    cand = candidate[:compare_len]
    joint_min = np.minimum(ref.min(axis=0), cand.min(axis=0))
    joint_max = np.maximum(ref.max(axis=0), cand.max(axis=0))
    scale = np.maximum(joint_max - joint_min, 1.0e-6)
    return float(np.mean(np.abs(ref - cand) / scale))


def main() -> None:
    args = parse_args()
    data = np.load(args.motion_file, allow_pickle=True)
    if "joint_pos" not in data:
        raise KeyError(f"{args.motion_file} does not contain 'joint_pos'. Available keys: {list(data.files)}")

    joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
    joint_names = ROBOT_JOINT_NAMES[args.robot]

    if joint_pos.ndim != 2:
        raise ValueError(f"'joint_pos' must be 2D, got shape {joint_pos.shape}")
    if joint_pos.shape[1] != len(joint_names):
        raise ValueError(
            f"Joint dimension mismatch for robot '{args.robot}': "
            f"npz has {joint_pos.shape[1]} joints, expected {len(joint_names)}."
        )

    data_source = "npz"
    companion_csv = args.companion_csv if args.companion_csv is not None else find_companion_csv(args.motion_file)
    if companion_csv is not None:
        csv_joint_pos = load_joint_data_from_csv(companion_csv, len(joint_names))
        if csv_joint_pos is not None:
            score = mismatch_score(csv_joint_pos, joint_pos)
            if score > 0.10:
                print(
                    f"[WARN] Detected likely joint-order mismatch in {args.motion_file.name} "
                    f"(compared against {companion_csv.name}, score={score:.3f})."
                )
                print("[WARN] Using joint columns from the companion CSV, which follow the training joint_names order.")
                joint_pos = csv_joint_pos
                data_source = f"csv:{companion_csv}"
            else:
                data_source = f"npz (validated by {companion_csv.name}, score={score:.3f})"

    if args.degrees:
        joint_pos = np.rad2deg(joint_pos)
        unit = "deg"
    else:
        unit = "rad"

    joint_min = joint_pos.min(axis=0)
    joint_max = joint_pos.max(axis=0)
    joint_range = joint_max - joint_min

    print(f"motion_file: {args.motion_file}")
    print(f"frames: {joint_pos.shape[0]}")
    print(f"robot: {args.robot}")
    print(f"unit: {unit}")
    print(f"data_source: {data_source}")
    print("")
    print(f"{'joint_name':32s} {'min':>12s} {'max':>12s} {'range':>12s}")
    print("-" * 72)
    for name, min_val, max_val, range_val in zip(joint_names, joint_min, joint_max, joint_range, strict=True):
        print(f"{name:32s} {min_val:12.6f} {max_val:12.6f} {range_val:12.6f}")


if __name__ == "__main__":
    main()
