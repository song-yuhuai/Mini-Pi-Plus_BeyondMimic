#!/usr/bin/env python3
"""Apply sliding-window smoothing to motion CSV data.

This utility is designed for numeric motion CSV files such as the X2 motions
under ``source/motion/x2/csv``. It supports common window types and can
renormalize quaternion columns after smoothing.

Examples:

    python scripts/smooth_motion_csv.py \
        --input_csv source/motion/x2/csv/daizuwu2_0.5-1.8-1.0_more_feet_dis.csv \
        --output_csv source/motion/x2/csv/daizuwu2_0.5-1.8-1.0_more_feet_dis_smooth.csv \
        --window_size 9 \
        --window_type hann \
        --quat_slice 3:7

    python scripts/smooth_motion_csv.py \
        --input_csv source/motion/x2/csv/daizuwu2_0.5-1.8-1.0_more_feet_dis.csv \
        --output_csv source/motion/x2/csv/daizuwu2_0.5-1.8-1.0_more_feet_dis_joint_smooth.csv \
        --window_size 7 \
        --window_type gaussian \
        --gaussian_sigma 1.5 \
        --smooth_slice 7:
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_slice(spec: str | None, width: int) -> slice | None:
    """Parse a ``start:end`` slice string."""
    if spec is None:
        return None
    if ":" not in spec:
        raise ValueError(f"Invalid slice '{spec}'. Expected format like '7:' or '3:7'.")

    start_str, end_str = spec.split(":", maxsplit=1)
    start = int(start_str) if start_str else 0
    end = int(end_str) if end_str else width
    if start < 0 or end < 0 or start > width or end > width or start >= end:
        raise ValueError(f"Slice '{spec}' is out of bounds for width {width}.")
    return slice(start, end)


def build_window(window_size: int, window_type: str, gaussian_sigma: float) -> np.ndarray:
    """Construct a normalized 1D smoothing window."""
    if window_size < 1:
        raise ValueError("window_size must be >= 1.")
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd so the smoothing stays centered.")

    if window_size == 1:
        return np.array([1.0], dtype=np.float64)

    if window_type == "box":
        weights = np.ones(window_size, dtype=np.float64)
    elif window_type == "hann":
        weights = np.hanning(window_size)
    elif window_type == "hamming":
        weights = np.hamming(window_size)
    elif window_type == "gaussian":
        if gaussian_sigma <= 0.0:
            raise ValueError("gaussian_sigma must be > 0 when using gaussian window.")
        radius = window_size // 2
        x = np.arange(-radius, radius + 1, dtype=np.float64)
        weights = np.exp(-(x**2) / (2.0 * gaussian_sigma**2))
    else:
        raise ValueError(f"Unsupported window_type '{window_type}'.")

    weights_sum = weights.sum()
    if weights_sum <= 0.0:
        raise ValueError("Window weights sum to zero.")
    return weights / weights_sum


def smooth_array(data: np.ndarray, weights: np.ndarray, target_slice: slice | None) -> np.ndarray:
    """Smooth selected columns in a 2D motion array."""
    result = data.copy()
    if target_slice is None:
        target_slice = slice(0, data.shape[1])

    view = data[:, target_slice]
    radius = len(weights) // 2
    padded = np.pad(view, ((radius, radius), (0, 0)), mode="edge")

    smoothed = np.empty_like(view, dtype=np.float64)
    for row_idx in range(view.shape[0]):
        window = padded[row_idx : row_idx + len(weights)]
        smoothed[row_idx] = np.tensordot(weights, window, axes=(0, 0))

    result[:, target_slice] = smoothed
    return result


def renormalize_quaternions(data: np.ndarray, quat_slice: slice | None) -> np.ndarray:
    """Normalize quaternion columns row by row."""
    if quat_slice is None:
        return data

    quat = data[:, quat_slice]
    if quat.shape[1] != 4:
        raise ValueError(
            f"quat_slice must select exactly 4 columns, got shape {quat.shape}."
        )

    norms = np.linalg.norm(quat, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("Encountered near-zero quaternion norm after smoothing.")

    data[:, quat_slice] = quat / norms
    return data


def maybe_load_header(input_csv: Path, has_header: bool) -> tuple[np.ndarray, str | None]:
    """Load motion CSV, preserving the header line when requested."""
    if has_header:
        with input_csv.open("r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n")
        data = np.loadtxt(input_csv, delimiter=",", skiprows=1)
        return np.atleast_2d(data), header

    data = np.loadtxt(input_csv, delimiter=",")
    return np.atleast_2d(data), None


def save_csv(output_csv: Path, data: np.ndarray, header: str | None, decimal_places: int) -> None:
    """Write the smoothed CSV back to disk."""
    fmt = f"%.{decimal_places}f"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output_csv,
        data,
        delimiter=",",
        fmt=fmt,
        header="" if header is None else header,
        comments="",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply sliding-window smoothing to motion CSV data.")
    parser.add_argument("--input_csv", required=True, help="Path to the input motion CSV.")
    parser.add_argument("--output_csv", required=True, help="Path to the output smoothed CSV.")
    parser.add_argument(
        "--window_size",
        type=int,
        default=7,
        help="Odd-number window size for smoothing. Default: 7.",
    )
    parser.add_argument(
        "--window_type",
        choices=["box", "hann", "hamming", "gaussian"],
        default="hann",
        help="Window type. Default: hann.",
    )
    parser.add_argument(
        "--gaussian_sigma",
        type=float,
        default=1.5,
        help="Sigma used for gaussian window. Default: 1.5.",
    )
    parser.add_argument(
        "--smooth_slice",
        default="0:",
        help="Column slice to smooth, e.g. '0:', '7:', or '3:20'. Default: all columns.",
    )
    parser.add_argument(
        "--quat_slice",
        default="3:7",
        help="Quaternion column slice to renormalize after smoothing. Use '' to disable. Default: 3:7.",
    )
    parser.add_argument(
        "--has_header",
        action="store_true",
        help="Set this if the CSV has a header row.",
    )
    parser.add_argument(
        "--decimal_places",
        type=int,
        default=6,
        help="Number of decimal places to save. Default: 6.",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()

    data, header = maybe_load_header(input_csv, args.has_header)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D motion matrix, got shape {data.shape}.")

    smooth_slice = parse_slice(args.smooth_slice, data.shape[1])
    quat_slice = parse_slice(args.quat_slice, data.shape[1]) if args.quat_slice else None
    weights = build_window(args.window_size, args.window_type, args.gaussian_sigma)

    smoothed = smooth_array(data, weights, smooth_slice)
    smoothed = renormalize_quaternions(smoothed, quat_slice)
    save_csv(output_csv, smoothed, header, args.decimal_places)

    diff = smoothed - data
    print(f"Loaded: {input_csv}")
    print(f"Shape: {data.shape}")
    print(
        "Smoothing:"
        f" window_size={args.window_size}, window_type={args.window_type},"
        f" smooth_slice={args.smooth_slice}, quat_slice={args.quat_slice or 'disabled'}"
    )
    print(f"Saved: {output_csv}")
    print(
        "Delta summary:"
        f" mean_abs={np.mean(np.abs(diff)):.6f},"
        f" max_abs={np.max(np.abs(diff)):.6f}"
    )


if __name__ == "__main__":
    main()
