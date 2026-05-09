#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from calib_utils import invert_transform, load_json, rotation_angle_deg, rotation_mean, write_json

DATA_DIR = Path(__file__).resolve().parent / "data"


def make_transform(xyz: tuple[float, float, float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return T


def load_front_in_base(path: Path) -> np.ndarray:
    data = load_json(path)
    if "T_front_in_base" in data:
        T = np.asarray(data["T_front_in_base"], dtype=np.float64)
    elif "T_camera_to_base" in data:
        T = np.asarray(data["T_camera_to_base"], dtype=np.float64)
    else:
        raise KeyError(f"Could not find T_front_in_base or T_camera_to_base in {path}")
    if T.shape != (4, 4):
        raise ValueError(f"Invalid transform shape in {path}: {T.shape}")
    return T


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert left/right arm-base front-camera calibrations into one body-frame result.")
    parser.add_argument("--left", type=Path, required=True, help="Front extrinsic solved from left arm samples.")
    parser.add_argument("--right", type=Path, required=True, help="Front extrinsic solved from right arm samples.")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "front_in_body_from_left_right.json")
    args = parser.parse_args()

    # From AgileX mobile_aloha_sim v2.0.0 aloha_tracer2_dabai_dark.urdf:
    # fl_base_joint origin xyz="0.23875 0.3 0.775", rpy="0 0 0"
    # fr_base_joint origin xyz="0.23875 -0.3 0.775", rpy="0 0 0"
    T_left_base_in_body = make_transform((0.23875, 0.3, 0.775))
    T_right_base_in_body = make_transform((0.23875, -0.3, 0.775))

    T_front_in_left_base = load_front_in_base(args.left)
    T_front_in_right_base = load_front_in_base(args.right)

    T_front_in_body_from_left = T_left_base_in_body @ T_front_in_left_base
    T_front_in_body_from_right = T_right_base_in_body @ T_front_in_right_base

    R_avg = rotation_mean([
        T_front_in_body_from_left[:3, :3],
        T_front_in_body_from_right[:3, :3],
    ])
    t_avg = 0.5 * (T_front_in_body_from_left[:3, 3] + T_front_in_body_from_right[:3, 3])
    T_front_in_body = np.eye(4, dtype=np.float64)
    T_front_in_body[:3, :3] = R_avg
    T_front_in_body[:3, 3] = t_avg
    T_body_in_front = invert_transform(T_front_in_body)

    left_rot_dev = rotation_angle_deg(R_avg, T_front_in_body_from_left[:3, :3])
    right_rot_dev = rotation_angle_deg(R_avg, T_front_in_body_from_right[:3, :3])
    left_trans_dev = float(np.linalg.norm(t_avg - T_front_in_body_from_left[:3, 3]))
    right_trans_dev = float(np.linalg.norm(t_avg - T_front_in_body_from_right[:3, 3]))
    left_right_rot_gap = rotation_angle_deg(
        T_front_in_body_from_left[:3, :3],
        T_front_in_body_from_right[:3, :3],
    )
    left_right_trans_gap = float(
        np.linalg.norm(T_front_in_body_from_left[:3, 3] - T_front_in_body_from_right[:3, 3])
    )

    payload = {
        "convention": "T_src_in_dst maps points from src frame into dst frame.",
        "camera_frame": "front_camera_optical",
        "robot_frame": "body_Link",
        "urdf_source": "agilexrobotics/mobile_aloha_sim v2.0.0 aloha_tracer2_dabai_dark.urdf fixed base joints",
        "T_left_base_in_body": T_left_base_in_body.tolist(),
        "T_right_base_in_body": T_right_base_in_body.tolist(),
        "T_front_in_body_from_left": T_front_in_body_from_left.tolist(),
        "T_front_in_body_from_right": T_front_in_body_from_right.tolist(),
        "T_front_in_body": T_front_in_body.tolist(),
        "T_body_in_front": T_body_in_front.tolist(),
        "T_camera_to_body": T_front_in_body.tolist(),
        "T_body_to_camera": T_body_in_front.tolist(),
        "left_rot_dev_deg": left_rot_dev,
        "right_rot_dev_deg": right_rot_dev,
        "left_trans_dev_m": left_trans_dev,
        "right_trans_dev_m": right_trans_dev,
        "left_right_rot_gap_deg": left_right_rot_gap,
        "left_right_trans_gap_m": left_right_trans_gap,
    }
    write_json(args.output, payload)

    print(f"Saved combined body-frame front extrinsic to {args.output}")
    print(f"left/right rotation gap: {left_right_rot_gap:.4f} deg")
    print(f"left/right translation gap: {left_right_trans_gap:.4f} m")


if __name__ == "__main__":
    main()
