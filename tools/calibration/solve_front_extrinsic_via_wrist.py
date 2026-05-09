#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from calib_utils import (
    add_target_arguments,
    as_transform,
    build_target_config_from_args,
    invert_transform,
    load_fk_callable,
    load_intrinsics,
    load_json,
    parse_index_list,
    rotation_angle_deg,
    rotation_mean,
    solve_target_pose,
    write_json,
)

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_wrist_in_eef(path: Path) -> np.ndarray:
    data = load_json(path)
    if "T_wrist_in_eef" in data:
        T = np.asarray(data["T_wrist_in_eef"], dtype=np.float64)
    elif "R_wrist_in_eef" in data and "t_wrist_in_eef" in data:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.asarray(data["R_wrist_in_eef"], dtype=np.float64)
        T[:3, 3] = np.asarray(data["t_wrist_in_eef"], dtype=np.float64).reshape(3)
    else:
        raise KeyError(f"Could not find T_wrist_in_eef in {path}")
    if T.shape != (4, 4):
        raise ValueError(f"Invalid wrist-in-eef transform from {path}: {T.shape}")
    return T


def main() -> None:
    parser = argparse.ArgumentParser(description="Transfer front-camera extrinsic through wrist camera observations.")
    parser.add_argument("--front-images-dir", type=Path, required=True)
    parser.add_argument("--wrist-images-dir", type=Path, required=True)
    parser.add_argument("--joints", type=Path, required=True)
    parser.add_argument("--front-intrinsics", type=Path, required=True)
    parser.add_argument("--wrist-intrinsics", type=Path, required=True)
    parser.add_argument("--wrist-eye-in-hand", type=Path, required=True)
    parser.add_argument("--fk-module", type=Path, default=Path(__file__).with_name("piper_fk_wrapper.py"))
    parser.add_argument("--fk-function", default="fk_full")
    add_target_arguments(parser, default_pattern=(8, 6), default_square=0.024)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "front_extrinsic_via_wrist.json")
    parser.add_argument(
        "--skip-indices",
        nargs="*",
        default=[],
        help="Transfer sample indices to exclude, e.g. '--skip-indices 0 1 2' or '--skip-indices 0,1,2'.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".png", ".jpg", ".jpeg", ".bmp"],
    )
    args = parser.parse_args()
    args.front_images_dir = args.front_images_dir.expanduser()
    args.wrist_images_dir = args.wrist_images_dir.expanduser()
    args.joints = args.joints.expanduser()
    args.front_intrinsics = args.front_intrinsics.expanduser()
    args.wrist_intrinsics = args.wrist_intrinsics.expanduser()
    args.wrist_eye_in_hand = args.wrist_eye_in_hand.expanduser()
    args.fk_module = args.fk_module.expanduser()
    args.output = args.output.expanduser()
    target_config = build_target_config_from_args(args)
    skip_indices = set(parse_index_list(args.skip_indices))

    front_paths = sorted(
        p for p in args.front_images_dir.iterdir() if p.is_file() and p.suffix.lower() in set(args.extensions)
    )
    wrist_paths = sorted(
        p for p in args.wrist_images_dir.iterdir() if p.is_file() and p.suffix.lower() in set(args.extensions)
    )
    joints = np.asarray(np.load(args.joints), dtype=np.float64)
    if not front_paths or not wrist_paths:
        raise FileNotFoundError("Front or wrist image directory is empty.")
    if len(front_paths) != len(wrist_paths):
        raise ValueError(f"Front image count ({len(front_paths)}) does not match wrist image count ({len(wrist_paths)})")
    if len(front_paths) != len(joints):
        raise ValueError(f"Image count ({len(front_paths)}) does not match joints count ({len(joints)})")

    front_K, front_D = load_intrinsics(args.front_intrinsics)
    wrist_K, wrist_D = load_intrinsics(args.wrist_intrinsics)
    T_wrist_in_eef = load_wrist_in_eef(args.wrist_eye_in_hand)
    fk_fn = load_fk_callable(args.fk_module, args.fk_function)

    candidates_front_in_base: list[np.ndarray] = []
    used_samples: list[dict] = []
    skipped_samples: list[dict] = []

    for idx, (front_path, wrist_path, joint_row) in enumerate(zip(front_paths, wrist_paths, joints)):
        if idx in skip_indices:
            skipped_samples.append(
                {
                    "index": idx,
                    "front_image": front_path.name,
                    "wrist_image": wrist_path.name,
                    "reason": "manual_skip",
                }
            )
            continue
        front_pose = solve_target_pose(image_path=front_path, target_config=target_config, camera_matrix=front_K, dist_coeffs=front_D)
        wrist_pose = solve_target_pose(image_path=wrist_path, target_config=target_config, camera_matrix=wrist_K, dist_coeffs=wrist_D)
        if front_pose is None or wrist_pose is None:
            skipped_samples.append(
                {
                    "index": idx,
                    "front_image": front_path.name,
                    "wrist_image": wrist_path.name,
                    "reason": "target_not_detected",
                    "front_detected": front_pose is not None,
                    "wrist_detected": wrist_pose is not None,
                }
            )
            continue

        T_eef_in_base = as_transform(fk_fn(np.asarray(joint_row[:6], dtype=np.float64)))
        T_target_in_front = front_pose["T_target_in_cam"]
        T_target_in_wrist = wrist_pose["T_target_in_cam"]
        T_front_in_base = T_eef_in_base @ T_wrist_in_eef @ T_target_in_wrist @ invert_transform(T_target_in_front)
        candidates_front_in_base.append(T_front_in_base)
        used_samples.append(
            {
                "index": idx,
                "front_image": front_path.name,
                "wrist_image": wrist_path.name,
                "front_reprojection_px": front_pose["reprojection_px"],
                "wrist_reprojection_px": wrist_pose["reprojection_px"],
                "T_front_in_base": T_front_in_base.tolist(),
            }
        )

    if len(candidates_front_in_base) < 3:
        raise RuntimeError("Need at least 3 valid transfer samples.")

    R_avg = rotation_mean([T[:3, :3] for T in candidates_front_in_base])
    t_avg = np.mean([T[:3, 3] for T in candidates_front_in_base], axis=0)
    T_front_in_base = np.eye(4, dtype=np.float64)
    T_front_in_base[:3, :3] = R_avg
    T_front_in_base[:3, 3] = t_avg
    T_base_in_front = invert_transform(T_front_in_base)

    rot_errs = []
    trans_errs = []
    for sample, candidate in zip(used_samples, candidates_front_in_base):
        rot_err = rotation_angle_deg(R_avg, candidate[:3, :3])
        trans_err = float(np.linalg.norm(t_avg - candidate[:3, 3]))
        sample["rotation_dev_deg"] = rot_err
        sample["translation_dev_m"] = trans_err
        rot_errs.append(rot_err)
        trans_errs.append(trans_err)

    payload = {
        "convention": "T_src_in_dst maps points from src frame into dst frame.",
        "camera_frame": "front_camera_optical",
        "robot_frame": "arm_base",
        "target_type": target_config["target_type"],
        "square_size_m": target_config["square_size_m"],
        "num_input_samples": len(front_paths),
        "num_used_samples": len(used_samples),
        "num_skipped_samples": len(skipped_samples),
        "manual_skip_indices": sorted(skip_indices),
        "used_samples": used_samples,
        "skipped_samples": skipped_samples,
        "T_front_in_base": T_front_in_base.tolist(),
        "T_base_in_front": T_base_in_front.tolist(),
        "T_camera_to_base": T_front_in_base.tolist(),
        "T_base_to_camera": T_base_in_front.tolist(),
        "rotation_dev_deg_mean": float(np.mean(rot_errs)),
        "rotation_dev_deg_max": float(np.max(rot_errs)),
        "translation_dev_m_mean": float(np.mean(trans_errs)),
        "translation_dev_m_max": float(np.max(trans_errs)),
    }
    if target_config["target_type"] == "checkerboard":
        payload["pattern_inner_corners"] = [target_config["pattern"][0], target_config["pattern"][1]]
    else:
        payload["charuco_squares"] = [target_config["charuco_squares_x"], target_config["charuco_squares_y"]]
        payload["marker_length_m"] = target_config["marker_length_m"]
        payload["aruco_dict_name"] = target_config["aruco_dict_name"]
    write_json(args.output, payload)

    print(f"Saved front-camera extrinsic to {args.output}")
    print(f"samples used: {len(used_samples)} / {len(front_paths)}")
    if skip_indices:
        print(f"manual skip indices: {sorted(skip_indices)}")
    print(f"rotation deviation mean/max: {np.mean(rot_errs):.4f} / {np.max(rot_errs):.4f} deg")
    print(f"translation deviation mean/max: {np.mean(trans_errs):.4f} / {np.max(trans_errs):.4f} m")


if __name__ == "__main__":
    main()
