#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from calib_utils import (
    add_target_arguments,
    as_transform,
    build_target_config_from_args,
    invert_transform,
    load_fk_callable,
    load_intrinsics,
    parse_index_list,
    solve_target_pose,
    write_json,
)


METHODS = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}

DATA_DIR = Path(__file__).resolve().parent / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve wrist camera eye-in-hand extrinsic against EEF.")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--joints", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--fk-module", type=Path, default=Path(__file__).with_name("piper_fk_wrapper.py"))
    parser.add_argument("--fk-function", default="fk_full")
    add_target_arguments(parser, default_pattern=(8, 6), default_square=0.024)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "wrist_left_eye_in_hand.json")
    parser.add_argument("--method", choices=sorted(METHODS), default="tsai")
    parser.add_argument(
        "--skip-indices",
        nargs="*",
        default=[],
        help="Sample indices to exclude, e.g. '--skip-indices 0 1 2' or '--skip-indices 0,1,2'.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".png", ".jpg", ".jpeg", ".bmp"],
    )
    args = parser.parse_args()
    args.images_dir = args.images_dir.expanduser()
    args.joints = args.joints.expanduser()
    args.intrinsics = args.intrinsics.expanduser()
    args.fk_module = args.fk_module.expanduser()
    args.output = args.output.expanduser()
    target_config = build_target_config_from_args(args)
    skip_indices = set(parse_index_list(args.skip_indices))

    image_paths = sorted(
        p for p in args.images_dir.iterdir() if p.is_file() and p.suffix.lower() in set(args.extensions)
    )
    joints = np.asarray(np.load(args.joints), dtype=np.float64)
    if not image_paths:
        raise FileNotFoundError(f"No images found under {args.images_dir}")
    if joints.ndim != 2:
        raise ValueError(f"Expected joints with shape (N, D), got {joints.shape}")
    if len(image_paths) != len(joints):
        raise ValueError(f"Image count ({len(image_paths)}) does not match joints count ({len(joints)})")

    camera_matrix, dist_coeffs = load_intrinsics(args.intrinsics)
    fk_fn = load_fk_callable(args.fk_module, args.fk_function)

    R_eef_in_base: list[np.ndarray] = []
    t_eef_in_base: list[np.ndarray] = []
    R_board_in_wrist: list[np.ndarray] = []
    t_board_in_wrist: list[np.ndarray] = []
    used_samples: list[dict] = []
    skipped_images: list[str] = []

    for idx, (image_path, joint_row) in enumerate(zip(image_paths, joints)):
        if idx in skip_indices:
            skipped_images.append(image_path.name)
            continue
        pose = solve_target_pose(image_path=image_path, target_config=target_config, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        if pose is None:
            skipped_images.append(image_path.name)
            continue

        T_eef_in_base = as_transform(fk_fn(np.asarray(joint_row[:6], dtype=np.float64)))
        T_target_in_wrist = pose["T_target_in_cam"]

        R_eef_in_base.append(T_eef_in_base[:3, :3])
        t_eef_in_base.append(T_eef_in_base[:3, 3].reshape(3, 1))
        R_board_in_wrist.append(T_target_in_wrist[:3, :3])
        t_board_in_wrist.append(T_target_in_wrist[:3, 3].reshape(3, 1))
        used_samples.append(
            {
                "index": idx,
                "image": image_path.name,
                "reprojection_px": pose["reprojection_px"],
            }
        )

    if len(used_samples) < 3:
        raise RuntimeError("Need at least 3 valid samples for eye-in-hand calibration.")

    R_wrist_in_eef, t_wrist_in_eef = cv2.calibrateHandEye(
        R_eef_in_base,
        t_eef_in_base,
        R_board_in_wrist,
        t_board_in_wrist,
        method=METHODS[args.method],
    )

    T_wrist_in_eef = np.eye(4, dtype=np.float64)
    T_wrist_in_eef[:3, :3] = R_wrist_in_eef
    T_wrist_in_eef[:3, 3] = t_wrist_in_eef.reshape(3)
    T_eef_in_wrist = invert_transform(T_wrist_in_eef)

    payload = {
        "convention": "T_src_in_dst maps points from src frame into dst frame.",
        "method": args.method,
        "target_type": target_config["target_type"],
        "square_size_m": target_config["square_size_m"],
        "num_input_samples": len(image_paths),
        "num_used_samples": len(used_samples),
        "num_skipped_samples": len(skipped_images),
        "manual_skip_indices": sorted(skip_indices),
        "used_samples": used_samples,
        "skipped_images": skipped_images,
        "T_wrist_in_eef": T_wrist_in_eef.tolist(),
        "T_eef_in_wrist": T_eef_in_wrist.tolist(),
        "R_wrist_in_eef": T_wrist_in_eef[:3, :3].tolist(),
        "t_wrist_in_eef": T_wrist_in_eef[:3, 3].tolist(),
    }
    if target_config["target_type"] == "checkerboard":
        payload["pattern_inner_corners"] = [target_config["pattern"][0], target_config["pattern"][1]]
    else:
        payload["charuco_squares"] = [target_config["charuco_squares_x"], target_config["charuco_squares_y"]]
        payload["marker_length_m"] = target_config["marker_length_m"]
        payload["aruco_dict_name"] = target_config["aruco_dict_name"]
    write_json(args.output, payload)

    mean_reproj = float(np.mean([item["reprojection_px"] for item in used_samples]))
    print(f"Saved wrist eye-in-hand calibration to {args.output}")
    print(f"samples used: {len(used_samples)} / {len(image_paths)}")
    if skip_indices:
        print(f"manual skip indices: {sorted(skip_indices)}")
    print(f"mean checkerboard reprojection: {mean_reproj:.4f} px")


if __name__ == "__main__":
    main()
