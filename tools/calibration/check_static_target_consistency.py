#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import importlib.util
from pathlib import Path

import numpy as np

from calib_utils import load_intrinsics, load_json, parse_index_list, rotation_angle_deg, rotation_mean, solve_target_pose


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether a static calibration target stays fixed in robot base coordinates.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--target-pose-json", type=Path, required=True, help="Usually wrist eye-in-hand JSON containing T_wrist_in_eef.")
    parser.add_argument("--fk-module", type=Path, required=True)
    parser.add_argument("--fk-function", default="fk_full")
    parser.add_argument("--target-type", choices=["checkerboard", "charuco"], required=True)
    parser.add_argument("--pattern", default="8x6")
    parser.add_argument("--charuco-squares-x", type=int, default=4)
    parser.add_argument("--charuco-squares-y", type=int, default=5)
    parser.add_argument("--aruco-dict", default="DICT_4X4_50")
    parser.add_argument("--square-size", type=float, required=True)
    parser.add_argument("--marker-length", type=float, default=None)
    parser.add_argument(
        "--skip-indices",
        nargs="*",
        default=[],
        help="Sample indices to exclude from the check, e.g. '--skip-indices 0 1 2' or '--skip-indices 0,1,2'.",
    )
    args = parser.parse_args()

    args.manifest = args.manifest.expanduser()
    args.images_dir = args.images_dir.expanduser()
    args.intrinsics = args.intrinsics.expanduser()
    args.target_pose_json = args.target_pose_json.expanduser()
    args.fk_module = args.fk_module.expanduser()
    skip_indices = set(parse_index_list(args.skip_indices))

    if args.target_type == "checkerboard":
        parts = args.pattern.lower().replace(",", "x").split("x")
        target_config = {
            "target_type": "checkerboard",
            "pattern": (int(parts[0]), int(parts[1])),
            "square_size_m": float(args.square_size),
        }
    else:
        if args.marker_length is None:
            raise ValueError("--marker-length is required for charuco")
        target_config = {
            "target_type": "charuco",
            "charuco_squares_x": int(args.charuco_squares_x),
            "charuco_squares_y": int(args.charuco_squares_y),
            "square_size_m": float(args.square_size),
            "marker_length_m": float(args.marker_length),
            "aruco_dict_name": args.aruco_dict,
        }

    manifest = load_json(args.manifest)
    intr_K, intr_D = load_intrinsics(args.intrinsics)
    target_pose_data = load_json(args.target_pose_json)
    T_wrist_in_eef = np.asarray(target_pose_data["T_wrist_in_eef"], dtype=np.float64)
    fk_module = load_module(args.fk_module)
    fk_fn = getattr(fk_module, args.fk_function)

    Ts = []
    reproj = []
    rows = []
    for sample in manifest["samples"]:
        if sample["index"] in skip_indices:
            continue
        img_name = sample.get("image", sample.get("wrist_image"))
        pose = solve_target_pose(args.images_dir / img_name, target_config, intr_K, intr_D)
        if pose is None:
            continue
        q = np.asarray(sample["joints_rad"], dtype=np.float64)
        T_eef_in_base = np.asarray(fk_fn(q), dtype=np.float64)
        T_target_in_base = T_eef_in_base @ T_wrist_in_eef @ pose["T_target_in_cam"]
        Ts.append(T_target_in_base)
        reproj.append(pose["reprojection_px"])
        rows.append(sample["index"])

    if not Ts:
        raise RuntimeError("No valid target detections.")

    Rs = [T[:3, :3] for T in Ts]
    ts = np.asarray([T[:3, 3] for T in Ts], dtype=np.float64)
    R_avg = rotation_mean(Rs)
    t_avg = ts.mean(axis=0)
    rdev = np.asarray([rotation_angle_deg(R_avg, R) for R in Rs], dtype=np.float64)
    tdev = np.linalg.norm(ts - t_avg, axis=1)

    print(f"valid samples: {len(Ts)}")
    if skip_indices:
        print(f"manual skip indices: {sorted(skip_indices)}")
    print(f"reprojection mean/max: {float(np.mean(reproj)):.4f} / {float(np.max(reproj)):.4f} px")
    print(f"static-target rotation deviation mean/max: {float(np.mean(rdev)):.4f} / {float(np.max(rdev)):.4f} deg")
    print(f"static-target translation deviation mean/max: {float(np.mean(tdev)):.4f} / {float(np.max(tdev)):.4f} m")
    print("worst samples:")
    worst = np.argsort(-(tdev + rdev / 100.0))[:8]
    for idx in worst:
        print(
            f"  sample {rows[int(idx)]}: rot_dev={float(rdev[idx]):.4f} deg "
            f"trans_dev={float(tdev[idx]):.4f} m reproj={float(reproj[idx]):.4f} px"
        )


if __name__ == "__main__":
    main()
