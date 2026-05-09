#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

from calib_utils import add_target_arguments, build_target_config_from_args, collect_calibration_points

DATA_DIR = Path(__file__).resolve().parent / "data"

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = load_json(path)
    camera_matrix = np.asarray(data.get("camera_matrix", data.get("K")), dtype=np.float64)
    dist_coeffs = np.asarray(data.get("dist_coeffs", data.get("D", [])), dtype=np.float64).reshape(-1, 1)
    if camera_matrix.shape != (3, 3):
        raise ValueError(f"Invalid camera matrix shape from {path}: {camera_matrix.shape}")
    return camera_matrix, dist_coeffs

def load_fk_callable(module_path: Path, function_name: str):
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        fk_fn = getattr(module, function_name)
    except AttributeError as exc:
        raise AttributeError(f"{module_path} does not define '{function_name}'") from exc
    return fk_fn


def as_transform(value) -> np.ndarray:
    if isinstance(value, tuple) and len(value) == 2:
        rot, trans = value
        rot = np.asarray(rot, dtype=np.float64)
        trans = np.asarray(trans, dtype=np.float64).reshape(3)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rot
        T[:3, 3] = trans
        return T

    T = np.asarray(value, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"FK must return a 4x4 transform or (R, t), got shape {T.shape}")
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


METHODS = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate a fixed front camera against robot base using checkerboard + FK.")
    parser.add_argument("--images-dir", type=Path, default=DATA_DIR / "handeye_imgs")
    parser.add_argument("--joints", type=Path, default=DATA_DIR / "handeye_joints_left.npy")
    parser.add_argument("--intrinsics", type=Path, default=DATA_DIR / "intrinsics_front.json")
    parser.add_argument(
        "--fk-module",
        type=Path,
        default=Path(__file__).with_name("piper_fk_wrapper.py"),
        help="Python file that defines the FK function.",
    )
    parser.add_argument("--fk-function", default="fk_full", help="Function that maps one 6D joint vector to base->gripper transform.")
    add_target_arguments(parser, default_pattern=(8, 6), default_square=0.024)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "extrinsics_front_base.json")
    parser.add_argument("--method", choices=sorted(METHODS), default="tsai")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".png", ".jpg", ".jpeg", ".bmp"],
        help="Image suffixes to scan under --images-dir.",
    )
    args = parser.parse_args()
    args.images_dir = args.images_dir.expanduser()
    args.joints = args.joints.expanduser()
    args.intrinsics = args.intrinsics.expanduser()
    args.fk_module = args.fk_module.expanduser()
    args.output = args.output.expanduser()
    target_config = build_target_config_from_args(args)

    camera_matrix, dist_coeffs = load_intrinsics(args.intrinsics)
    fk_fn = load_fk_callable(args.fk_module, args.fk_function)
    joints = np.asarray(np.load(args.joints), dtype=np.float64)
    if joints.ndim != 2:
        raise ValueError(f"Expected joints with shape (N, D), got {joints.shape}")

    image_paths = sorted(
        p for p in args.images_dir.iterdir() if p.is_file() and p.suffix.lower() in set(args.extensions)
    )
    if not image_paths:
        raise FileNotFoundError(f"No hand-eye images found under {args.images_dir}")
    if len(image_paths) != len(joints):
        raise ValueError(f"Image count ({len(image_paths)}) does not match joints count ({len(joints)})")

    R_gripper_to_base: list[np.ndarray] = []
    t_gripper_to_base: list[np.ndarray] = []
    R_target_to_cam: list[np.ndarray] = []
    t_target_to_cam: list[np.ndarray] = []
    used_samples: list[dict] = []
    skipped_images: list[str] = []

    for idx, (image_path, joint_row) in enumerate(zip(image_paths, joints)):
        result = collect_calibration_points(image_path, target_config)
        if result is None:
            skipped_images.append(image_path.name)
            continue
        objp, imgp, _ = result
        ok, rvec, tvec = cv2.solvePnP(objp, imgp.reshape(-1, 1, 2), camera_matrix, dist_coeffs)
        if not ok:
            skipped_images.append(image_path.name)
            continue
        R_tc, _ = cv2.Rodrigues(rvec)
        T_base_to_gripper = as_transform(fk_fn(np.asarray(joint_row[:6], dtype=np.float64)))
        T_gripper_to_base = invert_transform(T_base_to_gripper)

        R_gripper_to_base.append(T_gripper_to_base[:3, :3])
        t_gripper_to_base.append(T_gripper_to_base[:3, 3].reshape(3, 1))
        R_target_to_cam.append(R_tc)
        t_target_to_cam.append(tvec.reshape(3, 1))
        used_samples.append({"index": idx, "image": image_path.name})

    if len(used_samples) < 3:
        raise RuntimeError("Need at least 3 valid poses for hand-eye calibration.")

    R_camera_to_base, t_camera_to_base = cv2.calibrateHandEye(
        R_gripper_to_base,
        t_gripper_to_base,
        R_target_to_cam,
        t_target_to_cam,
        method=METHODS[args.method],
    )

    T_camera_to_base = np.eye(4, dtype=np.float64)
    T_camera_to_base[:3, :3] = R_camera_to_base
    T_camera_to_base[:3, 3] = t_camera_to_base.reshape(3)
    T_base_to_camera = invert_transform(T_camera_to_base)

    payload = {
        "convention": "T_src_to_dst maps a point expressed in src frame into dst frame.",
        "camera_frame": "front_camera_optical",
        "robot_frame": "robot_base",
        "method": args.method,
        "target_type": target_config["target_type"],
        "square_size_m": target_config["square_size_m"],
        "num_input_samples": len(image_paths),
        "num_used_samples": len(used_samples),
        "num_skipped_samples": len(skipped_images),
        "used_samples": used_samples,
        "skipped_images": skipped_images,
        "T_camera_to_base": T_camera_to_base.tolist(),
        "T_base_to_camera": T_base_to_camera.tolist(),
        "R_camera_to_base": R_camera_to_base.tolist(),
        "t_camera_to_base": t_camera_to_base.reshape(-1).tolist(),
        "R_base_to_camera": T_base_to_camera[:3, :3].tolist(),
        "t_base_to_camera": T_base_to_camera[:3, 3].tolist(),
        "legacy": {
            "R_cam_base": T_base_to_camera[:3, :3].tolist(),
            "t_cam_base": T_base_to_camera[:3, 3].tolist(),
        },
    }
    if target_config["target_type"] == "checkerboard":
        payload["pattern_inner_corners"] = [target_config["pattern"][0], target_config["pattern"][1]]
    else:
        payload["charuco_squares"] = [target_config["charuco_squares_x"], target_config["charuco_squares_y"]]
        payload["marker_length_m"] = target_config["marker_length_m"]
        payload["aruco_dict_name"] = target_config["aruco_dict_name"]
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved extrinsics to {args.output}")
    print(f"samples used: {len(used_samples)} / {len(image_paths)}")
    print("Returned transform for projection is T_base_to_camera.")


if __name__ == "__main__":
    main()
