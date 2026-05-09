#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from calib_utils import add_target_arguments, build_target_config_from_args, collect_calibration_points

DATA_DIR = Path(__file__).resolve().parent / "data"


def mean_reprojection_error(
    obj_points: list[np.ndarray],
    img_points: list[np.ndarray],
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    total_error = 0.0
    total_points = 0
    for objp, imgp, rvec, tvec in zip(obj_points, img_points, rvecs, tvecs):
        reproj, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        err = np.linalg.norm(imgp.reshape(-1, 2) - reproj.reshape(-1, 2), axis=1)
        total_error += float(err.sum())
        total_points += int(err.size)
    return total_error / max(total_points, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate camera intrinsics from checkerboard images.")
    parser.add_argument("--images-dir", type=Path, default=Path("calib_imgs"))
    add_target_arguments(parser, default_pattern=(8, 6), default_square=0.024)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "intrinsics_front.json")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".png", ".jpg", ".jpeg", ".bmp"],
        help="Image suffixes to scan under --images-dir.",
    )
    args = parser.parse_args()
    args.images_dir = args.images_dir.expanduser()
    args.output = args.output.expanduser()
    target_config = build_target_config_from_args(args)

    image_paths = sorted(
        p for p in args.images_dir.iterdir() if p.is_file() and p.suffix.lower() in set(args.extensions)
    )
    if not image_paths:
        raise FileNotFoundError(f"No calibration images found under {args.images_dir}")

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    used_images: list[str] = []
    skipped_images: list[str] = []

    for path in image_paths:
        result = collect_calibration_points(path, target_config)
        if result is None:
            skipped_images.append(path.name)
            continue
        objp, imgp, image_size = result
        obj_points.append(objp.reshape(-1, 1, 3))
        img_points.append(imgp.reshape(-1, 1, 2))
        used_images.append(path.name)

    if not obj_points:
        raise RuntimeError("Checkerboard corners were not detected in any image.")
    if image_size is None:
        raise RuntimeError("Failed to infer image size.")

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points,
        img_points,
        image_size,
        None,
        None,
    )
    mean_err = mean_reprojection_error(obj_points, img_points, rvecs, tvecs, camera_matrix, dist_coeffs)

    payload = {
        "convention": "camera_matrix maps camera-frame 3D points to pixels; dist_coeffs follow OpenCV calibrateCamera order.",
        "target_type": target_config["target_type"],
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
        "image_size": [image_size[0], image_size[1]],
        "square_size_m": target_config["square_size_m"],
        "num_input_images": len(image_paths),
        "num_used_images": len(used_images),
        "num_skipped_images": len(skipped_images),
        "used_images": used_images,
        "skipped_images": skipped_images,
        "rms_px": float(rms),
        "mean_reprojection_px": float(mean_err),
        "legacy": {
            "K": camera_matrix.tolist(),
            "D": dist_coeffs.reshape(-1).tolist(),
        },
    }
    if target_config["target_type"] == "checkerboard":
        payload["pattern_inner_corners"] = [target_config["pattern"][0], target_config["pattern"][1]]
    else:
        payload["charuco_squares"] = [target_config["charuco_squares_x"], target_config["charuco_squares_y"]]
        payload["marker_length_m"] = target_config["marker_length_m"]
        payload["aruco_dict_name"] = target_config["aruco_dict_name"]

    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved intrinsics to {args.output}")
    print(f"images used: {len(used_images)} / {len(image_paths)}")
    print(f"RMS reprojection error: {rms:.4f} px")
    print(f"Mean reprojection error: {mean_err:.4f} px")


if __name__ == "__main__":
    main()
