#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from calib_utils import build_charuco_board, collect_calibration_points


DICT_CANDIDATES = [
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_4X4_250",
    "DICT_4X4_1000",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Guess the correct ChArUco 4x4 dictionary from sample images.")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--charuco-squares-x", type=int, default=5)
    parser.add_argument("--charuco-squares-y", type=int, default=4)
    parser.add_argument("--square-size", type=float, required=True)
    parser.add_argument("--marker-length", type=float, required=True)
    parser.add_argument("--extensions", nargs="+", default=[".png", ".jpg", ".jpeg", ".bmp"])
    args = parser.parse_args()

    args.images_dir = args.images_dir.expanduser()
    image_paths = sorted(
        p for p in args.images_dir.iterdir() if p.is_file() and p.suffix.lower() in set(args.extensions)
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found under {args.images_dir}")

    print(f"images: {len(image_paths)}")
    best = None
    for dict_name in DICT_CANDIDATES:
        conf = {
            "target_type": "charuco",
            "charuco_squares_x": args.charuco_squares_x,
            "charuco_squares_y": args.charuco_squares_y,
            "square_size_m": args.square_size,
            "marker_length_m": args.marker_length,
            "aruco_dict_name": dict_name,
        }
        detected = 0
        points = []
        for path in image_paths:
            res = collect_calibration_points(path, conf)
            if res is None:
                continue
            objp, _, _ = res
            detected += 1
            points.append(len(objp))
        avg_points = float(np.mean(points)) if points else 0.0
        print(f"{dict_name}: detected {detected}/{len(image_paths)}  avg_points={avg_points:.1f}")
        score = (detected, avg_points)
        if best is None or score > best[0]:
            best = (score, dict_name)

    print(f"best_guess: {best[1]}")


if __name__ == "__main__":
    main()
