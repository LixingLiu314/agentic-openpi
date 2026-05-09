#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_pattern(text: str) -> tuple[int, int]:
    parts = text.lower().replace(",", "x").split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Invalid pattern '{text}', expected COLSxROWS such as 8x6.")
    cols, rows = (int(p.strip()) for p in parts)
    if cols <= 0 or rows <= 0:
        raise argparse.ArgumentTypeError("Pattern dimensions must be positive.")
    return cols, rows


def parse_target_type(text: str) -> str:
    text = text.strip().lower()
    if text not in {"checkerboard", "charuco"}:
        raise argparse.ArgumentTypeError("target type must be 'checkerboard' or 'charuco'")
    return text


def parse_index_list(values: list[str] | None) -> list[int]:
    if not values:
        return []
    result: set[int] = set()
    for raw in values:
        for part in raw.replace(",", " ").split():
            idx = int(part)
            if idx < 0:
                raise ValueError(f"Sample index must be non-negative, got {idx}")
            result.add(idx)
    return sorted(result)


def aruco_dict_from_name(name: str) -> Any:
    key = name.strip()
    if not key.startswith("DICT_"):
        key = f"DICT_{key}"
    if not hasattr(cv2.aruco, key):
        raise ValueError(f"Unknown ArUco dictionary '{name}'")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, key))


def add_target_arguments(parser: argparse.ArgumentParser, default_pattern: tuple[int, int] = (8, 6), default_square: float = 0.024) -> None:
    parser.add_argument("--target-type", type=parse_target_type, default="checkerboard")
    parser.add_argument("--pattern", type=parse_pattern, default=default_pattern, help="Checkerboard inner corners, e.g. 8x6.")
    parser.add_argument("--square-size", type=float, default=default_square, help="Checkerboard square size in meters, or ChArUco square length.")
    parser.add_argument("--charuco-squares-x", type=int, default=5, help="Number of ChArUco squares along X.")
    parser.add_argument("--charuco-squares-y", type=int, default=4, help="Number of ChArUco squares along Y.")
    parser.add_argument("--marker-length", type=float, default=None, help="ChArUco ArUco marker side length in meters.")
    parser.add_argument("--aruco-dict", default="DICT_4X4_50", help="OpenCV ArUco dictionary name, e.g. DICT_4X4_50.")


def build_target_config_from_args(args) -> dict:
    if args.target_type == "checkerboard":
        return {
            "target_type": "checkerboard",
            "pattern": tuple(args.pattern),
            "square_size_m": float(args.square_size),
        }

    marker_length = args.marker_length
    if marker_length is None:
        raise ValueError("--marker-length is required when --target-type charuco")
    if args.charuco_squares_x < 2 or args.charuco_squares_y < 2:
        raise ValueError("ChArUco squares must be at least 2x2.")
    if marker_length >= args.square_size:
        raise ValueError("For ChArUco, marker length must be smaller than square size.")
    return {
        "target_type": "charuco",
        "charuco_squares_x": int(args.charuco_squares_x),
        "charuco_squares_y": int(args.charuco_squares_y),
        "square_size_m": float(args.square_size),
        "marker_length_m": float(marker_length),
        "aruco_dict_name": args.aruco_dict,
    }


def load_json(path: Path) -> dict:
    path = path.expanduser()
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    path = path.expanduser()
    data = load_json(path)
    camera_matrix = np.asarray(data.get("camera_matrix", data.get("K")), dtype=np.float64)
    dist_coeffs = np.asarray(data.get("dist_coeffs", data.get("D", [])), dtype=np.float64).reshape(-1, 1)
    if camera_matrix.shape != (3, 3):
        raise ValueError(f"Invalid camera matrix shape from {path}: {camera_matrix.shape}")
    return camera_matrix, dist_coeffs


def build_object_points(pattern: tuple[int, int], square_size_m: float) -> np.ndarray:
    cols, rows = pattern
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_m
    return objp


def build_charuco_board(target_config: dict):
    dictionary = aruco_dict_from_name(target_config["aruco_dict_name"])
    return cv2.aruco.CharucoBoard(
        (target_config["charuco_squares_x"], target_config["charuco_squares_y"]),
        target_config["square_size_m"],
        target_config["marker_length_m"],
        dictionary,
    )


def detect_corners(gray: np.ndarray, pattern: tuple[int, int]) -> np.ndarray | None:
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not ok:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners


def detect_charuco(image: np.ndarray, target_config: dict) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    board = build_charuco_board(target_config)
    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(image)
    if charuco_ids is None or charuco_corners is None:
        return None, None
    if len(charuco_ids) < 4:
        return None, None
    try:
        collinear = board.checkCharucoCornersCollinear(charuco_ids)
    except Exception:
        collinear = False
    if collinear:
        return None, None
    return charuco_corners, charuco_ids


def collect_calibration_points(image_path: Path, target_config: dict) -> tuple[np.ndarray, np.ndarray, tuple[int, int]] | None:
    image_path = image_path.expanduser()
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_size = (gray.shape[1], gray.shape[0])

    if target_config["target_type"] == "checkerboard":
        corners = detect_corners(gray, target_config["pattern"])
        if corners is None:
            return None
        objp = build_object_points(target_config["pattern"], target_config["square_size_m"])
        return objp.reshape(-1, 3).astype(np.float32), corners.reshape(-1, 2).astype(np.float32), image_size

    charuco_corners, charuco_ids = detect_charuco(img, target_config)
    if charuco_ids is None:
        return None
    board = build_charuco_board(target_config)
    obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
    if obj_points is None or img_points is None:
        return None
    obj_points = np.asarray(obj_points, dtype=np.float32).reshape(-1, 3)
    img_points = np.asarray(img_points, dtype=np.float32).reshape(-1, 2)
    if len(obj_points) < 4:
        return None
    return obj_points, img_points, image_size


def solve_target_pose(image_path: Path, target_config: dict, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> dict | None:
    image_path = image_path.expanduser()
    result = collect_calibration_points(image_path, target_config)
    if result is None:
        return None
    objp, imgp, _ = result
    ok, rvec, tvec = cv2.solvePnP(objp, imgp, camera_matrix, dist_coeffs)
    if not ok:
        return None

    R, _ = cv2.Rodrigues(rvec)
    reproj, _ = cv2.projectPoints(objp.reshape(-1, 1, 3), rvec, tvec, camera_matrix, dist_coeffs)
    err = np.linalg.norm(imgp.reshape(-1, 2) - reproj.reshape(-1, 2), axis=1)
    T_target_in_cam = np.eye(4, dtype=np.float64)
    T_target_in_cam[:3, :3] = R
    T_target_in_cam[:3, 3] = tvec.reshape(3)
    return {
        "T_target_in_cam": T_target_in_cam,
        "reprojection_px": float(err.mean()),
        "num_points": int(len(objp)),
    }


def load_fk_callable(module_path: Path, function_name: str):
    module_path = module_path.expanduser()
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


def rotation_mean(rotations: list[np.ndarray]) -> np.ndarray:
    if not rotations:
        raise ValueError("No rotations to average.")
    M = np.zeros((3, 3), dtype=np.float64)
    for R in rotations:
        M += R
    U, _, Vt = np.linalg.svd(M)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1.0
        R_avg = U @ Vt
    return R_avg


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R_rel = R_a.T @ R_b
    trace = np.clip((np.trace(R_rel) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))
