#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


LEFT_ARM_SLICE = slice(0, 6)
RIGHT_ARM_SLICE = slice(7, 13)
ARM_SLICES = {
    "left": LEFT_ARM_SLICE,
    "right": RIGHT_ARM_SLICE,
}
GRIPPER_DIMS = {
    "left": 6,
    "right": 13,
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int] | None]:
    data = load_json(path)
    camera_matrix = np.asarray(data.get("camera_matrix", data.get("K")), dtype=np.float64)
    dist_coeffs = np.asarray(data.get("dist_coeffs", data.get("D", [])), dtype=np.float64).reshape(-1)
    image_size = data.get("image_size")
    if camera_matrix.shape != (3, 3):
        raise ValueError(f"Invalid camera matrix shape from {path}: {camera_matrix.shape}")
    if image_size is not None:
        image_size = (int(image_size[0]), int(image_size[1]))
    return camera_matrix, dist_coeffs, image_size


def invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def load_extrinsics(path: Path) -> np.ndarray:
    data = load_json(path)
    if "T_base_to_camera" in data:
        T = np.asarray(data["T_base_to_camera"], dtype=np.float64)
    elif "T_body_to_camera" in data:
        T = np.asarray(data["T_body_to_camera"], dtype=np.float64)
    elif "T_base_in_front" in data:
        T = np.asarray(data["T_base_in_front"], dtype=np.float64)
    elif "R_base_to_camera" in data and "t_base_to_camera" in data:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.asarray(data["R_base_to_camera"], dtype=np.float64)
        T[:3, 3] = np.asarray(data["t_base_to_camera"], dtype=np.float64).reshape(3)
    elif "R_cam_base" in data and "t_cam_base" in data:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.asarray(data["R_cam_base"], dtype=np.float64)
        T[:3, 3] = np.asarray(data["t_cam_base"], dtype=np.float64).reshape(3)
    elif "T_front_in_base" in data:
        T = invert_transform(np.asarray(data["T_front_in_base"], dtype=np.float64))
    elif "T_camera_to_base" in data:
        T = invert_transform(np.asarray(data["T_camera_to_base"], dtype=np.float64))
    else:
        raise KeyError(f"Could not find a base/body -> camera transform in {path}")
    if T.shape != (4, 4):
        raise ValueError(f"Invalid transform shape from {path}: {T.shape}")
    return T


def load_fk_callable(module_path: Path, function_name: str):
    module_path = module_path.expanduser()
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return getattr(module, function_name)
    except AttributeError as exc:
        raise AttributeError(f"{module_path} does not define '{function_name}'") from exc


def as_transform(value) -> np.ndarray:
    if isinstance(value, tuple) and len(value) == 2:
        rot, trans = value
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.asarray(rot, dtype=np.float64)
        T[:3, 3] = np.asarray(trans, dtype=np.float64).reshape(3)
        return T

    T = np.asarray(value, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"FK must return a 4x4 transform or (R, t), got shape {T.shape}")
    return T


def parse_xyz(text: str) -> np.ndarray:
    parts = [float(part.strip()) for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected xyz as 'x,y,z', got {text!r}")
    return np.asarray(parts, dtype=np.float64)


def episode_parquet_path(dataset_path: Path, episode_index: int, chunks_size: int) -> Path:
    chunk_index = episode_index // chunks_size
    return dataset_path / "data" / f"chunk-{chunk_index:03d}" / f"episode_{episode_index:06d}.parquet"


def episode_video_path(dataset_path: Path, episode_index: int, chunks_size: int, video_key: str) -> Path:
    chunk_index = episode_index // chunks_size
    return dataset_path / "videos" / f"chunk-{chunk_index:03d}" / video_key / f"episode_{episode_index:06d}.mp4"


def load_dataset_info(dataset_path: Path) -> dict[str, Any]:
    return load_json(dataset_path / "meta" / "info.json")


def load_episode_lengths(dataset_path: Path) -> dict[int, int]:
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    lengths: dict[int, int] = {}
    with episodes_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            lengths[int(item["episode_index"])] = int(item["length"])
    return lengths


def distort_normalized_points(x: float, y: float, dist_coeffs: np.ndarray) -> tuple[float, float]:
    coeffs = np.zeros(14, dtype=np.float64)
    n = min(len(dist_coeffs), len(coeffs))
    coeffs[:n] = dist_coeffs[:n]
    k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, _, _ = coeffs

    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    radial_num = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    radial_den = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
    radial = radial_num / radial_den if radial_den != 0.0 else radial_num

    x_tan = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_tan = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    x_prism = s1 * r2 + s2 * r4
    y_prism = s3 * r2 + s4 * r4
    return x * radial + x_tan + x_prism, y * radial + y_tan + y_prism


def project_point(
    point_base: np.ndarray,
    T_base_to_camera: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    point_cam = T_base_to_camera[:3, :3] @ point_base + T_base_to_camera[:3, 3]
    if point_cam[2] == 0.0:
        raise ZeroDivisionError("Projected point lies on the camera plane (z == 0).")

    x = float(point_cam[0] / point_cam[2])
    y = float(point_cam[1] / point_cam[2])
    x_dist, y_dist = distort_normalized_points(x, y, dist_coeffs)

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    skew = camera_matrix[0, 1]
    uv = np.asarray([fx * x_dist + skew * y_dist + cx, fy * y_dist + cy], dtype=np.float64)
    return uv, point_cam


def scale_uv_to_output(uv: np.ndarray, raw_image_size: tuple[int, int], output_size: tuple[int, int]) -> np.ndarray:
    raw_w, raw_h = raw_image_size
    out_w, out_h = output_size
    return np.asarray([uv[0] * out_w / raw_w, uv[1] * out_h / raw_h], dtype=np.float64)


def scale_uv_to_raw(uv: np.ndarray, raw_image_size: tuple[int, int], output_size: tuple[int, int]) -> np.ndarray:
    raw_w, raw_h = raw_image_size
    out_w, out_h = output_size
    return np.asarray([uv[0] * raw_w / out_w, uv[1] * raw_h / out_h], dtype=np.float64)


def in_image_bounds(uv: np.ndarray, raw_image_size: tuple[int, int], z_camera: float | None = None) -> bool:
    raw_w, raw_h = raw_image_size
    if z_camera is not None and z_camera <= 0.0:
        return False
    return bool(0.0 <= uv[0] < raw_w and 0.0 <= uv[1] < raw_h)


def path_length(points: list[list[float]]) -> float:
    if len(points) < 2:
        return 0.0
    pts = np.asarray(points, dtype=np.float64)
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
