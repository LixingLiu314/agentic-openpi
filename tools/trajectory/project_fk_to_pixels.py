#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


LEFT_ARM = slice(0, 6)
RIGHT_ARM = slice(7, 13)
DEFAULT_FK_MODULE = Path(__file__).resolve().with_name("mobile_aloha_link6_fk.py")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = load_json(path)
    camera_matrix = np.asarray(data.get("camera_matrix", data.get("K")), dtype=np.float64)
    dist_coeffs = np.asarray(data.get("dist_coeffs", data.get("D", [])), dtype=np.float64).reshape(-1)
    if camera_matrix.shape != (3, 3):
        raise ValueError(f"Invalid camera matrix shape from {path}: {camera_matrix.shape}")
    return camera_matrix, dist_coeffs


def load_extrinsics(path: Path) -> np.ndarray:
    data = load_json(path)
    if "T_base_to_camera" in data:
        T = np.asarray(data["T_base_to_camera"], dtype=np.float64)
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
        T_front_in_base = np.asarray(data["T_front_in_base"], dtype=np.float64)
        R = T_front_in_base[:3, :3]
        t = T_front_in_base[:3, 3]
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R.T
        T[:3, 3] = -R.T @ t
    else:
        raise KeyError(f"Could not find base->camera transform fields in {path}")
    if T.shape != (4, 4):
        raise ValueError(f"Invalid transform shape from {path}: {T.shape}")
    return T


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


def parse_xyz(text: str) -> np.ndarray:
    parts = [float(x.strip()) for x in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected x,y,z")
    return np.asarray(parts, dtype=np.float64)


def episode_parquet_path(dataset_root: Path, episode_index: int) -> Path:
    chunk_index = episode_index // 1000
    return dataset_root / "data" / f"chunk-{chunk_index:03d}" / f"episode_{episode_index:06d}.parquet"


def load_states(dataset_root: Path, episode_index: int) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(
        episode_parquet_path(dataset_root, episode_index),
        columns=["observation.state", "frame_index"],
    )
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    frame_indices = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    return states, frame_indices


def distort_normalized_points(x: float, y: float, dist_coeffs: np.ndarray) -> tuple[float, float]:
    coeffs = np.zeros(14, dtype=np.float64)
    coeffs[: min(len(dist_coeffs), len(coeffs))] = dist_coeffs[: min(len(dist_coeffs), len(coeffs))]
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

    x_dist = x * radial + x_tan + x_prism
    y_dist = y * radial + y_tan + y_prism
    return x_dist, y_dist


def project_point(
    point_base: np.ndarray,
    T_base_to_camera: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    point_cam = T_base_to_camera[:3, :3] @ point_base + T_base_to_camera[:3, 3]
    if point_cam[2] == 0.0:
        raise ZeroDivisionError("Projected point lies on the camera plane (z == 0).")

    x = point_cam[0] / point_cam[2]
    y = point_cam[1] / point_cam[2]
    x_dist, y_dist = distort_normalized_points(float(x), float(y), dist_coeffs)

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    skew = camera_matrix[0, 1]

    u = fx * x_dist + skew * y_dist + cx
    v = fy * y_dist + cy
    return np.asarray([u, v], dtype=np.float64), point_cam


def main() -> None:
    parser = argparse.ArgumentParser(description="Project FK points from Aloha dataset state into front-camera pixels.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--extrinsics", type=Path, required=True, help="Must contain T_base_to_camera.")
    parser.add_argument(
        "--fk-module",
        type=Path,
        default=DEFAULT_FK_MODULE,
        help="Python file that defines the FK function.",
    )
    parser.add_argument("--fk-function", default="fk_gripper_center", help="Function that maps one 6D joint vector to base->gripper transform.")
    parser.add_argument("--arm", choices=["left", "right"], default="left")
    parser.add_argument("--point-offset", type=parse_xyz, default=np.zeros(3), help="Point offset in EE frame, format x,y,z meters.")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stop", type=int, default=-1, help="-1 means until episode end.")
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("fk_pixels_episode.json"))
    args = parser.parse_args()
    args.dataset_root = args.dataset_root.expanduser()
    args.intrinsics = args.intrinsics.expanduser()
    args.extrinsics = args.extrinsics.expanduser()
    args.fk_module = args.fk_module.expanduser()
    args.output = args.output.expanduser()

    fk_fn = load_fk_callable(args.fk_module, args.fk_function)
    camera_matrix, dist_coeffs = load_intrinsics(args.intrinsics)
    T_base_to_camera = load_extrinsics(args.extrinsics)

    states, frame_indices = load_states(args.dataset_root, args.episode)
    if states.ndim != 2 or states.shape[1] != 14:
        raise ValueError(f"Expected states of shape (T, 14), got {states.shape}")

    frame_stop = len(states) if args.frame_stop < 0 else min(args.frame_stop, len(states))
    arm_slice = LEFT_ARM if args.arm == "left" else RIGHT_ARM

    results = []
    for row_idx in range(args.frame_start, frame_stop, args.frame_step):
        state = states[row_idx]
        joints = state[arm_slice]
        T_base_to_gripper = as_transform(fk_fn(joints))

        point_base = T_base_to_gripper[:3, :3] @ args.point_offset + T_base_to_gripper[:3, 3]
        uv, point_cam = project_point(point_base, T_base_to_camera, camera_matrix, dist_coeffs)
        u, v = uv
        results.append(
            {
                "row_index": int(row_idx),
                "frame_index": int(frame_indices[row_idx]),
                "arm": args.arm,
                "joints_rad": joints.tolist(),
                "point_base_m": point_base.tolist(),
                "point_camera_m": point_cam.tolist(),
                "pixel_uv": [float(u), float(v)],
                "z_camera_m": float(point_cam[2]),
                "visible": bool(point_cam[2] > 0.0),
            }
        )

    payload = {
        "convention": "T_base_to_camera maps base-frame 3D points into camera frame; pixel_uv uses OpenCV u,v.",
        "dataset_root": str(args.dataset_root),
        "episode": args.episode,
        "arm": args.arm,
        "point_offset_ee_m": args.point_offset.tolist(),
        "num_rows": len(results),
        "rows": results,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {len(results)} projected points to {args.output}")


if __name__ == "__main__":
    main()
