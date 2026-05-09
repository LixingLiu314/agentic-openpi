#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from aloha_fk_trajectory_utils import (
    ARM_SLICES,
    GRIPPER_DIMS,
    as_transform,
    episode_parquet_path,
    in_image_bounds,
    load_dataset_info,
    load_episode_lengths,
    load_extrinsics,
    load_fk_callable,
    load_intrinsics,
    parse_xyz,
    path_length,
    project_point,
    scale_uv_to_output,
    write_json,
)

THIS_DIR = Path(__file__).resolve().parent
TOOLS_ROOT = THIS_DIR.parent
DEFAULT_CALIBRATION_DATA_DIR = TOOLS_ROOT / "calibration" / "data"
DEFAULT_FK_MODULE = THIS_DIR / "mobile_aloha_link6_fk.py"


def parse_episode_selection(values: list[str] | None, available: list[int]) -> list[int]:
    if not values:
        return available

    selected: set[int] = set()
    for raw_value in values:
        for token in raw_value.replace(",", " ").split():
            if ":" in token:
                parts = token.split(":")
                if len(parts) != 2:
                    raise ValueError(f"Invalid episode range {token!r}; expected start:stop")
                start = int(parts[0]) if parts[0] else available[0]
                stop = int(parts[1]) if parts[1] else available[-1] + 1
                selected.update(range(start, stop))
            else:
                selected.add(int(token))
    available_set = set(available)
    missing = sorted(selected - available_set)
    if missing:
        raise ValueError(f"Requested episodes not found in dataset metadata: {missing[:10]}")
    return sorted(selected)


def rounded_pair(value: np.ndarray, digits: int) -> list[float]:
    return [round(float(value[0]), digits), round(float(value[1]), digits)]


def add_bool_argument(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    parser.add_argument(name, dest=name.lstrip("-").replace("-", "_"), action="store_true", help=help_text)
    parser.add_argument(
        f"--no-{name.lstrip('-')}",
        dest=name.lstrip("-").replace("-", "_"),
        action="store_false",
        help=f"Disable: {help_text}",
    )
    parser.set_defaults(**{name.lstrip("-").replace("-", "_"): default})


def project_episode(
    dataset_path: Path,
    episode_index: int,
    chunks_size: int,
    arms: list[str],
    primary_arm: str,
    fk_fn,
    T_base_to_camera_by_arm: dict[str, np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    raw_image_size: tuple[int, int],
    output_size: tuple[int, int],
    point_offset: np.ndarray,
    drop_invisible_for_ref: bool,
    digits: int,
    include_raw_points: bool,
    include_visibility: bool,
) -> dict:
    table = pq.read_table(
        episode_parquet_path(dataset_path, episode_index, chunks_size),
        columns=["observation.state", "frame_index"],
    )
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    frame_indices = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)

    positions_by_arm: dict[str, dict[str, list[float]]] = {arm: {} for arm in arms}
    raw_positions_by_arm: dict[str, dict[str, list[float]]] = {arm: {} for arm in arms}
    visibility_by_arm: dict[str, dict[str, bool]] = {arm: {} for arm in arms}
    z_camera_by_arm: dict[str, dict[str, float]] = {arm: {} for arm in arms}

    for row_index, state in enumerate(states):
        frame_key = str(int(frame_indices[row_index]))
        for arm in arms:
            joints = state[ARM_SLICES[arm]]
            T_base_to_tool = as_transform(fk_fn(joints))
            point_base = T_base_to_tool[:3, :3] @ point_offset + T_base_to_tool[:3, 3]
            uv_raw, point_cam = project_point(
                point_base,
                T_base_to_camera_by_arm[arm],
                camera_matrix,
                dist_coeffs,
            )
            uv_output = scale_uv_to_output(uv_raw, raw_image_size, output_size)
            visible = in_image_bounds(uv_raw, raw_image_size, float(point_cam[2]))

            positions_by_arm[arm][frame_key] = rounded_pair(uv_output, digits)
            if include_raw_points:
                raw_positions_by_arm[arm][frame_key] = rounded_pair(uv_raw, digits)
            if include_visibility:
                visibility_by_arm[arm][frame_key] = visible
                z_camera_by_arm[arm][frame_key] = round(float(point_cam[2]), digits)

    primary_positions = positions_by_arm[primary_arm]
    if drop_invisible_for_ref:
        primary_positions = {
            frame_key: value
            for frame_key, value in primary_positions.items()
            if visibility_by_arm.get(primary_arm, {}).get(frame_key, True)
        }

    result = {
        "episode_index": int(episode_index),
        "num_frames": int(len(states)),
        "primary_arm": primary_arm,
        "all_frame_eef_positions": primary_positions,
        "trajectory_stats": {
            arm: {
                "num_points": len(positions_by_arm[arm]),
                "num_visible": int(sum(visibility_by_arm.get(arm, {}).values())) if include_visibility else None,
                "path_length_output_px": round(path_length(list(positions_by_arm[arm].values())), digits),
            }
            for arm in arms
        },
    }

    if len(arms) > 1:
        result["all_frame_eef_positions_by_arm"] = positions_by_arm
    if include_raw_points:
        if len(arms) == 1:
            result["all_frame_eef_positions_raw"] = raw_positions_by_arm[primary_arm]
        else:
            result["all_frame_eef_positions_raw_by_arm"] = raw_positions_by_arm
    if include_visibility:
        if len(arms) == 1:
            result["visibility"] = visibility_by_arm[primary_arm]
            result["z_camera_m"] = z_camera_by_arm[primary_arm]
        else:
            result["visibility_by_arm"] = visibility_by_arm
            result["z_camera_m_by_arm"] = z_camera_by_arm

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate ref.py-compatible EEF trajectory JSON for Aloha Piper from FK + calibrated front camera."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("playground/Datasets/aloha_banana_lerobot_absolute"),
        help="LeRobot dataset root.",
    )
    parser.add_argument(
        "--tools-dir",
        type=Path,
        default=DEFAULT_CALIBRATION_DATA_DIR,
        help="Directory containing calibration JSON files. Kept as a legacy option; default: tools/calibration/data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Default writes under dataset_path/trajectory_data.",
    )
    parser.add_argument(
        "--episodes",
        nargs="*",
        default=None,
        help="Episode ids or Python-style ranges, e.g. 0 5 10:20. Default: all episodes.",
    )
    parser.add_argument("--primary-arm", choices=["left", "right"], default="right")
    parser.add_argument(
        "--include-arms",
        choices=["primary", "both"],
        default="primary",
        help="Store only the primary arm or both arms. ref.py reads the primary arm field either way.",
    )
    parser.add_argument(
        "--intrinsics",
        type=Path,
        default=None,
        help="Front-camera intrinsics JSON. Default: tools/calibration/data/intrinsics_front_charuco.json.",
    )
    parser.add_argument(
        "--left-extrinsics",
        type=Path,
        default=None,
        help="Left arm base -> front camera JSON. Default: tools/calibration/data/front_in_left_base_from_left_charuco_final.json.",
    )
    parser.add_argument(
        "--right-extrinsics",
        type=Path,
        default=None,
        help="Right arm base -> front camera JSON. Default: tools/calibration/data/front_in_right_base_from_left_charuco_final.json.",
    )
    parser.add_argument(
        "--fk-module",
        type=Path,
        default=None,
        help="FK module path. Default: tools/trajectory/mobile_aloha_link6_fk.py.",
    )
    parser.add_argument(
        "--fk-function",
        default="fk_gripper_center",
        help="FK function mapping 6 arm joints to a base->tool transform.",
    )
    parser.add_argument(
        "--point-offset",
        default="0,0,0",
        help="Point offset in the FK output frame, meters, formatted as x,y,z.",
    )
    parser.add_argument(
        "--output-image-size",
        nargs=2,
        type=int,
        default=[256, 256],
        metavar=("WIDTH", "HEIGHT"),
        help="Coordinate size stored in all_frame_eef_positions. Use ref.py --image_size WIDTH when square.",
    )
    parser.add_argument(
        "--raw-image-size",
        nargs=2,
        type=int,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Raw front video size. Default comes from the intrinsics JSON image_size.",
    )
    parser.add_argument(
        "--drop-invisible-for-ref",
        action="store_true",
        help="Omit primary-arm points outside the raw image from all_frame_eef_positions.",
    )
    parser.add_argument("--digits", type=int, default=3, help="Decimal digits kept in JSON coordinates.")
    add_bool_argument(
        parser,
        "--include-raw-points",
        default=True,
        help_text="Store raw 640x480 pixel coordinates for visualization.",
    )
    add_bool_argument(
        parser,
        "--include-visibility",
        default=True,
        help_text="Store per-frame visibility and camera z diagnostics.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    dataset_path = args.dataset_path.expanduser()
    tools_dir = args.tools_dir.expanduser()
    intrinsics_path = (args.intrinsics or tools_dir / "intrinsics_front_charuco.json").expanduser()
    left_extrinsics_path = (
        args.left_extrinsics or tools_dir / "front_in_left_base_from_left_charuco_final.json"
    ).expanduser()
    right_extrinsics_path = (
        args.right_extrinsics or tools_dir / "front_in_right_base_from_left_charuco_final.json"
    ).expanduser()
    fk_module_path = (args.fk_module or DEFAULT_FK_MODULE).expanduser()

    info = load_dataset_info(dataset_path)
    chunks_size = int(info.get("chunks_size", 1000))
    episode_lengths = load_episode_lengths(dataset_path)
    episodes = parse_episode_selection(args.episodes, sorted(episode_lengths))

    camera_matrix, dist_coeffs, intrinsics_image_size = load_intrinsics(intrinsics_path)
    raw_image_size = tuple(args.raw_image_size) if args.raw_image_size else intrinsics_image_size
    if raw_image_size is None:
        raise ValueError("Raw image size was not supplied and was not present in intrinsics JSON.")
    output_size = tuple(args.output_image_size)
    if output_size[0] <= 0 or output_size[1] <= 0:
        raise ValueError(f"Invalid output image size: {output_size}")

    arms = [args.primary_arm] if args.include_arms == "primary" else ["left", "right"]
    if args.primary_arm not in arms:
        arms.append(args.primary_arm)

    T_base_to_camera_by_arm = {}
    if "left" in arms:
        T_base_to_camera_by_arm["left"] = load_extrinsics(left_extrinsics_path)
    if "right" in arms:
        T_base_to_camera_by_arm["right"] = load_extrinsics(right_extrinsics_path)

    fk_fn = load_fk_callable(fk_module_path, args.fk_function)
    point_offset = parse_xyz(args.point_offset)

    if args.output is None:
        suffix = args.primary_arm if args.include_arms == "primary" else f"{args.primary_arm}_with_both_arms"
        args.output = dataset_path / "trajectory_data" / f"tapir_pnp_trajectory_fk_calibrated_{suffix}.json"
    output_path = args.output.expanduser()

    results = []
    failed_episodes = []
    for episode_index in tqdm(episodes, desc="Projecting FK trajectories"):
        try:
            result = project_episode(
                dataset_path=dataset_path,
                episode_index=episode_index,
                chunks_size=chunks_size,
                arms=arms,
                primary_arm=args.primary_arm,
                fk_fn=fk_fn,
                T_base_to_camera_by_arm=T_base_to_camera_by_arm,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                raw_image_size=raw_image_size,
                output_size=output_size,
                point_offset=point_offset,
                drop_invisible_for_ref=args.drop_invisible_for_ref,
                digits=args.digits,
                include_raw_points=args.include_raw_points,
                include_visibility=args.include_visibility,
            )
            results.append(result)
        except Exception as exc:
            failed_episodes.append({"episode_index": int(episode_index), "error": str(exc)})
            print(f"Failed episode {episode_index}: {exc}")

    gripper_dim = GRIPPER_DIMS[args.primary_arm]
    payload = {
        "metadata": {
            "method": "fk_calibrated_front_camera_projection",
            "ref_py_compatible": True,
            "dataset_path": str(dataset_path),
            "num_requested_episodes": len(episodes),
            "num_successful_episodes": len(results),
            "num_failed_episodes": len(failed_episodes),
            "primary_arm": args.primary_arm,
            "included_arms": arms,
            "state_slices": {"left": [0, 6], "right": [7, 13]},
            "gripper_dims": GRIPPER_DIMS,
            "primary_gripper_dim": gripper_dim,
            "fk_module": str(fk_module_path),
            "fk_function": args.fk_function,
            "point_offset_ee_m": point_offset.tolist(),
            "intrinsics": str(intrinsics_path),
            "left_extrinsics": str(left_extrinsics_path),
            "right_extrinsics": str(right_extrinsics_path),
            "raw_image_size": list(raw_image_size),
            "output_image_size": list(output_size),
            "all_frame_eef_positions_coordinate_space": "directly resized front camera pixels",
            "coordinate_note": (
                "Raw 640x480 front-camera pixels are scaled independently in x/y into output_image_size. "
                "This matches the dataset loader's direct resize-to-square behavior."
            ),
            "drop_invisible_for_ref": bool(args.drop_invisible_for_ref),
            "recommended_ref_py_args": {
                "image_size": output_size[0] if output_size[0] == output_size[1] else None,
                "gripper_dim": gripper_dim,
            },
        },
        "failed_episodes": failed_episodes,
        "results": results,
    }

    write_json(output_path, payload)
    print(f"Saved {len(results)} episodes to {output_path}")
    if failed_episodes:
        print(f"Failed episodes: {len(failed_episodes)}")
    if output_size[0] == output_size[1]:
        print(
            "ref.py example: "
            f"python ref.py --dataset_path {dataset_path} --traj_json {output_path} "
            f"--image_size {output_size[0]} --gripper_dim {gripper_dim}"
        )


if __name__ == "__main__":
    main()
