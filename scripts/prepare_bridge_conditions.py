#!/usr/bin/env python3
"""Prepare filtered Bridge condition data for conditional VLA training.

This script writes a new LeRobot-style dataset instead of editing the input
dataset in place. It adds chunk-start-aligned ``subtask`` and ``traj_cot``
columns, keeps only episodes with complete data for both columns, registers
Bridge subgoal video streams in metadata, symlinks regular videos, and
materializes chunk-start-aligned subgoal videos.
"""

from __future__ import annotations

import argparse
import copy
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is available in the repo env.
    tqdm = None


DEFAULT_SUBTASK_JSON_NAME = "cot_by_episode.json"
DEFAULT_SUBGOAL_VIDEO_KEYS = ("image_0_subgoal_gt", "image_0_subgoal_foreact")


def iter_progress(items, *, desc: str):
    if tqdm is None:
        return items
    return tqdm(items, desc=desc)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def episode_path(dataset_dir: Path, episode_index: int, chunks_size: int) -> Path:
    chunk = episode_index // chunks_size
    return dataset_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def video_path(dataset_dir: Path, episode_index: int, chunks_size: int, video_key: str) -> Path:
    chunk = episode_index // chunks_size
    return dataset_dir / "videos" / f"chunk-{chunk:03d}" / video_key / f"episode_{episode_index:06d}.mp4"


def is_valid_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def perpendicular_distance(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
    if np.allclose(line_start, line_end):
        return float(np.linalg.norm(point - line_start))
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_len = np.linalg.norm(line_vec)
    line_unit = line_vec / line_len
    proj_length = np.dot(point_vec, line_unit)
    if proj_length < 0:
        return float(np.linalg.norm(point_vec))
    if proj_length > line_len:
        return float(np.linalg.norm(point - line_end))
    proj = line_start + proj_length * line_unit
    return float(np.linalg.norm(point - proj))


def rdp_simplify(points: list[tuple[float, float]], epsilon: float) -> list[int]:
    """Ramer-Douglas-Peucker simplification, matching the ALOHA CoT script."""
    if len(points) <= 2:
        return list(range(len(points)))

    pts = np.asarray(points, dtype=np.float64)
    keep = {0, len(points) - 1}
    stack = [(0, len(points) - 1)]
    while stack:
        start, end = stack.pop()
        if end - start <= 1:
            continue
        max_dist = 0.0
        max_idx = start
        for idx in range(start + 1, end):
            dist = perpendicular_distance(pts[idx], pts[start], pts[end])
            if dist > max_dist:
                max_dist = dist
                max_idx = idx
        if max_dist > epsilon:
            keep.add(max_idx)
            stack.append((start, max_idx))
            stack.append((max_idx, end))
    return sorted(keep)


def detect_gripper_changes(gripper_values: np.ndarray, threshold: float) -> list[tuple[int, str]]:
    """Detect Bridge gripper transitions. Bridge uses 1.0=open and 0.0=closed."""
    if len(gripper_values) == 0:
        return []
    changes = []
    prev = "open" if gripper_values[0] >= threshold else "closed"
    for idx in range(1, len(gripper_values)):
        curr = "open" if gripper_values[idx] >= threshold else "closed"
        if curr != prev:
            changes.append((idx, "close gripper" if curr == "closed" else "open gripper"))
            prev = curr
    return changes


def get_chunk_start_frame(current_frame: int, chunk_size: int) -> int:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    return current_frame - (current_frame % chunk_size)


def normalize_xy(points_xy: np.ndarray, bounds: tuple[float, float, float, float]) -> np.ndarray:
    x_min, x_max, y_min, y_max = bounds
    x_den = max(x_max - x_min, 1e-6)
    y_den = max(y_max - y_min, 1e-6)
    out = np.empty_like(points_xy, dtype=np.float64)
    out[:, 0] = np.clip((points_xy[:, 0] - x_min) / x_den * 1000.0, 0.0, 1000.0)
    out[:, 1] = np.clip((points_xy[:, 1] - y_min) / y_den * 1000.0, 0.0, 1000.0)
    return out


def build_cot_prompt(
    future_points: list[tuple[int, float, float]],
    gripper_changes: list[tuple[int, str]],
    *,
    rdp_epsilon: float,
    min_dist_threshold: float,
    max_tokens: int,
) -> str:
    """Build ALOHA-style per-frame trajectory text from normalized Bridge XY."""
    if not future_points:
        return ""

    frame_to_point = {frame: (x, y) for frame, x, y in future_points}
    frames = [frame for frame, _, _ in future_points]
    start_frame = frames[0]
    end_frame = frames[-1]
    valid_gripper_changes = {
        frame: action for frame, action in gripper_changes if start_frame <= frame <= end_frame
    }
    anchor_frames = sorted(set([start_frame, end_frame] + list(valid_gripper_changes)))

    kept_frames: set[int] = set()
    if len(anchor_frames) == 1:
        kept_frames.add(anchor_frames[0])
    for anchor_idx in range(len(anchor_frames) - 1):
        seg_start = anchor_frames[anchor_idx]
        seg_end = anchor_frames[anchor_idx + 1]
        seg_frames = [frame for frame in frames if seg_start <= frame <= seg_end]
        if not seg_frames:
            continue
        seg_points = [frame_to_point[frame] for frame in seg_frames]
        for keep_idx in rdp_simplify(seg_points, rdp_epsilon):
            kept_frames.add(seg_frames[keep_idx])

    tokens: list[str] = []
    last_point = None
    for frame in sorted(kept_frames):
        x, y = frame_to_point[frame]
        is_gripper_change = frame in valid_gripper_changes

        skip_coord = False
        if last_point is not None:
            dist = np.linalg.norm(np.asarray([x, y]) - np.asarray(last_point))
            skip_coord = dist < min_dist_threshold

        if not skip_coord:
            tokens.append(f"({int(round(x))}, {int(round(y))})")
            last_point = (x, y)

        if is_gripper_change:
            tokens.append(valid_gripper_changes[frame])

    tokens = tokens[:max_tokens]
    if not tokens:
        return ""

    parts: list[str] = []
    has_started_moving = False
    for idx, token in enumerate(tokens):
        is_coord = token.startswith("(")
        if is_coord:
            if not has_started_moving:
                prefix = "Go along " if idx == 0 else "go along "
                parts.append(prefix + token)
                has_started_moving = True
            else:
                parts.append(token)
        else:
            parts.append(token.capitalize() if idx == 0 else token)
    return ", ".join(parts) + "."


def generate_traj_cot(
    table: pa.Table,
    *,
    coord_source: str,
    state_xy_dims: tuple[int, int],
    action_xy_dims: tuple[int, int],
    gripper_dim: int,
    gripper_threshold: float,
    coord_bounds: tuple[float, float, float, float],
    rdp_epsilon: float,
    min_dist_threshold: float,
    max_tokens: int,
) -> dict[str, str]:
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)

    if coord_source == "state":
        xy = state[:, list(state_xy_dims)]
    else:
        xy = actions[:, list(action_xy_dims)]
    xy_norm = normalize_xy(xy, coord_bounds)

    gripper_values = actions[:, gripper_dim]
    gripper_changes = detect_gripper_changes(gripper_values, gripper_threshold)

    num_frames = len(xy_norm)
    frame_prompts = {}
    for start_frame in range(num_frames):
        future_points = [
            (frame, float(xy_norm[frame, 0]), float(xy_norm[frame, 1]))
            for frame in range(start_frame, num_frames)
        ]
        future_gripper_changes = [(frame, action) for frame, action in gripper_changes if frame >= start_frame]
        frame_prompts[str(start_frame)] = build_cot_prompt(
            future_points,
            future_gripper_changes,
            rdp_epsilon=rdp_epsilon,
            min_dist_threshold=min_dist_threshold,
            max_tokens=max_tokens,
        )
    return frame_prompts


def load_coord_bounds(
    dataset_dir: Path,
    *,
    coord_source: str,
    state_xy_dims: tuple[int, int],
    action_xy_dims: tuple[int, int],
    stats_file: str,
) -> tuple[float, float, float, float]:
    stats_path = dataset_dir / "meta" / stats_file
    if not stats_path.exists():
        raise FileNotFoundError(f"Coordinate stats file not found: {stats_path}")

    stats = read_json(stats_path)
    feature = "observation.state" if coord_source == "state" else "action"
    dims = state_xy_dims if coord_source == "state" else action_xy_dims
    feature_stats = stats[feature]

    min_key = "q01" if "q01" in feature_stats else "min"
    max_key = "q99" if "q99" in feature_stats else "max"
    mins = feature_stats[min_key]
    maxs = feature_stats[max_key]
    x_min = float(mins[dims[0]])
    x_max = float(maxs[dims[0]])
    y_min = float(mins[dims[1]])
    y_max = float(maxs[dims[1]])
    return x_min, x_max, y_min, y_max


def table_length(table: pa.Table) -> int:
    return int(table.num_rows)


def count_missing_text(frame_map: dict[str, str] | None, num_frames: int) -> int:
    if frame_map is None:
        return num_frames
    missing = 0
    for frame in range(num_frames):
        if not is_valid_text(frame_map.get(str(frame))):
            missing += 1
    return missing


def load_subtask_prompts(path: Path) -> dict[str, dict[str, str]]:
    payload = read_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("prompts"), dict):
        return payload["prompts"]

    prompts: dict[str, dict[str, str]] = {}
    for episode_key, episode_payload in payload.items():
        if not isinstance(episode_payload, dict):
            continue
        step_to_subtask = episode_payload.get("step_to_subtask")
        if not isinstance(step_to_subtask, dict):
            continue
        prompts[str(episode_key)] = {
            str(frame): str(text)
            for frame, text in step_to_subtask.items()
            if is_valid_text(text)
        }
    return prompts


def align_text_to_chunk_starts(frame_map: dict[str, str], num_frames: int, chunk_size: int) -> list[str]:
    aligned = []
    for frame in range(num_frames):
        start_frame = get_chunk_start_frame(frame, chunk_size)
        aligned.append(frame_map[str(start_frame)])
    return aligned


def add_or_replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    if name in table.column_names:
        idx = table.column_names.index(name)
        return table.set_column(idx, name, values)
    return table.append_column(name, values)


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output dir already exists: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def symlink_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    target = os.path.relpath(src, start=dst.parent)
    dst.symlink_to(target)


def metadata_codec_name(encoder_name: str) -> str:
    return {
        "libx264": "h264",
        "h264": "h264",
        "libsvtav1": "av1",
    }.get(encoder_name, encoder_name)


def set_video_feature_codec(feature: dict[str, Any], codec_name: str) -> None:
    for info_key in ("info", "video_info"):
        if info_key in feature:
            feature[info_key]["video.codec"] = codec_name


def chunk_start_indices(num_frames: int, chunk_size: int) -> list[int]:
    return [get_chunk_start_frame(frame, chunk_size) for frame in range(num_frames)]


def write_chunk_aligned_video(
    src_video: Path,
    dst_video: Path,
    source_indices: list[int],
    *,
    fps: float,
    codec: str,
    crf: str,
    preset: str,
) -> None:
    """Write an output video whose frame t is source frame t - (t % chunk_size)."""
    import av

    frames = []
    with av.open(str(src_video)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))

    if len(frames) != len(source_indices):
        raise ValueError(
            f"Subgoal video frame count mismatch for {src_video}: "
            f"decoded={len(frames)} parquet={len(source_indices)}"
        )

    if not frames:
        raise ValueError(f"Subgoal video has no frames: {src_video}")

    dst_video.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    options = {}
    if codec in {"libx264", "h264"}:
        options = {"crf": str(crf), "preset": preset}

    rate = Fraction(str(fps)).limit_denominator()
    with av.open(str(dst_video), "w") as container:
        stream = container.add_stream(codec, rate=rate, options=options)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"

        for out_idx, src_idx in enumerate(source_indices):
            out_frame = av.VideoFrame.from_ndarray(frames[src_idx], format="rgb24")
            out_frame.pts = out_idx
            for packet in stream.encode(out_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def register_subgoal_videos(
    info: dict[str, Any],
    modality: dict[str, Any],
    *,
    subgoal_video_keys: tuple[str, ...],
    reference_video_key: str,
    subgoal_video_codec: str | None = None,
) -> None:
    features = info.setdefault("features", {})
    ref_feature_key = f"observation.images.{reference_video_key}"
    if ref_feature_key not in features:
        raise KeyError(f"Reference video feature is missing from info.json: {ref_feature_key}")

    modality_video = modality.setdefault("video", {})
    for short_key in subgoal_video_keys:
        feature_key = f"observation.images.{short_key}"
        if feature_key not in features:
            features[feature_key] = copy.deepcopy(features[ref_feature_key])
        if subgoal_video_codec is not None:
            set_video_feature_codec(features[feature_key], metadata_codec_name(subgoal_video_codec))
        modality_video[short_key] = {"original_key": feature_key}


def video_feature_keys(info: dict[str, Any]) -> list[str]:
    keys = []
    for feature_key, feature in info.get("features", {}).items():
        if feature.get("dtype") == "video":
            keys.append(feature_key)
    return sorted(keys)


def validate_subgoal_videos(
    dataset_dir: Path,
    episode_index: int,
    chunks_size: int,
    subgoal_video_keys: tuple[str, ...],
) -> bool:
    for short_key in subgoal_video_keys:
        feature_key = f"observation.images.{short_key}"
        if not video_path(dataset_dir, episode_index, chunks_size, feature_key).exists():
            return False
    return True


def validate_and_collect(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_dir = args.input_dir
    info = read_json(input_dir / "meta" / "info.json")
    episodes = read_jsonl(input_dir / "meta" / "episodes.jsonl")
    subtask_prompts = load_subtask_prompts(args.subtask_json)

    coord_bounds = args.coord_bounds
    if coord_bounds is None:
        coord_bounds = load_coord_bounds(
            input_dir,
            coord_source=args.coord_source,
            state_xy_dims=args.state_xy_dims,
            action_xy_dims=args.action_xy_dims,
            stats_file=args.stats_file,
        )

    chunks_size = int(info.get("chunks_size", 1000))
    total_original_episodes = int(info.get("total_episodes", len(episodes)))
    total_original_frames = int(info.get("total_frames", sum(int(ep["length"]) for ep in episodes)))

    stats = {
        "total_original_episodes": total_original_episodes,
        "total_original_frames": total_original_frames,
        "episodes_missing_subtask": 0,
        "frames_missing_subtask": 0,
        "episodes_missing_traj_cot": 0,
        "frames_missing_traj_cot": 0,
        "episodes_missing_subgoal_video": 0,
        "frames_missing_subgoal_video": 0,
        "strict_text_intersection_episodes": 0,
        "strict_text_intersection_frames": 0,
        "final_episodes": 0,
        "final_frames": 0,
        "chunk_size": args.chunk_size,
        "subgoal_alignment": "materialized_chunk_start_videos",
        "coord_bounds": {
            "x_min": coord_bounds[0],
            "x_max": coord_bounds[1],
            "y_min": coord_bounds[2],
            "y_max": coord_bounds[3],
        },
    }

    kept: list[dict[str, Any]] = []
    episodes_to_scan = episodes[: args.max_episodes] if args.max_episodes is not None else episodes
    if args.max_episodes is not None:
        total_original_episodes = len(episodes_to_scan)
        total_original_frames = sum(int(ep["length"]) for ep in episodes_to_scan)
        stats["total_original_episodes"] = total_original_episodes
        stats["total_original_frames"] = total_original_frames
        stats["debug_max_episodes"] = args.max_episodes

    for ep in iter_progress(episodes_to_scan, desc="Validating episodes"):
        episode_index = int(ep["episode_index"])
        declared_length = int(ep["length"])
        src_parquet = episode_path(input_dir, episode_index, chunks_size)

        try:
            table = pq.read_table(src_parquet, columns=["observation.state", "action"])
            num_frames = table_length(table)
            traj_cot = generate_traj_cot(
                table,
                coord_source=args.coord_source,
                state_xy_dims=args.state_xy_dims,
                action_xy_dims=args.action_xy_dims,
                gripper_dim=args.gripper_dim,
                gripper_threshold=args.gripper_threshold,
                coord_bounds=coord_bounds,
                rdp_epsilon=args.rdp_epsilon,
                min_dist_threshold=args.min_dist_threshold,
                max_tokens=args.max_tokens,
            )
        except Exception:
            num_frames = declared_length
            traj_cot = None

        subtask = subtask_prompts.get(str(episode_index))
        missing_subtask = count_missing_text(subtask, num_frames)
        missing_traj = count_missing_text(traj_cot, num_frames)
        if missing_subtask == 0:
            missing_subtask = sum(
                1
                for frame in range(num_frames)
                if not is_valid_text(subtask.get(str(get_chunk_start_frame(frame, args.chunk_size))))
            )
        if missing_traj == 0:
            missing_traj = sum(
                1
                for frame in range(num_frames)
                if not is_valid_text(traj_cot.get(str(get_chunk_start_frame(frame, args.chunk_size))))
            )

        if missing_subtask:
            stats["episodes_missing_subtask"] += 1
            stats["frames_missing_subtask"] += missing_subtask
        if missing_traj:
            stats["episodes_missing_traj_cot"] += 1
            stats["frames_missing_traj_cot"] += missing_traj

        text_complete = missing_subtask == 0 and missing_traj == 0
        if text_complete:
            stats["strict_text_intersection_episodes"] += 1
            stats["strict_text_intersection_frames"] += num_frames

        has_subgoal = validate_subgoal_videos(input_dir, episode_index, chunks_size, args.subgoal_video_keys)
        if args.require_subgoal_videos and not has_subgoal:
            stats["episodes_missing_subgoal_video"] += 1
            stats["frames_missing_subgoal_video"] += num_frames

        if text_complete and (has_subgoal or not args.require_subgoal_videos):
            kept.append(
                {
                    "source_episode_index": episode_index,
                    "length": num_frames,
                    "subtask": subtask,
                    "traj_cot": traj_cot,
                    "tasks": ep.get("tasks", []),
                }
            )
            stats["final_episodes"] += 1
            stats["final_frames"] += num_frames

    return kept, stats


def write_filtered_dataset(args: argparse.Namespace, kept: list[dict[str, Any]], stats: dict[str, Any]) -> None:
    input_dir = args.input_dir
    output_dir = args.output_dir
    info = read_json(input_dir / "meta" / "info.json")
    modality = read_json(input_dir / "meta" / "modality.json")
    chunks_size = int(info.get("chunks_size", 1000))

    register_subgoal_videos(
        info,
        modality,
        subgoal_video_keys=args.subgoal_video_keys,
        reference_video_key=args.subgoal_reference_key,
        subgoal_video_codec=args.subgoal_video_codec,
    )
    videos_to_link = video_feature_keys(info)
    subgoal_video_feature_keys = {f"observation.images.{key}" for key in args.subgoal_video_keys}
    videos_to_symlink = [key for key in videos_to_link if key not in subgoal_video_feature_keys]
    fps = float(info.get("fps", 30.0))

    prepare_output_dir(output_dir, args.overwrite)
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)

    # Copy metadata files first, then rewrite the files that must change.
    for meta_file in (input_dir / "meta").iterdir():
        if meta_file.is_file():
            shutil.copy2(meta_file, output_dir / "meta" / meta_file.name)

    source_map = []
    new_episodes = []
    global_frame_index = 0
    output_chunks_size = chunks_size

    for new_episode_index, record in enumerate(iter_progress(kept, desc="Writing filtered dataset")):
        src_episode_index = int(record["source_episode_index"])
        src_parquet = episode_path(input_dir, src_episode_index, chunks_size)
        dst_parquet = episode_path(output_dir, new_episode_index, output_chunks_size)
        dst_parquet.parent.mkdir(parents=True, exist_ok=True)

        table = pq.read_table(src_parquet)
        num_frames = table_length(table)
        frame_indices = list(range(num_frames))
        subtask_values = align_text_to_chunk_starts(record["subtask"], num_frames, args.chunk_size)
        traj_values = align_text_to_chunk_starts(record["traj_cot"], num_frames, args.chunk_size)
        subgoal_source_indices = chunk_start_indices(num_frames, args.chunk_size)

        table = add_or_replace_column(table, "episode_index", pa.array([new_episode_index] * num_frames, pa.int64()))
        table = add_or_replace_column(table, "frame_index", pa.array(frame_indices, pa.int64()))
        table = add_or_replace_column(
            table,
            "index",
            pa.array(list(range(global_frame_index, global_frame_index + num_frames)), pa.int64()),
        )
        table = add_or_replace_column(table, "subtask", pa.array(subtask_values, pa.string()))
        table = add_or_replace_column(table, "traj_cot", pa.array(traj_values, pa.string()))
        pq.write_table(table, dst_parquet)

        for video_key in videos_to_symlink:
            src_video = video_path(input_dir, src_episode_index, chunks_size, video_key)
            dst_video = video_path(output_dir, new_episode_index, output_chunks_size, video_key)
            if src_video.exists():
                symlink_file(src_video, dst_video)

        for video_key in sorted(subgoal_video_feature_keys):
            src_video = video_path(input_dir, src_episode_index, chunks_size, video_key)
            dst_video = video_path(output_dir, new_episode_index, output_chunks_size, video_key)
            if src_video.exists():
                write_chunk_aligned_video(
                    src_video,
                    dst_video,
                    subgoal_source_indices,
                    fps=fps,
                    codec=args.subgoal_video_codec,
                    crf=args.subgoal_video_crf,
                    preset=args.subgoal_video_preset,
                )

        source_map.append(
            {
                "episode_index": new_episode_index,
                "source_episode_index": src_episode_index,
                "length": num_frames,
            }
        )
        new_episodes.append(
            {
                "episode_index": new_episode_index,
                "tasks": record["tasks"],
                "length": num_frames,
            }
        )
        global_frame_index += num_frames

    info["total_episodes"] = stats["final_episodes"]
    info["total_frames"] = stats["final_frames"]
    info["total_chunks"] = int(math.ceil(stats["final_episodes"] / output_chunks_size)) if stats["final_episodes"] else 0
    info["splits"] = {"train": f"0:{stats['final_episodes']}"}
    info["total_videos"] = stats["final_episodes"] * len(videos_to_link)
    info["features"]["subtask"] = {"dtype": "string", "shape": [1], "names": ["subtask"]}
    info["features"]["traj_cot"] = {"dtype": "string", "shape": [1], "names": ["traj_cot"]}

    write_json(output_dir / "meta" / "info.json", info)
    write_json(output_dir / "meta" / "modality.json", modality)
    write_jsonl(output_dir / "meta" / "episodes.jsonl", new_episodes)
    write_jsonl(output_dir / "meta" / "source_episode_map.jsonl", source_map)
    write_json(output_dir / "meta" / "prepare_bridge_conditions_report.json", stats)


def print_report(stats: dict[str, Any]) -> None:
    print("\n=== Bridge Condition Preparation Report ===")
    print(f"Total original episodes: {stats['total_original_episodes']}")
    print(f"Total original frames:   {stats['total_original_frames']}")
    print(f"Episodes missing subtask: {stats['episodes_missing_subtask']}")
    print(f"Frames missing subtask:   {stats['frames_missing_subtask']}")
    print(f"Episodes missing traj_cot: {stats['episodes_missing_traj_cot']}")
    print(f"Frames missing traj_cot:   {stats['frames_missing_traj_cot']}")
    print(f"Text intersection episodes: {stats['strict_text_intersection_episodes']}")
    print(f"Text intersection frames:   {stats['strict_text_intersection_frames']}")
    print(f"Episodes missing subgoal video: {stats['episodes_missing_subgoal_video']}")
    print(f"Frames missing subgoal video:   {stats['frames_missing_subgoal_video']}")
    print(f"Final filtered episodes: {stats['final_episodes']}")
    print(f"Final filtered frames:   {stats['final_frames']}")
    print(f"Chunk size: {stats['chunk_size']}")
    print(f"Subgoal alignment: {stats['subgoal_alignment']}")
    print(f"Coordinate bounds: {stats['coord_bounds']}")


def parse_dims(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected two comma-separated dims, e.g. '0,1'")
    return int(parts[0]), int(parts[1])


def parse_bounds(value: str) -> tuple[float, float, float, float]:
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Expected x_min,x_max,y_min,y_max")
    return tuple(float(part) for part in parts)  # type: ignore[return-value]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("Datasets/bridge_orig_lerobot"))
    parser.add_argument("--output-dir", type=Path, default=Path("Datasets/bridge_cond_lerobot"))
    parser.add_argument(
        "--subtask-json",
        type=Path,
        default=None,
        help=f"Defaults to --input-dir/subtask_data/{DEFAULT_SUBTASK_JSON_NAME}.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=None, help="Debug only: scan the first N episodes.")

    parser.add_argument("--coord-source", choices=("state", "action"), default="state")
    parser.add_argument("--state-xy-dims", type=parse_dims, default=(0, 1))
    parser.add_argument("--action-xy-dims", type=parse_dims, default=(0, 1))
    parser.add_argument("--coord-bounds", type=parse_bounds, default=None, help="Override as x_min,x_max,y_min,y_max.")
    parser.add_argument("--stats-file", default="stats_gr00t.json")
    parser.add_argument("--gripper-dim", type=int, default=6)
    parser.add_argument("--gripper-threshold", type=float, default=0.5)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10,
        help="Condition hold interval. Each frame uses the condition from t - (t % chunk_size).",
    )
    parser.add_argument("--rdp-epsilon", type=float, default=3.0)
    parser.add_argument("--min-dist-threshold", type=float, default=20.0)
    parser.add_argument("--max-tokens", type=int, default=10)

    parser.add_argument("--subgoal-video-keys", nargs="+", default=list(DEFAULT_SUBGOAL_VIDEO_KEYS))
    parser.add_argument("--subgoal-reference-key", default="image_0")
    parser.add_argument("--subgoal-video-codec", default="libx264")
    parser.add_argument("--subgoal-video-crf", default="23")
    parser.add_argument("--subgoal-video-preset", default="medium")
    parser.add_argument("--no-require-subgoal-videos", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.input_dir = args.input_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.subtask_json is None:
        args.subtask_json = args.input_dir / "subtask_data" / DEFAULT_SUBTASK_JSON_NAME
    args.subtask_json = args.subtask_json.expanduser().resolve()
    if args.chunk_size <= 0:
        raise ValueError(f"--chunk-size must be positive, got {args.chunk_size}")
    args.subgoal_video_keys = tuple(args.subgoal_video_keys)
    args.require_subgoal_videos = not args.no_require_subgoal_videos

    if args.output_dir == args.input_dir:
        raise ValueError("Refusing to write into the input dataset. Use a separate --output-dir.")

    kept, stats = validate_and_collect(args)
    print_report(stats)

    if args.dry_run:
        print("\nDry run only; no dataset was written.")
        return

    write_filtered_dataset(args, kept, stats)
    print(f"\nWrote filtered dataset to: {args.output_dir}")


if __name__ == "__main__":
    main()
