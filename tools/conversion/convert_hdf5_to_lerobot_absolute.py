#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CAMERA_ORDER = ("cam_high", "cam_left_wrist", "cam_right_wrist")
DEFAULT_EXCLUDED_DIR_NAMES = {
    "_removed_bad_episodes",
    "removed_bad_episodes",
    "bad_episodes",
    ".trash",
}


@dataclass(frozen=True)
class SourceEpisode:
    instruction: str
    path: Path
    source_index: int | None


class ArrayStats:
    def __init__(self) -> None:
        self._chunks: list[Any] = []

    def add(self, array: Any) -> None:
        self._chunks.append(array.astype("float32", copy=False))

    def to_payload(self, np) -> dict[str, list[float]]:
        if not self._chunks:
            raise ValueError("No arrays were collected for statistics.")
        data = np.concatenate(self._chunks, axis=0)
        return {
            "mean": np.mean(data, axis=0).tolist(),
            "std": np.std(data, axis=0).tolist(),
            "min": np.min(data, axis=0).tolist(),
            "max": np.max(data, axis=0).tolist(),
            "q01": np.quantile(data, 0.01, axis=0).tolist(),
            "q99": np.quantile(data, 0.99, axis=0).tolist(),
        }


def import_array_deps():
    missing = []
    try:
        import h5py  # type: ignore
    except ModuleNotFoundError:
        h5py = None
        missing.append("h5py")
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError:
        np = None
        missing.append("numpy")
    try:
        import pandas as pd  # type: ignore
    except ModuleNotFoundError:
        pd = None
        missing.append("pandas")

    if missing:
        raise SystemExit(
            "Missing conversion dependencies: "
            + ", ".join(missing)
            + ". Install them in the data-processing environment, for example: "
            "pip install h5py numpy pandas pyarrow"
        )
    return h5py, np, pd


def import_cv2():
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing video dependency: opencv-python. Install it with: pip install opencv-python"
        ) from exc
    return cv2


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def move_preserved_entries(out_dir: Path, preserve_names: set[str]) -> tuple[Path | None, list[tuple[str, Path]]]:
    if not preserve_names or not out_dir.exists():
        return None, []

    preserve_root = Path(
        tempfile.mkdtemp(prefix=f".{out_dir.name}_preserve_", dir=str(out_dir.parent))
    )
    preserved = []
    for name in sorted(preserve_names):
        src = out_dir / name
        if not src.exists():
            continue
        dst = preserve_root / name
        shutil.move(str(src), str(dst))
        preserved.append((name, dst))
    return preserve_root, preserved


def restore_preserved_entries(out_dir: Path, preserve_root: Path | None, preserved: list[tuple[str, Path]]) -> None:
    try:
        for name, src in preserved:
            dst = out_dir / name
            if dst.exists():
                if dst.is_dir():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            shutil.move(str(src), str(dst))
    finally:
        if preserve_root is not None:
            shutil.rmtree(preserve_root, ignore_errors=True)


def ensure_empty_output_dir(out_dir: Path, overwrite: bool, preserve_names: set[str] | None = None) -> None:
    preserve_names = preserve_names or set()
    if out_dir.exists():
        if not out_dir.is_dir():
            raise SystemExit(f"Output path exists and is not a directory: {out_dir}")
        if overwrite:
            preserve_root, preserved = move_preserved_entries(out_dir, preserve_names)
            shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            restore_preserved_entries(out_dir, preserve_root, preserved)
        elif any(out_dir.iterdir()):
            blocking_entries = [
                child.name for child in out_dir.iterdir() if child.name not in preserve_names
            ]
            if blocking_entries:
                preview = ", ".join(sorted(blocking_entries)[:8])
                raise SystemExit(
                    f"Output directory already contains conversion outputs: {out_dir}\n"
                    f"Blocking entries: {preview}\n"
                    "Use --overwrite to rebuild it."
                )
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)


def infer_task_name_from_dir(task_dir: Path) -> str:
    return task_dir.name.replace("_", " ").strip() or "default_task"


def source_episode_index(path: Path) -> int | None:
    stem = path.stem
    for prefix in ("episode_", "episode-"):
        if stem.startswith(prefix):
            suffix = stem[len(prefix) :]
            if suffix.isdigit():
                return int(suffix)
    return None


def episode_sort_key(path: Path) -> tuple[int, str]:
    index = source_episode_index(path)
    if index is None:
        return (10**12, path.name)
    return (index, path.name)


def sorted_episode_files(directory: Path) -> list[Path]:
    files = list(directory.glob("episode_*.hdf5")) + list(directory.glob("episode-*.hdf5"))
    return sorted(set(files), key=episode_sort_key)


def parse_task_filter(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    tasks = set()
    for value in values:
        for part in value.split(","):
            if part.strip():
                tasks.add(part.strip())
    return tasks or None


def task_is_selected(task_name: str, task_filter: set[str] | None) -> bool:
    return task_filter is None or task_name in task_filter


def discover_episodes(src_dir: Path, task_name: str | None, task_filter: set[str] | None) -> list[SourceEpisode]:
    src_dir = src_dir.expanduser()
    meta_path = src_dir / "pipeline_meta.json"

    if task_name is not None:
        if task_filter and src_dir.name not in task_filter:
            return []
        return [
            SourceEpisode(task_name, path, source_episode_index(path))
            for path in sorted_episode_files(src_dir)
        ]

    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        slug_to_text = {slug: text for text, slug in meta.get("slugs", {}).items()}
        ordered_instructions = list(meta.get("instructions", []))
        grouped: dict[str, list[SourceEpisode]] = {}

        for slug_dir in sorted(src_dir.iterdir()):
            if not slug_dir.is_dir() or slug_dir.name in DEFAULT_EXCLUDED_DIR_NAMES:
                continue
            if not task_is_selected(slug_dir.name, task_filter):
                continue
            files = sorted_episode_files(slug_dir)
            if not files:
                continue
            instruction = slug_to_text.get(slug_dir.name, infer_task_name_from_dir(slug_dir))
            grouped[instruction] = [
                SourceEpisode(instruction, path, source_episode_index(path)) for path in files
            ]

        if not ordered_instructions:
            ordered_instructions = sorted(grouped)

        episodes = []
        for instruction in ordered_instructions:
            episodes.extend(grouped.get(instruction, []))
        return episodes

    flat_files = sorted_episode_files(src_dir)
    if flat_files:
        if not task_is_selected(src_dir.name, task_filter):
            return []
        instruction = infer_task_name_from_dir(src_dir)
        return [SourceEpisode(instruction, path, source_episode_index(path)) for path in flat_files]

    episodes = []
    for task_dir in sorted(src_dir.iterdir()):
        if not task_dir.is_dir() or task_dir.name in DEFAULT_EXCLUDED_DIR_NAMES:
            continue
        if not task_is_selected(task_dir.name, task_filter):
            continue
        files = sorted_episode_files(task_dir)
        if not files:
            continue
        instruction = infer_task_name_from_dir(task_dir)
        episodes.extend(SourceEpisode(instruction, path, source_episode_index(path)) for path in files)
    return episodes


def load_screen_report_bad_paths(screen_report: Path) -> tuple[set[str], set[str], dict[str, Any]]:
    screen_report = screen_report.expanduser()
    report = json.loads(screen_report.read_text(encoding="utf-8"))
    report_root = Path(report.get("root", "")).expanduser()
    bad_abs_paths: set[str] = set()
    bad_rel_paths: set[str] = set()

    for item in report.get("bad_episodes", []):
        path_text = item.get("path")
        if path_text:
            bad_abs_paths.add(str(Path(path_text).expanduser().resolve()))
        rel_text = item.get("relative_path")
        if rel_text:
            bad_rel_paths.add(str(rel_text))
            if str(report_root):
                bad_abs_paths.add(str((report_root / rel_text).expanduser().resolve()))

    return bad_abs_paths, bad_rel_paths, report


def filter_screen_report_bad_episodes(
    episodes: list[SourceEpisode],
    screen_report: Path | None,
    src_dir: Path,
) -> tuple[list[SourceEpisode], list[dict[str, Any]], dict[str, Any] | None]:
    if screen_report is None:
        return episodes, [], None

    bad_abs_paths, bad_rel_paths, report = load_screen_report_bad_paths(screen_report)
    report_root = Path(report.get("root", src_dir)).expanduser().resolve()
    kept = []
    skipped = []

    for ep in episodes:
        abs_key = str(ep.path.expanduser().resolve())
        try:
            rel_key = ep.path.expanduser().resolve().relative_to(report_root).as_posix()
        except ValueError:
            rel_key = ""
        if abs_key in bad_abs_paths or rel_key in bad_rel_paths:
            skipped.append(
                {
                    "path": str(ep.path),
                    "relative_path": rel_key,
                    "source_episode_index": ep.source_index,
                    "task": ep.instruction,
                }
            )
            continue
        kept.append(ep)

    return kept, skipped, report


def parse_episode_selection(values: list[str] | None) -> set[int]:
    selected: set[int] = set()
    if not values:
        return selected
    for raw_value in values:
        for token in raw_value.replace(",", " ").split():
            if not token:
                continue
            if ":" in token:
                parts = token.split(":")
                if len(parts) != 2 or not parts[0] or not parts[1]:
                    raise ValueError(f"Invalid episode range {token!r}; expected start:stop")
                start = int(parts[0])
                stop = int(parts[1])
                if stop < start:
                    raise ValueError(f"Invalid episode range {token!r}; stop must be >= start")
                selected.update(range(start, stop))
            else:
                selected.add(int(token))
    return selected


def filter_episodes(
    episodes: list[SourceEpisode],
    include_episode_ids: set[int],
    exclude_episode_ids: set[int],
) -> list[SourceEpisode]:
    if (include_episode_ids or exclude_episode_ids) and any(ep.source_index is None for ep in episodes):
        raise ValueError(
            "--include-episodes/--exclude-episodes require source files named like episode_000123.hdf5."
        )

    selected_episodes = []
    for ep in episodes:
        source_index = ep.source_index
        if include_episode_ids and source_index not in include_episode_ids:
            continue
        if exclude_episode_ids and source_index in exclude_episode_ids:
            continue
        selected_episodes.append(ep)
    return selected_episodes


def assign_output_indices(
    episodes: list[SourceEpisode],
    mode: str,
) -> tuple[dict[Path, int], str]:
    source_indices = [ep.source_index for ep in episodes]
    can_preserve = (
        all(index is not None for index in source_indices)
        and len(set(source_indices)) == len(source_indices)
    )

    resolved_mode = "preserve" if mode == "auto" and can_preserve else mode
    if mode == "auto" and not can_preserve:
        resolved_mode = "sequential"

    if resolved_mode == "preserve":
        if not can_preserve:
            raise ValueError(
                "Cannot preserve source episode ids because some ids are missing or duplicated. "
                "Use --episode-index-mode sequential for a multi-task source tree with repeated episode ids."
            )
        return {ep.path: int(ep.source_index) for ep in episodes}, resolved_mode

    return {ep.path: idx for idx, ep in enumerate(episodes)}, resolved_mode


def h5_key(key: str) -> str:
    return key.strip("/")


def read_required_array(h5_file, np, key: str, dtype: str):
    normalized = h5_key(key)
    if normalized not in h5_file:
        raise KeyError(f"Missing required HDF5 dataset: {key}")
    return np.asarray(h5_file[normalized], dtype=dtype)


def read_optional_array(h5_file, np, key: str | None, dtype: str):
    if key is None:
        return None
    normalized = h5_key(key)
    if normalized not in h5_file:
        return None
    return np.asarray(h5_file[normalized], dtype=dtype)


def validate_time_series(name: str, array: Any, expected_length: int | None, expected_dim: int | None) -> None:
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {array.shape}")
    if expected_length is not None and int(array.shape[0]) != expected_length:
        raise ValueError(
            f"{name} length mismatch: expected {expected_length}, got {int(array.shape[0])}"
        )
    if expected_dim is not None and int(array.shape[1]) != expected_dim:
        raise ValueError(f"{name} dim mismatch: expected {expected_dim}, got {int(array.shape[1])}")


def summarize_array(np, name: str, array: Any) -> dict[str, Any]:
    return {
        "name": name,
        "shape": [int(x) for x in array.shape],
        "min": np.min(array, axis=0).tolist(),
        "max": np.max(array, axis=0).tolist(),
        "mean": np.mean(array, axis=0).tolist(),
        "std": np.std(array, axis=0).tolist(),
    }


def resolve_camera_names(image_group, requested: list[str] | None) -> list[str]:
    available = list(image_group.keys())
    if requested:
        missing = [name for name in requested if name not in available]
        if missing:
            raise KeyError(f"Requested cameras not found: {missing}; available cameras: {available}")
        return requested

    preferred = [name for name in DEFAULT_CAMERA_ORDER if name in available]
    rest = sorted(name for name in available if name not in preferred)
    return preferred + rest


def trim_jpeg_padding(np, buffer: Any):
    arr = np.asarray(buffer, dtype=np.uint8).reshape(-1)
    if arr.size < 2:
        return arr
    eoi_positions = np.where((arr[:-1] == 0xFF) & (arr[1:] == 0xD9))[0]
    if len(eoi_positions):
        return arr[: int(eoi_positions[-1]) + 2]
    return arr


def decode_frame_to_bgr(cv2, np, item: Any, raw_image_color: str):
    arr = np.asarray(item)
    if arr.ndim == 3:
        frame = arr
        if frame.shape[-1] < 3:
            raise ValueError(f"Raw image frame must have at least 3 channels, got {frame.shape}")
        frame = frame[..., :3]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if raw_image_color == "rgb":
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    if isinstance(item, (bytes, bytearray, memoryview)):
        encoded = np.frombuffer(item, dtype=np.uint8)
    else:
        encoded = trim_jpeg_padding(np, item)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Failed to decode JPEG frame from HDF5 image dataset.")
    return image


def write_video_from_dataset(
    cv2,
    np,
    dataset,
    video_path: Path,
    fps: int,
    raw_image_color: str,
    ffmpeg_bin: str,
) -> tuple[int, int, int]:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    length = int(dataset.shape[0])
    if length <= 0:
        raise ValueError(f"Cannot write empty video: {video_path}")

    first_shape: tuple[int, int, int] | None = None
    with tempfile.TemporaryDirectory(prefix="aloha_lerobot_frames_") as tmp_dir_text:
        tmp_dir = Path(tmp_dir_text)
        for frame_idx in range(length):
            frame_bgr = decode_frame_to_bgr(cv2, np, dataset[frame_idx], raw_image_color)
            if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
                raise ValueError(f"Decoded frame has invalid shape {frame_bgr.shape}")
            shape = (int(frame_bgr.shape[0]), int(frame_bgr.shape[1]), int(frame_bgr.shape[2]))
            if first_shape is None:
                first_shape = shape
            elif shape != first_shape:
                raise ValueError(
                    f"Camera frames must have a fixed shape. First frame {first_shape}, "
                    f"frame {frame_idx} {shape}"
                )
            frame_path = tmp_dir / f"frame_{frame_idx:06d}.png"
            ok = cv2.imwrite(str(frame_path), frame_bgr)
            if not ok:
                raise RuntimeError(f"Failed to write temporary frame: {frame_path}")

        cmd = [
            ffmpeg_bin,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(tmp_dir / "frame_%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except FileNotFoundError as exc:
            raise RuntimeError(f"ffmpeg executable not found: {ffmpeg_bin}") from exc
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg encode failed for {video_path}:\n{result.stderr.strip()}"
            )

    assert first_shape is not None
    return first_shape


def data_path_for_episode(out_dir: Path, episode_index: int, chunks_size: int) -> Path:
    chunk_index = episode_index // chunks_size
    return out_dir / "data" / f"chunk-{chunk_index:03d}" / f"episode_{episode_index:06d}.parquet"


def video_path_for_episode(out_dir: Path, episode_index: int, chunks_size: int, camera_name: str) -> Path:
    chunk_index = episode_index // chunks_size
    video_key = f"observation.images.{camera_name}"
    return out_dir / "videos" / f"chunk-{chunk_index:03d}" / video_key / f"episode_{episode_index:06d}.mp4"


def relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def convert_episode(
    ep: SourceEpisode,
    output_episode_index: int,
    task_index: int,
    out_dir: Path,
    args: argparse.Namespace,
    h5py,
    np,
    pd,
    cv2,
    state_stats: ArrayStats,
    action_stats: ArrayStats,
) -> dict[str, Any]:
    with h5py.File(ep.path, "r") as h5_file:
        state = read_required_array(h5_file, np, args.state_key, "float32")
        action = read_required_array(h5_file, np, args.action_key, "float32")
        validate_time_series("observation.state", state, None, args.state_dim)
        validate_time_series("action", action, int(state.shape[0]), args.action_dim)

        qvel = read_optional_array(h5_file, np, args.qvel_key, "float32")
        effort = read_optional_array(h5_file, np, args.effort_key, "float32")
        base_action = read_optional_array(h5_file, np, args.base_action_key, "float32")
        for optional_name, optional_array in (
            ("observation.qvel", qvel),
            ("observation.effort", effort),
            ("base_action", base_action),
        ):
            if optional_array is not None:
                validate_time_series(optional_name, optional_array, int(state.shape[0]), None)

        fps = int(args.fps or h5_file.attrs.get("frame_rate", args.default_fps))
        image_group_key = h5_key(args.image_group_key)
        if image_group_key not in h5_file:
            raise KeyError(f"Missing HDF5 image group: {args.image_group_key}")
        image_group = h5_file[image_group_key]
        camera_names = resolve_camera_names(image_group, args.camera_names)
        num_frames = int(state.shape[0])
        for camera_name in camera_names:
            camera_length = int(image_group[camera_name].shape[0])
            if camera_length != num_frames:
                raise ValueError(
                    f"Image/state length mismatch for {camera_name}: "
                    f"{camera_length} images vs {num_frames} state rows"
                )

        rel_video_paths = {
            camera_name: relative_posix(
                video_path_for_episode(out_dir, output_episode_index, args.chunks_size, camera_name),
                out_dir,
            )
            for camera_name in camera_names
        }

        rows = []
        for frame_idx in range(num_frames):
            row: dict[str, Any] = {
                "episode_index": int(output_episode_index),
                "frame_index": int(frame_idx),
                "timestamp": float(frame_idx / max(fps, 1)),
                "task_index": int(task_index),
                "task": ep.instruction,
                "observation.state": state[frame_idx].tolist(),
                "action": action[frame_idx].tolist(),
            }
            if qvel is not None:
                row["observation.qvel"] = qvel[frame_idx].tolist()
            if effort is not None:
                row["observation.effort"] = effort[frame_idx].tolist()
            if base_action is not None:
                row["base_action"] = base_action[frame_idx].tolist()
            for camera_name, rel_video_path in rel_video_paths.items():
                row[f"observation.images.{camera_name}"] = rel_video_path
            rows.append(row)

        parquet_path = data_path_for_episode(out_dir, output_episode_index, args.chunks_size)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(parquet_path, index=False)

        camera_shapes = {}
        for camera_name in camera_names:
            video_path = video_path_for_episode(out_dir, output_episode_index, args.chunks_size, camera_name)
            camera_shapes[camera_name] = write_video_from_dataset(
                cv2=cv2,
                np=np,
                dataset=image_group[camera_name],
                video_path=video_path,
                fps=fps,
                raw_image_color=args.raw_image_color,
                ffmpeg_bin=args.ffmpeg_bin,
            )

    state_stats.add(state)
    action_stats.add(action)

    episode_meta = {
        "episode_index": int(output_episode_index),
        "source_episode_index": None if ep.source_index is None else int(ep.source_index),
        "source_path": str(ep.path),
        "task_index": int(task_index),
        "task": ep.instruction,
        "length": num_frames,
        "num_frames": num_frames,
        "fps": fps,
    }
    episode_stats = {
        "episode_index": int(output_episode_index),
        "source_episode_index": None if ep.source_index is None else int(ep.source_index),
        "task_index": int(task_index),
        "task": ep.instruction,
        "length": num_frames,
        "num_frames": num_frames,
        "observation.state": summarize_array(np, "observation.state", state),
        "action": summarize_array(np, "action", action),
    }
    return {
        "episode_meta": episode_meta,
        "episode_stats": episode_stats,
        "camera_names": camera_names,
        "camera_shapes": camera_shapes,
        "fps": fps,
        "columns": list(rows[0]) if rows else [],
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_modality_json(camera_names: list[str]) -> dict[str, Any]:
    state_part = {
        "left_arm": {"start": 0, "end": 6, "absolute": True, "dtype": "float32"},
        "left_gripper": {"start": 6, "end": 7, "absolute": True, "dtype": "float32"},
        "right_arm": {"start": 7, "end": 13, "absolute": True, "dtype": "float32"},
        "right_gripper": {"start": 13, "end": 14, "absolute": True, "dtype": "float32"},
    }
    action_part = {
        "left_arm": {"start": 0, "end": 6, "absolute": True, "dtype": "float32"},
        "left_gripper": {"start": 6, "end": 7, "absolute": True, "dtype": "float32"},
        "right_arm": {"start": 7, "end": 13, "absolute": True, "dtype": "float32"},
        "right_gripper": {"start": 13, "end": 14, "absolute": True, "dtype": "float32"},
    }
    for item in state_part.values():
        item["original_key"] = "observation.state"
    for item in action_part.values():
        item["original_key"] = "action"

    return {
        "state": state_part,
        "action": action_part,
        "video": {
            camera_name: {"original_key": f"observation.images.{camera_name}"}
            for camera_name in camera_names
        },
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"},
        },
    }


def build_features(
    camera_shapes: dict[str, tuple[int, int, int]],
    fps: int,
    state_dim: int,
    action_dim: int,
    optional_columns: set[str],
) -> dict[str, Any]:
    features: dict[str, Any] = {
        "episode_index": {"dtype": "int64", "shape": [1], "names": ["index"]},
        "frame_index": {"dtype": "int64", "shape": [1], "names": ["index"]},
        "timestamp": {"dtype": "float32", "shape": [1], "names": ["time"]},
        "task_index": {"dtype": "int64", "shape": [1], "names": ["index"]},
        "task": {"dtype": "string", "shape": [1], "names": ["task"]},
        "observation.state": {
            "dtype": "float32",
            "shape": [state_dim],
            "names": ["state"],
            "encoding": "absolute",
        },
        "action": {
            "dtype": "float32",
            "shape": [action_dim],
            "names": ["action"],
            "encoding": "absolute",
        },
    }

    optional_dims = {
        "observation.qvel": state_dim,
        "observation.effort": state_dim,
        "base_action": 2,
    }
    for column in sorted(optional_columns):
        features[column] = {
            "dtype": "float32",
            "shape": [optional_dims.get(column, -1)],
            "names": [column],
        }

    for camera_name, shape in sorted(camera_shapes.items()):
        height, width, channels = [int(x) for x in shape]
        video_info = {
            "video.fps": int(fps),
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        }
        features[f"observation.images.{camera_name}"] = {
            "dtype": "video",
            "shape": [height, width, channels],
            "names": ["height", "width", "channel"],
            "video_info": video_info,
            "info": {
                "video.fps": int(fps),
                "video.height": height,
                "video.width": width,
                "video.channels": channels,
            },
        }
    return features


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Aloha HDF5 episodes to a LeRobot v2.1-style local dataset. "
            "The converter copies /observations/qpos to observation.state and /action to action as absolute values."
        )
    )
    parser.add_argument("--src-dir", type=Path, required=True, help="Raw HDF5 root directory.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output LeRobot dataset directory.")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--robot-type", default="aloha_piper_absolute")
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional task directory names under --src-dir. Accepts repeated values or comma-separated values.",
    )
    parser.add_argument(
        "--screen-report",
        type=Path,
        default=None,
        help="HDF5 screening report JSON. Episodes listed in bad_episodes are skipped automatically.",
    )
    parser.add_argument(
        "--task-name",
        default=None,
        help="Optional fixed task text for all episodes. If omitted, task names come from pipeline_meta.json or folder names.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Delete and rebuild generated outputs in --out-dir. If --screen-report is under "
            "--out-dir/data_quality, that directory is preserved."
        ),
    )
    parser.add_argument(
        "--episode-index-mode",
        choices=["auto", "preserve", "sequential"],
        default="auto",
        help=(
            "auto preserves source ids when they are unique, otherwise writes contiguous ids. "
            "Use preserve for single-task datasets where episode_000123.hdf5 should become episode_000123.parquet."
        ),
    )
    parser.add_argument(
        "--include-episodes",
        nargs="*",
        default=None,
        help="Optional source episode ids/ranges to include, e.g. 0:100 120:200. Ranges are stop-exclusive.",
    )
    parser.add_argument(
        "--exclude-episodes",
        nargs="*",
        default=None,
        help="Optional source episode ids/ranges to exclude, e.g. 100:120 200:205. Ranges are stop-exclusive.",
    )
    parser.add_argument("--chunks-size", type=int, default=1000)
    parser.add_argument("--state-key", default="/observations/qpos")
    parser.add_argument("--action-key", default="/action")
    parser.add_argument("--qvel-key", default="/observations/qvel")
    parser.add_argument("--effort-key", default="/observations/effort")
    parser.add_argument("--base-action-key", default="/base_action")
    parser.add_argument("--image-group-key", default="/observations/images")
    parser.add_argument("--state-dim", type=int, default=14)
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--camera-names", nargs="*", default=None)
    parser.add_argument(
        "--raw-image-color",
        choices=["bgr", "rgb"],
        default="bgr",
        help="Color layout for uncompressed HDF5 image arrays. JPEG frames are always decoded by OpenCV.",
    )
    parser.add_argument("--fps", type=int, default=None, help="Override frame rate for all episodes.")
    parser.add_argument("--default-fps", type=int, default=30)
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--limit", type=int, default=None, help="Debug option: convert only the first N episodes.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    src_dir = args.src_dir.expanduser()
    out_dir = args.out_dir.expanduser()
    screen_report_path = args.screen_report.expanduser() if args.screen_report else None
    task_filter = parse_task_filter(args.tasks)
    include_episode_ids = parse_episode_selection(args.include_episodes)
    exclude_episode_ids = parse_episode_selection(args.exclude_episodes)

    episodes = discover_episodes(src_dir, args.task_name, task_filter)
    episodes = filter_episodes(episodes, include_episode_ids, exclude_episode_ids)
    episodes, skipped_by_screen_report, screen_report_payload = filter_screen_report_bad_episodes(
        episodes,
        screen_report_path,
        src_dir,
    )
    if args.limit is not None:
        episodes = episodes[: args.limit]
    if not episodes:
        raise SystemExit(f"No HDF5 episodes found under {src_dir}")

    output_indices, resolved_index_mode = assign_output_indices(episodes, args.episode_index_mode)
    preserve_names = set()
    if screen_report_path is not None and is_relative_to(
        screen_report_path.resolve(),
        out_dir.resolve(),
    ):
        preserve_names.add("data_quality")
    ensure_empty_output_dir(out_dir, args.overwrite, preserve_names=preserve_names)

    h5py, np, pd = import_array_deps()
    cv2 = import_cv2()

    task_to_index: dict[str, int] = {}
    for ep in episodes:
        if ep.instruction not in task_to_index:
            task_to_index[ep.instruction] = len(task_to_index)

    state_stats = ArrayStats()
    action_stats = ArrayStats()
    episodes_meta = []
    episodes_stats = []
    camera_names: list[str] | None = None
    camera_shapes: dict[str, tuple[int, int, int]] = {}
    optional_columns: set[str] = set()
    first_fps: int | None = None
    failed: list[dict[str, Any]] = []

    for position, ep in enumerate(episodes, start=1):
        output_episode_index = output_indices[ep.path]
        task_index = task_to_index[ep.instruction]
        print(f"[{position}/{len(episodes)}] episode_{output_episode_index:06d} <- {ep.path}")
        try:
            result = convert_episode(
                ep=ep,
                output_episode_index=output_episode_index,
                task_index=task_index,
                out_dir=out_dir,
                args=args,
                h5py=h5py,
                np=np,
                pd=pd,
                cv2=cv2,
                state_stats=state_stats,
                action_stats=action_stats,
            )
        except Exception as exc:
            failed.append(
                {
                    "source_path": str(ep.path),
                    "source_episode_index": ep.source_index,
                    "output_episode_index": output_episode_index,
                    "error": str(exc),
                }
            )
            print(f"[FAILED] {ep.path}: {exc}")
            continue

        episodes_meta.append(result["episode_meta"])
        episodes_stats.append(result["episode_stats"])
        if camera_names is None:
            camera_names = list(result["camera_names"])
        for camera_name, shape in result["camera_shapes"].items():
            camera_shapes.setdefault(camera_name, shape)
        for column in result["columns"]:
            if column in {"observation.qvel", "observation.effort", "base_action"}:
                optional_columns.add(column)
        if first_fps is None:
            first_fps = int(result["fps"])

    if failed:
        write_json(out_dir / "meta" / "conversion_failures.json", failed)
    if not episodes_meta:
        raise SystemExit(f"All conversions failed. See {out_dir / 'meta' / 'conversion_failures.json'}")

    camera_names = camera_names or []
    first_fps = first_fps or args.default_fps
    tasks_rows = [
        {"task_index": task_index, "task": task}
        for task, task_index in sorted(task_to_index.items(), key=lambda item: item[1])
    ]
    total_frames = int(sum(item["length"] for item in episodes_meta))

    write_jsonl(out_dir / "meta" / "episodes.jsonl", sorted(episodes_meta, key=lambda x: x["episode_index"]))
    write_jsonl(out_dir / "meta" / "episodes_stats.jsonl", sorted(episodes_stats, key=lambda x: x["episode_index"]))
    write_jsonl(out_dir / "meta" / "tasks.jsonl", tasks_rows)
    write_json(out_dir / "meta" / "modality.json", build_modality_json(camera_names))
    write_json(
        out_dir / "meta" / "stats_gr00t.json",
        {
            "observation.state": state_stats.to_payload(np),
            "action": action_stats.to_payload(np),
        },
    )

    info = {
        "repo_id": args.repo_id or f"local/{out_dir.name}",
        "robot_type": args.robot_type,
        "codebase_version": "v2.1",
        "format": "lerobot_v2.1_style_local",
        "source_format": "aloha_hdf5",
        "state_action_encoding": {
            "observation.state": "absolute",
            "action": "absolute",
        },
        "fps": int(first_fps),
        "chunks_size": int(args.chunks_size),
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "num_episodes": len(episodes_meta),
        "total_episodes": len(episodes_meta),
        "num_frames": total_frames,
        "total_frames": total_frames,
        "num_tasks": len(tasks_rows),
        "total_tasks": len(tasks_rows),
        "camera_names": camera_names,
        "features": build_features(
            camera_shapes=camera_shapes,
            fps=int(first_fps),
            state_dim=int(args.state_dim),
            action_dim=int(args.action_dim),
            optional_columns=optional_columns,
        ),
        "splits": {"train": f"0:{len(episodes_meta)}"},
    }
    write_json(out_dir / "meta" / "info.json", info)

    report = {
        "src_dir": str(src_dir),
        "out_dir": str(out_dir),
        "tasks": sorted(task_filter) if task_filter else None,
        "screen_report": str(screen_report_path) if screen_report_path else None,
        "screen_report_num_bad": screen_report_payload.get("num_bad") if screen_report_payload else None,
        "num_skipped_by_screen_report": len(skipped_by_screen_report),
        "skipped_by_screen_report": skipped_by_screen_report,
        "episode_index_mode_requested": args.episode_index_mode,
        "episode_index_mode_resolved": resolved_index_mode,
        "num_discovered_after_filters": len(episodes),
        "num_converted": len(episodes_meta),
        "num_failed": len(failed),
        "include_source_episodes": sorted(include_episode_ids),
        "exclude_source_episodes": sorted(exclude_episode_ids),
        "state_key": args.state_key,
        "action_key": args.action_key,
        "absolute_state_action": True,
    }
    write_json(out_dir / "meta" / "conversion_report.json", report)
    if screen_report_payload is not None:
        screen_output_dir = out_dir / "data_quality"
        write_json(screen_output_dir / "hdf5_screen_report.json", screen_report_payload)
        bad_lines = [
            str(item.get("path", ""))
            for item in screen_report_payload.get("bad_episodes", [])
            if item.get("path")
        ]
        (screen_output_dir / "bad_hdf5_episodes.txt").write_text(
            "\n".join(bad_lines) + ("\n" if bad_lines else ""),
            encoding="utf-8",
        )

    print(f"[DONE] saved LeRobot absolute dataset to: {out_dir}")
    print(f"[INFO] converted episodes: {len(episodes_meta)}; failures: {len(failed)}")
    print("[INFO] observation.state and action were copied as absolute values.")


if __name__ == "__main__":
    main()
