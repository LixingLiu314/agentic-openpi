#!/usr/bin/env python3
from __future__ import annotations

import copy
from fractions import Fraction
import json
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np
import pandas as pd


VIDEO_COLS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _decode_hdf5_strings(values: Any) -> list[str]:
    decoded = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        elif hasattr(value, "decode"):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def _first_hdf5_dataset(h5_file: Any, keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        clean = key.strip("/")
        if clean in h5_file:
            return h5_file[clean]
        if key in h5_file:
            return h5_file[key]
    return None


def patch_subtasks_from_hdf5(
    dataset_path: Path,
    *,
    subtask_keys: tuple[str, ...] = ("observations/subtask", "observation/subtask", "subtask"),
) -> dict[str, Any]:
    """Patch per-frame subtask strings into converted parquets using episode source_path."""
    import h5py  # type: ignore

    episodes = load_jsonl(dataset_path / "meta" / "episodes.jsonl")
    patched = 0
    skipped = []
    for ep in episodes:
        ep_idx = int(ep["episode_index"])
        source_path = Path(str(ep.get("source_path", ""))).expanduser()
        parquet = dataset_path / "data" / f"chunk-{ep_idx // 1000:03d}" / f"episode_{ep_idx:06d}.parquet"
        if not source_path.exists():
            skipped.append({"episode_index": ep_idx, "reason": "source_hdf5_missing", "source_path": str(source_path)})
            continue
        with h5py.File(source_path, "r") as h5_file:
            subtask_ds = _first_hdf5_dataset(h5_file, subtask_keys)
            if subtask_ds is None:
                skipped.append({"episode_index": ep_idx, "reason": "subtask_dataset_missing", "source_path": str(source_path)})
                continue
            subtasks = _decode_hdf5_strings(subtask_ds[()])

        df = pd.read_parquet(parquet)
        if len(df) != len(subtasks):
            skipped.append(
                {
                    "episode_index": ep_idx,
                    "reason": "frame_count_mismatch",
                    "parquet_frames": len(df),
                    "subtask_frames": len(subtasks),
                    "source_path": str(source_path),
                }
            )
            continue
        df["subtask"] = subtasks
        df.to_parquet(parquet, index=False)
        patched += 1

    info_path = dataset_path / "meta" / "info.json"
    info = read_json(info_path)
    info.setdefault("features", {})["subtask"] = {
        "dtype": "string",
        "shape": [1],
        "names": ["subtask"],
    }
    write_json(info_path, info)

    modality_path = dataset_path / "meta" / "modality.json"
    if modality_path.exists():
        modality = read_json(modality_path)
        modality.setdefault("annotation", {})["subtask"] = {"original_key": "subtask"}
        write_json(modality_path, modality)

    return {"patched": patched, "skipped": skipped}


def binarize_action(
    action: np.ndarray,
    *,
    left_gripper_joint: int,
    right_gripper_joint: int,
    close_threshold: float,
    open_value: float,
    close_value: float,
) -> np.ndarray:
    action = action.copy()
    for dim in (left_gripper_joint, right_gripper_joint):
        if 0 <= dim < action.shape[1]:
            action[:, dim] = np.where(action[:, dim] < close_threshold, close_value, open_value)
    return action


def _vec_stat(arr: np.ndarray) -> dict[str, Any]:
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [int(len(arr))],
    }


def _scalar_stat(arr: np.ndarray) -> dict[str, Any]:
    return {
        "min": [float(arr.min())],
        "max": [float(arr.max())],
        "mean": [float(arr.mean())],
        "std": [float(arr.std())],
        "count": [int(len(arr))],
    }


def _image_placeholder(shape: list[int]) -> dict[str, Any]:
    channels = int(shape[2]) if len(shape) >= 3 else 3
    return {
        "min": [[[0.0]] for _ in range(channels)],
        "max": [[[1.0]] for _ in range(channels)],
        "mean": [[[0.5]] for _ in range(channels)],
        "std": [[[0.25]] for _ in range(channels)],
        "count": [1],
    }


def _build_episode_stats(df: pd.DataFrame, info_features: dict[str, Any]) -> dict[str, Any]:
    scalar_keys = {"timestamp", "frame_index", "episode_index", "task_index"}
    stats: dict[str, Any] = {}
    for feature, meta in info_features.items():
        dtype = meta.get("dtype")
        if dtype == "video":
            stats[feature] = _image_placeholder(meta.get("shape", [1, 1, 3]))
        elif feature in scalar_keys and feature in df.columns:
            stats[feature] = _scalar_stat(df[feature].to_numpy(dtype=float))
        elif dtype == "float32" and feature in df.columns:
            stats[feature] = _vec_stat(np.asarray(df[feature].tolist(), dtype=np.float32))
    return stats


def create_gripper_binary_dataset(
    src: Path,
    dst: Path,
    *,
    repo_id: str,
    overwrite: bool,
    left_gripper_joint: int = 6,
    right_gripper_joint: int = 13,
    close_threshold: float = 0.05,
    open_value: float = 0.09,
    close_value: float = 0.0,
) -> dict[str, Any]:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Gripper-binary dataset already exists: {dst}")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    for name in ("videos", "data_quality", "trajectory_data"):
        src_path = src / name
        if src_path.exists():
            (dst / name).symlink_to(src_path.resolve(), target_is_directory=True)

    info = read_json(src / "meta" / "info.json")
    info_features = info["features"]
    episode_stats_lines: list[str] = []
    parquet_files = sorted((src / "data").glob("chunk-*/episode_*.parquet"))

    for src_parquet in parquet_files:
        dst_parquet = dst / "data" / src_parquet.relative_to(src / "data")
        dst_parquet.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_parquet(src_parquet)
        action = np.asarray(df["action"].tolist(), dtype=np.float32)
        df["action"] = list(
            binarize_action(
                action,
                left_gripper_joint=left_gripper_joint,
                right_gripper_joint=right_gripper_joint,
                close_threshold=close_threshold,
                open_value=open_value,
                close_value=close_value,
            )
        )
        df = df.drop(columns=[column for column in VIDEO_COLS if column in df.columns])
        ep_idx = int(df["episode_index"].iloc[0])
        episode_stats_lines.append(
            json.dumps({"episode_index": ep_idx, "stats": _build_episode_stats(df, info_features)}, ensure_ascii=False)
        )
        df.to_parquet(dst_parquet, index=False)

    dst_meta = dst / "meta"
    dst_meta.mkdir()
    for src_file in sorted((src / "meta").iterdir()):
        dst_file = dst_meta / src_file.name
        if src_file.name == "episodes_stats.jsonl":
            dst_file.write_text("\n".join(episode_stats_lines) + ("\n" if episode_stats_lines else ""), encoding="utf-8")
        elif src_file.suffix == ".json":
            payload = read_json(src_file)
            if src_file.name == "info.json":
                payload["repo_id"] = repo_id
            write_json(dst_file, payload)
        else:
            shutil.copy2(src_file, dst_file)

    return {"src": str(src), "dst": str(dst), "episodes": len(parquet_files)}


def _copy_or_break_symlink_dir(dataset_path: Path, name: str) -> None:
    path = dataset_path / name
    if path.is_symlink():
        source = path.resolve()
        path.unlink()
        shutil.copytree(source, path)


def _codec_metadata_name(codec: str) -> str:
    if codec in {"libsvtav1", "libaom-av1", "av1"}:
        return "av1"
    if codec in {"libx264", "h264"}:
        return "h264"
    return codec.removeprefix("lib")


def _add_stream(container: Any, codec: str, fps: Fraction | int, width: int, height: int, crf: int) -> Any:
    options = {"crf": str(crf)}
    if codec in {"libsvtav1", "libaom-av1", "av1", "libx264", "h264", "mpeg4"}:
        options["g"] = "2"
    if codec in {"libaom-av1", "av1"}:
        # Keep AV1 encoding practical when libsvtav1 is unavailable.
        options.setdefault("cpu-used", "8")
        options.setdefault("row-mt", "1")
    if codec in {"libx264", "h264"}:
        options.setdefault("preset", "veryfast")
    stream = container.add_stream(codec, rate=fps, options=options)
    stream.width = int(width)
    stream.height = int(height)
    stream.pix_fmt = "yuv420p"
    return stream


def reencode_videos(
    dataset_path: Path,
    *,
    codec: str = "libsvtav1",
    crf: int = 30,
) -> dict[str, Any]:
    import av  # type: ignore

    av.logging.set_level(av.logging.ERROR)
    videos_path = dataset_path / "videos"
    if videos_path.is_symlink():
        source = videos_path.resolve()
        videos_path.unlink()
        videos_path.mkdir(parents=True)
        src_videos = sorted(source.glob("chunk-*/*/episode_*.mp4"))
        source_root = source
    else:
        src_videos = sorted(videos_path.glob("chunk-*/*/episode_*.mp4"))
        source_root = videos_path

    failures = []
    for src_video in src_videos:
        rel = src_video.relative_to(source_root)
        dst_video = videos_path / rel
        tmp = dst_video.with_suffix(".tmp.mp4")
        dst_video.parent.mkdir(parents=True, exist_ok=True)
        try:
            with av.open(str(src_video)) as inp:
                in_stream = inp.streams.video[0]
                fps = in_stream.average_rate or Fraction(30, 1)
                with av.open(str(tmp), "w") as out:
                    out_stream = _add_stream(out, codec, fps, in_stream.width, in_stream.height, crf)
                    for frame in inp.decode(in_stream):
                        if frame.format.name != "yuv420p":
                            frame = frame.reformat(format="yuv420p")
                        for packet in out_stream.encode(frame):
                            out.mux(packet)
                    for packet in out_stream.encode():
                        out.mux(packet)
            tmp.replace(dst_video)
        except Exception as exc:  # noqa: BLE001
            if tmp.exists():
                tmp.unlink()
            failures.append({"video": str(src_video), "error": str(exc)})

    if failures:
        sample = "; ".join(f"{Path(item['video']).name}: {item['error']}" for item in failures[:5])
        raise RuntimeError(f"Failed to re-encode {len(failures)} / {len(src_videos)} videos with {codec}: {sample}")

    for name in ("data_quality", "trajectory_data"):
        _copy_or_break_symlink_dir(dataset_path, name)

    info_path = dataset_path / "meta" / "info.json"
    info = read_json(info_path)
    codec_name = _codec_metadata_name(codec)
    for feature in info.get("features", {}).values():
        if feature.get("dtype") == "video":
            feature.setdefault("video_info", {})["video.codec"] = codec_name
            feature.setdefault("info", {})["video.codec"] = codec_name
    write_json(info_path, info)
    return {"videos": len(src_videos), "failures": failures, "codec": codec}


def compute_subgoal_indices(total_frames: int, window: int) -> list[int]:
    window = max(1, int(window))
    return [min((frame // window + 1) * window, total_frames - 1) for frame in range(total_frames)]


def add_subgoal_videos(
    dataset_path: Path,
    *,
    camera: str = "cam_high",
    window: int = 60,
    codec: str = "libsvtav1",
    crf: int = 30,
    overwrite: bool = False,
) -> dict[str, Any]:
    import av  # type: ignore

    av.logging.set_level(av.logging.ERROR)
    key = f"observation.images.{camera}_subgoal"
    src_dir = dataset_path / "videos" / "chunk-000" / f"observation.images.{camera}"
    dst_dir = dataset_path / "videos" / "chunk-000" / key
    parquet_files = sorted((dataset_path / "data").glob("chunk-*/episode_*.parquet"))
    written = 0
    skipped = 0
    failures = []

    for parquet in parquet_files:
        ep_idx = int(parquet.stem.replace("episode_", ""))
        src_video = src_dir / f"episode_{ep_idx:06d}.mp4"
        dst_video = dst_dir / f"episode_{ep_idx:06d}.mp4"
        if dst_video.exists() and not overwrite:
            skipped += 1
            continue
        try:
            df = pd.read_parquet(parquet, columns=["frame_index"])
            total_frames = len(df)
            indices = compute_subgoal_indices(total_frames, window)
            frames = []
            with av.open(str(src_video)) as inp:
                stream = inp.streams.video[0]
                fps = stream.average_rate or Fraction(30, 1)
                for frame in inp.decode(stream):
                    frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) != total_frames:
                raise ValueError(f"video {len(frames)} frames != parquet {total_frames} frames")
            dst_video.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst_video.with_suffix(".tmp.mp4")
            with av.open(str(tmp), "w") as out:
                out_stream = _add_stream(out, codec, fps, frames[0].shape[1], frames[0].shape[0], crf)
                for subgoal_idx in indices:
                    frame = av.VideoFrame.from_ndarray(frames[subgoal_idx], format="rgb24")
                    for packet in out_stream.encode(frame):
                        out.mux(packet)
                for packet in out_stream.encode():
                    out.mux(packet)
            tmp.replace(dst_video)
            written += 1
        except Exception as exc:  # noqa: BLE001
            tmp = dst_video.with_suffix(".tmp.mp4")
            if tmp.exists():
                tmp.unlink()
            failures.append({"episode_index": ep_idx, "error": str(exc)})

    if failures:
        sample = "; ".join(f"episode_{item['episode_index']:06d}: {item['error']}" for item in failures[:5])
        raise RuntimeError(f"Failed to generate {len(failures)} / {len(parquet_files)} subgoal videos with {codec}: {sample}")

    info_path = dataset_path / "meta" / "info.json"
    info = read_json(info_path)
    features = info.setdefault("features", {})
    if key not in features and f"observation.images.{camera}" in features:
        feature = copy.deepcopy(features[f"observation.images.{camera}"])
        codec_name = _codec_metadata_name(codec)
        feature.setdefault("video_info", {})["video.codec"] = codec_name
        feature.setdefault("info", {})["video.codec"] = codec_name
        features[key] = feature
        info["total_videos"] = int(info.get("total_videos", 0)) + len(parquet_files)
        info["num_videos"] = int(info["total_videos"])
    write_json(info_path, info)

    modality_path = dataset_path / "meta" / "modality.json"
    if modality_path.exists():
        modality = read_json(modality_path)
        modality.setdefault("video", {})[f"{camera}_subgoal"] = {"original_key": key}
        write_json(modality_path, modality)

    return {"written": written, "skipped": skipped, "failures": failures, "key": key}


def coords_to_loc_tokens(text: str) -> str:
    def replace_coord(match: re.Match[str]) -> str:
        x = min(max(int(match.group(1)), 0), 1023)
        y = min(max(int(match.group(2)), 0), 1023)
        return f"<loc{x:04d}><loc{y:04d}>"

    return re.sub(r"\((\d+),\s*(\d+)\)", replace_coord, text)


def patch_traj_cot_column(
    dataset_path: Path,
    *,
    cot_json: Path,
    window: int = 60,
    column: str = "traj_cot",
) -> dict[str, Any]:
    cot_data = read_json(cot_json)
    all_prompts = cot_data.get("prompts", cot_data)
    episodes = load_jsonl(dataset_path / "meta" / "episodes.jsonl")
    patched = 0
    errors = []
    for ep in episodes:
        ep_idx = int(ep["episode_index"])
        prompts = all_prompts.get(str(ep_idx), {})
        if not prompts:
            errors.append({"episode_index": ep_idx, "reason": "missing_cot_prompts"})
            continue
        parquet = dataset_path / "data" / f"chunk-{ep_idx // 1000:03d}" / f"episode_{ep_idx:06d}.parquet"
        df = pd.read_parquet(parquet)
        values = []
        for frame_idx in range(len(df)):
            window_start = (frame_idx // max(1, int(window))) * max(1, int(window))
            key = str(min(window_start, len(df) - 1))
            values.append(coords_to_loc_tokens(str(prompts.get(key, ""))))
        df[column] = values
        df.to_parquet(parquet, index=False)
        patched += 1

    info_path = dataset_path / "meta" / "info.json"
    info = read_json(info_path)
    info.setdefault("features", {})[column] = {
        "dtype": "string",
        "shape": [1],
        "names": [column],
    }
    write_json(info_path, info)

    modality_path = dataset_path / "meta" / "modality.json"
    if modality_path.exists():
        modality = read_json(modality_path)
        modality.setdefault("annotation", {})[column] = {"original_key": column}
        write_json(modality_path, modality)

    return {"patched": patched, "errors": errors, "column": column}
