#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm


_G_EP_TO_RESULT: dict[int, dict] = {}
_G_DATASET_PATH = ""
_G_CHUNKS_SIZE = 1000


def perpendicular_distance(point, line_start, line_end) -> float:
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


def rdp_simplify(points, epsilon: float) -> list[int]:
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


def adaptive_gripper_threshold(values: np.ndarray, fallback: float = 0.5) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return fallback
    v_min = float(np.min(finite))
    v_max = float(np.max(finite))
    if v_max - v_min < 1e-4:
        return (v_min + v_max) * 0.5
    return (v_min + v_max) * 0.5


def detect_gripper_changes(gripper_values, threshold: float | None = None, min_gap: int = 2):
    if len(gripper_values) == 0:
        return []
    values = np.asarray(gripper_values, dtype=np.float64)
    threshold = adaptive_gripper_threshold(values) if threshold is None else float(threshold)
    changes = []
    prev = "open" if values[0] >= threshold else "closed"
    last_change = -min_gap
    for idx in range(1, len(values)):
        curr = "open" if values[idx] >= threshold else "closed"
        if curr != prev and idx - last_change >= min_gap:
            changes.append((idx, "close gripper" if curr == "closed" else "open gripper"))
            prev = curr
            last_change = idx
    return changes


def build_cot_prompt(
    future_points,
    gripper_changes,
    rdp_epsilon: float = 3.0,
    image_size: int = 256,
    normalize: bool = True,
    min_dist_threshold: float = 20.0,
    max_tokens: int = 10,
) -> str:
    if not future_points:
        return ""

    frame_to_pt = {frame: (x, y) for frame, x, y in future_points}
    frames = [frame for frame, _, _ in future_points]
    start_frame = frames[0]
    end_frame = frames[-1]
    valid_gripper_changes = {
        frame: action for frame, action in gripper_changes if start_frame <= frame <= end_frame
    }
    anchor_frames = sorted(set([start_frame, end_frame] + list(valid_gripper_changes)))

    kept_frames: set[int] = set()
    for anchor_idx in range(len(anchor_frames) - 1):
        seg_start = anchor_frames[anchor_idx]
        seg_end = anchor_frames[anchor_idx + 1]
        seg_frames = [frame for frame in frames if seg_start <= frame <= seg_end]
        if not seg_frames:
            continue
        seg_pts = [frame_to_pt[frame] for frame in seg_frames]
        for keep_idx in rdp_simplify(seg_pts, rdp_epsilon):
            kept_frames.add(seg_frames[keep_idx])

    if len(anchor_frames) == 1:
        kept_frames.add(anchor_frames[0])

    tokens = []
    last_pt = None
    for frame in sorted(kept_frames):
        x, y = frame_to_pt[frame]
        is_gripper_change = frame in valid_gripper_changes

        skip_coord = False
        if last_pt is not None:
            dist = np.linalg.norm(np.asarray([x, y]) - np.asarray(last_pt))
            if dist < min_dist_threshold:
                skip_coord = True

        if not skip_coord:
            if normalize:
                x_out = max(0, min(1000, int(round(x / image_size * 1000))))
                y_out = max(0, min(1000, int(round(y / image_size * 1000))))
            else:
                x_out, y_out = int(round(x)), int(round(y))
            tokens.append(f"({x_out}, {y_out})")
            last_pt = (x, y)

        if is_gripper_change:
            tokens.append(valid_gripper_changes[frame])

    tokens = tokens[:max_tokens]
    if not tokens:
        return ""

    parts = []
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


def get_trajectory_start_frame(current_frame: int, interval: int) -> int:
    return (current_frame // interval) * interval


def episode_parquet_path(dataset_path: Path, episode_index: int, chunks_size: int) -> Path:
    chunk_index = episode_index // chunks_size
    return dataset_path / "data" / f"chunk-{chunk_index:03d}" / f"episode_{episode_index:06d}.parquet"


def load_actions(dataset_path: Path, episode_index: int, chunks_size: int) -> np.ndarray:
    table = pq.read_table(episode_parquet_path(dataset_path, episode_index, chunks_size), columns=["action"])
    return np.asarray(table["action"].to_pylist(), dtype=np.float64)


def frame_points_for_start(all_frame_eef: dict, start_frame: int, num_frames: int):
    future_points = []
    for frame in range(start_frame, num_frames):
        value = all_frame_eef.get(str(frame))
        if value is not None:
            future_points.append((frame, float(value[0]), float(value[1])))
    return future_points


def process_arm_prompts(
    all_frame_eef: dict,
    gripper_values,
    num_frames: int,
    rdp_epsilon: float,
    image_size: int,
    interval: int,
    gripper_threshold: float | None,
    min_dist_threshold: float,
    max_tokens: int,
) -> dict[str, str]:
    gripper_changes = detect_gripper_changes(gripper_values, threshold=gripper_threshold)
    prompts_by_start: dict[int, str] = {}

    for current_frame in range(0, num_frames, interval):
        start_frame = get_trajectory_start_frame(current_frame, interval)
        future_points = frame_points_for_start(all_frame_eef, start_frame, num_frames)
        future_gripper_changes = [(frame, action) for frame, action in gripper_changes if frame >= start_frame]
        prompts_by_start[start_frame] = build_cot_prompt(
            future_points=future_points,
            gripper_changes=future_gripper_changes,
            rdp_epsilon=rdp_epsilon,
            image_size=image_size,
            min_dist_threshold=min_dist_threshold,
            max_tokens=max_tokens,
        )

    frame_prompts = {}
    for current_frame in range(num_frames):
        start_frame = get_trajectory_start_frame(current_frame, interval)
        frame_prompts[str(current_frame)] = prompts_by_start.get(start_frame, "")
    return frame_prompts


def combine_prompts(left_prompt: str, right_prompt: str, empty_prompt: str) -> str:
    left = left_prompt if left_prompt else empty_prompt
    right = right_prompt if right_prompt else empty_prompt
    return f"Left: {left} Right: {right}"


def _worker(args_tuple):
    (
        episode_index,
        rdp_epsilon,
        image_size,
        interval,
        left_gripper_dim,
        right_gripper_dim,
        left_gripper_threshold,
        right_gripper_threshold,
        min_dist_threshold,
        max_tokens,
        empty_prompt,
    ) = args_tuple

    result = _G_EP_TO_RESULT[episode_index]
    dataset_path = Path(_G_DATASET_PATH)
    actions = load_actions(dataset_path, episode_index, _G_CHUNKS_SIZE)
    num_frames = int(result["num_frames"])

    by_arm = result.get("all_frame_eef_positions_by_arm")
    if not by_arm:
        primary_arm = result.get("primary_arm", "right")
        by_arm = {primary_arm: result["all_frame_eef_positions"]}

    left_positions = by_arm.get("left", {})
    right_positions = by_arm.get("right", {})

    left_prompts = process_arm_prompts(
        all_frame_eef=left_positions,
        gripper_values=actions[:, left_gripper_dim],
        num_frames=num_frames,
        rdp_epsilon=rdp_epsilon,
        image_size=image_size,
        interval=interval,
        gripper_threshold=left_gripper_threshold,
        min_dist_threshold=min_dist_threshold,
        max_tokens=max_tokens,
    )
    right_prompts = process_arm_prompts(
        all_frame_eef=right_positions,
        gripper_values=actions[:, right_gripper_dim],
        num_frames=num_frames,
        rdp_epsilon=rdp_epsilon,
        image_size=image_size,
        interval=interval,
        gripper_threshold=right_gripper_threshold,
        min_dist_threshold=min_dist_threshold,
        max_tokens=max_tokens,
    )

    combined = {
        str(frame): combine_prompts(left_prompts[str(frame)], right_prompts[str(frame)], empty_prompt)
        for frame in range(num_frames)
    }
    return episode_index, combined


def parse_optional_float(value: str) -> float | None:
    if value.lower() in {"none", "adaptive", "auto"}:
        return None
    return float(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate bimanual Left/Right CoT prompts from FK trajectory JSON.")
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--traj_json", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--rdp_epsilon", type=float, default=3.0)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--interval", type=int, default=6)
    parser.add_argument("--left_gripper_dim", type=int, default=6)
    parser.add_argument("--right_gripper_dim", type=int, default=13)
    parser.add_argument(
        "--left_gripper_threshold",
        type=parse_optional_float,
        default=None,
        help="Use 'auto'/'none' for per-episode midpoint threshold.",
    )
    parser.add_argument(
        "--right_gripper_threshold",
        type=parse_optional_float,
        default=None,
        help="Use 'auto'/'none' for per-episode midpoint threshold.",
    )
    parser.add_argument("--min_dist_threshold", type=float, default=20.0)
    parser.add_argument("--max_tokens", type=int, default=10)
    parser.add_argument("--empty_prompt", default="stay.")
    parser.add_argument("--num_workers", type=int, default=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    dataset_path = args.dataset_path.expanduser()
    traj_json = args.traj_json.expanduser()
    if args.output_path is None:
        args.output_path = dataset_path / "trajectory_data" / "cot_text_prompts.json"
    output_path = args.output_path.expanduser()

    with traj_json.open("r", encoding="utf-8") as f:
        traj_data = json.load(f)
    results = traj_data.get("results", [])
    ep_to_result = {int(result["episode_index"]): result for result in results}

    info_path = dataset_path / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    global _G_EP_TO_RESULT, _G_DATASET_PATH, _G_CHUNKS_SIZE
    _G_EP_TO_RESULT = ep_to_result
    _G_DATASET_PATH = str(dataset_path)
    _G_CHUNKS_SIZE = int(info.get("chunks_size", 1000))

    tasks = [
        (
            episode_index,
            args.rdp_epsilon,
            args.image_size,
            args.interval,
            args.left_gripper_dim,
            args.right_gripper_dim,
            args.left_gripper_threshold,
            args.right_gripper_threshold,
            args.min_dist_threshold,
            args.max_tokens,
            args.empty_prompt,
        )
        for episode_index in sorted(ep_to_result)
    ]

    num_workers = args.num_workers or multiprocessing.cpu_count()
    ctx = multiprocessing.get_context("fork")
    cot_prompts = {}
    failed_episodes = []

    print(f"Generating bimanual CoT prompts for {len(tasks)} episodes with {num_workers} workers")
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        futures = {executor.submit(_worker, task): task[0] for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating CoT prompts"):
            episode_index = futures[future]
            try:
                episode_index_ret, prompts = future.result()
                cot_prompts[str(episode_index_ret)] = prompts
            except Exception as exc:
                failed_episodes.append({"episode_index": int(episode_index), "error": str(exc)})
                print(f"Failed episode {episode_index}: {exc}")

    payload = {
        "metadata": {
            "format": "bimanual_text",
            "prompt_template": "Left: {left_prompt} Right: {right_prompt}",
            "rdp_epsilon": args.rdp_epsilon,
            "image_size": args.image_size,
            "interval": args.interval,
            "left_gripper_dim": args.left_gripper_dim,
            "right_gripper_dim": args.right_gripper_dim,
            "left_gripper_threshold": args.left_gripper_threshold if args.left_gripper_threshold is not None else "adaptive_midpoint",
            "right_gripper_threshold": args.right_gripper_threshold if args.right_gripper_threshold is not None else "adaptive_midpoint",
            "min_dist_threshold": args.min_dist_threshold,
            "max_tokens_per_arm": args.max_tokens,
            "empty_prompt": args.empty_prompt,
            "total_episodes": len(cot_prompts),
            "failed_episodes": failed_episodes,
            "source_traj_json": str(traj_json),
            "coordinate_range": "0-1000 normalized per arm",
        },
        "prompts": cot_prompts,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved bimanual CoT prompts to {output_path}")
    print(f"Successful episodes: {len(cot_prompts)}, failed: {len(failed_episodes)}")

    for episode in list(sorted(cot_prompts, key=int))[:2]:
        print(f"\nEpisode {episode}:")
        for frame in ["0", "5", "10", "30"]:
            if frame in cot_prompts[episode]:
                print(f"  Frame {frame}: {cot_prompts[episode][frame][:220]}")


if __name__ == "__main__":
    main()
