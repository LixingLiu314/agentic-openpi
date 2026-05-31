"""
================================================================================
Pipeline: put_the_correct_object_into_the_hole 数据处理（共 6 步）

输入数据（两个来源，缺一不可）：
  1. 原始 HDF5 文件（机器人采集的原始数据）：
       Datasets/correct_object/put_the_correct_object_into_the_hole_hdf5/
  2. Hongyu 处理好的 lerobot 格式文件（含 traj_cot 等 trajectory 信息，但存在
     bad episode，需要本 pipeline 清理）：
       Datasets/correct_object/aloha_object_lerobot/

六步流程：
  step1  检查并删除 bad episode，重建连续索引          → 操作 aloha_object_lerobot
  step2  从 HDF5 读取 subtask 标签写入 parquet         → 操作 aloha_object_lerobot
  step3  生成 gripper 二值化数据集                     → 输出 aloha_object_lerobot_gripper_binary
  step4  视频 H264→AV1 重编码，消除软链接              → 操作 aloha_object_lerobot_gripper_binary
  step5  生成 subgoal 视频（cam_high，window=60）       → 操作 aloha_object_lerobot_gripper_binary
  step6  将 traj_cot 按窗口写入 parquet                → 操作 aloha_object_lerobot_gripper_binary
================================================================================

Step 1 — Strict quality check for lerobot-format datasets.

Checks per episode:
  1. NaN / Inf in state and action.
  2. Frame count below minimum (too short to be useful).
  3. Active arm frozen: the arm that should be moving has total range < ACTIVE_ARM_MIN_RANGE.
     This catches episodes where the robot barely moved at all (e.g. recording started while
     the arm was stationary). The previous quick check used a large-jump threshold (> 2 rad)
     which only fires when the robot moves TOO MUCH; it never fires when the robot doesn't
     move enough.
  4. Frozen joint in active arm: within the arm that is supposed to be moving, any individual
     joint whose state OR action never changes (range < FROZEN_JOINT_THRESHOLD) throughout the
     whole episode. The inactive arm's joints are naturally constant and are excluded.
     Up to 1 frozen joint per arm is tolerated (gripper can stay open/closed the whole time).
  5. Large per-frame jump: any joint moves > MAX_JUMP_PER_FRAME rad in a single step
     (catches catastrophic teleport artifacts).

Usage:
  # Check only (dry-run, no changes):
  python check_and_remove_bad_episodes.py --dataset /path/to/lerobot_dataset

  # Delete bad episodes and reindex all remaining episodes:
  python check_and_remove_bad_episodes.py --dataset /path/to/lerobot_dataset --remove
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

# ── tuneable thresholds ────────────────────────────────────────────────────────
MIN_FRAMES = 50               # fewer frames than this → too short
ACTIVE_ARM_MIN_RANGE = 0.15   # rad; at least one arm must exceed this range
FROZEN_JOINT_THRESHOLD = 1e-3 # rad; joint range below this → considered frozen
MAX_FROZEN_IN_ACTIVE_ARM = 1  # allow ≤ this many frozen joints in active arm (e.g. gripper)
MAX_JUMP_PER_FRAME = 1.5      # rad; single-frame jump larger than this → artifact

# joints per arm (aloha-piper dual-arm: 7 joints each, 14 total)
LEFT_ARM_JOINTS  = slice(0, 7)
RIGHT_ARM_JOINTS = slice(7, 14)
# ──────────────────────────────────────────────────────────────────────────────


def check_episode(parquet_path: pathlib.Path) -> list[str]:
    """Return a list of problem strings (empty = OK)."""
    df = pd.read_parquet(parquet_path)
    problems = []

    state  = np.array(df["observation.state"].tolist(), dtype=np.float32)
    action = np.array(df["action"].tolist(), dtype=np.float32)

    # 1. NaN / Inf
    if np.isnan(state).any() or np.isinf(state).any():
        problems.append("NaN/Inf in observation.state")
    if np.isnan(action).any() or np.isinf(action).any():
        problems.append("NaN/Inf in action")

    # 2. Too short
    if len(df) < MIN_FRAMES:
        problems.append(f"too short: {len(df)} frames < {MIN_FRAMES}")

    # 3. At least one arm must be active (supports both bimanual and single-arm tasks)
    #    Both arms frozen simultaneously → bad episode.
    state_range  = state.max(axis=0) - state.min(axis=0)
    action_range = action.max(axis=0) - action.min(axis=0)
    left_max  = float(state_range[LEFT_ARM_JOINTS].max())
    right_max = float(state_range[RIGHT_ARM_JOINTS].max())

    if left_max < ACTIVE_ARM_MIN_RANGE and right_max < ACTIVE_ARM_MIN_RANGE:
        problems.append(
            f"both arms frozen: left max {left_max:.4f} rad, right max {right_max:.4f} rad "
            f"(threshold {ACTIVE_ARM_MIN_RANGE})"
        )

    arm_active = {"left": left_max >= ACTIVE_ARM_MIN_RANGE, "right": right_max >= ACTIVE_ARM_MIN_RANGE}
    for arm_name, arm_slice in [("left", LEFT_ARM_JOINTS), ("right", RIGHT_ARM_JOINTS)]:
        if not arm_active[arm_name]:
            continue  # inactive arm in single-arm task — skip frozen-joint check
        base = arm_slice.start
        frozen_state_joints = [
            base + i for i, r in enumerate(state_range[arm_slice])
            if r < FROZEN_JOINT_THRESHOLD
        ]
        frozen_action_joints = [
            base + i for i, r in enumerate(action_range[arm_slice])
            if r < FROZEN_JOINT_THRESHOLD
        ]
        if len(frozen_state_joints) > MAX_FROZEN_IN_ACTIVE_ARM:
            problems.append(
                f"frozen state joints in {arm_name} arm: {frozen_state_joints} "
                f"(range < {FROZEN_JOINT_THRESHOLD} rad)"
            )
        if len(frozen_action_joints) > MAX_FROZEN_IN_ACTIVE_ARM:
            problems.append(
                f"frozen action joints in {arm_name} arm: {frozen_action_joints} "
                f"(range < {FROZEN_JOINT_THRESHOLD} rad)"
            )

    # 5. Large per-frame jump
    if len(action) > 1:
        jumps = np.abs(np.diff(action, axis=0))
        max_jump = float(jumps.max())
        if max_jump > MAX_JUMP_PER_FRAME:
            loc = np.unravel_index(jumps.argmax(), jumps.shape)
            problems.append(
                f"large action jump: {max_jump:.4f} rad at frame {loc[0]}→{loc[0]+1}, "
                f"joint {loc[1]}"
            )

    return problems


# ── deletion ──────────────────────────────────────────────────────────────────

def _delete_files_for_episode(dataset_dir: pathlib.Path, ep_idx: int) -> None:
    """Delete parquet, video mp4s, and trajectory visualizations for one episode."""
    chunk = ep_idx // 1000
    tag   = f"episode_{ep_idx:06d}"

    p = dataset_dir / "data" / f"chunk-{chunk:03d}" / f"{tag}.parquet"
    if p.exists():
        p.unlink()
        print(f"    deleted {p.relative_to(dataset_dir)}")

    vid_root = dataset_dir / "videos" / f"chunk-{chunk:03d}"
    if vid_root.exists():
        for cam_dir in vid_root.iterdir():
            if cam_dir.is_dir():
                mp4 = cam_dir / f"{tag}.mp4"
                if mp4.exists():
                    mp4.unlink()
                    print(f"    deleted {mp4.relative_to(dataset_dir)}")

    viz_dir = dataset_dir / "trajectory_data" / "visualizations_all"
    if viz_dir.exists():
        for f in viz_dir.glob(f"{tag}*"):
            f.unlink()
            print(f"    deleted {f.relative_to(dataset_dir)}")


def _filter_meta(dataset_dir: pathlib.Path, bad_set: set[int]) -> list[dict]:
    """Remove bad episodes from all meta/traj JSON files. Returns surviving episodes list."""
    meta_dir = dataset_dir / "meta"

    # episodes.jsonl
    ep_path = meta_dir / "episodes.jsonl"
    surviving = []
    with open(ep_path) as f:
        for line in f:
            ep = json.loads(line)
            if ep["episode_index"] not in bad_set:
                surviving.append(ep)
    with open(ep_path, "w") as f:
        for ep in surviving:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    # episodes_stats.jsonl
    stats_path = meta_dir / "episodes_stats.jsonl"
    if stats_path.exists():
        kept = []
        with open(stats_path) as f:
            for line in f:
                s = json.loads(line)
                if s["episode_index"] not in bad_set:
                    kept.append(s)
        with open(stats_path, "w") as f:
            for s in kept:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # trajectory_data/fk_bimanual.json
    fk_path = dataset_dir / "trajectory_data" / "fk_bimanual.json"
    if fk_path.exists():
        fk = json.loads(fk_path.read_text())
        fk["results"] = [r for r in fk["results"] if r["episode_index"] not in bad_set]
        fk_path.write_text(json.dumps(fk, ensure_ascii=False))

    # trajectory_data/cot_text_prompts.json
    cot_path = dataset_dir / "trajectory_data" / "cot_text_prompts.json"
    if cot_path.exists():
        cot = json.loads(cot_path.read_text())
        for key in [str(i) for i in bad_set]:
            cot.get("prompts", {}).pop(key, None)
        cot_path.write_text(json.dumps(cot, ensure_ascii=False))

    return surviving


# ── reindexing ────────────────────────────────────────────────────────────────

def _reindex_dataset(dataset_dir: pathlib.Path, old_indices: list[int]) -> None:
    """
    Rename all episode files from old indices to new contiguous indices (0, 1, 2, …)
    and update every reference in meta and trajectory JSON files.

    old_indices: sorted list of surviving episode indices (may have gaps).
    Files are processed from lowest to highest old index so that renaming-down
    never overwrites a file that hasn't been moved yet.
    """
    old_to_new: dict[int, int] = {old: new for new, old in enumerate(old_indices)}

    # ── rename files ─────────────────────────────────────────────────────────
    for old_idx in sorted(old_to_new):
        new_idx = old_to_new[old_idx]
        if old_idx == new_idx:
            continue

        old_chunk = old_idx // 1000
        new_chunk = new_idx // 1000
        old_tag   = f"episode_{old_idx:06d}"
        new_tag   = f"episode_{new_idx:06d}"

        # parquet: reload to update episode_index + video path strings, then save
        old_pq = dataset_dir / "data" / f"chunk-{old_chunk:03d}" / f"{old_tag}.parquet"
        new_pq = dataset_dir / "data" / f"chunk-{new_chunk:03d}" / f"{new_tag}.parquet"
        if old_pq.exists():
            df = pd.read_parquet(old_pq)
            df["episode_index"] = new_idx
            vid_cols = [c for c in df.columns if c.startswith("observation.images.")]
            for col in vid_cols:
                df[col] = df[col].str.replace(old_tag, new_tag, regex=False)
            new_pq.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(new_pq, index=False)
            old_pq.unlink()

        # videos: scan old chunk dir for all camera subdirs
        old_vid_root = dataset_dir / "videos" / f"chunk-{old_chunk:03d}"
        if old_vid_root.exists():
            for cam_dir in old_vid_root.iterdir():
                if not cam_dir.is_dir():
                    continue
                old_mp4 = cam_dir / f"{old_tag}.mp4"
                if old_mp4.exists():
                    new_mp4 = dataset_dir / "videos" / f"chunk-{new_chunk:03d}" / cam_dir.name / f"{new_tag}.mp4"
                    new_mp4.parent.mkdir(parents=True, exist_ok=True)
                    old_mp4.rename(new_mp4)

        # trajectory visualizations
        viz_dir = dataset_dir / "trajectory_data" / "visualizations_all"
        if viz_dir.exists():
            for old_file in list(viz_dir.glob(f"{old_tag}*")):
                suffix = old_file.name[len(old_tag):]
                old_file.rename(viz_dir / f"{new_tag}{suffix}")

    print(f"  Renamed {sum(1 for o, n in old_to_new.items() if o != n)} episode file groups.")

    # ── update meta/episodes.jsonl ────────────────────────────────────────────
    meta_dir = dataset_dir / "meta"
    ep_path  = meta_dir / "episodes.jsonl"
    episodes = []
    with open(ep_path) as f:
        for line in f:
            ep = json.loads(line)
            ep["episode_index"] = old_to_new[ep["episode_index"]]
            episodes.append(ep)
    episodes.sort(key=lambda e: e["episode_index"])
    with open(ep_path, "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")
    print(f"  Updated meta/episodes.jsonl")

    # ── update meta/episodes_stats.jsonl ──────────────────────────────────────
    stats_path = meta_dir / "episodes_stats.jsonl"
    if stats_path.exists():
        stats = []
        with open(stats_path) as f:
            for line in f:
                s = json.loads(line)
                s["episode_index"] = old_to_new[s["episode_index"]]
                stats.append(s)
        stats.sort(key=lambda s: s["episode_index"])
        with open(stats_path, "w") as f:
            for s in stats:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  Updated meta/episodes_stats.jsonl")

    # ── update trajectory_data/fk_bimanual.json ───────────────────────────────
    fk_path = dataset_dir / "trajectory_data" / "fk_bimanual.json"
    if fk_path.exists():
        fk = json.loads(fk_path.read_text())
        for r in fk["results"]:
            r["episode_index"] = old_to_new[r["episode_index"]]
        fk["results"].sort(key=lambda r: r["episode_index"])
        fk_path.write_text(json.dumps(fk, ensure_ascii=False))
        print(f"  Updated trajectory_data/fk_bimanual.json")

    # ── update trajectory_data/cot_text_prompts.json ──────────────────────────
    cot_path = dataset_dir / "trajectory_data" / "cot_text_prompts.json"
    if cot_path.exists():
        cot = json.loads(cot_path.read_text())
        new_prompts = {
            str(old_to_new[int(k)]): v
            for k, v in cot.get("prompts", {}).items()
            if int(k) in old_to_new
        }
        cot["prompts"] = new_prompts
        cot_path.write_text(json.dumps(cot, ensure_ascii=False))
        print(f"  Updated trajectory_data/cot_text_prompts.json")

    # ── update meta/info.json ─────────────────────────────────────────────────
    info_path = meta_dir / "info.json"
    info      = json.loads(info_path.read_text())
    total_eps    = len(episodes)
    total_frames = sum(ep["num_frames"] for ep in episodes)
    info["num_episodes"]   = total_eps
    info["total_episodes"] = total_eps
    info["num_frames"]     = total_frames
    info["total_frames"]   = total_frames
    info["splits"]         = {"train": f"0:{total_eps}"}
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print(f"  Updated meta/info.json (num_episodes={total_eps}, num_frames={total_frames})")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check quality of lerobot episodes; optionally delete bad ones and reindex."
    )
    parser.add_argument("--dataset", required=True, type=pathlib.Path,
                        help="Path to lerobot dataset root directory")
    parser.add_argument("--remove", action="store_true",
                        help="Delete bad episodes and reindex remaining (default: dry-run)")
    args = parser.parse_args()

    dataset_dir = args.dataset.resolve()
    if not dataset_dir.exists():
        print(f"ERROR: dataset not found: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    parquets = sorted(dataset_dir.glob("data/chunk-*/episode_*.parquet"))
    if not parquets:
        print("No parquet files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Checking {len(parquets)} episodes in {dataset_dir}")
    print(f"Thresholds: min_frames={MIN_FRAMES}, active_arm_min_range={ACTIVE_ARM_MIN_RANGE} rad, "
          f"frozen_joint={FROZEN_JOINT_THRESHOLD} rad, max_jump={MAX_JUMP_PER_FRAME} rad/frame\n")

    bad_set:      set[int]                    = set()
    bad_episodes: list[tuple[int, list[str]]] = []
    all_indices:  list[int]                   = []

    for p in parquets:
        ep_idx = int(pd.read_parquet(p, columns=["episode_index"])["episode_index"].iloc[0])
        all_indices.append(ep_idx)
        problems = check_episode(p)
        if problems:
            bad_set.add(ep_idx)
            bad_episodes.append((ep_idx, problems))
            print(f"  [BAD] Episode {ep_idx}:")
            for prob in problems:
                print(f"        - {prob}")

    print(f"\n{'─'*60}")
    print(f"Result: {len(bad_episodes)} bad episode(s) out of {len(parquets)} checked.")

    if not bad_episodes:
        print("All episodes passed.")
        return

    if not args.remove:
        print("\nDry-run mode. Re-run with --remove to delete and reindex.")
        return

    # ── Step 1: delete files ──────────────────────────────────────────────────
    print("\nDeleting bad episode files...")
    for ep_idx, problems in bad_episodes:
        print(f"  Episode {ep_idx} ({len(problems)} problem(s)):")
        _delete_files_for_episode(dataset_dir, ep_idx)

    # ── Step 2: filter meta JSON files ───────────────────────────────────────
    print("\nUpdating meta (removing bad episodes)...")
    surviving = _filter_meta(dataset_dir, bad_set)
    print(f"  {len(surviving)} episodes remaining.")

    # ── Step 3: reindex ───────────────────────────────────────────────────────
    surviving_old = sorted(ep["episode_index"] for ep in surviving)
    needs_reindex = surviving_old != list(range(len(surviving_old)))
    if needs_reindex:
        print(f"\nReindexing {len(surviving_old)} episodes to fill gaps...")
        _reindex_dataset(dataset_dir, surviving_old)
    else:
        print("\nNo gaps in indices — skipping reindex.")

    print("\nDone.")


if __name__ == "__main__":
    main()
