"""Merge two LeRobot v2.1 datasets into one.

Primary dataset episodes come first (indices 0..N_primary-1).
Secondary dataset episodes are appended (indices N_primary..N_primary+N_secondary-1).
Secondary task_index values are shifted by the number of primary tasks.
Secondary episode video files are symlinked (not copied).

Usage:
    python scripts/merge_datasets.py \
        --primary   aloha_lerobot_actionequalstate_absinfo_butinstatsgroot_armdelta_gripperabs \
        --secondary aloha_cube \
        --output    aloha_merged_banana_cube
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def load_info(root: Path) -> dict:
    with open(root / "meta" / "info.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(primary_dir: str, secondary_dir: str, output_dir: str) -> None:
    primary  = Path(primary_dir)
    secondary = Path(secondary_dir)
    out      = Path(output_dir)

    if out.exists():
        print(f"Output directory {out} already exists — aborting.")
        return

    # ---- load meta --------------------------------------------------------
    primary_info    = load_info(primary)
    secondary_info  = load_info(secondary)

    primary_tasks   = load_jsonl(primary  / "meta" / "tasks.jsonl")
    secondary_tasks = load_jsonl(secondary / "meta" / "tasks.jsonl")

    primary_episodes   = load_jsonl(primary  / "meta" / "episodes.jsonl")
    secondary_episodes = load_jsonl(secondary / "meta" / "episodes.jsonl")

    N_primary      = primary_info["total_episodes"]
    N_primary_tasks = len(primary_tasks)
    N_secondary    = secondary_info["total_episodes"]

    print(f"Primary   : {N_primary} episodes, {N_primary_tasks} tasks  ({primary})")
    print(f"Secondary : {N_secondary} episodes, {len(secondary_tasks)} tasks  ({secondary})")

    # ---- create output dirs -----------------------------------------------
    vid_base_p = primary  / "videos" / "chunk-000"
    vid_base_s = secondary / "videos" / "chunk-000"
    # use actual directory names (e.g. "observation.images.cam_high")
    cam_names = sorted(d.name for d in vid_base_p.iterdir() if d.is_dir())

    (out / "data" / "chunk-000").mkdir(parents=True)
    (out / "meta").mkdir()
    for cam in cam_names:
        (out / "videos" / "chunk-000" / cam).mkdir(parents=True)

    # ---- 1. tasks.jsonl ---------------------------------------------------
    merged_tasks = list(primary_tasks)  # task_index 0..(N_primary_tasks-1)
    for t in secondary_tasks:
        merged_tasks.append({
            "task_index": t["task_index"] + N_primary_tasks,
            "task": t["task"],
        })
    write_jsonl(out / "meta" / "tasks.jsonl", merged_tasks)
    print(f"tasks.jsonl  : {len(merged_tasks)} tasks")

    # Columns to drop from all parquets: unused by training, cause schema conflicts
    DROP_COLS = {"observation.velocity", "observation.qvel", "base_action", "task"}

    # ---- 2. parquet files — primary (drop unused cols) --------------------
    print("Copying primary parquets ...")
    primary_data_dir = primary / "data" / "chunk-000"
    out_data_dir     = out / "data" / "chunk-000"

    global_frame_offset = 0
    primary_frame_counts = []
    for ep_file in sorted(primary_data_dir.glob("episode_*.parquet")):
        df = pd.read_parquet(ep_file)
        primary_frame_counts.append(len(df))
        cols_to_drop = [c for c in df.columns if c in DROP_COLS]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        df.to_parquet(out_data_dir / ep_file.name, index=False)
        global_frame_offset += len(df)

    total_primary_frames = global_frame_offset

    # ---- 3. parquet files — secondary (offset indices, schema normalise) --
    print("Processing secondary parquets ...")
    secondary_data_dir = secondary / "data" / "chunk-000"

    for ep_file in sorted(secondary_data_dir.glob("episode_*.parquet")):
        df = pd.read_parquet(ep_file)

        old_ep_idx = int(df["episode_index"].iloc[0])
        new_ep_idx = old_ep_idx + N_primary

        # update indices
        df["episode_index"] = new_ep_idx
        df["task_index"]    = df["task_index"] + N_primary_tasks

        # add global `index` column (matches reference format)
        if "index" not in df.columns:
            df.insert(0, "index", global_frame_offset + df["frame_index"].values)

        # drop image path columns (video loaded via path pattern) and unused cols
        img_cols = [c for c in df.columns if c.startswith("observation.images.")]
        extra_cols = [c for c in df.columns if c in DROP_COLS]
        df = df.drop(columns=img_cols + extra_cols)

        # cast float arrays and float scalars from float64 → float32 to match primary schema
        for col in df.columns:
            if col in ("episode_index", "frame_index", "task_index", "index"):
                continue  # keep as int64
            sample = df[col].iloc[0]
            if hasattr(sample, "__len__"):
                # array column: cast each element list
                import numpy as np
                df[col] = df[col].apply(lambda v: np.asarray(v, dtype=np.float32))
            elif df[col].dtype == "float64":
                df[col] = df[col].astype("float32")

        global_frame_offset += len(df)
        out_name = f"episode_{new_ep_idx:06d}.parquet"
        df.to_parquet(out_data_dir / out_name, index=False)

    total_frames = global_frame_offset
    print(f"  total frames: {total_primary_frames} (primary) + "
          f"{total_frames - total_primary_frames} (secondary) = {total_frames}")

    # ---- 4. video symlinks — primary -------------------------------------
    print("Symlinking primary videos ...")
    for cam in cam_names:
        src_dir = vid_base_p / cam
        dst_dir = out / "videos" / "chunk-000" / cam
        for mp4 in sorted(src_dir.glob("episode_*.mp4")):
            (dst_dir / mp4.name).symlink_to(mp4.resolve())

    # ---- 5. video symlinks — secondary (rename episode index) ------------
    print("Symlinking secondary videos ...")
    for cam in cam_names:
        src_dir = vid_base_s / cam
        dst_dir = out / "videos" / "chunk-000" / cam
        for mp4 in sorted(src_dir.glob("episode_*.mp4")):
            old_idx = int(mp4.stem.split("_")[1])
            new_idx = old_idx + N_primary
            new_name = f"episode_{new_idx:06d}.mp4"
            (dst_dir / new_name).symlink_to(mp4.resolve())

    # ---- 6. episodes.jsonl -----------------------------------------------
    # normalise both to: {episode_index, tasks: [task_str], length}
    task_lookup = {t["task_index"]: t["task"] for t in merged_tasks}

    def normalise_episode(ep: dict, ep_idx_offset: int, task_idx_offset: int) -> dict:
        new_idx = ep["episode_index"] + ep_idx_offset
        length  = ep.get("length") or ep.get("num_frames")
        # derive task string list
        if "tasks" in ep:                   # reference format
            raw_tasks = ep["tasks"]
        elif "task_index" in ep:            # aloha_cube format
            raw_tasks = [ep.get("task", task_lookup.get(ep["task_index"] + task_idx_offset, ""))]
        else:
            raw_tasks = []
        return {"episode_index": new_idx, "tasks": raw_tasks, "length": length}

    merged_episodes = (
        [normalise_episode(e, 0,        0)              for e in primary_episodes] +
        [normalise_episode(e, N_primary, N_primary_tasks) for e in secondary_episodes]
    )
    write_jsonl(out / "meta" / "episodes.jsonl", merged_episodes)
    print(f"episodes.jsonl: {len(merged_episodes)} entries")

    # ---- 7. episodes_stats.jsonl ----------------------------------------
    primary_stats   = load_jsonl(primary  / "meta" / "episodes_stats.jsonl")
    secondary_stats = load_jsonl(secondary / "meta" / "episodes_stats.jsonl")

    def normalise_stats(entry: dict, ep_idx_offset: int, task_idx_offset: int,
                        frame_offset: int) -> dict:
        """Convert either format to {episode_index, stats} reference format."""
        new_ep_idx = entry["episode_index"] + ep_idx_offset

        if "stats" in entry:
            # reference format — patch indices in-place
            stats = entry["stats"]
            if "episode_index" in stats:
                stats["episode_index"] = _shift_scalar_stat(stats["episode_index"], ep_idx_offset)
            if "task_index" in stats:
                stats["task_index"] = _shift_scalar_stat(stats["task_index"], task_idx_offset)
            if "index" in stats:
                stats["index"] = _shift_scalar_stat(stats["index"], frame_offset)
            return {"episode_index": new_ep_idx, "stats": stats}
        else:
            # aloha_cube format: keys are feature names at top level
            stats = {}
            skip = {"episode_index", "source_episode_index", "task_index", "task",
                    "length", "num_frames"}
            n_frames = entry.get("length") or entry.get("num_frames", 1)

            for key, val in entry.items():
                if key in skip:
                    continue
                if not isinstance(val, dict):
                    continue
                # val has keys: name, shape, min, max, mean, std
                stats[key] = {
                    "min":   val.get("min", []),
                    "max":   val.get("max", []),
                    "mean":  val.get("mean", []),
                    "std":   val.get("std", []),
                    "count": [n_frames],
                }

            # synthesise index/episode_index/task_index stats
            new_task_idx = entry.get("task_index", 0) + task_idx_offset
            stats["episode_index"] = {"min": [new_ep_idx], "max": [new_ep_idx],
                                      "mean": [float(new_ep_idx)], "std": [0.0],
                                      "count": [n_frames]}
            stats["task_index"]    = {"min": [new_task_idx], "max": [new_task_idx],
                                      "mean": [float(new_task_idx)], "std": [0.0],
                                      "count": [n_frames]}
            return {"episode_index": new_ep_idx, "stats": stats}

    def _shift_scalar_stat(stat: dict, delta: float) -> dict:
        """Add delta to min/max/mean of a scalar stat (std/count unchanged)."""
        out = dict(stat)
        for k in ("min", "max", "mean"):
            if k in out:
                out[k] = [v + delta for v in out[k]]
        return out

    # compute per-episode frame offsets for primary (needed for `index` stat shift)
    primary_ep_frame_start = {}
    offset = 0
    for ep, n in zip(primary_stats, primary_frame_counts):
        primary_ep_frame_start[ep["episode_index"]] = offset
        offset += n

    merged_stats = []
    for entry in primary_stats:
        frame_off = primary_ep_frame_start.get(entry["episode_index"], 0)
        merged_stats.append(normalise_stats(entry, 0, 0, frame_off))

    # secondary frame offsets start after all primary frames
    sec_frame_offset = total_primary_frames
    for entry in secondary_stats:
        merged_stats.append(normalise_stats(entry, N_primary, N_primary_tasks, sec_frame_offset))
        n = entry.get("length") or entry.get("num_frames", 0)
        sec_frame_offset += n

    write_jsonl(out / "meta" / "episodes_stats.jsonl", merged_stats)
    print(f"episodes_stats.jsonl: {len(merged_stats)} entries")

    # ---- 8. info.json ----------------------------------------------------
    # Merge features: primary features are canonical; add secondary-only features.
    # Drop features that are unused by training and stripped from parquets.
    DROP_FEATURES = {"observation.velocity", "observation.qvel", "base_action"}
    merged_features = {k: v for k, v in primary_info["features"].items()
                       if k not in DROP_FEATURES}
    for feat_name, feat_val in secondary_info.get("features", {}).items():
        if feat_name not in merged_features and feat_name not in DROP_FEATURES:
            merged_features[feat_name] = feat_val

    merged_info = {
        **primary_info,
        "total_episodes": N_primary + N_secondary,
        "total_frames":   total_frames,
        "total_videos":   (N_primary + N_secondary) * len(cam_names),
        "splits":         {"train": f"0:{N_primary + N_secondary}"},
        "features":       merged_features,
    }
    # drop fields that are specific to one dataset
    for key in ("repo_id", "source_format", "state_action_encoding"):
        merged_info.pop(key, None)

    with open(out / "meta" / "info.json", "w") as f:
        json.dump(merged_info, f, indent=2)

    # ---- 9. modality.json (copy from primary) ----------------------------
    shutil.copy2(primary / "meta" / "modality.json", out / "meta" / "modality.json")

    # ---- summary ----------------------------------------------------------
    print()
    print("=== Done ===")
    print(f"Output : {out}")
    print(f"Episodes: {N_primary + N_secondary}  (primary {N_primary} + secondary {N_secondary})")
    print(f"Frames  : {total_frames}")
    print(f"Tasks   : {len(merged_tasks)}")
    for t in merged_tasks:
        print(f"  [{t['task_index']}] {t['task']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary",   required=True, help="Primary dataset root dir")
    parser.add_argument("--secondary", required=True, help="Secondary dataset root dir")
    parser.add_argument("--output",    required=True, help="Output merged dataset dir")
    args = parser.parse_args()
    main(args.primary, args.secondary, args.output)
