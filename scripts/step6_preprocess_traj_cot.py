"""
Step 6 — Add per-frame traj_cot column in-place to aloha_object_lerobot_gripper_binary.

COT key mapping: episode_index (lerobot index after reindexing) → cot_text_prompts.json key.

Window strategy (WINDOW_SIZE=60):
  frame i uses the COT text anchored at window_start = (i // 60) * 60
  e.g. frames 0-59  → COT of frame 0
       frames 60-119 → COT of frame 60
       ...

Coordinate pairs (x, y) are converted to PaliGemma <loc> tokens:
  (x, y) → <loc{x:04d}><loc{y:04d}>

Writes traj_cot column in-place to each parquet file.

Usage:
  python Datasets/step6_preprocess_traj_cot.py
  python Datasets/step6_preprocess_traj_cot.py --dry_run
"""

import argparse
import json
import pathlib
import re

import pandas as pd
import tqdm

DATASET_DIR    = pathlib.Path(__file__).parent / "correct_object" / "aloha_object_lerobot_gripper_binary"
COT_JSON       = DATASET_DIR / "trajectory_data" / "cot_text_prompts.json"
PARQUET_DIR    = DATASET_DIR / "data" / "chunk-000"
EPISODES_JSONL = DATASET_DIR / "meta" / "episodes.jsonl"

WINDOW_SIZE = 60


def coords_to_loc_tokens(text: str) -> str:
    def replace_coord(m: re.Match) -> str:
        x = min(max(int(m.group(1)), 0), 1023)
        y = min(max(int(m.group(2)), 0), 1023)
        return f"<loc{x:04d}><loc{y:04d}>"
    return re.sub(r"\((\d+),\s*(\d+)\)", replace_coord, text)


def compute_windowed_traj_cots(ep_prompts: dict, total_frames: int) -> list[str]:
    result = []
    for frame_idx in range(total_frames):
        window_start = (frame_idx // WINDOW_SIZE) * WINDOW_SIZE
        key  = str(min(window_start, total_frames - 1))
        text = ep_prompts.get(key, "")
        result.append(coords_to_loc_tokens(text))
    return result


def main(dry_run: bool) -> None:
    cot_data   = json.loads(COT_JSON.read_text())
    all_prompts = cot_data["prompts"]   # str(episode_index) → {str(frame_idx): text}

    episodes = [json.loads(l) for l in EPISODES_JSONL.read_text().splitlines() if l.strip()]

    print(f"Episodes: {len(episodes)}  COT entries: {len(all_prompts)}")
    if dry_run:
        print("[DRY RUN] No files will be modified.")

    errors = []
    for ep in tqdm.tqdm(episodes, desc="episodes"):
        ep_idx  = ep["episode_index"]
        cot_key = str(ep_idx)

        ep_prompts = all_prompts.get(cot_key, {})
        if not ep_prompts:
            errors.append(f"ep {ep_idx}: COT key '{cot_key}' not found")
            continue

        parquet_path = PARQUET_DIR / f"episode_{ep_idx:06d}.parquet"
        df           = pd.read_parquet(parquet_path)
        total_frames = len(df)

        traj_cots = compute_windowed_traj_cots(ep_prompts, total_frames)

        if len(traj_cots) != total_frames:
            errors.append(f"ep {ep_idx}: frame mismatch parquet={total_frames} cot={len(traj_cots)}")
            continue

        if not dry_run:
            df["traj_cot"] = traj_cots
            df.to_parquet(parquet_path, index=False)

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors:
            print(f"  {e}")
    else:
        action = "Would write" if dry_run else "Written"
        print(f"\n{action} traj_cot to {len(episodes)} parquet files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
