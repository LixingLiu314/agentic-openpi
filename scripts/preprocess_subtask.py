"""
Offline subtask preprocessing for the aloha banana two-tasks dataset.

Background
----------
The original annotation files (subtask_annotations/) use the old episode numbering
that includes episodes 248-255 (which were later removed from the video/parquet data).
As a result, annotation 256 corresponds to video episode 248, and so on (+8 offset).

This script:
  1. Writes renumbered annotation JSON files to OUTPUT_ANN_DIR
     (old idx 256-398 → new idx 248-390, episodes 0-247 unchanged).
  2. For each episode, computes a per-frame subtask label using a sliding window
     of WINDOW_SIZE=45 frames:
       frame 0-44   → subtask at frame 0
       frame 45-89  → subtask at frame 45
       ...
  3. Writes new parquet files (with an added 'subtask' column) to OUTPUT_PARQUET_DIR.
  4. Creates symlinks for meta/ and videos/ so the output forms a valid LeRobot dataset.

Usage
-----
  uv run python scripts/preprocess_subtask.py [--output_dir PATH]

After running, create the HuggingFace cache symlink and start training:
  ln -sfn <output_dir> ~/.cache/huggingface/lerobot/lixing/aloha_banana_subtask
  uv run python scripts/train_pytorch.py pi05_aloha_banana_subtask --exp_name <name>
"""

import argparse
import json
import os
import pathlib

import pandas as pd

DATASET_DIR = pathlib.Path(
    "/media/raid/workspace/surongpeng/ws_lixing/agentic-openpi"
    "/aloha_lerobot_actionequalstate_absinfo_butinstatsgroot_armdelta_gripperabs"
)
ANN_DIR = DATASET_DIR / "subtask_banana_two_tasks" / "subtask_annotations"
PARQUET_DIR = DATASET_DIR / "data" / "chunk-000"

WINDOW_SIZE = 45
NUM_EPISODES = 391  # video episodes 0-390


def ann_path_for_episode(ep_idx: int) -> pathlib.Path:
    """Return the original annotation file for a given video episode index."""
    old_idx = ep_idx if ep_idx < 248 else ep_idx + 8
    return ANN_DIR / f"annotation_episode_{old_idx:06d}.json"


def subtask_at_frame(segments: list[dict], frame_idx: int) -> str:
    """Return the subtask label for the segment that contains frame_idx."""
    for seg in segments:
        if seg["start_frame"] <= frame_idx <= seg["end_frame"]:
            return seg["label"]
    return segments[-1]["label"]


def compute_windowed_subtasks(segments: list[dict], total_frames: int) -> list[str]:
    """
    For each frame, find the window start (floor(frame / WINDOW_SIZE) * WINDOW_SIZE)
    and return the subtask label at that window start.
    """
    subtasks = []
    for frame_idx in range(total_frames):
        window_start = (frame_idx // WINDOW_SIZE) * WINDOW_SIZE
        subtasks.append(subtask_at_frame(segments, window_start))
    return subtasks


def main(output_dir: pathlib.Path) -> None:
    out_ann_dir = output_dir / "subtask_annotations_renumbered"
    out_parquet_dir = output_dir / "data" / "chunk-000"
    out_ann_dir.mkdir(parents=True, exist_ok=True)
    out_parquet_dir.mkdir(parents=True, exist_ok=True)

    # Symlink meta/ and videos/ so LeRobot sees a complete dataset.
    for name in ("meta", "videos"):
        link = output_dir / name
        if not link.exists():
            link.symlink_to(DATASET_DIR / name)

    errors = []
    for ep_idx in range(NUM_EPISODES):
        ann_path = ann_path_for_episode(ep_idx)
        with open(ann_path) as f:
            ann = json.load(f)

        segments = ann["subtask_segments"]
        total_frames = ann["total_frames"]

        # Write renumbered annotation.
        new_ann = dict(ann)
        new_ann["episode_idx"] = ep_idx
        out_ann_path = out_ann_dir / f"annotation_episode_{ep_idx:06d}.json"
        with open(out_ann_path, "w") as f:
            json.dump(new_ann, f, indent=2)

        # Load parquet and validate frame count.
        parquet_path = PARQUET_DIR / f"episode_{ep_idx:06d}.parquet"
        df = pd.read_parquet(parquet_path)
        if len(df) != total_frames:
            errors.append(
                f"Episode {ep_idx}: parquet has {len(df)} frames, annotation says {total_frames}"
            )
            continue

        # Add windowed subtask column and save.
        df["subtask"] = compute_windowed_subtasks(segments, total_frames)
        out_parquet_path = out_parquet_dir / f"episode_{ep_idx:06d}.parquet"
        df.to_parquet(out_parquet_path, index=False)

        if ep_idx % 50 == 0:
            print(f"  episode {ep_idx:3d}/{NUM_EPISODES - 1}")

    if errors:
        print("\nERRORS (frame count mismatch):")
        for e in errors:
            print(f"  {e}")
    else:
        print(f"\nAll {NUM_EPISODES} episodes processed successfully.")

    print(f"\nOutput directory : {output_dir}")
    print(f"  annotations    : {out_ann_dir}")
    print(f"  parquets       : {out_parquet_dir}")
    print(f"  meta (symlink) : {output_dir / 'meta'}")
    print(f"  videos (symlink): {output_dir / 'videos'}")
    print()
    print("Next steps:")
    print(f"  ln -sfn {output_dir} ~/.cache/huggingface/lerobot/lixing/aloha_banana_subtask")
    print("  uv run python scripts/train_pytorch.py pi05_aloha_banana_subtask --exp_name <name>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=DATASET_DIR.parent / "aloha_lerobot_subtask",
        help="Directory where the new dataset will be written.",
    )
    args = parser.parse_args()
    main(args.output_dir)
