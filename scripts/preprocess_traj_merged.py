"""
Offline trajectory COT preprocessing for the merged banana+cube dataset.

Episode mapping in aloha_merged_banana_cube:
  episodes 0-390   (391 eps) → banana traj COT
  episodes 391-765 (375 eps) → cube traj COT  (cube index = merged_index - 391)

For each frame, assigns a per-frame traj_cot string using a fixed window strategy
(WINDOW_SIZE=60): all frames in a window share the COT text of the window's first frame.

Coordinate pairs (x, y) in the COT text (0-1000 range) are converted to PaliGemma
<loc> special tokens: (x, y) → <loc{x:04d}><loc{y:04d}>.

Usage
-----
  uv run python scripts/preprocess_traj_merged.py

  # custom output directory
  uv run python scripts/preprocess_traj_merged.py --output_dir /path/to/dir

After running, create the HuggingFace cache symlink:
  ln -sfn <output_dir> ~/.cache/huggingface/lerobot/lixing/aloha_merged_banana_cube_traj

Then start training:
  uv run python scripts/train_pytorch.py pi05_aloha_banana_cube_traj --exp_name <name>
"""

import argparse
import json
import pathlib
import re

import pandas as pd

PROJECT_ROOT = pathlib.Path("/media/raid/workspace/surongpeng/ws_lixing/agentic-openpi")

DATASET_DIR = PROJECT_ROOT / "aloha_merged_banana_cube"
PARQUET_DIR = DATASET_DIR / "data" / "chunk-000"

BANANA_COT_JSON = PROJECT_ROOT / "traj_data/banana_traj_origin_data/trajectory_data/cot_text_prompts.json"
CUBE_COT_JSON = PROJECT_ROOT / "traj_data/cube_origin_trajectory_data/cot_text_prompts.json"

WINDOW_SIZE = 60
BANANA_NUM_EPISODES = 391   # merged episodes 0-390
CUBE_NUM_EPISODES = 375     # merged episodes 391-765
TOTAL_EPISODES = BANANA_NUM_EPISODES + CUBE_NUM_EPISODES  # 766

REPO_ID = "lixing/aloha_merged_banana_cube_traj"
TRAIN_CONFIG = "pi05_aloha_banana_cube_traj"
DEFAULT_OUTPUT = PROJECT_ROOT / "aloha_merged_banana_cube_traj"


def coords_to_loc_tokens(text: str) -> str:
    """Replace '(x, y)' coordinate pairs with PaliGemma <loc> special tokens."""
    def replace_coord(m: re.Match) -> str:
        x = min(max(int(m.group(1)), 0), 1023)
        y = min(max(int(m.group(2)), 0), 1023)
        return f"<loc{x:04d}><loc{y:04d}>"
    return re.sub(r"\((\d+),\s*(\d+)\)", replace_coord, text)


def compute_windowed_traj_cots(ep_prompts: dict, total_frames: int) -> list[str]:
    result = []
    for frame_idx in range(total_frames):
        window_start = (frame_idx // WINDOW_SIZE) * WINDOW_SIZE
        key = str(min(window_start, total_frames - 1))
        text = ep_prompts.get(key, "")
        result.append(coords_to_loc_tokens(text))
    return result


def main(output_dir: pathlib.Path) -> None:
    with open(BANANA_COT_JSON) as f:
        banana_prompts = json.load(f)["prompts"]
    with open(CUBE_COT_JSON) as f:
        cube_prompts = json.load(f)["prompts"]

    out_parquet_dir = output_dir / "data" / "chunk-000"
    out_parquet_dir.mkdir(parents=True, exist_ok=True)

    for name in ("meta", "videos"):
        link = output_dir / name
        if not link.exists():
            link.symlink_to(DATASET_DIR / name)

    errors = []
    for merged_idx in range(TOTAL_EPISODES):
        if merged_idx < BANANA_NUM_EPISODES:
            ep_prompts = banana_prompts.get(str(merged_idx), {})
            source = "banana"
        else:
            cube_idx = merged_idx - BANANA_NUM_EPISODES
            ep_prompts = cube_prompts.get(str(cube_idx), {})
            source = f"cube(orig={cube_idx})"

        parquet_path = PARQUET_DIR / f"episode_{merged_idx:06d}.parquet"
        df = pd.read_parquet(parquet_path)
        total_frames = len(df)

        traj_cots = compute_windowed_traj_cots(ep_prompts, total_frames)
        if len(traj_cots) != total_frames:
            errors.append(f"Episode {merged_idx} [{source}]: frame count mismatch ({total_frames} vs {len(traj_cots)})")
            continue

        df["traj_cot"] = traj_cots
        df.to_parquet(out_parquet_dir / f"episode_{merged_idx:06d}.parquet", index=False)

        if merged_idx % 50 == 0:
            print(f"  episode {merged_idx:3d}/{TOTAL_EPISODES - 1} [{source}]")

    if errors:
        print("\nERRORS:")
        for e in errors:
            print(f"  {e}")
    else:
        print(f"\nAll {TOTAL_EPISODES} episodes processed successfully.")

    hf_cache = (
        pathlib.Path("/media/raid/workspace/surongpeng/ws_lixing/.cache/huggingface/lerobot")
        / REPO_ID
    )
    print(f"\nOutput directory: {output_dir}")
    print("\nNext steps:")
    print(f"  ln -sfn {output_dir} {hf_cache}")
    print(f"  uv run python scripts/train_pytorch.py {TRAIN_CONFIG} --exp_name <name>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory. Defaults to {DEFAULT_OUTPUT}.",
    )
    args = parser.parse_args()
    main(args.output_dir)
