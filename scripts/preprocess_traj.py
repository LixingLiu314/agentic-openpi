"""
Offline trajectory COT preprocessing for the aloha banana two-tasks dataset.

For each frame, assigns a per-frame traj_cot string derived from cot_text_prompts.json
using a fixed window strategy (WINDOW_SIZE=60):
  frame 0-59   → cot text of frame 0
  frame 60-119 → cot text of frame 60
  frame 120-179→ cot text of frame 120  ...

Coordinate pairs (x, y) in the COT text (0-1000 range) are converted to PaliGemma
<loc> special tokens: (x, y) → <loc{x:04d}><loc{y:04d}>, which the SentencePiece
tokenizer encodes as single tokens (IDs 256000-257023).

Usage
-----
  uv run python scripts/preprocess_traj.py

  # custom output directory
  uv run python scripts/preprocess_traj.py --output_dir /path/to/dir

After running, create the HuggingFace cache symlink:
  ln -sfn <output_dir> ~/.cache/huggingface/lerobot/lixing/aloha_banana_traj

Then start training:
  uv run python scripts/train_pytorch.py pi05_aloha_banana_traj --exp_name <name>
"""

import argparse
import json
import pathlib
import re

import pandas as pd

DATASET_DIR = pathlib.Path(
    "/media/raid/workspace/surongpeng/ws_lixing/agentic-openpi"
    "/aloha_lerobot_actionequalstate_absinfo_butinstatsgroot_armdelta_gripperabs"
)
COT_JSON = pathlib.Path(
    "/media/raid/workspace/surongpeng/ws_lixing/agentic-openpi"
    "/banana_traj_origin_data/trajectory_data/cot_text_prompts.json"
)
PARQUET_DIR = DATASET_DIR / "data" / "chunk-000"

WINDOW_SIZE = 60
NUM_EPISODES = 391  # video episodes 0-390

REPO_ID = "lixing/aloha_banana_traj"
TRAIN_CONFIG = "pi05_aloha_banana_traj"
DEFAULT_OUTPUT = DATASET_DIR.parent / "aloha_lerobot_traj_cot"


def coords_to_loc_tokens(text: str) -> str:
    """Replace '(x, y)' coordinate pairs with PaliGemma <loc> special tokens."""
    def replace_coord(m: re.Match) -> str:
        x = min(max(int(m.group(1)), 0), 1023)
        y = min(max(int(m.group(2)), 0), 1023)
        return f"<loc{x:04d}><loc{y:04d}>"
    return re.sub(r"\((\d+),\s*(\d+)\)", replace_coord, text)


def compute_windowed_traj_cots(ep_prompts: dict, total_frames: int) -> list[str]:
    """
    Window strategy: all frames in a window of WINDOW_SIZE share the COT text
    of the window's first frame.
    """
    result = []
    for frame_idx in range(total_frames):
        window_start = (frame_idx // WINDOW_SIZE) * WINDOW_SIZE
        # Clamp to last available frame if window_start exceeds episode length.
        key = str(min(window_start, total_frames - 1))
        text = ep_prompts.get(key, "")
        result.append(coords_to_loc_tokens(text))
    return result


def main(output_dir: pathlib.Path) -> None:
    with open(COT_JSON) as f:
        cot_data = json.load(f)
    all_prompts = cot_data["prompts"]  # dict: str(ep_idx) -> dict(str(frame_idx) -> text)

    out_parquet_dir = output_dir / "data" / "chunk-000"
    out_parquet_dir.mkdir(parents=True, exist_ok=True)

    # Symlink meta/ and videos/ so LeRobot sees a complete dataset.
    for name in ("meta", "videos"):
        link = output_dir / name
        if not link.exists():
            link.symlink_to(DATASET_DIR / name)

    errors = []
    for ep_idx in range(NUM_EPISODES):
        ep_prompts = all_prompts.get(str(ep_idx), all_prompts.get(ep_idx, {}))

        parquet_path = PARQUET_DIR / f"episode_{ep_idx:06d}.parquet"
        df = pd.read_parquet(parquet_path)
        total_frames = len(df)

        traj_cots = compute_windowed_traj_cots(ep_prompts, total_frames)
        if len(traj_cots) != total_frames:
            errors.append(f"Episode {ep_idx}: frame count mismatch ({total_frames} vs {len(traj_cots)})")
            continue

        df["traj_cot"] = traj_cots
        df.to_parquet(out_parquet_dir / f"episode_{ep_idx:06d}.parquet", index=False)

        if ep_idx % 50 == 0:
            print(f"  episode {ep_idx:3d}/{NUM_EPISODES - 1}")

    if errors:
        print("\nERRORS:")
        for e in errors:
            print(f"  {e}")
    else:
        print(f"\nAll {NUM_EPISODES} episodes processed successfully.")

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
