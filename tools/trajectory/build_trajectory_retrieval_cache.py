#!/usr/bin/env python3
# ruff: noqa: E402
"""Build the trajectory-reference image embedding cache used by the eval GUI."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.aloha_real.eval_gui.trajectory_retrieval import DEFAULT_CAMERA_KEY
from examples.aloha_real.eval_gui.trajectory_retrieval import DEFAULT_TRAJ_COLUMN
from examples.aloha_real.eval_gui.trajectory_retrieval import build_retrieval_cache
from examples.aloha_real.eval_gui.trajectory_retrieval import default_cache_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute main-camera image embeddings for fast Top-1 trajectory "
            "reference retrieval in the PyQt eval GUI."
        )
    )
    parser.add_argument("--dataset-root", type=pathlib.Path, required=True, help="Local LeRobot dataset root.")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Output .pt cache path. Defaults to <dataset-root>/trajectory_data/cam_high_traj_reference_cache.pt.",
    )
    parser.add_argument(
        "--camera-key",
        default=DEFAULT_CAMERA_KEY,
        help="Dataset video key to embed, usually observation.images.cam_high.",
    )
    parser.add_argument(
        "--traj-column",
        default=DEFAULT_TRAJ_COLUMN,
        help="Parquet column containing GT trajectory text. Falls back to trajectory_data/cot_text_prompts.json.",
    )
    parser.add_argument(
        "--cot-json",
        type=pathlib.Path,
        default=None,
        help="Optional cot_text_prompts.json path if the parquet does not contain --traj-column.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Embed every Nth frame. Keep 1 to search all main-camera frames.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-frames", type=int, default=None, help="Debug limit; omit for the full dataset.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output = args.output or default_cache_path(args.dataset_root)
    summary = build_retrieval_cache(
        args.dataset_root,
        output_path=output,
        camera_key=args.camera_key,
        traj_column=args.traj_column,
        cot_json_path=args.cot_json,
        frame_stride=args.frame_stride,
        batch_size=args.batch_size,
        max_frames=args.max_frames,
        progress=not args.quiet,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
