"""
Offline subgoal image preprocessing for the merged banana+cube dataset.

For each episode, new video files are created for subgoal camera streams.
The subgoal for frame i is the frame at index min((i // SUBGOAL_WINDOW + 1) * SUBGOAL_WINDOW, last_frame),
so frames 0-29 all share frame 30 as their subgoal, frames 30-59 share frame 60, etc.

Episodes 0-390 (banana) reuse precomputed subgoal videos from aloha_lerobot_subgoal_{version}
via per-file symlinks. Only episodes 391-765 (cube) are freshly encoded.

Two output versions are supported:
  base  -- only cam_high_subgoal  (new dataset: aloha_merged_banana_cube_subgoal_base)
  all   -- cam_high_subgoal + cam_left_wrist_subgoal + cam_right_wrist_subgoal
           (new dataset: aloha_merged_banana_cube_subgoal_all)

Usage:
  uv run python scripts/preprocess_subgoal_merged.py --version base
  uv run python scripts/preprocess_subgoal_merged.py --version all
  uv run python scripts/preprocess_subgoal_merged.py --version base --output_dir /custom/path

After running, symlink the output directory into the HuggingFace cache:
  ln -sfn <output_dir> $HF_HOME/lerobot/lixing/aloha_merged_banana_cube_subgoal_base

Then start training:
  uv run python scripts/train_pytorch.py pi05_aloha_banana_cube_subgoal_base --exp_name <name>
"""

import argparse
import copy
import json
import pathlib

import av
import pandas as pd

PROJECT_ROOT = pathlib.Path("/media/raid/workspace/surongpeng/ws_lixing/agentic-openpi")

DATASET_DIR = PROJECT_ROOT / "aloha_merged_banana_cube"
PARQUET_DIR = DATASET_DIR / "data" / "chunk-000"
VIDEO_DIR = DATASET_DIR / "videos" / "chunk-000"

# Precomputed banana subgoal datasets (reused via symlinks for episodes 0-390)
BANANA_SUBGOAL_DIRS = {
    "base": PROJECT_ROOT / "aloha_lerobot_subgoal_base",
    "all": PROJECT_ROOT / "aloha_lerobot_subgoal_all",
}

SUBGOAL_WINDOW = 30
BANANA_NUM_EPISODES = 391   # merged episodes 0-390, reuse from banana subgoal
TOTAL_EPISODES = 766

SOURCE_CAMERAS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]

VERSION_CAMERAS = {
    "base": ["cam_high"],
    "all": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
}

DEFAULT_OUTPUT_DIRS = {
    "base": PROJECT_ROOT / "aloha_merged_banana_cube_subgoal_base",
    "all": PROJECT_ROOT / "aloha_merged_banana_cube_subgoal_all",
}

REPO_IDS = {
    "base": "lixing/aloha_merged_banana_cube_subgoal_base",
    "all": "lixing/aloha_merged_banana_cube_subgoal_all",
}

TRAIN_CONFIGS = {
    "base": "pi05_aloha_banana_cube_subgoal_base",
    "all": "pi05_aloha_banana_cube_subgoal_all",
}


def compute_subgoal_indices(total_frames: int) -> list[int]:
    result = []
    for i in range(total_frames):
        sg_idx = min((i // SUBGOAL_WINDOW + 1) * SUBGOAL_WINDOW, total_frames - 1)
        result.append(sg_idx)
    return result


def read_video_frames(video_path: pathlib.Path) -> list:
    frames = []
    with av.open(str(video_path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def write_subgoal_video(
    frames: list,
    subgoal_indices: list[int],
    dst_path: pathlib.Path,
    fps: int = 30,
) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]

    with av.open(str(dst_path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "fast"}

        for i, sg_idx in enumerate(subgoal_indices):
            out_frame = av.VideoFrame.from_ndarray(frames[sg_idx], format="rgb24")
            out_frame.pts = i
            for packet in stream.encode(out_frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)


def build_updated_info(src_info: dict, subgoal_cameras: list[str]) -> dict:
    info = copy.deepcopy(src_info)
    ref_key = "observation.images.cam_high"
    ref_feature = copy.deepcopy(info["features"][ref_key])

    new_video_count = 0
    for cam in subgoal_cameras:
        new_key = f"observation.images.{cam}_subgoal"
        new_feature = copy.deepcopy(ref_feature)
        new_feature["info"]["video.codec"] = "h264"
        info["features"][new_key] = new_feature
        new_video_count += 1

    info["total_videos"] += TOTAL_EPISODES * new_video_count
    return info


def build_updated_modality(src_modality: dict, subgoal_cameras: list[str]) -> dict:
    modality = copy.deepcopy(src_modality)
    for cam in subgoal_cameras:
        modality["video"][f"{cam}_subgoal"] = {
            "original_key": f"observation.images.{cam}_subgoal"
        }
    return modality


def setup_output_directory(output_dir: pathlib.Path, subgoal_cameras: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Symlink data/ (parquet files unchanged)
    data_link = output_dir / "data"
    if not data_link.exists():
        data_link.symlink_to(DATASET_DIR / "data")

    # Create videos/chunk-000 directory
    out_video_chunk = output_dir / "videos" / "chunk-000"
    out_video_chunk.mkdir(parents=True, exist_ok=True)

    # Symlink original camera video directories
    for cam in SOURCE_CAMERAS:
        cam_key = f"observation.images.{cam}"
        link = out_video_chunk / cam_key
        if not link.exists():
            link.symlink_to(VIDEO_DIR / cam_key)

    # Update meta files
    out_meta = output_dir / "meta"
    out_meta.mkdir(parents=True, exist_ok=True)

    src_meta = DATASET_DIR / "meta"

    for fname in ("episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl"):
        link = out_meta / fname
        if not link.exists():
            link.symlink_to(src_meta / fname)

    with open(src_meta / "info.json") as f:
        src_info = json.load(f)
    updated_info = build_updated_info(src_info, subgoal_cameras)
    with open(out_meta / "info.json", "w") as f:
        json.dump(updated_info, f, indent=4)

    with open(src_meta / "modality.json") as f:
        src_modality = json.load(f)
    updated_modality = build_updated_modality(src_modality, subgoal_cameras)
    with open(out_meta / "modality.json", "w") as f:
        json.dump(updated_modality, f, indent=4)

    print(f"Output directory ready: {output_dir}")


def main(version: str, output_dir: pathlib.Path) -> None:
    subgoal_cameras = VERSION_CAMERAS[version]
    banana_subgoal_dir = BANANA_SUBGOAL_DIRS[version]

    setup_output_directory(output_dir, subgoal_cameras)

    out_video_chunk = output_dir / "videos" / "chunk-000"
    banana_video_chunk = banana_subgoal_dir / "videos" / "chunk-000"

    # Episodes 0-390: symlink precomputed banana subgoal videos
    print(f"Symlinking {BANANA_NUM_EPISODES} banana subgoal episodes from {banana_subgoal_dir.name} ...")
    for ep_idx in range(BANANA_NUM_EPISODES):
        for cam in subgoal_cameras:
            cam_subgoal_key = f"observation.images.{cam}_subgoal"
            src = banana_video_chunk / cam_subgoal_key / f"episode_{ep_idx:06d}.mp4"
            dst = out_video_chunk / cam_subgoal_key / f"episode_{ep_idx:06d}.mp4"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                dst.symlink_to(src)

    # Episodes 391-765: encode cube subgoal videos
    print(f"Encoding cube subgoal episodes ({BANANA_NUM_EPISODES}-{TOTAL_EPISODES - 1}) ...")
    for ep_idx in range(BANANA_NUM_EPISODES, TOTAL_EPISODES):
        parquet_path = PARQUET_DIR / f"episode_{ep_idx:06d}.parquet"
        df = pd.read_parquet(parquet_path)
        total_frames = len(df)

        subgoal_indices = compute_subgoal_indices(total_frames)

        for cam in subgoal_cameras:
            src_video = VIDEO_DIR / f"observation.images.{cam}" / f"episode_{ep_idx:06d}.mp4"
            dst_video = out_video_chunk / f"observation.images.{cam}_subgoal" / f"episode_{ep_idx:06d}.mp4"

            frames = read_video_frames(src_video)
            assert len(frames) == total_frames, (
                f"Episode {ep_idx}: video has {len(frames)} frames, parquet has {total_frames}"
            )
            write_subgoal_video(frames, subgoal_indices, dst_video)

        if (ep_idx - BANANA_NUM_EPISODES) % 20 == 0:
            print(f"  episode {ep_idx:3d}/{TOTAL_EPISODES - 1}")

    repo_id = REPO_IDS[version]
    train_cfg = TRAIN_CONFIGS[version]
    hf_cache = (
        pathlib.Path("/media/raid/workspace/surongpeng/ws_lixing/.cache/huggingface/lerobot")
        / "lixing"
        / repo_id.split("/")[1]
    )

    print(f"\nDone.")
    print(f"Version         : {version}")
    print(f"Subgoal cameras : {subgoal_cameras}")
    print(f"Banana episodes : 0-{BANANA_NUM_EPISODES - 1} (symlinked from {banana_subgoal_dir.name})")
    print(f"Cube episodes   : {BANANA_NUM_EPISODES}-{TOTAL_EPISODES - 1} (freshly encoded)")
    print(f"Output directory: {output_dir}")
    print()
    print("Next steps:")
    print(f"  ln -sfn {output_dir} {hf_cache}")
    print(f"  uv run python scripts/train_pytorch.py {train_cfg} --exp_name <name>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--version",
        choices=["base", "all"],
        default="base",
        help="base: only cam_high subgoal; all: all 3 cameras as subgoal",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=None,
        help="Output directory (defaults to aloha_merged_banana_cube_subgoal_{version})",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or DEFAULT_OUTPUT_DIRS[args.version]
    main(args.version, output_dir)
