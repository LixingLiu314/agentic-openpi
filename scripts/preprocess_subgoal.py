"""
Offline subgoal image preprocessing for the Aloha banana dataset.

For each episode, new video files are created for subgoal camera streams.
The subgoal for frame i is the frame at index min((i // SUBGOAL_WINDOW + 1) * SUBGOAL_WINDOW, last_frame),
so frames 0-29 all share frame 30 as their subgoal, frames 30-59 share frame 60, etc.

Only the base-camera output is supported by the active training config:
  base  -- only cam_high_subgoal  (new dataset: aloha_lerobot_subgoal_base)

Usage:
  uv run python scripts/preprocess_subgoal.py
  uv run python scripts/preprocess_subgoal.py --output_dir /custom/path

After running, symlink the output directory into the HuggingFace cache:
  ln -sfn <output_dir> ~/.cache/huggingface/lerobot/lixing/aloha_banana_subgoal_base

Then start training:
  uv run torchrun ... scripts/train_pytorch.py pi05_aloha_banana_subgoal_base ...
"""

import argparse
import copy
import json
import pathlib

import av
import pandas as pd

DATASET_DIR = pathlib.Path(
    "/media/raid/workspace/surongpeng/ws_lixing/agentic-openpi"
    "/aloha_lerobot_actionequalstate_absinfo_butinstatsgroot_armdelta_gripperabs"
)
PARQUET_DIR = DATASET_DIR / "data" / "chunk-000"
VIDEO_DIR = DATASET_DIR / "videos" / "chunk-000"

SUBGOAL_WINDOW = 30
NUM_EPISODES = 391

SOURCE_CAMERAS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
SUBGOAL_CAMERAS = ["cam_high"]
DEFAULT_OUTPUT_DIR = DATASET_DIR.parent / "aloha_lerobot_subgoal_base"
REPO_ID = "lixing/aloha_banana_subgoal_base"
TRAIN_CONFIG = "pi05_aloha_banana_subgoal_base"


def compute_subgoal_indices(total_frames: int) -> list[int]:
    """Return subgoal frame index for each frame in the episode.

    Frame i maps to min((i // SUBGOAL_WINDOW + 1) * SUBGOAL_WINDOW, total_frames - 1).
    """
    result = []
    for i in range(total_frames):
        sg_idx = min((i // SUBGOAL_WINDOW + 1) * SUBGOAL_WINDOW, total_frames - 1)
        result.append(sg_idx)
    return result


def read_video_frames(video_path: pathlib.Path) -> list:
    """Read all frames from a video file as RGB numpy arrays."""
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
    """Write a new video where output frame i is frames[subgoal_indices[i]]."""
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
    """Return updated info.json dict with new subgoal video features."""
    info = copy.deepcopy(src_info)

    # Find a reference video feature to copy metadata from
    ref_key = "observation.images.cam_high"
    ref_feature = copy.deepcopy(info["features"][ref_key])

    new_video_count = 0
    for cam in subgoal_cameras:
        new_key = f"observation.images.{cam}_subgoal"
        new_feature = copy.deepcopy(ref_feature)
        # Update codec to h264 since we re-encode
        new_feature["info"]["video.codec"] = "h264"
        info["features"][new_key] = new_feature
        new_video_count += 1

    info["total_videos"] += NUM_EPISODES * new_video_count
    return info


def build_updated_modality(src_modality: dict, subgoal_cameras: list[str]) -> dict:
    """Return updated modality.json dict with new subgoal camera entries."""
    modality = copy.deepcopy(src_modality)
    for cam in subgoal_cameras:
        modality["video"][f"{cam}_subgoal"] = {
            "original_key": f"observation.images.{cam}_subgoal"
        }
    return modality


def setup_output_directory(output_dir: pathlib.Path, subgoal_cameras: list[str]) -> None:
    """Create output directory structure with symlinks and updated meta files."""
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

    # Symlink files that don't need updating
    for fname in ("episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl"):
        link = out_meta / fname
        if not link.exists():
            link.symlink_to(src_meta / fname)

    # Write updated info.json
    with open(src_meta / "info.json") as f:
        src_info = json.load(f)
    updated_info = build_updated_info(src_info, subgoal_cameras)
    with open(out_meta / "info.json", "w") as f:
        json.dump(updated_info, f, indent=4)

    # Write updated modality.json
    with open(src_meta / "modality.json") as f:
        src_modality = json.load(f)
    updated_modality = build_updated_modality(src_modality, subgoal_cameras)
    with open(out_meta / "modality.json", "w") as f:
        json.dump(updated_modality, f, indent=4)

    print(f"Output directory ready: {output_dir}")


def main(output_dir: pathlib.Path) -> None:
    setup_output_directory(output_dir, SUBGOAL_CAMERAS)

    out_video_chunk = output_dir / "videos" / "chunk-000"

    for ep_idx in range(NUM_EPISODES):
        # Get total frames from parquet
        parquet_path = PARQUET_DIR / f"episode_{ep_idx:06d}.parquet"
        df = pd.read_parquet(parquet_path)
        total_frames = len(df)

        subgoal_indices = compute_subgoal_indices(total_frames)

        for cam in SUBGOAL_CAMERAS:
            src_video = VIDEO_DIR / f"observation.images.{cam}" / f"episode_{ep_idx:06d}.mp4"
            dst_video = out_video_chunk / f"observation.images.{cam}_subgoal" / f"episode_{ep_idx:06d}.mp4"

            frames = read_video_frames(src_video)
            assert len(frames) == total_frames, (
                f"Episode {ep_idx}: video has {len(frames)} frames, parquet has {total_frames}"
            )
            write_subgoal_video(frames, subgoal_indices, dst_video)

        if ep_idx % 20 == 0:
            print(f"  episode {ep_idx:3d}/{NUM_EPISODES - 1}")

    hf_cache = (
        pathlib.Path("/media/raid/workspace/surongpeng/ws_lixing/.cache/huggingface/lerobot")
        / "lixing"
        / REPO_ID.split("/")[1]
    )

    print("\nDone.")
    print("Version         : base")
    print(f"Subgoal cameras : {SUBGOAL_CAMERAS}")
    print(f"Output directory: {output_dir}")
    print()
    print("Next steps:")
    print(f"  ln -sfn {output_dir} {hf_cache}")
    print(f"  uv run torchrun --nproc_per_node=<N> scripts/train_pytorch.py {TRAIN_CONFIG} \\")
    print("    --exp_name <run_name>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess subgoal images for Aloha banana dataset.")
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=None,
        help="Output directory (default: aloha_lerobot_subgoal_base next to base dataset)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    main(output_dir)
