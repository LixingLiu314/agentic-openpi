"""
Combined CoT preprocessing: subtask (segment) + trajectory COT + subgoal (cam_high).

Produces a single dataset containing all three CoT information types:
  - subtask  : per-frame segment-strategy subtask label (string column in parquet)
  - traj_cot : per-frame trajectory COT string with <loc> tokens (string column in parquet)
  - subgoal  : cam_high subgoal video (frame i shows the next-window boundary frame)

Output dataset repo_id : lixing/aloha_banana_all_cot
Training config name   : pi05_aloha_banana_all_cot

Dataset layout
--------------
  aloha_lerobot_all_cot/
    data/chunk-000/episode_XXXXXX.parquet  ← new files: base + subtask + traj_cot
    videos/chunk-000/
      observation.images.cam_high          → symlink to original
      observation.images.cam_left_wrist    → symlink to original
      observation.images.cam_right_wrist   → symlink to original
      observation.images.cam_high_subgoal/ ← newly created subgoal videos
    meta/
      info.json                            ← updated (adds cam_high_subgoal video feature)
      modality.json                        ← updated (adds cam_high_subgoal entry)
      episodes.jsonl                       → symlink to original
      episodes_stats.jsonl                 → symlink to original
      tasks.jsonl                          → symlink to original

Usage
-----
  uv run python scripts/preprocess_all_cot.py
  uv run python scripts/preprocess_all_cot.py --output_dir /custom/path

After running, create the HuggingFace cache symlink:
  ln -sfn <output_dir> ~/.cache/huggingface/lerobot/lixing/aloha_banana_all_cot

Then start training:
  uv run python scripts/train_pytorch.py pi05_aloha_banana_all_cot --exp_name <name>
"""

import argparse
import copy
import json
import pathlib
import re

import av
import pandas as pd

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------

DATASET_DIR = pathlib.Path(
    "/media/raid/workspace/surongpeng/ws_lixing/agentic-openpi"
    "/aloha_lerobot_actionequalstate_absinfo_butinstatsgroot_armdelta_gripperabs"
)
ANN_DIR = DATASET_DIR / "subtask_banana_two_tasks" / "subtask_annotations"
PARQUET_DIR = DATASET_DIR / "data" / "chunk-000"
VIDEO_DIR = DATASET_DIR / "videos" / "chunk-000"

COT_JSON = pathlib.Path(
    "/media/raid/workspace/surongpeng/ws_lixing/agentic-openpi"
    "/traj_data/banana_traj_origin_data/trajectory_data/cot_text_prompts.json"
)

NUM_EPISODES = 391
SOURCE_CAMERAS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
SUBGOAL_CAMERAS = ["cam_high"]  # base version: only cam_high subgoal

SUBTASK_WINDOW_SIZE = 45  # not used for segment strategy, kept for reference
TRAJ_WINDOW_SIZE = 60
SUBGOAL_WINDOW = 30

REPO_ID = "lixing/aloha_banana_all_cot"
TRAIN_CONFIG = "pi05_aloha_banana_all_cot"
DEFAULT_OUTPUT = DATASET_DIR.parent / "aloha_lerobot_all_cot"

# ---------------------------------------------------------------------------
# Subtask helpers (segment strategy)
# ---------------------------------------------------------------------------

def ann_path_for_episode(ep_idx: int) -> pathlib.Path:
    """Return the original annotation file for a given video episode index."""
    old_idx = ep_idx if ep_idx < 248 else ep_idx + 8
    return ANN_DIR / f"annotation_episode_{old_idx:06d}.json"


def subtask_at_frame(segments: list[dict], frame_idx: int) -> str:
    for seg in segments:
        if seg["start_frame"] <= frame_idx <= seg["end_frame"]:
            return seg["label"]
    return segments[-1]["label"]


def compute_segment_subtasks(segments: list[dict], total_frames: int) -> list[str]:
    return [subtask_at_frame(segments, i) for i in range(total_frames)]


# ---------------------------------------------------------------------------
# Trajectory COT helpers
# ---------------------------------------------------------------------------

def coords_to_loc_tokens(text: str) -> str:
    def replace_coord(m: re.Match) -> str:
        x = min(max(int(m.group(1)), 0), 1023)
        y = min(max(int(m.group(2)), 0), 1023)
        return f"<loc{x:04d}><loc{y:04d}>"
    return re.sub(r"\((\d+),\s*(\d+)\)", replace_coord, text)


def compute_windowed_traj_cots(ep_prompts: dict, total_frames: int) -> list[str]:
    result = []
    for frame_idx in range(total_frames):
        window_start = (frame_idx // TRAJ_WINDOW_SIZE) * TRAJ_WINDOW_SIZE
        key = str(min(window_start, total_frames - 1))
        text = ep_prompts.get(key, "")
        result.append(coords_to_loc_tokens(text))
    return result


# ---------------------------------------------------------------------------
# Subgoal video helpers
# ---------------------------------------------------------------------------

def compute_subgoal_indices(total_frames: int) -> list[int]:
    return [min((i // SUBGOAL_WINDOW + 1) * SUBGOAL_WINDOW, total_frames - 1)
            for i in range(total_frames)]


def read_video_frames(video_path: pathlib.Path) -> list:
    frames = []
    with av.open(str(video_path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def write_subgoal_video(frames: list, subgoal_indices: list[int], dst_path: pathlib.Path, fps: int = 30) -> None:
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


# ---------------------------------------------------------------------------
# Meta file helpers
# ---------------------------------------------------------------------------

def build_updated_info(src_info: dict) -> dict:
    info = copy.deepcopy(src_info)
    ref_key = "observation.images.cam_high"
    ref_feature = copy.deepcopy(info["features"][ref_key])
    for cam in SUBGOAL_CAMERAS:
        new_key = f"observation.images.{cam}_subgoal"
        new_feature = copy.deepcopy(ref_feature)
        new_feature["info"]["video.codec"] = "h264"
        info["features"][new_key] = new_feature
    info["total_videos"] += NUM_EPISODES * len(SUBGOAL_CAMERAS)
    return info


def build_updated_modality(src_modality: dict) -> dict:
    modality = copy.deepcopy(src_modality)
    for cam in SUBGOAL_CAMERAS:
        modality["video"][f"{cam}_subgoal"] = {
            "original_key": f"observation.images.{cam}_subgoal"
        }
    return modality


def setup_output_directory(output_dir: pathlib.Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # videos/chunk-000: create directory, symlink original cameras
    out_video_chunk = output_dir / "videos" / "chunk-000"
    out_video_chunk.mkdir(parents=True, exist_ok=True)
    for cam in SOURCE_CAMERAS:
        cam_key = f"observation.images.{cam}"
        link = out_video_chunk / cam_key
        if not link.exists():
            link.symlink_to(VIDEO_DIR / cam_key)

    # meta/: symlink static files, write updated info.json and modality.json
    out_meta = output_dir / "meta"
    out_meta.mkdir(parents=True, exist_ok=True)
    src_meta = DATASET_DIR / "meta"
    for fname in ("episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl"):
        link = out_meta / fname
        if not link.exists():
            link.symlink_to(src_meta / fname)

    with open(src_meta / "info.json") as f:
        src_info = json.load(f)
    with open(out_meta / "info.json", "w") as f:
        json.dump(build_updated_info(src_info), f, indent=4)

    with open(src_meta / "modality.json") as f:
        src_modality = json.load(f)
    with open(out_meta / "modality.json", "w") as f:
        json.dump(build_updated_modality(src_modality), f, indent=4)

    print(f"Output directory structure ready: {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(output_dir: pathlib.Path) -> None:
    setup_output_directory(output_dir)

    # Load traj COT data once
    with open(COT_JSON) as f:
        cot_data = json.load(f)
    all_prompts = cot_data["prompts"]  # dict: str(ep_idx) -> dict(str(frame_idx) -> text)

    out_parquet_dir = output_dir / "data" / "chunk-000"
    out_parquet_dir.mkdir(parents=True, exist_ok=True)
    out_video_chunk = output_dir / "videos" / "chunk-000"

    errors = []
    for ep_idx in range(NUM_EPISODES):
        # --- subtask (segment strategy) ---
        ann_path = ann_path_for_episode(ep_idx)
        with open(ann_path) as f:
            ann = json.load(f)
        segments = ann["subtask_segments"]
        total_frames = ann["total_frames"]

        # --- parquet: load, add subtask + traj_cot ---
        parquet_path = PARQUET_DIR / f"episode_{ep_idx:06d}.parquet"
        df = pd.read_parquet(parquet_path)
        if len(df) != total_frames:
            errors.append(
                f"Episode {ep_idx}: parquet has {len(df)} frames, annotation says {total_frames}"
            )
            continue

        df["subtask"] = compute_segment_subtasks(segments, total_frames)

        ep_prompts = all_prompts.get(str(ep_idx), all_prompts.get(ep_idx, {}))
        traj_cots = compute_windowed_traj_cots(ep_prompts, total_frames)
        df["traj_cot"] = traj_cots

        df.to_parquet(out_parquet_dir / f"episode_{ep_idx:06d}.parquet", index=False)

        # --- subgoal videos (cam_high only) ---
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

    if errors:
        print("\nERRORS:")
        for e in errors:
            print(f"  {e}")
    else:
        print(f"\nAll {NUM_EPISODES} episodes processed successfully.")

    hf_cache = (
        pathlib.Path("/media/raid/workspace/surongpeng/ws_lixing/.cache/huggingface/lerobot")
        / "lixing"
        / REPO_ID.split("/")[1]
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
