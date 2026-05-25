"""
Step 5 — Add subgoal images in-place to aloha_object_lerobot_gripper_binary.

Subgoal rule (window=60):
  frame i → subgoal frame = min((i // 60 + 1) * 60, last_frame)
  e.g.  0-59  → frame 60 (or last frame if episode is shorter than 60)
        60-119 → frame 120
        ...

Writes one subgoal video per episode into:
  videos/chunk-000/observation.images.cam_high_subgoal/episode_XXXXXX.mp4
Updates meta/info.json and meta/modality.json in-place.
Supports resuming: skips already-written episodes.

Usage:
  python Datasets/step5_preprocess_subgoal.py
  python Datasets/step5_preprocess_subgoal.py --dry_run
"""

import argparse
import copy
import json
import pathlib

import av
import pandas as pd
import tqdm

DATASET_DIR = pathlib.Path("/media/raid/workspace/surongpeng/ws_lixing/agentic-openpi/Datasets/three_object/three_object_lerobot_binary")
PARQUET_DIR = DATASET_DIR / "data" / "chunk-000"
VIDEO_DIR   = DATASET_DIR / "videos" / "chunk-000"

SUBGOAL_WINDOW = 60
NUM_EPISODES   = json.loads((DATASET_DIR / "meta" / "info.json").read_text())["num_episodes"]
SUBGOAL_CAM    = "cam_high"
SUBGOAL_KEY    = f"observation.images.{SUBGOAL_CAM}_subgoal"


def compute_subgoal_indices(total_frames: int) -> list[int]:
    """frame i → min((i // window + 1) * window, last_frame)"""
    return [min((i // SUBGOAL_WINDOW + 1) * SUBGOAL_WINDOW, total_frames - 1)
            for i in range(total_frames)]


def read_video_frames(video_path: pathlib.Path) -> list:
    frames = []
    with av.open(str(video_path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def write_subgoal_video(frames: list, subgoal_indices: list[int],
                        dst_path: pathlib.Path, fps: int = 30) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    with av.open(str(dst_path), "w") as container:
        stream = container.add_stream("libsvtav1", rate=fps, options={"g": "2", "crf": "30"})
        stream.width  = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        for i, sg_idx in enumerate(subgoal_indices):
            out_frame = av.VideoFrame.from_ndarray(frames[sg_idx], format="rgb24")
            out_frame.pts = i
            for pkt in stream.encode(out_frame):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)


def update_meta(dry_run: bool) -> None:
    meta_dir = DATASET_DIR / "meta"

    # info.json
    info_path = meta_dir / "info.json"
    info = json.loads(info_path.read_text())
    if SUBGOAL_KEY not in info["features"]:
        ref = copy.deepcopy(info["features"]["observation.images.cam_high"])
        ref["video_info"]["video.codec"] = "av1"
        ref["info"]["video.codec"]       = "av1"
        info["features"][SUBGOAL_KEY]    = ref
        info["total_videos"] = info.get("total_videos", 0) + NUM_EPISODES
        if not dry_run:
            info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2))
        print(f"{'[DRY] ' if dry_run else ''}info.json: added {SUBGOAL_KEY}")
    else:
        print(f"info.json already has {SUBGOAL_KEY}")

    # modality.json
    modality_path = meta_dir / "modality.json"
    modality = json.loads(modality_path.read_text())
    cam_key = f"{SUBGOAL_CAM}_subgoal"
    if cam_key not in modality.get("video", {}):
        modality.setdefault("video", {})[cam_key] = {"original_key": SUBGOAL_KEY}
        if not dry_run:
            modality_path.write_text(json.dumps(modality, ensure_ascii=False, indent=2))
        print(f"{'[DRY] ' if dry_run else ''}modality.json: added {cam_key}")
    else:
        print(f"modality.json already has {cam_key}")


def main(dry_run: bool) -> None:
    subgoal_video_dir = VIDEO_DIR / SUBGOAL_KEY
    existing  = sorted(subgoal_video_dir.glob("episode_*.mp4")) if subgoal_video_dir.exists() else []
    start_ep  = len(existing)

    if start_ep > 0:
        print(f"Resuming from episode {start_ep} ({start_ep}/{NUM_EPISODES} already done)")

    update_meta(dry_run=dry_run)

    if dry_run:
        print(f"[DRY] Would generate {NUM_EPISODES - start_ep} subgoal videos → {subgoal_video_dir}")
        # show a sample of subgoal indices
        sample_frames = 200
        indices = compute_subgoal_indices(sample_frames)
        print(f"[DRY] Sample subgoal indices (first 130 of {sample_frames}-frame episode):")
        print(f"       frame 0   → subgoal {indices[0]}")
        print(f"       frame 59  → subgoal {indices[59]}")
        print(f"       frame 60  → subgoal {indices[60]}")
        print(f"       frame 119 → subgoal {indices[min(119, sample_frames-1)]}")
        print(f"       frame 199 → subgoal {indices[199]} (last frame, clamped)")
        return

    for ep_idx in tqdm.tqdm(range(start_ep, NUM_EPISODES), desc="episodes"):
        parquet_path = PARQUET_DIR / f"episode_{ep_idx:06d}.parquet"
        df = pd.read_parquet(parquet_path)
        total_frames   = len(df)
        subgoal_indices = compute_subgoal_indices(total_frames)

        src_video = VIDEO_DIR / f"observation.images.{SUBGOAL_CAM}" / f"episode_{ep_idx:06d}.mp4"
        dst_video = subgoal_video_dir / f"episode_{ep_idx:06d}.mp4"

        frames = read_video_frames(src_video)
        assert len(frames) == total_frames, (
            f"Episode {ep_idx}: video {len(frames)} frames != parquet {total_frames}"
        )
        write_subgoal_video(frames, subgoal_indices, dst_video)

    print(f"\nDone. {NUM_EPISODES} subgoal videos → {subgoal_video_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
