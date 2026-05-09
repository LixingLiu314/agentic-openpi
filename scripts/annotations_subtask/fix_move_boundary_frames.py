"""
fix_move_boundary_frames.py

For specific hard episodes, extract a sequence of cam_high frames at regular
intervals and send them as multiple images to Doubao, asking which is the
FIRST frame where the banana is above the target plate.

This is more accurate than video analysis for tricky cases.

Usage:
    python fix_move_boundary_frames.py --episodes 19 234 339 361 364 376 387 \
        --search-start-offset -30 --search-end-offset 90 --step 3
"""

import argparse
import base64
import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DOUBAO_API_KEY  = "8dc53ee3-fddc-47b5-b419-5888d5e057ff"
DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DOUBAO_MODEL    = "doubao-seed-2-0-pro-260215"

ANNOTATION_DIR = Path(__file__).parent / "subtask_annotations"
SUBGOAL_DIR    = ANNOTATION_DIR / "subgoal_images"
DATASET_ROOT   = Path("/home/lixingliu/Agent-VLA-eval/Agent-VLA/playground/Datasets/aloha_lerobot")
VIDEO_DIR      = DATASET_ROOT / "videos" / "chunk-000" / "observation.images.cam_high"

TASK_META = {
    "put_banana_in_the_green_plate": {
        "object_name": "banana", "target_name": "green plate",
        "wrist_cam": "observation.images.cam_right_wrist",
    },
    "put_banana_in_the_red_plate": {
        "object_name": "banana", "target_name": "red plate",
        "wrist_cam": "observation.images.cam_left_wrist",
    },
}

_CLIENT = None

def _get_client():
    global _CLIENT
    if _CLIENT is None:
        for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
                    "all_proxy", "ALL_PROXY"):
            os.environ.pop(var, None)
        import openai
        _CLIENT = openai.OpenAI(api_key=DOUBAO_API_KEY, base_url=DOUBAO_BASE_URL)
    return _CLIENT


def _img_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_frame(mp4_path: Path, frame_idx: int, fps: float, out_path: Path) -> bool:
    seek = frame_idx / fps
    ret = os.system(
        f'ffmpeg -ss {seek:.4f} -i "{mp4_path}" '
        f'-frames:v 1 -q:v 2 "{out_path}" -y -loglevel error'
    )
    return ret == 0 and out_path.exists()


def find_boundary_with_frames(episode_idx: int, annotation: dict,
                               start_frame: int, end_frame: int,
                               step: int = 3, retry: int = 2) -> int | None:
    """
    Extract cam_high frames at [start_frame, start_frame+step, ...end_frame]
    and send them all to Doubao as numbered images.
    Ask: which is the FIRST frame where the banana is above the target plate?
    Returns the identified frame index, or None on failure.
    """
    task   = annotation["task"]
    fps    = annotation["fps"]
    total  = annotation["total_frames"]
    obj    = TASK_META[task]["object_name"]
    target = TASK_META[task]["target_name"]
    mp4    = VIDEO_DIR / f"episode_{episode_idx:06d}.mp4"

    start_frame = max(0, start_frame)
    end_frame   = min(total - 1, end_frame)
    frames      = list(range(start_frame, end_frame + 1, step))

    logger.info("Episode %d: extracting %d frames [%d..%d step %d]",
                episode_idx, len(frames), start_frame, end_frame, step)

    # Extract frames to /tmp
    tmp_dir = Path(f"/tmp/frame_seq_ep{episode_idx:06d}")
    tmp_dir.mkdir(exist_ok=True)

    frame_paths = []
    for f in frames:
        out = tmp_dir / f"frame_{f:05d}.jpg"
        if extract_frame(mp4, f, fps, out):
            frame_paths.append((f, out))
        else:
            logger.warning("  Could not extract frame %d", f)

    if not frame_paths:
        logger.error("Episode %d: no frames extracted", episode_idx)
        return None

    # Build prompt
    system_prompt = (
        f"You are a robot manipulation expert.\n\n"
        f"You will be shown a sequence of numbered overhead camera frames from a robot demo. "
        f"The robot is transporting a {obj} toward a {target}.\n\n"
        f"Your task: identify the FIRST frame number where the {obj} has arrived directly "
        f"above the {target} (the banana's projection overlaps with the plate area when "
        f"viewed from overhead). The banana does NOT need to be lowered yet, but it must "
        f"be spatially over the plate — not still in transit.\n\n"
        f"Output ONLY a JSON object: {{\"first_above_frame\": <frame_number>}}\n"
        f"Use the exact frame numbers shown in the labels (not the image index).\n"
        f"No explanation, no other text."
    )

    content = [{"type": "text",
                "text": f"Overhead camera frames for episode {episode_idx}. "
                        f"Find the FIRST frame where the {obj} is above the {target}:"}]
    for frame_idx, path in frame_paths:
        content.append({"type": "text", "text": f"Frame {frame_idx}:"})
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{_img_b64(path)}"}})

    client = _get_client()
    for attempt in range(retry + 1):
        try:
            resp = client.chat.completions.create(
                model=DOUBAO_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": content},
                ],
                extra_body={"thinking": {"type": "disabled"}},
            )
            raw = resp.choices[0].message.content.strip()
            logger.debug("Doubao frame-seq response: %s", raw[:300])
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                result = json.loads(match.group())
                f = int(result.get("first_above_frame", -1))
                if f >= 0:
                    # Snap to nearest extracted frame
                    nearest = min(frame_paths, key=lambda x: abs(x[0] - f))[0]
                    logger.info("  Doubao says first_above_frame=%d → snapped to %d", f, nearest)
                    return nearest
            logger.warning("Attempt %d: could not parse response: %r", attempt + 1, raw[:200])
        except Exception as e:
            logger.warning("Attempt %d failed: %s", attempt + 1, e)
            if attempt < retry:
                time.sleep(2)

    return None


def _re_extract_subgoal_images(episode_idx: int, annotation: dict):
    task      = annotation["task"]
    fps       = annotation["fps"]
    duration  = annotation["duration_sec"]
    mp4_path  = VIDEO_DIR / f"episode_{episode_idx:06d}.mp4"
    wrist_cam = TASK_META[task]["wrist_cam"]
    wrist_path = mp4_path.parent.parent / wrist_cam / mp4_path.name
    img_dir   = SUBGOAL_DIR / f"episode_{episode_idx:06d}"
    img_dir.mkdir(parents=True, exist_ok=True)

    for seg in annotation["subtask_segments"]:
        if seg["subtask_id"] not in (3, 4):
            continue
        end_sec   = seg["end_sec"]
        label     = seg["label"]
        seg_id    = seg["subtask_id"]
        end_frame = seg["end_frame"]
        label_slug = label.replace(" ", "_")
        seek_sec   = max(0.0, min(end_sec, duration - 0.1))

        for old in img_dir.glob(f"subtask_{seg_id}_*"):
            old.unlink(missing_ok=True)

        img_name = f"subtask_{seg_id}_{label_slug}_end_frame{end_frame:04d}_camhigh.jpg"
        os.system(f'ffmpeg -ss {seek_sec:.3f} -i "{mp4_path}" '
                  f'-frames:v 1 -q:v 2 "{img_dir / img_name}" -y -loglevel error')

        if wrist_path.exists():
            wrist_img_name = f"subtask_{seg_id}_{label_slug}_end_frame{end_frame:04d}_wrist.jpg"
            os.system(f'ffmpeg -ss {seek_sec:.3f} -i "{wrist_path}" '
                      f'-frames:v 1 -q:v 2 "{img_dir / wrist_img_name}" -y -loglevel error')


def fix_episode(episode_idx: int, start_offset: int, end_offset: int,
                step: int, retry: int):
    json_path = ANNOTATION_DIR / f"annotation_episode_{episode_idx:06d}.json"
    with open(json_path) as f:
        annotation = json.load(f)

    move_seg  = next(s for s in annotation["subtask_segments"] if s["subtask_id"] == 3)
    grasp_seg = next(s for s in annotation["subtask_segments"] if s["subtask_id"] == 2)
    place_seg = next(s for s in annotation["subtask_segments"] if s["subtask_id"] == 4)
    fps       = annotation["fps"]

    current_move_end = move_seg["end_frame"]
    grasp_end        = grasp_seg["end_frame"]
    place_end        = place_seg["end_frame"]

    search_start = max(grasp_end + 1, current_move_end + start_offset)
    search_end   = min(place_end - 1, current_move_end + end_offset)

    logger.info("Episode %d: current move_end=%d, searching [%d..%d]",
                episode_idx, current_move_end, search_start, search_end)

    new_move_end = find_boundary_with_frames(
        episode_idx, annotation, search_start, search_end, step=step, retry=retry
    )

    if new_move_end is None:
        logger.error("Episode %d: frame-sequence detection failed", episode_idx)
        return False

    new_move_end = max(grasp_end + 1, min(new_move_end, place_end - 1))
    logger.info("Episode %d: move_end  %d → %d  (%.2fs → %.2fs)",
                episode_idx, current_move_end, new_move_end,
                current_move_end / fps, new_move_end / fps)

    for seg in annotation["subtask_segments"]:
        if seg["subtask_id"] == 3:
            seg["end_frame"] = new_move_end
            seg["end_sec"]   = round(new_move_end / fps, 3)
        elif seg["subtask_id"] == 4:
            seg["start_frame"] = new_move_end + 1
            seg["start_sec"]   = round((new_move_end + 1) / fps, 3)

    annotation["move_place_method"] = "frame_sequence_revalidated"

    with open(json_path, "w") as f:
        json.dump(annotation, f, indent=2)
    logger.info("Episode %d: annotation updated", episode_idx)

    _re_extract_subgoal_images(episode_idx, annotation)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--search-start-offset", type=int, default=-30,
                        help="Start search at current_move_end + offset (default -30)")
    parser.add_argument("--search-end-offset", type=int, default=90,
                        help="End search at current_move_end + offset (default +90 = 3s)")
    parser.add_argument("--step", type=int, default=3,
                        help="Frame step between samples (default 3 = every 0.1s at 30fps)")
    parser.add_argument("--retry", type=int, default=2)
    args = parser.parse_args()

    results = {}
    for ep in args.episodes:
        ok = fix_episode(ep, args.search_start_offset, args.search_end_offset,
                         args.step, args.retry)
        results[ep] = "fixed" if ok else "failed"

    print("\n" + "=" * 60)
    print("FRAME-SEQUENCE FIX REPORT")
    print("=" * 60)
    for ep, status in results.items():
        print(f"  episode {ep:>4d}: {status}")


if __name__ == "__main__":
    main()
