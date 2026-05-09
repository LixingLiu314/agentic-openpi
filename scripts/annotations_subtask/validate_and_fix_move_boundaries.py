"""
validate_and_fix_move_boundaries.py

For every annotated episode:
1. Find the move-subtask (subtask_id=3) subgoal images (cam_high + wrist).
2. Send both images to Doubao and ask: "Is the banana already above the target plate?"
3. If the answer is "no" (boundary annotated too early), report it.
4. For bad episodes, call Doubao on the overhead video to re-find the move→place boundary.
5. Update the annotation JSON in place and re-extract the subgoal images.

Usage:
    python validate_and_fix_move_boundaries.py [--episodes 0 1 2] [--dry-run]
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

# ---------------------------------------------------------------------------
# Config (same API as annotate_subtasks.py)
# ---------------------------------------------------------------------------

DOUBAO_API_KEY  = "8dc53ee3-fddc-47b5-b419-5888d5e057ff"
DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DOUBAO_MODEL    = "doubao-seed-2-0-pro-260215"

ANNOTATION_DIR = Path(__file__).parent / "subtask_annotations"
SUBGOAL_DIR    = ANNOTATION_DIR / "subgoal_images"

DATASET_ROOT = Path("/home/lixingliu/Agent-VLA-eval/Agent-VLA/playground/Datasets/aloha_lerobot")
VIDEO_DIR    = DATASET_ROOT / "videos" / "chunk-000" / "observation.images.cam_high"

TASK_META = {
    "put_banana_in_the_green_plate": {
        "object_name": "banana",
        "target_name": "green plate",
        "wrist_cam":   "observation.images.cam_right_wrist",
    },
    "put_banana_in_the_red_plate": {
        "object_name": "banana",
        "target_name": "red plate",
        "wrist_cam":   "observation.images.cam_left_wrist",
    },
}

# ---------------------------------------------------------------------------
# Doubao client
# ---------------------------------------------------------------------------

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


def _encode_mp4(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Step 1: validate move-end subgoal images via Doubao
# ---------------------------------------------------------------------------

def validate_move_subgoal(camhigh_img: Path, wrist_img: Path | None,
                           task: str) -> tuple[bool, str]:
    """
    Ask Doubao whether the banana has already reached above the target plate
    in the move-end subgoal frame.

    Returns (is_above_plate: bool, raw_answer: str).
    """
    obj    = TASK_META[task]["object_name"]
    target = TASK_META[task]["target_name"]

    content = []
    content.append({"type": "text",
                    "text": f"Overhead camera view (end of 'move' phase — the last frame before 'place' begins):"})
    content.append({"type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{_img_b64(camhigh_img)}"}})

    if wrist_img and wrist_img.exists():
        content.append({"type": "text", "text": "Wrist camera view (same frame):"})
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{_img_b64(wrist_img)}"}})

    content.append({"type": "text", "text": (
        f"This frame is the END of the 'move' phase (move→place boundary).\n\n"
        f"Definition: the move phase ends the moment the {obj} is positioned DIRECTLY ABOVE "
        f"the {target} — meaning when viewed from overhead, the {obj} overlaps with "
        f"the {target}'s area. The {obj} does not need to be lowered or touching yet, "
        f"but it must be over the plate, not still approaching it from the side.\n\n"
        f"Answer 'yes' if: the {obj} is directly above the {target} (overhead view shows "
        f"the {obj} overlapping the plate area).\n"
        f"Answer 'no' if: the {obj} is still in transit and has NOT yet reached directly "
        f"above the {target} — e.g. it is above the table but not yet over the plate, "
        f"or it is approaching but the plate is still clearly off to one side.\n\n"
        f"Answer with ONLY one word: 'yes' or 'no'."
    )})

    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=DOUBAO_MODEL,
            messages=[{"role": "user", "content": content}],
            extra_body={"thinking": {"type": "disabled"}},
        )
        answer = resp.choices[0].message.content.strip().lower()
        logger.debug("Doubao move-validate answer: %r", answer)
        is_above = "yes" in answer
        return is_above, answer
    except Exception as e:
        logger.warning("Doubao validate call failed: %s — assuming valid", e)
        return True, f"error: {e}"


# ---------------------------------------------------------------------------
# Step 2: re-detect move→place boundary on the overhead video
# ---------------------------------------------------------------------------

def redetect_move_place_boundary(episode_idx: int, task: str,
                                  grasp_end_frame: int, fps: float,
                                  duration: float, retry: int = 2) -> int | None:
    """
    Call Doubao on the cam_high overhead video to re-find the move→place transition.
    Returns the new move_end_frame, or None on failure.
    """
    obj    = TASK_META[task]["object_name"]
    target = TASK_META[task]["target_name"]
    mp4_path = VIDEO_DIR / f"episode_{episode_idx:06d}.mp4"
    if not mp4_path.exists():
        logger.error("Episode %d: video not found: %s", episode_idx, mp4_path)
        return None

    b64 = _encode_mp4(mp4_path)
    grasp_end_sec = round(grasp_end_frame / fps, 2)

    system_prompt = (
        f"You are a robot manipulation expert analyzing a robot demo video.\n\n"
        f"The video shows the overhead camera view of a table scene.\n"
        f"The robot has already grasped the {obj} and is transporting it.\n\n"
        f"Your task: find the exact timestamp when the robot transitions from MOVING the {obj} to PLACING it.\n\n"
        f"Definition:\n"
        f"- move  : {obj} is held in the air and has NOT yet reached above the {target}.\n"
        f"- place : {obj} has moved above the {target} (directly overhead or in contact). "
        f"The transition happens the moment the {obj} is above the {target}, even before it is lowered.\n\n"
        f"Output ONLY a single JSON object:\n"
        f'{{\"move_end_sec\": <float>, \"place_start_sec\": <float>}}\n\n'
        f"Where move_end_sec == place_start_sec (the transition moment).\n"
        f"No explanation, no other text."
    )

    for attempt in range(retry + 1):
        try:
            client = _get_client()
            resp = client.chat.completions.create(
                model=DOUBAO_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "video_url",
                         "video_url": {"url": f"data:video/mp4;base64,{b64}"}},
                        {"type": "text", "text": (
                            f"Video: {duration:.2f}s total, {fps:.0f} FPS. "
                            f"Overhead camera view. "
                            f"The robot grasped the object at ~{grasp_end_sec}s. "
                            f"Find when the object first moves above the {target}."
                        )},
                    ]},
                ],
                extra_body={"thinking": {"type": "disabled"}},
            )
            raw = resp.choices[0].message.content.strip()
            logger.debug("Doubao re-detect response: %s", raw[:300])
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                result = json.loads(match.group())
                t = float(result.get("place_start_sec", result.get("move_end_sec", -1)))
                if t > 0:
                    frame = min(round(t * fps), int(duration * fps) - 2)
                    return frame
            logger.warning("Attempt %d/%d: could not parse re-detect response: %r",
                           attempt + 1, retry + 1, raw[:200])
        except Exception as e:
            logger.warning("Attempt %d/%d failed: %s", attempt + 1, retry + 1, e)
            if attempt < retry:
                time.sleep(2)

    return None


# ---------------------------------------------------------------------------
# Helper: get video info
# ---------------------------------------------------------------------------

def _get_video_info(mp4_path: Path):
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=nb_frames,r_frame_rate,duration",
             "-of", "json", str(mp4_path)],
            stderr=subprocess.DEVNULL,
        )
        info = json.loads(out)["streams"][0]
        num, den = map(int, info["r_frame_rate"].split("/"))
        fps = num / den if den else 30.0
        if "nb_frames" in info and info["nb_frames"] != "N/A":
            total_frames = int(info["nb_frames"])
        else:
            total_frames = round(float(info["duration"]) * fps)
        duration = total_frames / fps
        return total_frames, fps, duration
    except Exception as e:
        logger.warning("ffprobe failed (%s), using fallback fps=30", e)
        return None, 30.0, None


# ---------------------------------------------------------------------------
# Helper: re-extract subgoal images after fix
# ---------------------------------------------------------------------------

def _re_extract_subgoal_images(episode_idx: int, annotation: dict):
    """Re-extract the subgoal images for segments that changed."""
    task       = annotation["task"]
    fps        = annotation["fps"]
    duration   = annotation["duration_sec"]
    mp4_path   = VIDEO_DIR / f"episode_{episode_idx:06d}.mp4"
    wrist_cam  = TASK_META[task]["wrist_cam"]
    wrist_path = mp4_path.parent.parent / wrist_cam / mp4_path.name
    img_dir    = SUBGOAL_DIR / f"episode_{episode_idx:06d}"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Only re-extract move (seg id 3) and place (seg id 4) images
    for seg in annotation["subtask_segments"]:
        if seg["subtask_id"] not in (3, 4):
            continue
        end_sec  = seg["end_sec"]
        label    = seg["label"]
        seg_id   = seg["subtask_id"]
        end_frame = seg["end_frame"]
        label_slug = label.replace(" ", "_")
        seek_sec   = max(0.0, min(end_sec, duration - 0.1))

        # Remove old images for this segment
        for old in img_dir.glob(f"subtask_{seg_id}_*"):
            old.unlink(missing_ok=True)

        # cam_high
        img_name = f"subtask_{seg_id}_{label_slug}_end_frame{end_frame:04d}_camhigh.jpg"
        cmd = (f'ffmpeg -ss {seek_sec:.3f} -i "{mp4_path}" '
               f'-frames:v 1 -q:v 2 "{img_dir / img_name}" -y -loglevel error')
        os.system(cmd)

        # wrist
        if wrist_path.exists():
            wrist_img_name = f"subtask_{seg_id}_{label_slug}_end_frame{end_frame:04d}_wrist.jpg"
            cmd = (f'ffmpeg -ss {seek_sec:.3f} -i "{wrist_path}" '
                   f'-frames:v 1 -q:v 2 "{img_dir / wrist_img_name}" -y -loglevel error')
            os.system(cmd)


# ---------------------------------------------------------------------------
# Find move-end subgoal images for an episode
# ---------------------------------------------------------------------------

def _find_move_subgoal_images(episode_idx: int) -> tuple[Path | None, Path | None]:
    img_dir = SUBGOAL_DIR / f"episode_{episode_idx:06d}"
    if not img_dir.exists():
        return None, None
    camhigh = next(img_dir.glob("subtask_3_*_camhigh.jpg"), None)
    wrist   = next(img_dir.glob("subtask_3_*_wrist.jpg"),   None)
    return camhigh, wrist


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate and optionally fix move→place boundaries using Doubao.\n\n"
                    "Typical workflow:\n"
                    "  1. python validate_and_fix_move_boundaries.py --dry-run\n"
                    "     → reports bad episodes to move_boundary_validation_report.json\n"
                    "  2. Review the report, then:\n"
                    "     python validate_and_fix_move_boundaries.py --fix-from-report\n"
                    "     → fixes only the episodes listed as bad in the report"
    )
    parser.add_argument("--episodes", type=int, nargs="+", default=None,
                        help="Specific episode indices to check (default: all annotated)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only validate and report; do not update annotation files")
    parser.add_argument("--fix-from-report", action="store_true",
                        help="Skip validation; fix only episodes listed as bad in the existing report")
    parser.add_argument("--retry", type=int, default=2,
                        help="API retries on failure (default: 2)")
    args = parser.parse_args()

    report_path = ANNOTATION_DIR / "move_boundary_validation_report.json"

    # --fix-from-report: load bad episodes from existing report and jump straight to fix
    if args.fix_from_report:
        if not report_path.exists():
            logger.error("No report found at %s — run with --dry-run first", report_path)
            return
        with open(report_path) as f:
            prev = json.load(f)
        bad_list = prev.get("bad_episodes", [])
        logger.info("Fixing %d episodes from existing report...", len(bad_list))
        fixed_episodes, failed_fix = [], []
        for info in bad_list:
            episode_idx = info["episode_idx"]
            task        = info["task"]
            json_path   = ANNOTATION_DIR / f"annotation_episode_{episode_idx:06d}.json"
            with open(json_path) as f:
                annotation = json.load(f)
            grasp_seg = next((s for s in annotation["subtask_segments"] if s["subtask_id"] == 2), None)
            grasp_end_frame = grasp_seg["end_frame"] if grasp_seg else 0
            fps      = annotation["fps"]
            duration = annotation["duration_sec"]
            move_seg = next((s for s in annotation["subtask_segments"] if s["subtask_id"] == 3), None)
            old_move_end = move_seg["end_frame"] if move_seg else 0
            new_move_end = redetect_move_place_boundary(
                episode_idx, task, grasp_end_frame, fps, duration, retry=args.retry
            )
            if new_move_end is None:
                logger.error("Episode %d: re-detection failed", episode_idx)
                failed_fix.append(episode_idx)
                continue
            place_seg  = next((s for s in annotation["subtask_segments"] if s["subtask_id"] == 4), None)
            place_end  = place_seg["end_frame"] if place_seg else annotation["total_frames"] - 1
            new_move_end = max(grasp_end_frame + 1, min(new_move_end, place_end - 1))
            logger.info("Episode %d: move_end  %d → %d  (%.2fs → %.2fs)",
                        episode_idx, old_move_end, new_move_end,
                        old_move_end / fps, new_move_end / fps)
            for seg in annotation["subtask_segments"]:
                if seg["subtask_id"] == 3:
                    seg["end_frame"] = new_move_end
                    seg["end_sec"]   = round(new_move_end / fps, 3)
                elif seg["subtask_id"] == 4:
                    seg["start_frame"] = new_move_end + 1
                    seg["start_sec"]   = round((new_move_end + 1) / fps, 3)
            annotation["move_place_method"] = "vlm_revalidated"
            with open(json_path, "w") as f:
                json.dump(annotation, f, indent=2)
            logger.info("Episode %d: annotation updated", episode_idx)
            _re_extract_subgoal_images(episode_idx, annotation)
            fixed_episodes.append({"episode_idx": episode_idx, "task": task,
                                    "old_move_end": old_move_end, "new_move_end": new_move_end})
        # Print fix summary
        print("\n" + "=" * 70)
        print("FIX REPORT")
        print("=" * 70)
        print(f"Fixed  : {len(fixed_episodes)}")
        print(f"Failed : {len(failed_fix)}")
        if fixed_episodes:
            for info in fixed_episodes:
                print(f"  episode {info['episode_idx']:>4d}  "
                      f"{info['old_move_end']} → {info['new_move_end']}  task={info['task']!r}")
        if failed_fix:
            print(f"Failed to fix: {failed_fix}")
        return

    # Collect annotation files
    all_json = sorted(ANNOTATION_DIR.glob("annotation_episode_*.json"))
    if args.episodes is not None:
        ep_set  = set(args.episodes)
        all_json = [p for p in all_json
                    if int(re.search(r"(\d+)", p.stem).group(1)) in ep_set]

    logger.info("Checking %d episodes...", len(all_json))

    bad_episodes      = []   # list of dicts with episode info
    fixed_episodes    = []
    failed_fix        = []

    for json_path in all_json:
        with open(json_path) as f:
            annotation = json.load(f)

        episode_idx = annotation["episode_idx"]
        task        = annotation["task"]

        # Find move segment
        move_seg = next((s for s in annotation["subtask_segments"] if s["subtask_id"] == 3), None)
        if move_seg is None:
            logger.warning("Episode %d: no move segment found, skipping", episode_idx)
            continue

        move_end_frame = move_seg["end_frame"]

        # Find subgoal images
        camhigh_img, wrist_img = _find_move_subgoal_images(episode_idx)
        if camhigh_img is None:
            logger.warning("Episode %d: no move subgoal cam_high image found, skipping", episode_idx)
            continue

        # Ask Doubao
        is_above, answer = validate_move_subgoal(camhigh_img, wrist_img, task)

        obj    = TASK_META[task]["object_name"]
        target = TASK_META[task]["target_name"]

        if is_above:
            logger.info("Episode %d [OK]  move_end=%d — %s is above %s  (answer: %r)",
                        episode_idx, move_end_frame, obj, target, answer)
            continue

        # --- BAD: banana not yet above plate ---
        logger.warning("Episode %d [BAD] move_end=%d — %s NOT above %s yet  (answer: %r)",
                       episode_idx, move_end_frame, obj, target, answer)
        bad_episodes.append({
            "episode_idx":   episode_idx,
            "task":          task,
            "old_move_end":  move_end_frame,
            "doubao_answer": answer,
        })

        if args.dry_run:
            continue

        # --- Re-detect move→place boundary ---
        grasp_seg = next((s for s in annotation["subtask_segments"] if s["subtask_id"] == 2), None)
        grasp_end_frame = grasp_seg["end_frame"] if grasp_seg else 0
        fps      = annotation["fps"]
        duration = annotation["duration_sec"]

        new_move_end = redetect_move_place_boundary(
            episode_idx, task, grasp_end_frame, fps, duration, retry=args.retry
        )

        if new_move_end is None:
            logger.error("Episode %d: re-detection failed", episode_idx)
            failed_fix.append(episode_idx)
            continue

        # Clamp
        place_seg   = next((s for s in annotation["subtask_segments"] if s["subtask_id"] == 4), None)
        place_end   = place_seg["end_frame"] if place_seg else annotation["total_frames"] - 1
        new_move_end = max(grasp_end_frame + 1, min(new_move_end, place_end - 1))

        logger.info("Episode %d: move_end  %d → %d  (%.2fs → %.2fs)",
                    episode_idx, move_end_frame, new_move_end,
                    move_end_frame / fps, new_move_end / fps)

        # Update annotation
        for seg in annotation["subtask_segments"]:
            if seg["subtask_id"] == 3:
                seg["end_frame"] = new_move_end
                seg["end_sec"]   = round(new_move_end / fps, 3)
            elif seg["subtask_id"] == 4:
                seg["start_frame"] = new_move_end + 1
                seg["start_sec"]   = round((new_move_end + 1) / fps, 3)

        annotation["move_place_method"] = "vlm_revalidated"

        with open(json_path, "w") as f:
            json.dump(annotation, f, indent=2)
        logger.info("Episode %d: annotation updated -> %s", episode_idx, json_path.name)

        # Re-extract subgoal images
        _re_extract_subgoal_images(episode_idx, annotation)

        fixed_episodes.append({
            "episode_idx":   episode_idx,
            "task":          task,
            "old_move_end":  move_end_frame,
            "new_move_end":  new_move_end,
        })

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"VALIDATION REPORT  (dry_run={args.dry_run})")
    print("=" * 70)
    print(f"Total episodes checked : {len(all_json)}")
    print(f"BAD boundaries found   : {len(bad_episodes)}")

    if bad_episodes:
        print("\n--- Episodes with move_end annotated TOO EARLY ---")
        for info in bad_episodes:
            print(f"  episode {info['episode_idx']:>4d}  task={info['task']!r}  "
                  f"old_move_end={info['old_move_end']}  doubao_answer={info['doubao_answer']!r}")

    if not args.dry_run:
        print(f"\nFixed  : {len(fixed_episodes)}")
        print(f"Failed : {len(failed_fix)}")
        if fixed_episodes:
            print("\n--- Fixed episodes ---")
            for info in fixed_episodes:
                print(f"  episode {info['episode_idx']:>4d}  "
                      f"{info['old_move_end']} → {info['new_move_end']}  task={info['task']!r}")
        if failed_fix:
            print(f"\nFailed to fix: {failed_fix}")

    # Save report to file
    report_path = ANNOTATION_DIR / "move_boundary_validation_report.json"
    report = {
        "dry_run":        args.dry_run,
        "total_checked":  len(all_json),
        "bad_count":      len(bad_episodes),
        "bad_episodes":   bad_episodes,
        "fixed_episodes": fixed_episodes if not args.dry_run else [],
        "failed_fix":     failed_fix     if not args.dry_run else [],
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to: {report_path}")


if __name__ == "__main__":
    main()
