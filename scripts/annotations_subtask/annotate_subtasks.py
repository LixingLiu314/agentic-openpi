"""
annotate_subtasks.py — Subtask annotation for aloha_banana_lerobot_delta dataset.

Uses gripper joint state from parquet data to segment each episode into 4 subtasks:
  1. reach  — gripper open, arm moving toward the banana
  2. grasp  — gripper closing around the banana
  3. move   — gripper fully closed, banana lifted and transported through the air
  4. place  — banana above or in the target plate

Boundaries:
  reach→grasp : derivative-based detection (rapid closing, MIN_RUN=5 consecutive frames)
  grasp→move  : gripper reaches fully closed (within 15% of min)
  move→place  : VLM (cam_high overhead video), validated by single-frame check

Subgoal images (cam_high + wrist camera) are saved for visual inspection.

Output per episode:
  subtask_annotations/annotation_episode_{idx:06d}.json
  subtask_annotations/subgoal_images/episode_{idx:06d}/subtask_{i}_{label}_end_frame{f}_{cam}.jpg
  subtask_annotations/vlm_grasp_fallback_episodes.txt
  subtask_annotations/vlm_move_retry_episodes.txt
"""

import argparse
import base64
import json
import logging
import os
import re
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DOUBAO_API_KEY = "8dc53ee3-fddc-47b5-b419-5888d5e057ff"
DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DOUBAO_MODEL = "doubao-seed-2-0-pro-260215"

DATASET_ROOT = Path("/home/lixingliu/Agent-VLA-eval/Agent-VLA/playground/Datasets/aloha_lerobot")
VIDEO_DIR = DATASET_ROOT / "videos" / "chunk-000" / "observation.images.cam_high"
EPISODES_JSONL = DATASET_ROOT / "meta" / "episodes.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "subtask_annotations"

# ---------------------------------------------------------------------------
# Per-task subtask configs
# ---------------------------------------------------------------------------

TASK_CONFIGS = {
    "put_banana_in_the_green_plate": {
        "task_description": "put the banana in the green plate",
        "object_name": "banana",
        "target_name": "green plate",
        "gripper_state_dim": 13,        # right arm active
        "wrist_cam": "observation.images.cam_right_wrist",
        "subtask_names": [
            "reach the banana",
            "grasp the banana",
            "move the banana to the green plate",
            "place the banana in the green plate",
        ],
        "subtask_criteria": [
            "The robot gripper moves from its initial resting position toward the yellow banana "
            "lying on the white table. The banana has not been touched yet — it remains flat on "
            "the table surface. In the RIGHT view (wrist camera), the gripper fingers are fully "
            "open with a wide gap between them. (At the very start the gripper may not be visible "
            "yet; this still counts as reach.)",
            "The gripper fingers begin closing around the yellow banana. Use the RIGHT view (wrist "
            "camera) as the primary signal: grasp starts the moment the gap between the two black "
            "gripper fingers visibly narrows and they make contact with the banana. The banana is "
            "still resting on the table (not yet lifted) in the LEFT view (overhead camera).",
            "The gripper holds the banana firmly and lifts it off the table. In the LEFT view "
            "(overhead camera) the banana is visibly in the air. In the RIGHT view (wrist camera) "
            "the fingers remain closed around the banana while it is transported toward the green plate.",
            "The banana is being lowered into the green plate and released. In the LEFT view "
            "(overhead camera) the banana is on or inside the green plate. In the RIGHT view "
            "(wrist camera) the gripper fingers open and withdraw.",
        ],
    },
    "put_banana_in_the_red_plate": {
        "task_description": "put the banana in the red plate",
        "object_name": "banana",
        "target_name": "red plate",
        "gripper_state_dim": 6,         # left arm active
        "wrist_cam": "observation.images.cam_left_wrist",
        "subtask_names": [
            "reach the banana",
            "grasp the banana",
            "move the banana to the red plate",
            "place the banana in the red plate",
        ],
        "subtask_criteria": [
            "The robot gripper moves from its initial resting position toward the yellow banana "
            "lying on the white table. The banana has not been touched yet — it remains flat on "
            "the table surface. In the RIGHT view (wrist camera), the gripper fingers are fully "
            "open with a wide gap between them. (At the very start the gripper may not be visible "
            "yet; this still counts as reach.)",
            "The gripper fingers begin closing around the yellow banana. Use the RIGHT view (wrist "
            "camera) as the primary signal: grasp starts the moment the gap between the two black "
            "gripper fingers visibly narrows and they make contact with the banana. The banana is "
            "still resting on the table (not yet lifted) in the LEFT view (overhead camera).",
            "The gripper holds the banana firmly and lifts it off the table. In the LEFT view "
            "(overhead camera) the banana is visibly in the air. In the RIGHT view (wrist camera) "
            "the fingers remain closed around the banana while it is transported toward the red plate.",
            "The banana is being lowered into the red plate and released. In the LEFT view "
            "(overhead camera) the banana is on or inside the red plate. In the RIGHT view "
            "(wrist camera) the gripper fingers open and withdraw.",
        ],
    },
}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def build_move_place_prompt(task: str) -> str:
    """Focused prompt: ask VLM only for the move→place transition timestamp.
    Sends cam_high (overhead) video only.
    """
    cfg = TASK_CONFIGS[task]
    obj = cfg["object_name"]
    target = cfg["target_name"]
    return f"""You are a robot manipulation expert analyzing a robot demo video.

The video shows the overhead camera view of a table scene.

The robot has already grasped the {obj} and is transporting it.
Your task: find the exact timestamp when the robot transitions from MOVING the {obj} to PLACING it.

Definition:
- move  : {obj} is held in the air and has NOT yet reached above the {target}.
- place : {obj} has moved above the {target} (directly overhead or in contact). The transition happens the moment the {obj} is above the {target}, even before it is lowered.

Output ONLY a single JSON object:
{{"move_end_sec": <float>, "place_start_sec": <float>}}

Where move_end_sec == place_start_sec (the transition moment).
No explanation, no other text."""


def build_system_prompt(task: str) -> str:
    cfg = TASK_CONFIGS[task]
    task_desc = cfg["task_description"]
    obj = cfg["object_name"]
    target = cfg["target_name"]
    names = cfg["subtask_names"]
    criteria = cfg["subtask_criteria"]
    return f"""You are a robot manipulation expert analyzing a robot arm demo video.
Task: "{task_desc}"

The video shows TWO camera views side by side:
- LEFT half: overhead (top-down) camera showing the full table scene with the {obj}, plates, and robot arm.
- RIGHT half: wrist camera mounted on the active gripper, giving a close-up view of the gripper fingers and the {obj}.

Use BOTH views together to identify the 4 subtask phases:
1. "{names[0]}"  : {criteria[0]}
2. "{names[1]}"  : {criteria[1]}
3. "{names[2]}" : {criteria[2]}
4. "{names[3]}" : {criteria[3]}

Key decision rules:
- Reach vs Grasp boundary: determined by the RIGHT view (wrist camera).
  Watch for the moment the gripper fingers START to close (gap between fingers narrows) — that is when grasp begins.
- Grasp vs Move boundary: determined by the LEFT view (overhead camera).
  Move begins when the {obj} visibly lifts off the table surface.
- Move vs Place boundary: determined by the LEFT view (overhead camera).
  Place begins when the {obj} is above or in contact with the {target}.

The robot may fail to grasp and retry — backtracking (1 → 2 → 1 → 2) is valid.

Output ONLY a JSON array of contiguous segments covering the full video from 0.0 s to the end.
Each segment: {{"subtask_label": <int 1-4>, "start_sec": <float>, "end_sec": <float>}}

Constraints:
- Segments are contiguous: each start_sec equals the previous end_sec.
- First segment start_sec = 0.0. Last segment end_sec = total video duration.
- No gaps, no overlaps.
- Use the minimum number of segments that accurately describes the motion."""


# ---------------------------------------------------------------------------
# Doubao client (singleton)
# ---------------------------------------------------------------------------

_CLIENT = None


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
            os.environ.pop(var, None)
        import openai
        _CLIENT = openai.OpenAI(api_key=DOUBAO_API_KEY, base_url=DOUBAO_BASE_URL)
    return _CLIENT


# ---------------------------------------------------------------------------
# Video utilities
# ---------------------------------------------------------------------------

def _encode_mp4(mp4_path: Path) -> str:
    with open(mp4_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _make_composite_video(cam_high_path: Path, wrist_path: Path, out_path: Path) -> bool:
    """Side-by-side composite: LEFT = cam_high, RIGHT = wrist cam."""
    cmd = (
        f'ffmpeg -i "{cam_high_path}" -i "{wrist_path}" '
        f'-filter_complex "[0:v][1:v]hstack=inputs=2" '
        f'-c:v libx264 -crf 23 -preset fast '
        f'"{out_path}" -y -loglevel error'
    )
    return os.system(cmd) == 0


def _get_video_info(mp4_path: Path):
    """Return (total_frames, fps, duration_sec) using ffprobe, with cv2 as fallback."""
    import subprocess
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=nb_frames,r_frame_rate,duration",
                "-of", "json", str(mp4_path),
            ],
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
        logger.warning("ffprobe failed (%s), falling back to cv2", e)
        import cv2
        cap = cv2.VideoCapture(str(mp4_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        duration = total_frames / fps
        return total_frames, fps, duration


# ---------------------------------------------------------------------------
# VLM-based detection
# ---------------------------------------------------------------------------

def detect_move_place_boundary(mp4_path: Path, task: str,
                                grasp_end_frame: int, fps: float,
                                duration: float, retry: int = 2):
    """Use VLM to find the move→place transition timestamp.

    Sends the cam_high overhead video with a focused prompt.
    Returns move_end_frame (inclusive), or None on failure.
    """
    system_prompt = build_move_place_prompt(task)
    b64 = _encode_mp4(mp4_path)
    client = _get_client()
    grasp_end_sec = round(grasp_end_frame / fps, 2)

    for attempt in range(retry + 1):
        try:
            resp = client.chat.completions.create(
                model=DOUBAO_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "video_url",
                             "video_url": {"url": f"data:video/mp4;base64,{b64}"}},
                            {"type": "text",
                             "text": (
                                 f"Video: {duration:.2f}s total, {fps:.0f} FPS. "
                                 f"Overhead camera view. "
                                 f"The robot grasped the object at ~{grasp_end_sec}s. "
                                 f"Find when the object first moves above the target plate."
                             )},
                        ],
                    },
                ],
                extra_body={"thinking": {"type": "disabled"}},
            )
            raw = resp.choices[0].message.content.strip()
            logger.debug("VLM move→place response: %s", raw[:300])
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                result = json.loads(match.group())
                t = float(result.get("place_start_sec", result.get("move_end_sec", -1)))
                if t > 0:
                    frame = min(round(t * fps), int(duration * fps) - 2)
                    return frame
            logger.warning("Attempt %d/%d: could not parse move→place response: %r",
                           attempt + 1, retry + 1, raw[:200])
        except Exception as e:
            logger.warning("Attempt %d/%d failed: %s", attempt + 1, retry + 1, e)
            if attempt < retry:
                time.sleep(2)

    logger.warning("VLM move→place failed, falling back to gripper-open signal")
    return None


def detect_reach_grasp_by_vlm(mp4_path: Path, task: str,
                               fps: float, duration: float,
                               retry: int = 2):
    """Use VLM (composite video) to find reach→grasp and grasp→move boundaries.

    Called as fallback when gripper-state detection is unreliable.
    Returns (reach_end_frame, grasp_end_frame) or None on failure.
    """
    wrist_cam = TASK_CONFIGS[task]["wrist_cam"]
    wrist_path = mp4_path.parent.parent / wrist_cam / mp4_path.name
    composite_path = Path(f"/tmp/composite_{mp4_path.stem}.mp4")

    if wrist_path.exists():
        ok = _make_composite_video(mp4_path, wrist_path, composite_path)
        video_to_send = composite_path if ok else mp4_path
    else:
        video_to_send = mp4_path

    cfg = TASK_CONFIGS[task]
    obj = cfg["object_name"]

    system_prompt = f"""You are a robot manipulation expert analyzing a robot demo video.

The video shows TWO views side by side:
- LEFT half: overhead camera (full table scene).
- RIGHT half: wrist camera (close-up of gripper fingers and {obj}).

Find two transition timestamps:
1. reach→grasp: the moment the gripper fingers BEGIN to close around the {obj}.
   Use the RIGHT view (wrist camera): look for when the gap between the black gripper fingers starts narrowing.
2. grasp→move: the moment the gripper is FULLY closed and the {obj} is about to be lifted.
   The gripper fingers stop moving closer together and the {obj} begins to rise off the table.

Output ONLY a JSON object:
{{"reach_end_sec": <float>, "grasp_end_sec": <float>}}

No explanation, no other text."""

    b64 = _encode_mp4(video_to_send)
    client = _get_client()

    for attempt in range(retry + 1):
        try:
            resp = client.chat.completions.create(
                model=DOUBAO_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "video_url",
                         "video_url": {"url": f"data:video/mp4;base64,{b64}"}},
                        {"type": "text",
                         "text": (f"Video: {duration:.2f}s total, {fps:.0f} FPS. "
                                  f"LEFT=overhead, RIGHT=wrist camera. "
                                  f"Find reach→grasp and grasp→move transition timestamps.")},
                    ]},
                ],
                extra_body={"thinking": {"type": "disabled"}},
            )
            raw = resp.choices[0].message.content.strip()
            logger.debug("VLM reach/grasp response: %s", raw[:300])
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                result = json.loads(match.group())
                reach_end = min(round(result["reach_end_sec"] * fps), int(duration * fps) - 2)
                grasp_end = min(round(result["grasp_end_sec"] * fps), int(duration * fps) - 2)
                grasp_end = max(grasp_end, reach_end + 1)
                return reach_end, grasp_end
            logger.warning("Attempt %d/%d: could not parse reach/grasp response: %r",
                           attempt + 1, retry + 1, raw[:200])
        except Exception as e:
            logger.warning("Attempt %d/%d failed: %s", attempt + 1, retry + 1, e)
            if attempt < retry:
                time.sleep(2)

    logger.error("VLM reach/grasp detection failed for %s", mp4_path)
    return None


def validate_reach_end_by_vlm(mp4_path: Path, task: str,
                               reach_end_frame: int, fps: float,
                               duration: float) -> tuple:
    """Ask VLM whether the reach_end frame shows the gripper near the banana.

    Extracts a single frame at reach_end (cam_high + wrist) and sends to VLM.
    Returns (is_valid: bool, reason: str).
    """
    cfg = TASK_CONFIGS[task]
    obj = cfg["object_name"]

    reach_sec = max(0.0, min(reach_end_frame / fps, duration - 0.1))
    tmp_high = Path(f"/tmp/validate_camhigh_{mp4_path.stem}.jpg")
    tmp_wrist_path = mp4_path.parent.parent / cfg["wrist_cam"] / mp4_path.name
    tmp_wrist = Path(f"/tmp/validate_wrist_{mp4_path.stem}.jpg")

    os.system(f'ffmpeg -ss {reach_sec:.3f} -i "{mp4_path}" '
              f'-frames:v 1 -q:v 2 "{tmp_high}" -y -loglevel error')

    has_wrist = False
    if tmp_wrist_path.exists():
        ret = os.system(f'ffmpeg -ss {reach_sec:.3f} -i "{tmp_wrist_path}" '
                        f'-frames:v 1 -q:v 2 "{tmp_wrist}" -y -loglevel error')
        has_wrist = (ret == 0)

    def _img_b64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    content = []
    if tmp_high.exists():
        content.append({"type": "text", "text": "Overhead camera view (reach phase end frame):"})
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{_img_b64(tmp_high)}"}})
    if has_wrist and tmp_wrist.exists():
        content.append({"type": "text", "text": "Wrist camera view (same frame):"})
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{_img_b64(tmp_wrist)}"}})

    content.append({"type": "text", "text": (
        f"This frame is supposed to be the END of the reach phase, "
        f"i.e. the moment just before the gripper starts closing around the {obj}.\n"
        f"Question: Is the gripper already close to / nearly touching the {obj}?\n"
        f"Answer with ONLY one word: 'yes' (gripper is close to the {obj}) "
        f"or 'no' (gripper is still far from the {obj})."
    )})

    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=DOUBAO_MODEL,
            messages=[{"role": "user", "content": content}],
            extra_body={"thinking": {"type": "disabled"}},
        )
        answer = resp.choices[0].message.content.strip().lower()
        logger.debug("VLM reach validation answer: %r", answer)
        if "yes" in answer:
            return True, "vlm confirmed gripper near banana"
        else:
            return False, f"vlm says gripper still far from banana (response: {answer!r})"
    except Exception as e:
        logger.warning("VLM reach validation failed: %s — assuming valid", e)
        return True, f"vlm error, assuming valid: {e}"


def validate_move_end_by_vlm(mp4_path: Path, task: str,
                              move_end_frame: int, fps: float,
                              duration: float) -> tuple:
    """Ask VLM whether the move_end frame shows the banana above the target plate.

    Extracts a single cam_high frame at move_end and sends to VLM.
    Returns (is_valid: bool, reason: str).
    """
    cfg = TASK_CONFIGS[task]
    obj = cfg["object_name"]
    target = cfg["target_name"]

    move_sec = max(0.0, min(move_end_frame / fps, duration - 0.1))
    tmp_high = Path(f"/tmp/validate_move_camhigh_{mp4_path.stem}.jpg")
    os.system(f'ffmpeg -ss {move_sec:.3f} -i "{mp4_path}" '
              f'-frames:v 1 -q:v 2 "{tmp_high}" -y -loglevel error')

    if not tmp_high.exists():
        return True, "could not extract frame, assuming valid"

    def _img_b64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    content = [
        {"type": "text", "text": "Overhead camera view (move phase end frame):"},
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{_img_b64(tmp_high)}"}},
        {"type": "text", "text": (
            f"This frame is supposed to be the END of the move phase, "
            f"i.e. the moment the {obj} has just arrived above the {target}.\n"
            f"Question: Is the {obj} clearly above or on the {target} at this frame?\n"
            f"Answer with ONLY one word: 'yes' (the {obj} is above/on the {target}) "
            f"or 'no' (the {obj} is NOT yet above the {target}, it is still in transit)."
        )},
    ]

    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=DOUBAO_MODEL,
            messages=[{"role": "user", "content": content}],
            extra_body={"thinking": {"type": "disabled"}},
        )
        answer = resp.choices[0].message.content.strip().lower()
        logger.debug("VLM move validation answer: %r", answer)
        if "yes" in answer:
            return True, "vlm confirmed banana above plate"
        else:
            return False, f"vlm says banana NOT above plate (response: {answer!r})"
    except Exception as e:
        logger.warning("VLM move validation failed: %s — assuming valid", e)
        return True, f"vlm error, assuming valid: {e}"


# ---------------------------------------------------------------------------
# Gripper-state based detection
# ---------------------------------------------------------------------------

def _smooth(signal, window: int = 5):
    """Simple box-filter smoothing (no scipy dependency)."""
    import numpy as np
    out = signal.copy().astype(float)
    half = window // 2
    for i in range(len(signal)):
        lo, hi = max(0, i - half), min(len(signal), i + half + 1)
        out[i] = signal[lo:hi].mean()
    return out


def detect_grasp_boundaries(gripper) -> tuple:
    """Detect reach→grasp and grasp→move boundaries from gripper state signal.

    Requires MIN_RUN=5 consecutive frames above threshold to filter noise spikes.

    Args:
        gripper: 1-D numpy array of gripper joint angles (high=open, low=closed).

    Returns:
        (reach_end, grasp_end): inclusive frame indices.
    """
    import numpy as np

    sig = _smooth(gripper, window=7)
    n = len(sig)

    open_val  = sig.max()
    close_val = sig.min()
    span = open_val - close_val

    # --- reach→grasp: detect rapid closing via derivative ---
    velocity = -np.diff(sig, prepend=sig[0])  # positive when closing
    rate_thresh = 0.02 * span

    # Require at least MIN_RUN consecutive frames above threshold to filter noise
    MIN_RUN = 5
    onset = None
    run_len = 0
    for i, v in enumerate(velocity):
        if v > rate_thresh:
            run_len += 1
            if run_len >= MIN_RUN:
                onset = i - MIN_RUN + 1
                break
        else:
            run_len = 0

    if onset is not None:
        reach_end = max(0, onset - 1)
    else:
        # Fallback: 40% threshold from max
        open_thresh = open_val - 0.40 * span
        closing = np.where(sig < open_thresh)[0]
        reach_end = int(closing[0]) - 1 if len(closing) > 0 else n - 2

    # --- grasp→move: first frame gripper plateaus at minimum (within 15% of min) ---
    close_thresh = close_val + 0.15 * span
    start = max(reach_end, 0)
    closed_frames = np.where(sig[start:] <= close_thresh)[0]
    grasp_end = int(start + closed_frames[0]) if len(closed_frames) > 0 else reach_end + 1

    # Clamp
    reach_end = max(0, min(reach_end, n - 2))
    grasp_end = max(reach_end + 1, min(grasp_end, n - 2))

    return reach_end, grasp_end


# ---------------------------------------------------------------------------
# Process one episode
# ---------------------------------------------------------------------------

def process_episode(episode_idx: int, task: str, output_dir: Path,
                    retry: int = 2) -> tuple:
    """Process one episode.

    Returns (success: bool, used_vlm_for_grasp: bool, used_vlm_for_move: bool).
    """
    mp4_path = VIDEO_DIR / f"episode_{episode_idx:06d}.mp4"
    if not mp4_path.exists():
        logger.error("Episode %d: video not found: %s", episode_idx, mp4_path)
        return False, False, False

    json_path = output_dir / f"annotation_episode_{episode_idx:06d}.json"
    if json_path.exists():
        logger.info("Episode %d: already annotated, skipping", episode_idx)
        return True, False, False

    logger.info("Episode %d | task: %s", episode_idx, task)

    import numpy as np
    import pandas as pd

    parquet_path = DATASET_ROOT / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet"
    if not parquet_path.exists():
        logger.error("Episode %d: parquet not found: %s", episode_idx, parquet_path)
        return False, False, False

    df = pd.read_parquet(parquet_path)
    states = np.stack(df["observation.state"].values)
    gripper_dim = TASK_CONFIGS[task]["gripper_state_dim"]
    gripper = states[:, gripper_dim]

    total_frames, fps, duration = _get_video_info(mp4_path)

    # --- Step 1: gripper state → reach/grasp boundaries ---
    used_vlm_for_grasp = False
    reach_end, grasp_end = detect_grasp_boundaries(gripper)

    # Validate reach_end: ask VLM if gripper is near banana at that frame
    is_valid, reason = validate_reach_end_by_vlm(
        mp4_path, task, reach_end_frame=reach_end, fps=fps, duration=duration,
    )
    if not is_valid:
        logger.warning("Episode %d: gripper detection unreliable (%s) → using VLM",
                       episode_idx, reason)
        vlm_result = detect_reach_grasp_by_vlm(mp4_path, task, fps=fps,
                                                duration=duration, retry=retry)
        if vlm_result is not None:
            reach_end, grasp_end = vlm_result
            used_vlm_for_grasp = True
            logger.info("Episode %d: VLM reach/grasp → reach_end=%d grasp_end=%d",
                        episode_idx, reach_end, grasp_end)
        else:
            logger.warning("Episode %d: VLM reach/grasp also failed, keeping gripper result",
                           episode_idx)

    # --- Step 2: VLM → move→place boundary ---
    used_vlm_for_move = False

    def _detect_move_with_fallback():
        m = detect_move_place_boundary(
            mp4_path, task, grasp_end_frame=grasp_end, fps=fps,
            duration=duration, retry=retry,
        )
        if m is None:
            sig = _smooth(gripper, window=7)
            close_val = sig.min()
            open_val  = sig.max()
            close_thresh = close_val + 0.15 * (open_val - close_val)
            start2 = max(grasp_end, 0)
            opening = np.where(sig[start2:] > close_thresh)[0]
            m = int(start2 + opening[0]) - 1 if len(opening) > 0 else total_frames - 2
            logger.info("  move→place fallback (gripper-open): move_end=%d", m)
        return m

    move_end = _detect_move_with_fallback()

    # Validate move_end: ask VLM if banana is above target plate at that frame
    move_valid, move_reason = validate_move_end_by_vlm(
        mp4_path, task, move_end_frame=move_end, fps=fps, duration=duration,
    )
    if not move_valid:
        logger.warning("Episode %d: move_end=%d invalid (%s) → retrying VLM",
                       episode_idx, move_end, move_reason)
        move_end_retry = _detect_move_with_fallback()
        move_valid2, _ = validate_move_end_by_vlm(
            mp4_path, task, move_end_frame=move_end_retry, fps=fps, duration=duration,
        )
        if move_valid2:
            move_end = move_end_retry
            logger.info("Episode %d: retry move_end=%d accepted", episode_idx, move_end)
        else:
            move_end = move_end_retry
            logger.warning("Episode %d: retry move_end=%d still uncertain, keeping it",
                           episode_idx, move_end)
        used_vlm_for_move = True

    place_end = total_frames - 1
    move_end = max(grasp_end + 1, min(move_end, place_end - 1))

    subtask_names = TASK_CONFIGS[task]["subtask_names"]

    def _seg(label, start_f, end_f):
        return {
            "subtask_id": label,
            "label": subtask_names[label - 1],
            "subtask_label": label,
            "start_frame": int(start_f),
            "end_frame": int(end_f),
            "start_sec": round(start_f / fps, 3),
            "end_sec": round(end_f / fps, 3),
        }

    boundaries = [
        _seg(1, 0,             reach_end),
        _seg(2, reach_end + 1, grasp_end),
        _seg(3, grasp_end + 1, move_end),
        _seg(4, move_end  + 1, place_end),
    ]

    logger.info(
        "  reach=[0,%d] grasp=[%d,%d] move=[%d,%d] place=[%d,%d]",
        reach_end, reach_end+1, grasp_end, grasp_end+1, move_end, move_end+1, place_end,
    )

    annotation = {
        "episode_idx": episode_idx,
        "task": task,
        "total_frames": total_frames,
        "duration_sec": duration,
        "fps": fps,
        "gripper_state_dim": gripper_dim,
        "reach_grasp_method": "vlm" if used_vlm_for_grasp else "gripper_state",
        "move_place_method": "vlm_retry" if used_vlm_for_move else "vlm",
        "subtask_segments": boundaries,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(annotation, f, indent=2)
    logger.info("  Saved: %s", json_path.name)

    # Extract subgoal images (end frame of each segment) via ffmpeg
    wrist_cam = TASK_CONFIGS[task]["wrist_cam"]
    wrist_path = mp4_path.parent.parent / wrist_cam / mp4_path.name

    img_dir = output_dir / "subgoal_images" / f"episode_{episode_idx:06d}"
    img_dir.mkdir(parents=True, exist_ok=True)

    for seg in boundaries:
        end_sec = seg["end_sec"]
        label_name = seg["label"]
        seg_id = seg["subtask_id"]
        end_frame = seg["end_frame"]
        label_slug = label_name.replace(" ", "_")
        seek_sec = max(0.0, min(end_sec, duration - 0.1))

        # cam_high frame
        img_name = f"subtask_{seg_id}_{label_slug}_end_frame{end_frame:04d}_camhigh.jpg"
        cmd = (f'ffmpeg -ss {seek_sec:.3f} -i "{mp4_path}" '
               f'-frames:v 1 -q:v 2 "{img_dir / img_name}" -y -loglevel error')
        if os.system(cmd) == 0:
            logger.info("  Subgoal cam_high: %s", img_name)
        else:
            logger.warning("  ffmpeg failed cam_high: subtask %d end_sec=%.2f", seg_id, end_sec)

        # wrist camera frame
        if wrist_path.exists():
            wrist_img_name = f"subtask_{seg_id}_{label_slug}_end_frame{end_frame:04d}_wrist.jpg"
            cmd = (f'ffmpeg -ss {seek_sec:.3f} -i "{wrist_path}" '
                   f'-frames:v 1 -q:v 2 "{img_dir / wrist_img_name}" -y -loglevel error')
            if os.system(cmd) == 0:
                logger.info("  Subgoal wrist:    %s", wrist_img_name)
            else:
                logger.warning("  ffmpeg failed wrist: subtask %d end_sec=%.2f", seg_id, end_sec)

    return True, used_vlm_for_grasp, used_vlm_for_move


# ---------------------------------------------------------------------------
# Load episodes from meta
# ---------------------------------------------------------------------------

def load_episodes():
    episodes = []
    with open(EPISODES_JSONL) as f:
        for line in f:
            ep = json.loads(line)
            episodes.append((ep["episode_index"], ep["tasks"][0]))
    return episodes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Annotate aloha_banana_lerobot_delta subtasks via Doubao VLM."
    )
    parser.add_argument(
        "--episodes", type=int, nargs="+", default=None,
        help="Specific episode indices to process (e.g. --episodes 0 1 2)"
    )
    parser.add_argument(
        "--range", type=int, nargs=2, metavar=("START", "END"), default=None,
        help="Episode index range [START, END)  e.g. --range 0 10"
    )
    parser.add_argument(
        "--task-filter", choices=["green", "red"], default=None,
        help="Only process one task: 'green' or 'red'"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--retry", type=int, default=2,
        help="API retries per episode on failure (default: 2)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    all_episodes = load_episodes()

    if args.episodes:
        ep_set = set(args.episodes)
        episodes = [(idx, task) for idx, task in all_episodes if idx in ep_set]
    elif args.range:
        start, end = args.range
        episodes = [(idx, task) for idx, task in all_episodes if start <= idx < end]
    else:
        episodes = all_episodes

    if args.task_filter == "green":
        episodes = [(idx, task) for idx, task in episodes if "green" in task]
    elif args.task_filter == "red":
        episodes = [(idx, task) for idx, task in episodes if "red" in task]

    logger.info("Processing %d episodes -> %s", len(episodes), output_dir)

    success, failed, vlm_grasp_episodes, vlm_move_episodes = 0, [], [], []
    for episode_idx, task in episodes:
        ok, used_vlm_grasp, used_vlm_move = process_episode(
            episode_idx, task, output_dir, retry=args.retry
        )
        if ok:
            success += 1
            if used_vlm_grasp:
                vlm_grasp_episodes.append(episode_idx)
            if used_vlm_move:
                vlm_move_episodes.append(episode_idx)
        else:
            failed.append(episode_idx)

    logger.info("Done. %d succeeded, %d failed.", success, len(failed))
    if failed:
        logger.warning("Failed episodes: %s", failed)

    if vlm_grasp_episodes:
        logger.warning("%d episodes used VLM for reach/grasp: %s",
                       len(vlm_grasp_episodes), vlm_grasp_episodes)
        vlm_log = output_dir / "vlm_grasp_fallback_episodes.txt"
        with open(vlm_log, "w") as f:
            f.write("Episodes where gripper detection was unreliable and VLM was used for reach/grasp:\n")
            for ep in vlm_grasp_episodes:
                f.write(f"  {ep}\n")
        logger.info("Saved: %s", vlm_log)
    else:
        logger.info("All episodes used gripper state for reach/grasp detection.")

    if vlm_move_episodes:
        logger.warning("%d episodes had move_end retried by VLM validation: %s",
                       len(vlm_move_episodes), vlm_move_episodes)
        move_log = output_dir / "vlm_move_retry_episodes.txt"
        with open(move_log, "w") as f:
            f.write("Episodes where move_end was invalid and retried:\n")
            for ep in vlm_move_episodes:
                f.write(f"  {ep}\n")
        logger.info("Saved: %s", move_log)
    else:
        logger.info("All episodes passed move_end VLM validation.")


if __name__ == "__main__":
    main()
