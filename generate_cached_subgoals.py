#!/usr/bin/env python3
"""Generate cached GPT subgoal images for the shape-matching task.

Extracts the first frame from each scene's base camera video, then calls the
GPT image generation API to produce 6 sequential subgoal images representing
the full task execution plan.

[Updated]: Uses autoregressive/sequential generation. The output of Step N
is used as the base image input for Step N+1 to ensure visual continuity.

Usage:
    python generate_cached_subgoals.py --api-key sk-xxxxx
    python generate_cached_subgoals.py --scenes scene41 scene42
    GPT_SUBGOAL_API_KEY=sk-xxxxx python generate_cached_subgoals.py
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import pathlib
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from examples.aloha_real.eval_gui.gpt_subgoal_client import GptSubgoalClient

logger = logging.getLogger(__name__)

TEST_VIDEO_DIR = pathlib.Path(__file__).resolve().parent / "test_video"
TASK_DESCRIPTION = "put the shapes into the matching holes"

VIDEO_PATTERN = "*_video_base.mp4"

# ---------------------------------------------------------------------------
# Prompt templates for the 6 sequential subgoals.
#
# [Updated Logic]: Because we are feeding the PREVIOUS generated image into the
# next step, the prompts are now written to describe the *incremental delta*
# (the next immediate action) rather than describing the whole history.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# [Updated] Config-Driven Prompt Templates
# 由于大模型缺乏几何匹配推理能力，我们在这里显式地定义每一步的绝对指令。
# ---------------------------------------------------------------------------

# 1. 显式定义当前场景的静态物理元素（防幻觉）
_SCENE_CONTEXT = (
    "This is a top-down camera view of a dual-arm robot workspace during an ongoing task. "
    "Originally, there are 4 objects. Below them are 3 black plates with holes. "
    "CRITICAL: Strictly respect the current visual state (the task is ongoing). ONLY modify the specific object and arm mentioned in the next action. "
    "Other objects and completed holes MUST remain exactly as they are."
)

# 2. 用户自定义动作序列（你可以根据不同视频随意修改这里）
# 示例：假设顺序是 -> 左手抓红方块 -> 放中间方孔 -> 右手抓蓝长方体 -> 放右侧长方孔 ...
# (请根据你实际的孔位匹配逻辑修改 target_object 和 target_hole)
STEPS_CONFIG = [
    {
        "step_name": "subgoal_step1_grasp",
        "action": "GRASP",
        "arm": "RIGHT arm",
        "target_object": "YELLOW HEXAGON PRISM",
        "target_hole": "LEFT HEXAGON hole" # 仅用于逻辑上下文，实际抓取时不需要画孔
    },
    {
        "step_name": "subgoal_step2_place",
        "action": "PLACE",
        "arm": "RIGHT arm",
        "target_object": "YELLOW HEXAGON PRISM",
        "target_hole": "LEFT HEXAGON hole"
    },
    {
        "step_name": "subgoal_step3_grasp",
        "action": "GRASP",
        "arm": "LEFT arm",
        "target_object": "RED CUBE",
        "target_hole": "MIDDLE SQUARE hole",
    },
    {
        "step_name": "subgoal_step4_place",
        "action": "PLACE",
        "arm": "LEFT arm",
        "target_object": "RED CUBE",
        "target_hole": "MIDDLE SQUARE hole",
    },
    {
        "step_name": "subgoal_step5_grasp",
        "action": "GRASP",
        "arm": "LEFT arm",
        "target_object": "BLUE CUBOID",
        "target_hole": "RIGHTMOST RECTANGLE hole"
    },
    {
        "step_name": "subgoal_step6_place",
        "action": "PLACE",
        "arm": "LEFT arm",
        "target_object": "BLUE CUBOID",
        "target_hole": "RIGHTMOST RECTANGLE hole"
    },
]

# 3. 动态生成最终的 Prompts 列表
SUBGOAL_PROMPTS = []
for config in STEPS_CONFIG:
    name = config["step_name"]
    action = config["action"]
    arm = config["arm"]
    obj = config["target_object"]
    hole = config["target_hole"]

    if action == "GRASP":
        prompt_text = (
            f"{_SCENE_CONTEXT} Based on the provided image, generate the next state: "
            f"The robot uses its {arm} to GRASP the {obj}. "
            f"The gripper of the {arm} is securely closed around the {obj}, lifting it slightly off the table. "
            f"The {obj} is no longer resting flat on the table."
        )
    elif action == "PLACE":
        prompt_text = (
            f"{_SCENE_CONTEXT} Based on the provided image (where the robot is holding the {obj}), "
            f"generate the next state: The robot successfully PLACES the {obj} strictly inside the {hole}. "
            f"The {obj} is now perfectly seated inside the black plate. "
            f"The {arm} has released the object and moved back to a neutral position out of the way."
        )
    else:
        raise ValueError(f"Unknown action: {action}")

    SUBGOAL_PROMPTS.append((name, prompt_text))


# ---------------------------------------------------------------------------
def extract_first_frame(video_path: pathlib.Path) -> np.ndarray:
    """Read the first frame from a video file, return as HWC uint8 RGB."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    ret, frame_bgr = cap.read()
    cap.release()
    if not ret or frame_bgr is None:
        raise RuntimeError(f"Cannot read first frame from: {video_path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def find_base_video(scene_dir: pathlib.Path) -> pathlib.Path:
    """Find the first *_video_base.mp4 in a scene directory."""
    candidates = sorted(scene_dir.glob(VIDEO_PATTERN))
    if not candidates:
        raise FileNotFoundError(
            f"No base video ({VIDEO_PATTERN}) found in {scene_dir}"
        )
    return candidates[0]


def generate_subgoals(
    client: GptSubgoalClient,
    base_image: np.ndarray,
    output_dir: pathlib.Path,
) -> list[pathlib.Path]:
    """Generate 6 subgoal images sequentially and save them to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    # [CRITICAL UPDATE]: Initialize current_image with the base_image.
    # This variable will be updated after every successful step.
    current_image = base_image.copy()

    for i, (filename, prompt) in enumerate(SUBGOAL_PROMPTS, start=1):
        print(f"  [{i}/6] Generating {filename} ...")
        t0 = time.time()

        client._custom_prompt = prompt

        # Pass the CURRENT image (output of the previous step), not the base image
        subgoal_img = client.predict_subgoal(current_image, TASK_DESCRIPTION)

        if subgoal_img is None:
            print(f"  [{i}/6] FAILED - API returned no image.")
            print("  Aborting the sequence because the next step relies on this image.")
            break  # [CRITICAL UPDATE]: If step N fails, we CANNOT generate step N+1.

        elapsed = time.time() - t0
        out_path = output_dir / f"{filename}.jpg"
        img_bgr = cv2.cvtColor(subgoal_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved_paths.append(out_path)
        print(f"  [{i}/6] Saved -> {out_path}  ({elapsed:.1f}s)")

        # [CRITICAL UPDATE]: Set the newly generated image as the input for the NEXT step
        current_image = subgoal_img

    return saved_paths


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate cached GPT subgoal images sequentially.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GPT_SUBGOAL_API_KEY", ""),
        help="GPT image API key (or set GPT_SUBGOAL_API_KEY env)",
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Scene directories to process (e.g. scene41 scene42). "
             "Default: all scene4* directories.",
    )
    parser.add_argument(
        "--video-dir",
        type=pathlib.Path,
        default=TEST_VIDEO_DIR,
        help=f"Root test_video directory (default: {TEST_VIDEO_DIR})",
    )
    parser.add_argument(
        "--output-subdir",
        default="gpt_cached_subgoals",
        help="Subdirectory name for cached outputs (default: gpt_cached_subgoals)",
    )
    parser.add_argument(
        "--model",
        default="gpt-image-2",
        help="GPT image model name (default: gpt-image-2)",
    )
    parser.add_argument(
        "--size",
        default="1024x1024",
        help="Generated image size (default: 1024x1024)",
    )
    parser.add_argument(
        "--log",
        default="INFO",
        help="Log level (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.api_key:
        print("ERROR: No API key provided.")
        print("  Use --api-key sk-xxxxx or set GPT_SUBGOAL_API_KEY env.")
        sys.exit(1)

    # Discover scene directories
    video_dir = args.video_dir.resolve()
    if args.scenes:
        scene_dirs = [video_dir / s for s in args.scenes]
    else:
        scene_dirs = sorted(video_dir.glob("scene4*"))

    if not scene_dirs:
        print(f"No scene directories found in {video_dir}")
        sys.exit(1)

    # Initialize GPT client
    client = GptSubgoalClient(
        api_key=args.api_key,
        model=args.model,
        size=args.size,
    )

    print(f"Processing {len(scene_dirs)} scene(s)...")
    print(f"Model: {args.model}, Size: {args.size}")
    print("-" * 60)

    total_generated = 0
    for scene_dir in scene_dirs:
        if not scene_dir.is_dir():
            print(f"\nSkipping {scene_dir.name}: directory not found")
            continue

        print(f"\n{'='*60}")
        print(f"Scene: {scene_dir.name}")
        print(f"{'='*60}")

        try:
            video_path = find_base_video(scene_dir)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        print(f"  Video: {video_path.name}")
        base_image = extract_first_frame(video_path)
        print(f"  Frame shape: {base_image.shape}")

        output_dir = scene_dir / args.output_subdir
        saved = generate_subgoals(client, base_image, output_dir)
        total_generated += len(saved)

    print(f"\n{'='*60}")
    print(f"Done. Generated {total_generated} subgoal images total.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()