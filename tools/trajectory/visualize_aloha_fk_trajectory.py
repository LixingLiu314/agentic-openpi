#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from aloha_fk_trajectory_utils import (
    episode_video_path,
    in_image_bounds,
    load_dataset_info,
    scale_uv_to_raw,
)


COLORS = {
    "text_bg": (0, 0, 0),
    "text": (255, 255, 255),
}

ARM_COLORS = {
    "left": {
        "full": (190, 90, 255),
        "history": (255, 120, 210),
        "current": (255, 35, 170),
        "future": (155, 75, 245),
    },
    "right": {
        "full": (255, 196, 0),
        "history": (0, 220, 120),
        "current": (255, 40, 40),
        "future": (40, 150, 255),
    },
}


def load_trajectory_result(traj_json: Path, episode_index: int) -> tuple[dict, dict]:
    payload = json.loads(traj_json.expanduser().read_text(encoding="utf-8"))
    for result in payload.get("results", []):
        if int(result["episode_index"]) == episode_index:
            return payload.get("metadata", {}), result
    raise KeyError(f"Episode {episode_index} not found in {traj_json}")


def positions_for_arm(result: dict, arm: str | None, metadata: dict) -> dict[int, np.ndarray]:
    raw_key_by_arm = result.get("all_frame_eef_positions_raw_by_arm")
    raw_primary = result.get("all_frame_eef_positions_raw")
    output_by_arm = result.get("all_frame_eef_positions_by_arm")

    primary_arm = result.get("primary_arm", metadata.get("primary_arm", "right"))
    arm = arm or primary_arm

    if raw_key_by_arm and arm in raw_key_by_arm:
        raw_positions = raw_key_by_arm[arm]
        return {int(frame): np.asarray(value, dtype=np.float64) for frame, value in raw_positions.items()}
    if raw_primary and arm == primary_arm:
        return {int(frame): np.asarray(value, dtype=np.float64) for frame, value in raw_primary.items()}

    if output_by_arm and arm in output_by_arm:
        output_positions = output_by_arm[arm]
    elif arm == primary_arm:
        output_positions = result["all_frame_eef_positions"]
    else:
        raise KeyError(f"No trajectory positions for arm={arm!r}. Regenerate with --include-arms both.")

    raw_image_size = tuple(metadata["raw_image_size"])
    output_image_size = tuple(metadata["output_image_size"])
    return {
        int(frame): scale_uv_to_raw(np.asarray(value, dtype=np.float64), raw_image_size, output_image_size)
        for frame, value in output_positions.items()
    }


def sorted_points_until(points: dict[int, np.ndarray], frame_index: int) -> list[tuple[int, tuple[float, float]]]:
    return [
        (frame, (float(uv[0]), float(uv[1])))
        for frame, uv in sorted(points.items())
        if frame <= frame_index
    ]


def sorted_points_from(points: dict[int, np.ndarray], frame_index: int, limit: int) -> list[tuple[int, tuple[float, float]]]:
    selected = [
        (frame, (float(uv[0]), float(uv[1])))
        for frame, uv in sorted(points.items())
        if frame >= frame_index
    ]
    return selected[:limit] if limit > 0 else selected


def draw_polyline(draw: ImageDraw.ImageDraw, pts: list[tuple[float, float]], fill: tuple[int, int, int], width: int) -> None:
    if len(pts) >= 2:
        draw.line(pts, fill=fill, width=width, joint="curve")


def draw_marker(draw: ImageDraw.ImageDraw, uv: np.ndarray, radius: int, fill: tuple[int, int, int]) -> None:
    x, y = float(uv[0]), float(uv[1])
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=(255, 255, 255), width=2)
    draw.line((x - radius - 4, y, x + radius + 4, y), fill=(255, 255, 255), width=1)
    draw.line((x, y - radius - 4, x, y + radius + 4), fill=(255, 255, 255), width=1)


def draw_arm_trajectory(
    draw: ImageDraw.ImageDraw,
    arm: str,
    positions: dict[int, np.ndarray],
    frame_idx: int,
    image_size: tuple[int, int],
    future_frames: int,
    point_stride: int,
    line_width: int,
    marker_radius: int,
) -> tuple[str, bool]:
    colors = ARM_COLORS[arm]
    all_pts = [(float(uv[0]), float(uv[1])) for _, uv in sorted(positions.items())]
    draw_polyline(draw, all_pts, colors["full"] + (85,), max(1, line_width - 1))

    history = [pt for _, pt in sorted_points_until(positions, frame_idx)]
    future = [pt for _, pt in sorted_points_from(positions, frame_idx, future_frames)]
    draw_polyline(draw, history, colors["history"] + (210,), line_width)
    draw_polyline(draw, future, colors["future"] + (150,), max(1, line_width - 1))

    if point_stride > 0:
        for point_frame, uv in positions.items():
            if point_frame % point_stride == 0 and in_image_bounds(uv, image_size):
                x, y = float(uv[0]), float(uv[1])
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=colors["full"] + (190,))

    current_uv = positions.get(frame_idx)
    visible = False
    uv_text = "missing"
    if current_uv is not None:
        uv_text = f"({current_uv[0]:.1f}, {current_uv[1]:.1f})"
        visible = in_image_bounds(current_uv, image_size)
        if visible:
            draw_marker(draw, current_uv, marker_radius, colors["current"])
            label_xy = (float(current_uv[0]) + marker_radius + 5, float(current_uv[1]) - marker_radius - 5)
            draw.text(label_xy, arm[0].upper(), fill=colors["current"] + (255,))

    return uv_text, visible


def draw_text_box(draw: ImageDraw.ImageDraw, xy: tuple[int, int], lines: list[str], font) -> None:
    x, y = xy
    line_height = 16
    widths = [draw.textlength(line, font=font) for line in lines]
    box_w = int(max(widths, default=0)) + 12
    box_h = line_height * len(lines) + 10
    draw.rectangle((x, y, x + box_w, y + box_h), fill=COLORS["text_bg"])
    for i, line in enumerate(lines):
        draw.text((x + 6, y + 5 + i * line_height), line, fill=COLORS["text"], font=font)


def add_stream(output_container, codec: str, fps: Fraction | int, width: int, height: int):
    try:
        stream = output_container.add_stream(codec, rate=fps)
    except av.FFmpegError:
        stream = output_container.add_stream("mpeg4", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    if codec == "libx264":
        stream.options = {"crf": "18", "preset": "veryfast"}
    return stream


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render front-camera videos annotated with generated FK trajectories.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("playground/Datasets/aloha_banana_lerobot_absolute"),
        help="LeRobot dataset root.",
    )
    parser.add_argument("--traj-json", type=Path, required=True, help="JSON produced by generate_aloha_fk_trajectory.py.")
    parser.add_argument("--episode", type=int, required=True, help="Episode index to visualize.")
    parser.add_argument("--arm", choices=["left", "right", "both"], default=None, help="Arm to draw. Default: JSON primary arm.")
    parser.add_argument(
        "--video-key",
        default="observation.images.cam_high",
        help="Video key under dataset videos/chunk-xxx.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output video path.")
    parser.add_argument("--codec", default="libx264", help="Output codec for PyAV, e.g. libx264 or mpeg4.")
    parser.add_argument(
        "--future-frames",
        type=int,
        default=60,
        help="Number of future trajectory points to draw from the current frame. 0 means all future points.",
    )
    parser.add_argument("--point-stride", type=int, default=10, help="Draw small dots on the full path every N frames.")
    parser.add_argument("--line-width", type=int, default=3)
    parser.add_argument("--marker-radius", type=int, default=7)
    parser.add_argument("--max-frames", type=int, default=None, help="Optional debugging limit.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    dataset_path = args.dataset_path.expanduser()
    metadata, result = load_trajectory_result(args.traj_json, args.episode)
    arm = args.arm or result.get("primary_arm", metadata.get("primary_arm", "right"))
    arms = ["left", "right"] if arm == "both" else [arm]
    positions_by_arm = {arm_name: positions_for_arm(result, arm_name, metadata) for arm_name in arms}

    info = load_dataset_info(dataset_path)
    chunks_size = int(info.get("chunks_size", 1000))
    video_path = episode_video_path(dataset_path, args.episode, chunks_size, args.video_key)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    output_path = args.output
    if output_path is None:
        output_path = dataset_path / "trajectory_data" / "visualizations" / f"episode_{args.episode:06d}_{arm}_trajectory.mp4"
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_container = av.open(str(video_path))
    in_stream = input_container.streams.video[0]
    fps = in_stream.average_rate or Fraction(int(info.get("fps", 30)), 1)
    width = int(in_stream.width)
    height = int(in_stream.height)

    output_container = av.open(str(output_path), "w")
    out_stream = add_stream(output_container, args.codec, fps, width, height)

    font = ImageFont.load_default()

    try:
        for frame_idx, frame in enumerate(tqdm(input_container.decode(in_stream), desc="Rendering video")):
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break

            image = Image.fromarray(frame.to_ndarray(format="rgb24"))
            draw = ImageDraw.Draw(image, "RGBA")

            text_lines = [f"episode {args.episode:06d}  frame {frame_idx}"]
            for arm_name in arms:
                uv_text, visible = draw_arm_trajectory(
                    draw=draw,
                    arm=arm_name,
                    positions=positions_by_arm[arm_name],
                    frame_idx=frame_idx,
                    image_size=(width, height),
                    future_frames=args.future_frames,
                    point_stride=args.point_stride,
                    line_width=args.line_width,
                    marker_radius=args.marker_radius,
                )
                text_lines.append(f"{arm_name}: uv {uv_text}  visible {visible}")

            draw_text_box(
                draw,
                (10, 10),
                text_lines,
                font,
            )

            out_frame = av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")
            for packet in out_stream.encode(out_frame):
                output_container.mux(packet)

        for packet in out_stream.encode():
            output_container.mux(packet)
    finally:
        input_container.close()
        output_container.close()

    print(f"Saved annotated video to {output_path}")


if __name__ == "__main__":
    main()
