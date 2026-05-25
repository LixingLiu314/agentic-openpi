#!/usr/bin/env python3
"""Visualize the discrete trajectory points that are actually present in prompts.

This is intentionally different from ``visualize_aloha_fk_trajectory.py``:
that tool draws the dense FK trajectory. This tool parses ``traj_cot`` /
``cot_text_prompts.json`` and overlays only the sparse points and gripper
events that the model sees in its prompt.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from aloha_fk_trajectory_utils import episode_parquet_path
from aloha_fk_trajectory_utils import episode_video_path
from aloha_fk_trajectory_utils import load_dataset_info


_LOC_PAIR_RE = re.compile(r"<loc(\d{4})><loc(\d{4})>", re.IGNORECASE)
_COORD_RE = re.compile(r"\((\d+),\s*(\d+)\)")
_GRIPPER_RE = re.compile(r"\b(?:open gripper|close gripper)\b", re.IGNORECASE)
_ITEM_RE = re.compile(
    r"<loc(\d{4})><loc(\d{4})>|\((\d+),\s*(\d+)\)|\b(?:open gripper|close gripper)\b",
    re.IGNORECASE,
)


ARM_COLORS = {
    "left": {
        "line": (190, 90, 255),
        "point": (255, 120, 210),
        "first": (255, 35, 170),
        "gripper": (255, 210, 255),
    },
    "right": {
        "line": (40, 150, 255),
        "point": (255, 196, 0),
        "first": (255, 40, 40),
        "gripper": (185, 230, 255),
    },
}


@dataclass(frozen=True)
class PromptPoint:
    arm: str
    index: int
    token_x: int
    token_y: int
    xy: tuple[float, float]


@dataclass(frozen=True)
class GripperEvent:
    arm: str
    action: str
    order: int
    anchor_index: int | None
    xy: tuple[float, float] | None


@dataclass(frozen=True)
class PromptOverlay:
    prompt_text: str
    prompt_frame: int
    points_by_arm: dict[str, list[PromptPoint]]
    grippers_by_arm: dict[str, list[GripperEvent]]

    @property
    def point_count(self) -> int:
        return sum(len(points) for points in self.points_by_arm.values())


def _arm_segment(text: str, arm: str) -> str:
    lower = text.lower()
    left_idx = lower.find("left:")
    right_idx = lower.find("right:")
    if arm == "left":
        if left_idx < 0:
            return ""
        end = right_idx if right_idx > left_idx else len(text)
        return text[left_idx + len("left:") : end]
    if right_idx < 0:
        return ""
    return text[right_idx + len("right:") :]


def _to_pixel(
    x_value: int,
    y_value: int,
    *,
    width: int,
    height: int,
    token_range: int,
) -> tuple[float, float]:
    token_range = max(1, int(token_range))
    x = max(0, min(int(x_value), token_range))
    y = max(0, min(int(y_value), token_range))
    return (
        x / token_range * max(0, width - 1),
        y / token_range * max(0, height - 1),
    )


def parse_prompt_overlay(
    prompt_text: str,
    *,
    prompt_frame: int,
    image_size: tuple[int, int],
    coord_range: int = 1000,
    loc_range: int = 1000,
) -> PromptOverlay:
    width, height = image_size
    points_by_arm: dict[str, list[PromptPoint]] = {"left": [], "right": []}
    grippers_by_arm: dict[str, list[GripperEvent]] = {"left": [], "right": []}

    for arm in ("left", "right"):
        segment = _arm_segment(prompt_text or "", arm)
        for order, match in enumerate(_ITEM_RE.finditer(segment)):
            loc_x, loc_y, coord_x, coord_y = match.group(1), match.group(2), match.group(3), match.group(4)
            if loc_x is not None and loc_y is not None:
                token_x, token_y = int(loc_x), int(loc_y)
                xy = _to_pixel(
                    token_x,
                    token_y,
                    width=width,
                    height=height,
                    token_range=loc_range,
                )
                points_by_arm[arm].append(
                    PromptPoint(
                        arm=arm,
                        index=len(points_by_arm[arm]),
                        token_x=token_x,
                        token_y=token_y,
                        xy=xy,
                    )
                )
                continue
            if coord_x is not None and coord_y is not None:
                token_x, token_y = int(coord_x), int(coord_y)
                xy = _to_pixel(
                    token_x,
                    token_y,
                    width=width,
                    height=height,
                    token_range=coord_range,
                )
                points_by_arm[arm].append(
                    PromptPoint(
                        arm=arm,
                        index=len(points_by_arm[arm]),
                        token_x=token_x,
                        token_y=token_y,
                        xy=xy,
                    )
                )
                continue

            gripper = _GRIPPER_RE.fullmatch(match.group(0).strip())
            if gripper is None:
                continue
            points = points_by_arm[arm]
            anchor = points[-1] if points else None
            grippers_by_arm[arm].append(
                GripperEvent(
                    arm=arm,
                    action=gripper.group(0).lower(),
                    order=order,
                    anchor_index=anchor.index if anchor is not None else None,
                    xy=anchor.xy if anchor is not None else None,
                )
            )

    return PromptOverlay(
        prompt_text=prompt_text or "",
        prompt_frame=int(prompt_frame),
        points_by_arm=points_by_arm,
        grippers_by_arm=grippers_by_arm,
    )


def resolve_prompt_frame(
    display_frame: int,
    *,
    total_frames: int,
    update_every: int,
    frame_policy: str,
) -> int:
    display_frame = max(0, min(int(display_frame), max(0, total_frames - 1)))
    update_every = max(1, int(update_every))
    if frame_policy == "per-frame":
        return display_frame

    slot = display_frame // update_every
    if frame_policy == "eval-blocking":
        # Matches the current GUI trajectory pipeline: the annotation captured
        # at boundary n is applied from boundary n+1 onward. Frames in the
        # first two slots therefore still use the initial prompt.
        slot = max(0, slot - 1)
    elif frame_policy != "direct-window":
        raise ValueError(f"Unknown frame policy: {frame_policy}")
    return min(slot * update_every, max(0, total_frames - 1))


def _read_prompt_from_parquet(
    dataset_path: Path,
    episode: int,
    chunks_size: int,
    frame: int,
    traj_column: str,
) -> str:
    path = episode_parquet_path(dataset_path, episode, chunks_size)
    table = pq.read_table(path, columns=[traj_column])
    if frame < 0 or frame >= table.num_rows:
        raise IndexError(f"Frame {frame} is outside {path} with {table.num_rows} rows")
    value = table[traj_column][frame].as_py()
    return "" if value is None else str(value)


def _load_cot_prompts(cot_json: Path) -> dict[str, Any]:
    payload = json.loads(cot_json.expanduser().read_text(encoding="utf-8"))
    if "prompts" in payload:
        return payload
    return {"metadata": {}, "prompts": payload}


def _prompt_from_cot(
    cot_payload: dict[str, Any],
    episode: int,
    frame: int,
) -> str:
    prompts = cot_payload.get("prompts", {})
    ep_prompts = prompts.get(str(episode), {})
    if str(frame) in ep_prompts:
        return str(ep_prompts[str(frame)])
    if not ep_prompts:
        return ""
    available = sorted(int(key) for key in ep_prompts if str(key).isdigit())
    if not available:
        return ""
    nearest = max((item for item in available if item <= frame), default=available[0])
    return str(ep_prompts.get(str(nearest), ""))


def parquet_has_column(dataset_path: Path, episode: int, chunks_size: int, column: str) -> bool:
    path = episode_parquet_path(dataset_path, episode, chunks_size)
    if not path.exists():
        return False
    schema = pq.read_schema(path)
    return column in schema.names


class PromptProvider:
    def __init__(
        self,
        *,
        dataset_path: Path,
        cot_payload: dict[str, Any] | None,
        episode: int,
        chunks_size: int,
        prompt_source: str,
        traj_column: str,
    ) -> None:
        self.dataset_path = dataset_path
        self.cot_payload = cot_payload
        self.episode = int(episode)
        self.chunks_size = int(chunks_size)
        self.traj_column = traj_column
        self.source = prompt_source
        if self.source == "auto":
            self.source = (
                "parquet"
                if parquet_has_column(dataset_path, episode, chunks_size, traj_column)
                else "cot-json"
            )

        self._parquet_values: list[Any] | None = None
        if self.source == "parquet":
            path = episode_parquet_path(dataset_path, episode, chunks_size)
            table = pq.read_table(path, columns=[traj_column])
            self._parquet_values = table[traj_column].to_pylist()
        elif self.source == "cot-json":
            if cot_payload is None:
                raise ValueError("cot_payload is required for cot-json prompt source")
        else:
            raise ValueError(f"Unknown prompt source: {prompt_source}")

    @property
    def label(self) -> str:
        if self.source == "parquet":
            return f"parquet:{self.traj_column}"
        return "cot-json"

    def get(self, frame: int) -> tuple[str, str]:
        frame = int(frame)
        if self.source == "parquet":
            assert self._parquet_values is not None
            if frame < 0 or frame >= len(self._parquet_values):
                raise IndexError(
                    f"Frame {frame} is outside episode {self.episode} with {len(self._parquet_values)} rows"
                )
            value = self._parquet_values[frame]
            return ("" if value is None else str(value)), self.label
        assert self.cot_payload is not None
        return _prompt_from_cot(self.cot_payload, self.episode, frame), self.label


def load_prompt_text(
    *,
    dataset_path: Path,
    cot_payload: dict[str, Any] | None,
    episode: int,
    chunks_size: int,
    prompt_frame: int,
    prompt_source: str,
    traj_column: str,
) -> tuple[str, str]:
    source = prompt_source
    if source == "auto":
        source = "parquet" if parquet_has_column(dataset_path, episode, chunks_size, traj_column) else "cot-json"
    if source == "parquet":
        return (
            _read_prompt_from_parquet(dataset_path, episode, chunks_size, prompt_frame, traj_column),
            f"parquet:{traj_column}",
        )
    if cot_payload is None:
        raise ValueError("cot_payload is required for cot-json prompt source")
    return _prompt_from_cot(cot_payload, episode, prompt_frame), "cot-json"


def parse_frame_tokens(values: list[str] | None) -> list[int]:
    frames: list[int] = []
    if not values:
        return frames
    for raw_value in values:
        for token in raw_value.replace(",", " ").split():
            if ":" not in token:
                frames.append(int(token))
                continue
            parts = token.split(":")
            if len(parts) not in (2, 3):
                raise ValueError(f"Invalid frame range {token!r}; expected start:stop[:step]")
            start = int(parts[0]) if parts[0] else 0
            stop = int(parts[1])
            step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
            if step <= 0:
                raise ValueError(f"Frame range step must be positive: {token!r}")
            frames.extend(range(start, stop, step))
    return sorted(dict.fromkeys(frames))


def selected_frames_for_episode(
    *,
    total_frames: int,
    requested_frames: list[int],
    update_every: int,
    max_images: int | None,
) -> list[int]:
    if requested_frames:
        frames = [frame for frame in requested_frames if 0 <= frame < total_frames]
    else:
        frames = list(range(0, total_frames, max(1, update_every)))
    if max_images is not None:
        frames = frames[: max(0, int(max_images))]
    return frames


def draw_text_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    lines: list[str],
    *,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int] = (0, 0, 0, 215),
) -> None:
    x, y = xy
    line_height = 16
    widths = [draw.textlength(line, font=font) for line in lines]
    box_w = int(max(widths, default=0)) + 12
    box_h = line_height * len(lines) + 10
    draw.rectangle((x, y, x + box_w, y + box_h), fill=fill)
    for idx, line in enumerate(lines):
        draw.text((x + 6, y + 5 + idx * line_height), line, fill=(255, 255, 255, 255), font=font)


def draw_prompt_points(
    image: Image.Image,
    overlay: PromptOverlay,
    *,
    episode: int,
    display_frame: int,
    frame_policy: str,
    prompt_source_label: str,
    font: ImageFont.ImageFont,
    marker_radius: int = 8,
    line_width: int = 4,
    show_prompt_text: bool = False,
) -> Image.Image:
    out = image.convert("RGB")
    draw = ImageDraw.Draw(out, "RGBA")

    for arm in ("left", "right"):
        colors = ARM_COLORS[arm]
        points = overlay.points_by_arm[arm]
        xy_points = [point.xy for point in points]
        if len(xy_points) >= 2:
            draw.line(xy_points, fill=colors["line"] + (190,), width=line_width, joint="curve")

        for point in points:
            x, y = point.xy
            fill = colors["first"] if point.index == 0 else colors["point"]
            r = marker_radius + 1 if point.index == 0 else marker_radius
            draw.ellipse((x - r, y - r, x + r, y + r), fill=fill + (235,), outline=(255, 255, 255, 255), width=2)
            label = f"{arm[0].upper()}{point.index}"
            draw.text((x + r + 4, y - r - 4), label, fill=fill + (255,), font=font)

        for event_idx, event in enumerate(overlay.grippers_by_arm[arm]):
            if event.xy is None:
                continue
            x, y = event.xy
            y_offset = 18 + 14 * (event_idx % 3)
            if arm == "left":
                y_offset = -y_offset
            label = "open" if event.action.startswith("open") else "close"
            text_xy = (int(x + 12), int(y + y_offset))
            draw.line((x, y, text_xy[0] - 2, text_xy[1] + 7), fill=colors["gripper"] + (210,), width=2)
            draw_text_box(
                draw,
                text_xy,
                [f"{arm[0].upper()} {label}"],
                font=font,
                fill=colors["line"] + (215,),
            )

    counts = {arm: len(overlay.points_by_arm[arm]) for arm in ("left", "right")}
    gripper_counts = {arm: len(overlay.grippers_by_arm[arm]) for arm in ("left", "right")}
    lines = [
        f"episode {episode:06d}  frame {display_frame}",
        f"prompt_frame {overlay.prompt_frame}  policy {frame_policy}",
        f"source {prompt_source_label}",
        f"left {counts['left']} pts / {gripper_counts['left']} grip",
        f"right {counts['right']} pts / {gripper_counts['right']} grip",
    ]
    if overlay.point_count == 0:
        lines.append("no trajectory points parsed")
    draw_text_box(draw, (10, 10), lines, font=font)

    if show_prompt_text:
        wrapped = wrap_text(overlay.prompt_text, width=95)
        draw_text_box(draw, (10, out.height - 16 * len(wrapped) - 18), wrapped, font=font)
    return out


def wrap_text(text: str, width: int) -> list[str]:
    words = str(text or "").split()
    lines: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        extra = 1 if current else 0
        if current and current_len + extra + len(word) > width:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += extra + len(word)
    if current:
        lines.append(" ".join(current))
    return lines[-10:] if len(lines) > 10 else lines


def frame_to_image_map(video_path: Path, frame_indices: list[int]) -> tuple[dict[int, Image.Image], tuple[int, int], Fraction]:
    wanted = set(frame_indices)
    if not wanted:
        return {}, (0, 0), Fraction(30, 1)
    images: dict[int, Image.Image] = {}
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    fps = stream.average_rate or Fraction(30, 1)
    width, height = int(stream.width), int(stream.height)
    last_needed = max(wanted)
    try:
        for idx, frame in enumerate(tqdm(container.decode(stream), desc="Decoding selected frames")):
            if idx > last_needed:
                break
            if idx in wanted:
                images[idx] = Image.fromarray(frame.to_ndarray(format="rgb24"))
    finally:
        container.close()
    missing = sorted(wanted.difference(images))
    if missing:
        raise RuntimeError(f"Could not decode requested frames: {missing[:10]}")
    return images, (width, height), fps


def make_contact_sheet(
    rendered: list[tuple[int, Image.Image]],
    output_path: Path,
    *,
    thumb_width: int = 480,
    columns: int = 2,
) -> None:
    if not rendered:
        return
    columns = max(1, int(columns))
    thumbs: list[Image.Image] = []
    for _, image in rendered:
        scale = thumb_width / max(1, image.width)
        thumb_height = max(1, int(round(image.height * scale)))
        thumbs.append(image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS))
    rows = int(math.ceil(len(thumbs) / columns))
    sheet_w = columns * thumb_width
    sheet_h = rows * max(thumb.height for thumb in thumbs)
    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(18, 18, 18))
    for idx, thumb in enumerate(thumbs):
        row = idx // columns
        col = idx % columns
        sheet.paste(thumb, (col * thumb_width, row * thumb.height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


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


def render_video(
    *,
    video_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    info: dict[str, Any],
    cot_payload: dict[str, Any] | None,
    total_frames: int,
    chunks_size: int,
) -> None:
    input_container = av.open(str(video_path))
    in_stream = input_container.streams.video[0]
    fps = in_stream.average_rate or Fraction(int(info.get("fps", 30)), 1)
    width, height = int(in_stream.width), int(in_stream.height)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_container = av.open(str(output_path), "w")
    out_stream = add_stream(output_container, args.codec, fps, width, height)
    font = ImageFont.load_default()
    prompt_provider = PromptProvider(
        dataset_path=args.dataset_path,
        cot_payload=cot_payload,
        episode=args.episode,
        chunks_size=chunks_size,
        prompt_source=args.prompt_source,
        traj_column=args.traj_column,
    )

    try:
        for frame_idx, frame in enumerate(tqdm(input_container.decode(in_stream), desc="Rendering prompt video")):
            if args.max_video_frames is not None and frame_idx >= args.max_video_frames:
                break
            prompt_frame = resolve_prompt_frame(
                frame_idx,
                total_frames=total_frames,
                update_every=args.update_every,
                frame_policy=args.frame_policy,
            )
            prompt_text, source_label = prompt_provider.get(prompt_frame)
            overlay = parse_prompt_overlay(
                prompt_text,
                prompt_frame=prompt_frame,
                image_size=(width, height),
                coord_range=args.coord_range,
                loc_range=args.loc_range,
            )
            image = Image.fromarray(frame.to_ndarray(format="rgb24"))
            image = draw_prompt_points(
                image,
                overlay,
                episode=args.episode,
                display_frame=frame_idx,
                frame_policy=args.frame_policy,
                prompt_source_label=source_label,
                font=font,
                marker_radius=args.marker_radius,
                line_width=args.line_width,
                show_prompt_text=args.show_prompt_text,
            )
            out_frame = av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")
            for packet in out_stream.encode(out_frame):
                output_container.mux(packet)
        for packet in out_stream.encode():
            output_container.mux(packet)
    finally:
        input_container.close()
        output_container.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Overlay the sparse trajectory points from traj prompts onto dataset camera frames."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("playground/Datasets/aloha_letter"),
        help="LeRobot dataset root.",
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--cot-json",
        type=Path,
        default=None,
        help="Defaults to <dataset-path>/trajectory_data/cot_text_prompts.json.",
    )
    parser.add_argument(
        "--prompt-source",
        choices=["auto", "cot-json", "parquet"],
        default="auto",
        help="auto prefers parquet traj_cot if present, otherwise cot_text_prompts.json.",
    )
    parser.add_argument("--traj-column", default="traj_cot")
    parser.add_argument(
        "--frame-policy",
        choices=["direct-window", "eval-blocking", "per-frame"],
        default="direct-window",
        help=(
            "direct-window shows the prompt anchored at the current annotation window; "
            "eval-blocking mimics the GUI semi-blocking prompt lag; per-frame uses the exact frame key."
        ),
    )
    parser.add_argument("--update-every", type=int, default=60, help="Trajectory annotation/control interval.")
    parser.add_argument("--frames", nargs="*", default=None, help="Frame ids/ranges, e.g. 0 60 120 or 0:600:60.")
    parser.add_argument("--max-images", type=int, default=None, help="Limit still-image/contact-sheet frames.")
    parser.add_argument("--video-key", default="observation.images.cam_high")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--contact-sheet", type=Path, default=None)
    parser.add_argument("--no-contact-sheet", action="store_true")
    parser.add_argument("--render-video", action="store_true")
    parser.add_argument("--video-output", type=Path, default=None)
    parser.add_argument("--max-video-frames", type=int, default=None)
    parser.add_argument("--codec", default="libx264")
    parser.add_argument("--coord-range", type=int, default=1000)
    parser.add_argument("--loc-range", type=int, default=1000)
    parser.add_argument("--marker-radius", type=int, default=8)
    parser.add_argument("--line-width", type=int, default=4)
    parser.add_argument("--sheet-columns", type=int, default=2)
    parser.add_argument("--thumb-width", type=int, default=480)
    parser.add_argument("--show-prompt-text", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.dataset_path = args.dataset_path.expanduser()
    if args.cot_json is None:
        args.cot_json = args.dataset_path / "trajectory_data" / "cot_text_prompts.json"
    args.cot_json = args.cot_json.expanduser()

    info = load_dataset_info(args.dataset_path)
    chunks_size = int(info.get("chunks_size", 1000))
    video_path = episode_video_path(args.dataset_path, args.episode, chunks_size, args.video_key)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    parquet_path = episode_parquet_path(args.dataset_path, args.episode, chunks_size)
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)
    total_frames = pq.read_metadata(parquet_path).num_rows

    cot_payload = None
    if args.prompt_source in ("auto", "cot-json") and args.cot_json.exists():
        cot_payload = _load_cot_prompts(args.cot_json)
    elif args.prompt_source == "cot-json":
        raise FileNotFoundError(args.cot_json)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            args.dataset_path
            / "trajectory_data"
            / "prompt_point_visualizations"
            / f"episode_{args.episode:06d}_{args.frame_policy}"
        )
    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_frames = parse_frame_tokens(args.frames)
    selected_frames = selected_frames_for_episode(
        total_frames=total_frames,
        requested_frames=requested_frames,
        update_every=args.update_every,
        max_images=args.max_images,
    )
    frame_images, image_size, _ = frame_to_image_map(video_path, selected_frames)
    font = ImageFont.load_default()
    prompt_provider = PromptProvider(
        dataset_path=args.dataset_path,
        cot_payload=cot_payload,
        episode=args.episode,
        chunks_size=chunks_size,
        prompt_source=args.prompt_source,
        traj_column=args.traj_column,
    )

    rendered: list[tuple[int, Image.Image]] = []
    report_items: list[dict[str, Any]] = []
    for display_frame in selected_frames:
        prompt_frame = resolve_prompt_frame(
            display_frame,
            total_frames=total_frames,
            update_every=args.update_every,
            frame_policy=args.frame_policy,
        )
        prompt_text, source_label = prompt_provider.get(prompt_frame)
        overlay = parse_prompt_overlay(
            prompt_text,
            prompt_frame=prompt_frame,
            image_size=image_size,
            coord_range=args.coord_range,
            loc_range=args.loc_range,
        )
        image = draw_prompt_points(
            frame_images[display_frame],
            overlay,
            episode=args.episode,
            display_frame=display_frame,
            frame_policy=args.frame_policy,
            prompt_source_label=source_label,
            font=font,
            marker_radius=args.marker_radius,
            line_width=args.line_width,
            show_prompt_text=args.show_prompt_text,
        )
        image_path = output_dir / f"episode_{args.episode:06d}_frame_{display_frame:06d}_prompt_{prompt_frame:06d}.png"
        image.save(image_path)
        rendered.append((display_frame, image))
        report_items.append(
            {
                "display_frame": int(display_frame),
                "prompt_frame": int(prompt_frame),
                "source": source_label,
                "output": str(image_path),
                "left_points": len(overlay.points_by_arm["left"]),
                "right_points": len(overlay.points_by_arm["right"]),
                "left_gripper_events": [event.action for event in overlay.grippers_by_arm["left"]],
                "right_gripper_events": [event.action for event in overlay.grippers_by_arm["right"]],
                "prompt_text": prompt_text,
            }
        )

    if rendered and not args.no_contact_sheet:
        sheet_path = args.contact_sheet
        if sheet_path is None:
            sheet_path = output_dir / f"episode_{args.episode:06d}_prompt_points_sheet.jpg"
        sheet_path = sheet_path.expanduser()
        make_contact_sheet(
            rendered,
            sheet_path,
            thumb_width=args.thumb_width,
            columns=args.sheet_columns,
        )
        print(f"Saved contact sheet -> {sheet_path}")

    if args.render_video:
        video_output = args.video_output
        if video_output is None:
            video_output = output_dir / f"episode_{args.episode:06d}_{args.frame_policy}_prompt_points.mp4"
        video_output = video_output.expanduser()
        render_video(
            video_path=video_path,
            output_path=video_output,
            args=args,
            info=info,
            cot_payload=cot_payload,
            total_frames=total_frames,
            chunks_size=chunks_size,
        )
        print(f"Saved prompt-point video -> {video_output}")

    report = {
        "dataset_path": str(args.dataset_path),
        "episode": int(args.episode),
        "video_path": str(video_path),
        "cot_json": str(args.cot_json),
        "prompt_source": args.prompt_source,
        "traj_column": args.traj_column,
        "frame_policy": args.frame_policy,
        "update_every": int(args.update_every),
        "coord_range": int(args.coord_range),
        "loc_range": int(args.loc_range),
        "num_images": len(report_items),
        "frames": report_items,
    }
    report_path = output_dir / "prompt_point_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {len(report_items)} prompt-point frames -> {output_dir}")
    print(f"Saved report -> {report_path}")


if __name__ == "__main__":
    main()
