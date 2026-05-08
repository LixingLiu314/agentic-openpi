"""Draw Doubao loc-token trajectories on the front camera image."""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


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


_LOC_PAIR_RE = re.compile(r"<loc(\d{4})><loc(\d{4})>")
_LEFT_SEGMENT_RE = re.compile(r"Left:\s*Go along\s*(.*?)(?:\.\s*Right:|$)", re.IGNORECASE | re.DOTALL)
_RIGHT_SEGMENT_RE = re.compile(r"Right:\s*Go along\s*(.*)$", re.IGNORECASE | re.DOTALL)


Point = Tuple[float, float]


def _loc_to_uv(x_token: str, y_token: str, image_size: tuple[int, int]) -> Point:
    width, height = image_size
    x = max(0, min(int(x_token), 1023))
    y = max(0, min(int(y_token), 1023))
    u = x / 1023.0 * max(0, width - 1)
    v = y / 1023.0 * max(0, height - 1)
    return u, v


def _segment_for_arm(loc_token_text: str, arm: str) -> str:
    regex = _LEFT_SEGMENT_RE if arm == "left" else _RIGHT_SEGMENT_RE
    match = regex.search(loc_token_text or "")
    return match.group(1) if match else ""


def parse_loc_token_trajectory(loc_token_text: str, image_size: tuple[int, int]) -> Dict[str, List[Point]]:
    """Parse canonical Mode 2 ``<locXXXX><locYYYY>`` text into image pixels."""
    return {
        arm: [_loc_to_uv(x, y, image_size) for x, y in _LOC_PAIR_RE.findall(_segment_for_arm(loc_token_text, arm))]
        for arm in ("left", "right")
    }


def _draw_polyline(draw: ImageDraw.ImageDraw, pts: list[Point], fill: tuple[int, int, int, int], width: int) -> None:
    if len(pts) >= 2:
        draw.line(pts, fill=fill, width=width, joint="curve")


def _draw_marker(
    draw: ImageDraw.ImageDraw,
    uv: Point,
    radius: int,
    fill: tuple[int, int, int],
    label: str,
) -> None:
    x, y = uv
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill + (245,), outline=(255, 255, 255), width=2)
    draw.line((x - radius - 4, y, x + radius + 4, y), fill=(255, 255, 255, 230), width=1)
    draw.line((x, y - radius - 4, x, y + radius + 4), fill=(255, 255, 255, 230), width=1)
    draw.text((x + radius + 5, y - radius - 5), label, fill=fill + (255,))


def _draw_arm_trajectory(draw: ImageDraw.ImageDraw, arm: str, points: list[Point]) -> None:
    if not points:
        return
    colors = ARM_COLORS[arm]
    _draw_polyline(draw, points, colors["full"] + (90,), width=3)
    if len(points) >= 2:
        _draw_polyline(draw, points[:2], colors["history"] + (210,), width=4)
        _draw_polyline(draw, points[1:], colors["future"] + (165,), width=3)
    for x, y in points:
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=colors["full"] + (210,))
    _draw_marker(draw, points[0], radius=6, fill=colors["history"], label=f"{arm[0].upper()}0")
    _draw_marker(draw, points[-1], radius=7, fill=colors["current"], label=f"{arm[0].upper()}{len(points) - 1}")


def _draw_text_box(draw: ImageDraw.ImageDraw, xy: tuple[int, int], lines: list[str]) -> None:
    font = ImageFont.load_default()
    x, y = xy
    line_height = 16
    widths = [draw.textlength(line, font=font) for line in lines]
    box_w = int(max(widths, default=0)) + 12
    box_h = line_height * len(lines) + 10
    draw.rectangle((x, y, x + box_w, y + box_h), fill=COLORS["text_bg"] + (210,))
    for i, line in enumerate(lines):
        draw.text((x + 6, y + 5 + i * line_height), line, fill=COLORS["text"] + (255,), font=font)


def render_trajectory_overlay(image_hwc_rgb: np.ndarray, loc_token_text: str) -> np.ndarray:
    """Return an RGB image with the latest Doubao trajectory drawn on top."""
    arr = np.asarray(image_hwc_rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("trajectory visualization requires an HxWx3 RGB image")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    height, width = arr.shape[:2]
    trajectories = parse_loc_token_trajectory(loc_token_text, (width, height))

    image = Image.fromarray(arr.copy())
    draw = ImageDraw.Draw(image, "RGBA")
    for arm in ("left", "right"):
        _draw_arm_trajectory(draw, arm, trajectories[arm])

    counts = {arm: len(points) for arm, points in trajectories.items()}
    lines = [
        "Doubao trajectory",
        f"left: {counts['left']} pts",
        f"right: {counts['right']} pts",
    ]
    if not any(counts.values()):
        lines.append("no <loc> points")
    _draw_text_box(draw, (10, 10), lines)
    return np.asarray(image, dtype=np.uint8)
