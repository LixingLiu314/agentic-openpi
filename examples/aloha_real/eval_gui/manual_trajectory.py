"""Helpers for human-in-the-loop trajectory annotation."""
from __future__ import annotations

from collections.abc import Iterable
import re
import time

from .doubao_predictor import TrajectoryPrediction
from .doubao_predictor import coords_to_loc_tokens
from .doubao_predictor import format_traj_string

_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_LOC_PAIR_RE = re.compile(r"<loc\d{4}><loc\d{4}>", re.IGNORECASE)
_COORD_RE = re.compile(r"\(\d+,\s*\d+\)")
_GRIPPER_RE = re.compile(r"\b(?:close gripper|open gripper)\b", re.IGNORECASE)
_ITEM_RE = re.compile(
    r"<loc\d{4}><loc\d{4}>|\(\d+,\s*\d+\)|\b(?:close gripper|open gripper)\b",
    re.IGNORECASE,
)


def pixel_to_loc_tokens(x: float, y: float, image_width: int, image_height: int) -> str:
    """Map image pixel coordinates to Paligemma ``<locXXXX><locYYYY>`` tokens.

    The inverse mapping is used by ``trajectory_visualizer``:
    ``pixel = loc / 1000 * (image_size - 1)``.
    """
    width = max(1, int(image_width))
    height = max(1, int(image_height))
    x_clamped = min(max(float(x), 0.0), float(width - 1))
    y_clamped = min(max(float(y), 0.0), float(height - 1))
    loc_x = int(round(x_clamped / max(1, width - 1) * 1000.0))
    loc_y = int(round(y_clamped / max(1, height - 1) * 1000.0))
    loc_x = max(0, min(loc_x, 1000))
    loc_y = max(0, min(loc_y, 1000))
    return f"<loc{loc_x:04d}><loc{loc_y:04d}>"


def build_manual_prediction(
    left_items: Iterable[str],
    right_items: Iterable[str],
    *,
    timestamp: float | None = None,
) -> TrajectoryPrediction:
    """Build a cached trajectory prediction from manual annotation items."""
    text = format_traj_string(list(left_items), list(right_items))
    return TrajectoryPrediction(
        raw_cot=text,
        loc_token_text=text,
        timestamp=time.time() if timestamp is None else float(timestamp),
    )


def normalize_trajectory_text(text: str) -> str:
    """Return prompt-ready trajectory text with coords converted to loc tokens."""
    text = _BR_RE.sub(" ", str(text or "").strip())
    text = coords_to_loc_tokens(text)
    return " ".join(text.split())


def build_prediction_from_text(
    text: str,
    *,
    timestamp: float | None = None,
) -> TrajectoryPrediction:
    """Build a trajectory prediction from an editable/reference text field."""
    loc_text = normalize_trajectory_text(text)
    return TrajectoryPrediction(
        raw_cot=str(text or "").strip(),
        loc_token_text=loc_text,
        timestamp=time.time() if timestamp is None else float(timestamp),
    )


def trajectory_text_to_arm_items(text: str) -> dict[str, list[str]]:
    """Best-effort parse of canonical Left/Right trajectory text into UI items."""
    normalized = normalize_trajectory_text(text)
    return {
        "left": _parse_items(_segment_for_arm(normalized, "left")),
        "right": _parse_items(_segment_for_arm(normalized, "right")),
    }


def _segment_for_arm(text: str, arm: str) -> str:
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


def _parse_items(segment: str) -> list[str]:
    items = []
    for match in _ITEM_RE.finditer(segment or ""):
        item = match.group(0).strip()
        if not item:
            continue
        if _LOC_PAIR_RE.fullmatch(item):
            items.append(item)
        elif _COORD_RE.fullmatch(item):
            items.append(coords_to_loc_tokens(item))
        else:
            grip = _GRIPPER_RE.search(item)
            if grip:
                items.append(grip.group(0).lower())
    return items
