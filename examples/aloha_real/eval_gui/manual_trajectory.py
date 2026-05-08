"""Helpers for human-in-the-loop trajectory annotation."""
from __future__ import annotations

import time
from typing import Iterable

from .doubao_predictor import TrajectoryPrediction, format_traj_string


def pixel_to_loc_tokens(x: float, y: float, image_width: int, image_height: int) -> str:
    """Map image pixel coordinates to Paligemma ``<locXXXX><locYYYY>`` tokens.

    The inverse mapping is used by ``trajectory_visualizer``:
    ``pixel = loc / 1023 * (image_size - 1)``.
    """
    width = max(1, int(image_width))
    height = max(1, int(image_height))
    x_clamped = min(max(float(x), 0.0), float(width - 1))
    y_clamped = min(max(float(y), 0.0), float(height - 1))
    loc_x = int(round(x_clamped / max(1, width - 1) * 1023.0))
    loc_y = int(round(y_clamped / max(1, height - 1) * 1023.0))
    loc_x = max(0, min(loc_x, 1023))
    loc_y = max(0, min(loc_y, 1023))
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
