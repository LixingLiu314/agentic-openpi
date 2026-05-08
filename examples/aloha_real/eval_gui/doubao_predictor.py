"""Doubao Vision-Language model trajectory predictor.

Used by Mode 2 ("Add Trajectory"). Given the current cam_high frame and a
task description, asks Doubao to produce a per-arm motion plan in the
**canonical internal format** used by the trajectory tooling (from
``scripts/preprocess_traj.py``):

    Left: Go along (xL,yL), (xL,yL), close gripper, (xL,yL).
    Right: Go along <br/>  (xR,yR), (xR,yR), close gripper, (xR,yR)

After the model returns this CoT string, ``coords_to_loc_tokens`` replaces
each ``(x, y)`` with PaliGemma ``<loc{x:04d}><loc{y:04d}>`` tokens — using
the **same regex** as the training preprocessor. The GUI later strips any
HTML break tags before constructing the final VLA prompt, so the model sees
clean text plus ``<locXXXX>`` tokens only.

The predictor is best-effort: any API/parsing failure returns ``None`` and
the caller is expected to reuse the previous prediction (or skip the
suffix entirely).
"""
from __future__ import annotations

import base64
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Coord -> <locXXXX> mirror of scripts/preprocess_traj.py:coords_to_loc_tokens
# ---------------------------------------------------------------------------
_COORD_RE = re.compile(r"\((\d+),\s*(\d+)\)")


def coords_to_loc_tokens(text: str) -> str:
    """Replace ``(x, y)`` pairs with PaliGemma ``<locXXXX>`` token pairs.

    Mirrors ``scripts/preprocess_traj.py:coords_to_loc_tokens`` exactly.
    """
    def replace(m: re.Match) -> str:
        x = max(0, min(int(m.group(1)), 1023))
        y = max(0, min(int(m.group(2)), 1023))
        return f"<loc{x:04d}><loc{y:04d}>"
    return _COORD_RE.sub(replace, text)


# ---------------------------------------------------------------------------
# Strict trajectory string format for the banana two-arm dataset.
# ---------------------------------------------------------------------------
# The internal trajectory string still uses the canonical left/right separator
# expected by the trajectory visualizer and parsers. The GUI sanitizes any
# HTML break tags before the text is passed to the VLA prompt.
TRAJ_LEFT_PREFIX = "Left: Go along "
TRAJ_RIGHT_PREFIX = ". Right: Go along <br/>  "   # NB: two spaces after <br/>


def format_traj_string(left_items: List[str], right_items: List[str]) -> str:
    """Combine per-arm sequences into the canonical training string.

    Each ``items`` is a list whose entries are either ``"(x, y)"`` style
    coordinate strings or action keywords (``"close gripper"``, ``"open
    gripper"``).
    """
    left = ", ".join(s.strip() for s in left_items if s and s.strip())
    right = ", ".join(s.strip() for s in right_items if s and s.strip())
    return f"{TRAJ_LEFT_PREFIX}{left}{TRAJ_RIGHT_PREFIX}{right}"


# ---------------------------------------------------------------------------
@dataclass
class TrajectoryPrediction:
    """Output of one Doubao call.

    ``raw_cot`` is the trajectory CoT *before* loc-token substitution
    (kept for inspection); ``loc_token_text`` is the substituted form
    that gets appended to the prompt.
    """

    raw_cot: str = ""
    loc_token_text: str = ""
    timestamp: float = 0.0

    def is_empty(self) -> bool:
        return not self.loc_token_text.strip()

    def to_loc_token_text(self) -> str:
        return self.loc_token_text


# ---------------------------------------------------------------------------
# _SYSTEM_PROMPT = (
#     "You are an expert Vision-Language model that generates a low-level motion "
#     "plan for a bimanual robot manipulator. The image shows the current "
#     "scene; coordinates are normalised to a 0-1000 grid with (0,0) at the "
#     "top-left and (1000,1000) at the bottom-right.\n\n"
#     "For BOTH the left and right end-effectors, output an ordered list of "
#     "items. Each item is either:\n"
#     "  - a coordinate pair in the form '(x, y)', or\n"
#     "  - one of these gripper action keywords: 'close gripper', 'open gripper'.\n\n"
#     "Respond with EXACTLY this XML block at the very end of your reply, "
#     "with no extra punctuation, no surrounding code fences, and one item per "
#     "<item>...</item> tag:\n"
#     "<plan>\n"
#     "<left>\n"
#     "<item>(x, y)</item>\n"
#     "<item>close gripper</item>\n"
#     "<item>(x, y)</item>\n"
#     "</left>\n"
#     "<right>\n"
#     "<item>(x, y)</item>\n"
#     "<item>(x, y)</item>\n"
#     "</right>\n"
#     "</plan>\n"
#     "If an arm is idle for this task, still emit at least one waypoint that "
#     "represents its current resting position."
# )

_SYSTEM_PROMPT = (
    "You are an expert Vision-Language-Action model generating low-level motion "
    "plans for a bimanual robot manipulator. The image shows the current "
    "scene; coordinates are normalised to a 0-1000 grid with (0,0) at the "
    "top-left and (1000,1000) at the bottom-right.\n\n"

    "Please follow these steps strictly to build your logic before outputting the plan:\n"
    "1. SCENE ANALYSIS: Briefly describe the spatial relationship between the left/right "
    "end-effectors, the target object(s), and any obstacles. Identify current gripper states.\n"
    "2. STRATEGY: Explain the bimanual movement strategy. Think in terms of action keyframes "
    "(e.g., pre-grasp, grasp, lift, place, handover) and assign roles to the Left and Right arms.\n"
    "3. WAYPOINTS: Predict the trajectory based on your strategy.\n\n"

    "For BOTH the left and right end-effectors, the waypoints must be an ordered list of items. "
    "Each item is either:\n"
    "  - a coordinate pair in the form '(x, y)', or\n"
    "  - one of these gripper action keywords: 'close gripper', 'open gripper'.\n\n"

    "Respond with EXACTLY this XML block at the VERY END of your reply, "
    "with no extra punctuation, no surrounding code fences, and one item per "
    "<item>...</item> tag:\n"
    "<plan>\n"
    "<left>\n"
    "<item>(x, y)</item>\n"
    "<item>close gripper</item>\n"
    "<item>(x, y)</item>\n"
    "</left>\n"
    "<right>\n"
    "<item>(x, y)</item>\n"
    "<item>(x, y)</item>\n"
    "</right>\n"
    "</plan>\n"
    "If an arm is idle for this task, still emit at least one waypoint that "
    "represents its current resting position."
)

_LEFT_BLOCK_RE = re.compile(r"<left>(.*?)</left>", re.DOTALL | re.IGNORECASE)
_RIGHT_BLOCK_RE = re.compile(r"<right>(.*?)</right>", re.DOTALL | re.IGNORECASE)
_ITEM_RE = re.compile(r"<item>\s*(.*?)\s*</item>", re.DOTALL | re.IGNORECASE)
_GRIPPER_RE = re.compile(r"\b(close gripper|open gripper)\b", re.IGNORECASE)


def _parse_arm_block(block: str) -> List[str]:
    items: List[str] = []
    for m in _ITEM_RE.finditer(block or ""):
        s = m.group(1).strip()
        if not s:
            continue
        coord = _COORD_RE.search(s)
        if coord:
            items.append(f"({int(coord.group(1))}, {int(coord.group(2))})")
            continue
        g = _GRIPPER_RE.search(s)
        if g:
            items.append(g.group(1).lower())
    return items


class DoubaoTrajectoryPredictor:
    """Thin wrapper around the Doubao (Volcengine Ark) chat completion API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "doubao-seed-2-0-lite-260215",
        base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
        timeout: float = 30.0,
        max_tokens: int = 400,
        jpeg_quality: int = 85,
    ) -> None:
        import openai  # noqa: F401  (validates installation up-front)

        self._api_key = api_key or os.environ.get("VOLCENKEY") or os.environ.get("ARK_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "Doubao API key not provided. Set VOLCENKEY (or ARK_API_KEY), "
                "or pass api_key=..."
            )
        self._client = openai.OpenAI(api_key=self._api_key, base_url=base_url, timeout=timeout)
        self._model = model
        self._max_tokens = max_tokens
        self._jpeg_quality = jpeg_quality

    def predict(self, image: np.ndarray, task_description: str) -> Optional[TrajectoryPrediction]:
        try:
            jpeg_b64 = self._encode_jpeg(image)
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64}"}},
                            {"type": "text",
                             "text": (f"Robot task: \"{task_description}\".\n"
                                      "Please analyze the scene, plan the bimanual strategy, "
                                      "and then output the motion plan in the required XML format at the end.")},
                        ],
                    },
                ],
            )
            text = response.choices[0].message.content or ""
        except Exception as e:
            logger.warning("Doubao API call failed: %s", e)
            return None
        return self._parse(text)

    def _encode_jpeg(self, image: np.ndarray) -> str:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
        if not ok:
            raise RuntimeError("Failed to JPEG-encode camera image")
        return base64.b64encode(buf).decode("utf-8")

    @staticmethod
    def _parse(text: str) -> Optional[TrajectoryPrediction]:
        lm = _LEFT_BLOCK_RE.search(text)
        rm = _RIGHT_BLOCK_RE.search(text)
        left_items = _parse_arm_block(lm.group(1)) if lm else []
        right_items = _parse_arm_block(rm.group(1)) if rm else []

        # Fallback: bare (x,y) pairs when XML missing — split halves at the
        # first 'right' keyword if present, otherwise put them all on left.
        if not left_items and not right_items:
            pairs = _COORD_RE.findall(text)
            if not pairs:
                return None
            mid = max(1, len(pairs) // 2)
            left_items = [f"({int(x)}, {int(y)})" for x, y in pairs[:mid]]
            right_items = [f"({int(x)}, {int(y)})" for x, y in pairs[mid:]]

        if not left_items:
            left_items = ["(500, 500)"]
        if not right_items:
            right_items = ["(500, 500)"]

        cot = format_traj_string(left_items, right_items)
        return TrajectoryPrediction(
            raw_cot=cot,
            loc_token_text=coords_to_loc_tokens(cot),
            timestamp=time.time(),
        )


# ---------------------------------------------------------------------------
class CachedTrajectoryPredictor:
    """Calls ``DoubaoTrajectoryPredictor`` every N inference steps.

    Two execution modes:
      * **non-blocking** (default): API call dispatched on a worker thread;
        ``step()`` returns the previous (or empty) prediction immediately.
      * **blocking**: ``step()`` calls ``predict()`` on the calling thread
        every ``update_every`` steps and returns the *fresh* result.
    """

    def __init__(
        self,
        predictor: Optional[DoubaoTrajectoryPredictor] = None,
        update_every: int = 6,
        blocking: bool = False,
    ) -> None:
        self._predictor = predictor or DoubaoTrajectoryPredictor()
        self._update_every = max(1, int(update_every))
        self._blocking = bool(blocking)
        self._last: Optional[TrajectoryPrediction] = None
        self._initialized = False   # True once first valid result received
        self._step = 0
        self._lock = threading.Lock()
        self._inflight: Optional[threading.Thread] = None
        self._on_update: Optional[Callable[[TrajectoryPrediction, np.ndarray], None]] = None
        self._stats = {"requests": 0, "successes": 0, "failures": 0, "cache_hits": 0}

    def reset(self) -> None:
        with self._lock:
            self._last = None
            self._initialized = False
            self._step = 0
            self._inflight = None

    @property
    def last(self) -> Optional[TrajectoryPrediction]:
        with self._lock:
            return self._last

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    @property
    def blocking(self) -> bool:
        with self._lock:
            return self._blocking

    @property
    def update_every(self) -> int:
        with self._lock:
            return self._update_every

    # --- runtime knobs (toggled from the GUI) ------------------------- #
    def set_blocking(self, blocking: bool) -> None:
        with self._lock:
            self._blocking = bool(blocking)

    def set_update_every(self, n: int) -> None:
        with self._lock:
            self._update_every = max(1, int(n))

    def set_update_callback(
        self,
        callback: Optional[Callable[[TrajectoryPrediction, np.ndarray], None]],
    ) -> None:
        with self._lock:
            self._on_update = callback

    def set_cached(
        self,
        result: Optional[TrajectoryPrediction],
        image: Optional[np.ndarray] = None,
    ) -> Optional[TrajectoryPrediction]:
        """Inject a cached trajectory result from an external source."""
        img_copy = None if image is None else image.copy()
        with self._lock:
            self._stats["requests"] += 1
        last, callback = self._store_result(result)
        if result is not None and not result.is_empty() and callback is not None and img_copy is not None:
            try:
                callback(result, img_copy)
            except Exception:
                logger.exception("Trajectory update callback failed")
        return last

    # ----------------------------------------------------------------- #
    def _store_result(self, result: Optional[TrajectoryPrediction]) -> tuple[Optional[TrajectoryPrediction], Optional[Callable]]:
        callback = None
        with self._lock:
            if result is not None and not result.is_empty():
                self._last = result
                self._initialized = True
                self._stats["successes"] += 1
                callback = self._on_update
            else:
                self._stats["failures"] += 1
            return self._last, callback

    def refresh(self, image: np.ndarray, task_description: str) -> Optional[TrajectoryPrediction]:
        """Synchronously fetch a fresh trajectory and update the cache."""
        img_copy = image.copy()
        with self._lock:
            self._stats["requests"] += 1

        result = self._predictor.predict(img_copy, task_description)
        last, callback = self._store_result(result)
        if result is not None and not result.is_empty() and callback is not None:
            try:
                callback(result, img_copy)
            except Exception:
                logger.exception("Trajectory update callback failed")
        return last

    def step(self, image: np.ndarray, task_description: str) -> Optional[TrajectoryPrediction]:
        with self._lock:
            # First-step wait: always block synchronously until we have an
            # initial valid result, regardless of the configured blocking flag.
            first_step = not self._initialized
            should_request = first_step or (self._step % self._update_every == 0)
            self._step += 1
            if not should_request:
                self._stats["cache_hits"] += 1
                return self._last
            img_copy = image.copy()
            blocking = first_step or self._blocking   # force-block on step 0

        if blocking:
            return self.refresh(img_copy, task_description)

        # Non-blocking: only spawn if no request currently in flight.
        with self._lock:
            if self._inflight is not None and self._inflight.is_alive():
                return self._last
            self._stats["requests"] += 1

        def _run() -> None:
            result = self._predictor.predict(img_copy, task_description)
            _, callback = self._store_result(result)
            if result is not None and not result.is_empty() and callback is not None:
                try:
                    callback(result, img_copy)
                except Exception:
                    logger.exception("Trajectory update callback failed")

        t = threading.Thread(target=_run, daemon=True, name="doubao-predict")
        t.start()
        with self._lock:
            self._inflight = t
            return self._last


# ---------------------------------------------------------------------------
# Optional: a Doubao-driven *subtask* suggester (Mode 3).
# Right now this is a placeholder hook — the GUI's keyboard shortcuts remain
# the canonical trigger. The interface lets us plug a real VLM later without
# re-wiring the rest of the pipeline.
# ---------------------------------------------------------------------------
class SubtaskPredictor:
    """Hook for an automatic subtask suggester.

    Concrete implementations override ``suggest()`` to return one of the
    keys in ``runtime.subtask_labels`` (or ``None`` to abstain). The
    default impl is a no-op so wiring it in costs nothing until you fill
    it with an actual VLM call.
    """

    def suggest(self, image: np.ndarray, task_description: str,
                labels: dict) -> Optional[int]:
        return None

    def reset(self) -> None:
        ...
