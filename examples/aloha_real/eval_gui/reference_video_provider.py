"""Reference video subgoal provider.

Pre-extracts frames from a successful evaluation video at fixed step
intervals, then serves them via lookahead logic: given the current step,
always return the frame for the *next* milestone we are aiming for.

    target_step = ((current_step // N) + 1) * N

If the target exceeds the maximum step in the reference log, clamp to the
final step's frame.

All video I/O happens at construction time so the runner thread never blocks
on disk reads.

Frame-index resolution priority:
  1. If the JSONL log contains non-null ``video_frame_index`` values, use them
     directly (exact, no drift).
  2. If the JSONL log exists but has no frame indices, compute from timestamps:
     ``frame_idx = (timestamp_unix - t0 + video_offset_s) * video_fps``
  3. If NO JSONL log is available (fallback mode), compute purely from math:
     ``frame_idx = int((target_step / control_hz + video_offset_s) * video_fps)``
"""
from __future__ import annotations

import json
import logging
import pathlib
import threading
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ReferenceVideoProvider:
    """Drop-in subgoal client that serves pre-extracted reference frames
    using lookahead interval mapping."""

    def __init__(
        self,
        video_path: str,
        jsonl_path: Optional[str],
        step_interval: int,
        video_offset_s: float = 0.0,
        control_hz: float = 30.0,
    ) -> None:
        self._video_path = pathlib.Path(video_path)
        self._step_interval = max(1, int(step_interval))
        self._video_offset_s = float(video_offset_s)
        self._control_hz = max(1.0, float(control_hz))
        self._lock = threading.Lock()
        self._current_step = 0

        if not self._video_path.is_file():
            raise ValueError(f"Video file not found: {self._video_path}")

        jsonl_path = pathlib.Path(jsonl_path) if jsonl_path else None
        has_jsonl = jsonl_path is not None and jsonl_path.is_file()

        if has_jsonl:
            step_ts_map, step_frame_map, video_fps = self._parse_jsonl(jsonl_path)
            self._max_step = max(step_ts_map.keys())
            self._has_frame_indices = bool(step_frame_map)
            self._frames: Dict[int, np.ndarray] = self._extract_frames(
                step_ts_map, step_frame_map, video_fps
            )
        else:
            video_fps = self._get_video_fps()
            self._has_frame_indices = False
            self._frames, self._max_step = self._extract_frames_fallback(video_fps)

        if not self._frames:
            raise ValueError("No frames could be extracted from the reference video.")
        if has_jsonl:
            mode = "frame_index" if self._has_frame_indices else f"timestamp+offset({self._video_offset_s:+.1f}s)"
        else:
            mode = f"fallback(ctrl={self._control_hz}Hz, offset={self._video_offset_s:+.1f}s)"
        logger.info(
            "ReferenceVideoProvider loaded %d frames from %s "
            "(interval=%d, max_step=%d, mode=%s)",
            len(self._frames), self._video_path.name,
            self._step_interval, self._max_step, mode,
        )

    def connect(self) -> dict:
        return {
            "client": "reference_video",
            "frames": len(self._frames),
            "max_step": self._max_step,
        }

    def close(self) -> None:
        pass

    def reset(self) -> None:
        with self._lock:
            self._current_step = 0

    def set_current_step(self, step: int) -> None:
        with self._lock:
            self._current_step = int(step)

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def predict_subgoal(
        self,
        image: np.ndarray,
        task_description: str,
    ) -> Optional[np.ndarray]:
        with self._lock:
            target = self._lookahead_target(self._current_step)
        return self._frames.get(target, self._frames[self._max_step])

    def _lookahead_target(self, current_step: int) -> int:
        N = self._step_interval
        target = ((current_step // N) + 1) * N
        if target > self._max_step:
            target = self._max_step
        if target not in self._frames:
            available = sorted(self._frames.keys())
            target = available[-1]
        return target

    def _parse_jsonl(self, jsonl_path: pathlib.Path) -> Tuple[Dict[int, float], Dict[int, int], float]:
        """Parse JSONL log into step->timestamp and step->frame_index maps.

        Returns (step_ts, step_frames, video_fps).
        step_frames is empty if no entries have non-null video_frame_index.
        Only uses ``video_frame_index`` (the actual video frame number);
        ``frame_index`` is the subgoal sequence index and must NOT be used here.
        """
        step_ts: Dict[int, float] = {}
        step_frames: Dict[int, int] = {}
        video_fps: float = 30.0
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                step = entry.get("step")
                ts = entry.get("timestamp_unix")
                if step is not None and ts is not None:
                    step_ts[int(step)] = float(ts)
                fi = entry.get("video_frame_index")
                if step is not None and fi is not None:
                    step_frames[int(step)] = int(fi)
                cfg = entry.get("config")
                if cfg and "video_fps" in cfg:
                    video_fps = float(cfg["video_fps"])
        if not step_ts:
            raise ValueError(f"No valid step/timestamp entries in {jsonl_path}")
        if step_frames:
            logger.info(
                "JSONL contains frame_index for %d/%d steps; using direct frame lookup.",
                len(step_frames), len(step_ts),
            )
        return step_ts, step_frames, video_fps

    def _extract_frames(
        self,
        step_ts: Dict[int, float],
        step_frames: Dict[int, int],
        video_fps: float,
    ) -> Dict[int, np.ndarray]:
        max_step = max(step_ts.keys())
        first_step = min(step_ts.keys())
        t0 = step_ts[first_step]

        N = self._step_interval
        target_steps = list(range(N, max_step + 1, N))
        if max_step not in target_steps:
            target_steps.append(max_step)
        if not target_steps:
            target_steps = [max_step]

        use_frame_idx = bool(step_frames)
        frame_targets: List[Tuple[int, int]] = []
        for s in target_steps:
            closest = s if s in step_ts else min(
                step_ts.keys(), key=lambda k: abs(k - s)
            )
            if use_frame_idx and closest in step_frames:
                frame_idx = step_frames[closest]
            else:
                ts = step_ts[closest]
                frame_idx = max(
                    0, int(round((ts - t0 + self._video_offset_s) * video_fps))
                )
            frame_targets.append((s, frame_idx))

        cap = cv2.VideoCapture(str(self._video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self._video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames: Dict[int, np.ndarray] = {}
        for step, frame_idx in frame_targets:
            clamped = min(frame_idx, total_frames - 1)
            if clamped < 0:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, clamped)
            ret, bgr = cap.read()
            if not ret:
                logger.warning("Failed to read frame %d for step %d", clamped, step)
                break
            frames[step] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        cap.release()
        return frames

    def _get_video_fps(self) -> float:
        cap = cv2.VideoCapture(str(self._video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self._video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps if fps > 0 else 30.0

    def _extract_frames_fallback(
        self, video_fps: float
    ) -> Tuple[Dict[int, np.ndarray], int]:
        """Extract frames using pure math when no JSONL is available.

        target_time = target_step / control_hz + video_offset_s
        frame_idx = int(target_time * video_fps)
        """
        cap = cv2.VideoCapture(str(self._video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self._video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration_s = total_frames / video_fps if video_fps > 0 else 0.0
        max_step = int(video_duration_s * self._control_hz)
        if max_step <= 0:
            cap.release()
            raise ValueError("Video too short or invalid FPS/control_hz.")

        N = self._step_interval
        target_steps = list(range(N, max_step + 1, N))
        if max_step not in target_steps:
            target_steps.append(max_step)
        if not target_steps:
            target_steps = [max_step]

        frames: Dict[int, np.ndarray] = {}
        for s in target_steps:
            target_time = s / self._control_hz + self._video_offset_s
            frame_idx = int(target_time * video_fps)
            clamped = max(0, min(frame_idx, total_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, clamped)
            ret, bgr = cap.read()
            if not ret:
                logger.warning("Failed to read frame %d for step %d", clamped, s)
                break
            frames[s] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        cap.release()
        return frames, max_step
