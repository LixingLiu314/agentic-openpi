"""Background thread that runs the policy<->robot loop for the GUI.

The runner does NOT use ``openpi_client.runtime.Runtime``: the GUI needs
fine-grained interactive control (start / pause / e-stop / mode-switch on
the fly) which the canonical Runtime does not expose. Instead we drive
``PiperRealEnv`` directly and reuse ``ActionChunkBroker`` only for the
chunk-streaming logic (kept identical to ``examples/aloha_real/piper_main.py``
so timing matches the reference implementation).

The runner also implements the Return-to-Zero (回零) action exactly the same
way ``scripts/eval_banana.sh --reset-only`` does (``environment.reset()``),
working both while the loop is running (drained on the next iteration) and
while it is idle (synchronous call from the GUI thread, guarded by an env
lock so we never race with ``env.step()``).
"""
from __future__ import annotations

import json
import logging
import pathlib
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
from openpi_client import action_chunk_broker as _broker
from openpi_client import websocket_client_policy as _ws

from . import modes as _modes

logger = logging.getLogger(__name__)


_DEFAULT_HZ = 30.0
_CONTINUOUS_VIDEO_CAMERAS: Dict[str, str] = {
    "base": "cam_high",
    "left_wrist": "cam_left_wrist",
    "right_wrist": "cam_right_wrist",
}


@dataclass
class RunnerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    action_horizon: int = 25
    max_steps: int = 1000
    max_hz: float = _DEFAULT_HZ
    dry_run: bool = False
    img_front_topic: str = "/camera_f/color/image_raw"
    img_left_topic: str = "/camera_l/color/image_raw"
    img_right_topic: str = "/camera_r/color/image_raw"
    joint_left_topic: str = "/puppet/joint_left"
    joint_right_topic: str = "/puppet/joint_right"
    cmd_left_topic: str = "/master/joint_left"
    cmd_right_topic: str = "/master/joint_right"
    gripper_open: float = 4.0
    gripper_close: float = 0.0
    reset_move_time: float = 2.0
    dump_dir: str = "debug_inputs"   # under cwd; created on first dump
    record_video: bool = False
    video_dir: str = ""
    video_name: str = "eval"
    video_fps: float = _DEFAULT_HZ
    log_model_inputs: bool = True


def _safe_filename_part(text: str) -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch in "._-" else "_"
        for ch in str(text).strip()
    )
    return (cleaned.strip("._-") or "eval")[:160]


class _VideoRecorder:
    """Asynchronous RGB frame writer for VS Code-compatible H.264 MP4."""

    def __init__(
        self,
        path: pathlib.Path,
        fps: float,
        frame_size: Optional[tuple[int, int]] = None,
    ) -> None:
        self.path = path
        self._fps = float(fps)
        self._frame_size = tuple(frame_size) if frame_size is not None else None
        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=256)
        self._thread = threading.Thread(target=self._run, daemon=True, name="eval-video-writer")
        self._proc: Optional[subprocess.Popen] = None
        self._dropped = 0
        self._error = ""

    def start(self) -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found; cannot write VS Code-compatible MP4")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread.start()

    def write(self, frame_rgb: Optional[np.ndarray]) -> None:
        if frame_rgb is None:
            return
        arr = np.asarray(frame_rgb)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        try:
            self._queue.put_nowait(arr.copy())
        except queue.Full:
            self._dropped += 1

    def stop(self) -> int:
        while True:
            try:
                self._queue.put_nowait(None)
                break
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self._dropped += 1
                except queue.Empty:
                    pass
        self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            self._error = "video writer did not finish within 10 seconds"
            self._terminate_process()
        return self._dropped

    @property
    def error(self) -> str:
        return self._error

    def _start_ffmpeg(self, frame_shape: tuple[int, int, int]) -> None:
        if self._frame_size is not None:
            h, w = self._frame_size
        else:
            h, w = frame_shape[:2]
            self._frame_size = (h, w)
        fps = max(1.0, self._fps)
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
            "-r",
            f"{fps:.3f}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv444p",
            "-movflags",
            "+faststart",
            str(self.path),
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    @staticmethod
    def _fit_frame_to_size(frame: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        target_h, target_w = target_shape
        src_h, src_w = frame.shape[:2]
        if (src_h, src_w) == (target_h, target_w):
            return frame
        out = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        copy_h = min(src_h, target_h)
        copy_w = min(src_w, target_w)
        src_y0 = max(0, (src_h - target_h) // 2)
        src_x0 = max(0, (src_w - target_w) // 2)
        dst_y0 = max(0, (target_h - src_h) // 2)
        dst_x0 = max(0, (target_w - src_w) // 2)
        out[dst_y0 : dst_y0 + copy_h, dst_x0 : dst_x0 + copy_w] = frame[
            src_y0 : src_y0 + copy_h,
            src_x0 : src_x0 + copy_w,
        ]
        return out

    def _finish_process(self) -> None:
        proc = self._proc
        if proc is None:
            return
        stderr = b""
        try:
            if proc.stdin is not None:
                try:
                    proc.stdin.close()
                except OSError:
                    pass
            returncode = proc.wait(timeout=10.0)
            stderr = proc.stderr.read() if proc.stderr is not None else b""
        except subprocess.TimeoutExpired:
            self._error = "ffmpeg did not exit within 10 seconds"
            self._terminate_process()
            return
        finally:
            self._proc = None
        if returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            self._error = f"ffmpeg exited with code {returncode}"
            if detail:
                self._error += f": {detail[-500:]}"

    def _terminate_process(self) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            self._proc = None
            return
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2.0)
        self._proc = None

    def _run(self) -> None:
        try:
            while True:
                frame = self._queue.get()
                if frame is None:
                    break
                if self._frame_size is not None:
                    frame = self._fit_frame_to_size(frame, self._frame_size)
                if self._proc is None:
                    self._start_ffmpeg(frame.shape)
                if self._proc.stdin is None:
                    self._error = "ffmpeg stdin is not available"
                    break
                try:
                    self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
                except BrokenPipeError:
                    self._error = "ffmpeg pipe closed while writing video frames"
                    break
                if self._proc.poll() is not None:
                    self._error = f"ffmpeg exited early with code {self._proc.returncode}"
                    break
        except Exception as e:                       # noqa: BLE001
            self._error = f"video writer failed: {e}"
        finally:
            self._finish_process()


class _MultiCameraVideoRecorder:
    """Samples one observation stream and writes selected cameras to MP4s."""

    def __init__(
        self,
        paths: Dict[str, pathlib.Path],
        fps: float,
        obs_provider: Callable[[], Optional[dict]],
        obs_callback: Callable[[dict], None],
        *,
        camera_keys: Dict[str, str],
    ) -> None:
        self.paths = dict(paths)
        self._fps = max(1.0, float(fps))
        self._obs_provider = obs_provider
        self._obs_callback = obs_callback
        self._camera_keys = dict(camera_keys)
        self._writers = {
            name: _VideoRecorder(path=self.paths[name], fps=self._fps)
            for name in self._camera_keys
        }
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="eval-video-sampler")
        self._lock = threading.Lock()
        self._started = False
        self._stopped = False
        self._dropped: Dict[str, int] = {name: 0 for name in self._camera_keys}
        self._sampler_error = ""

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            started: list[_VideoRecorder] = []
            try:
                for writer in self._writers.values():
                    writer.start()
                    started.append(writer)
            except Exception:
                for writer in started:
                    try:
                        writer.stop()
                    except Exception:                  # noqa: BLE001
                        pass
                raise
            self._started = True
            self._thread.start()

    def stop(self) -> Dict[str, int]:
        with self._lock:
            if self._stopped:
                return dict(self._dropped)
            self._stopped = True
            started = self._started
        if not started:
            return dict(self._dropped)
        self._stop_evt.set()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            self._sampler_error = "video sampler did not finish within 5 seconds"
        self._dropped = {
            name: writer.stop()
            for name, writer in self._writers.items()
        }
        return dict(self._dropped)

    @property
    def error(self) -> str:
        parts = [self._sampler_error]
        parts.extend(
            f"{name}: {writer.error}"
            for name, writer in self._writers.items()
            if writer.error
        )
        return "; ".join(part for part in parts if part)

    @property
    def writer_errors(self) -> Dict[str, str]:
        return {
            name: writer.error
            for name, writer in self._writers.items()
            if writer.error
        }

    @property
    def sampler_error(self) -> str:
        return self._sampler_error

    def _run(self) -> None:
        period = 1.0 / self._fps
        next_tick = time.monotonic()
        last_frames: Dict[str, np.ndarray] = {}

        while not self._stop_evt.is_set():
            try:
                obs = self._obs_provider()
                if obs is not None:
                    self._obs_callback(obs)
                    images = obs.get("images") or {}
                    for name, camera_key in self._camera_keys.items():
                        frame = images.get(camera_key)
                        if frame is not None:
                            last_frames[name] = frame
            except Exception as e:                       # noqa: BLE001
                if not self._sampler_error:
                    self._sampler_error = f"video sampler failed to read observation: {e}"

            for name, frame in last_frames.items():
                self._writers[name].write(frame)

            next_tick += period
            delay = next_tick - time.monotonic()
            if delay <= 0:
                next_tick = time.monotonic()
                continue
            self._stop_evt.wait(delay)


class _StepInputLogger:
    """Append-only JSONL logger for each control step.

    The logger keeps a single per-run MP4 for subgoal images instead of
    writing one JPG per inference. The JSONL trace is written once per
    control step so it matches the runtime step count rather than the
    chunked policy-inference cadence.
    """

    def __init__(self, path: pathlib.Path, artifact_prefix: str, video_fps: float) -> None:
        self.path = path
        self._artifact_prefix = artifact_prefix
        self._video_fps = max(1.0, float(video_fps))
        self._fh = None
        self._lock = threading.Lock()
        self._closed = False
        self._error = ""
        self._subgoal_video_path = self.path.parent / f"{artifact_prefix}_subgoal.mp4"
        self._subgoal_recorder: Optional[_VideoRecorder] = None
        self._subgoal_frames = 0
        self._subgoal_error = ""
        self._subgoal_camera_order: list[str] = []
        self._subgoal_tile_sizes: Dict[str, tuple[int, int]] = {}
        self._subgoal_tile_offsets: Dict[str, tuple[int, int]] = {}
        self._subgoal_canvas_shape: Optional[tuple[int, int]] = None
        self._subgoal_frame_index: Optional[int] = None

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._fh is not None:
                self._fh.close()
                self._fh = None
            recorder = self._subgoal_recorder
            self._subgoal_recorder = None
        if recorder is not None:
            try:
                dropped = recorder.stop()
            except Exception as e:                       # noqa: BLE001
                self._subgoal_error = f"failed to stop subgoal video recorder: {e}"
            else:
                if recorder.error:
                    self._subgoal_error = recorder.error
                if dropped and not self._subgoal_error:
                    self._subgoal_error = f"{dropped} subgoal frames dropped"

    @property
    def error(self) -> str:
        return self._error

    @property
    def subgoal_video_path(self) -> pathlib.Path:
        return self._subgoal_video_path

    @property
    def subgoal_frame_count(self) -> int:
        return self._subgoal_frames

    @property
    def subgoal_error(self) -> str:
        return self._subgoal_error

    @property
    def subgoal_frame_index(self) -> Optional[int]:
        return self._subgoal_frame_index

    def record_subgoal_images(self, payload: Dict[str, Any]) -> Optional[int]:
        subgoal_images = payload.get("subgoal_images") or {}
        if not subgoal_images:
            return None

        frame = self._compose_subgoal_frame(subgoal_images)
        if frame is None:
            return None

        with self._lock:
            frame_index = self._subgoal_frames
            recorder = self._subgoal_recorder
            if recorder is None:
                try:
                    recorder = _VideoRecorder(
                        path=self._subgoal_video_path,
                        fps=self._video_fps,
                        frame_size=frame.shape[:2],
                    )
                    recorder.start()
                except Exception as e:                   # noqa: BLE001
                    self._subgoal_error = f"failed to start subgoal video recorder: {e}"
                    return None
                self._subgoal_recorder = recorder
            self._subgoal_frames += 1
            self._subgoal_frame_index = frame_index

        try:
            recorder.write(frame)
        except Exception as e:                           # noqa: BLE001
            self._subgoal_error = f"failed to write subgoal video frame: {e}"
        return frame_index

    def log_step(
        self,
        payload: Dict[str, Any],
        raw_obs: Dict[str, Any],
        *,
        step: int,
        inference_index: int,
        mode: str,
        runtime_snapshot: Dict[str, Any],
        cfg: RunnerConfig,
        frame_index: Optional[int],
    ) -> None:
        state = np.asarray(payload.get("state", []), dtype=np.float32).reshape(-1)
        images = payload.get("images") or {}
        raw_images = (raw_obs.get("images") or {}) if isinstance(raw_obs, dict) else {}
        subgoal_images = payload.get("subgoal_images") or {}
        entry = {
            "type": "vla_input",
            "inference_index": int(inference_index),
            "step": int(step),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp_unix": time.time(),
            "mode": mode,
            "text_prompt": str(payload.get("prompt", "") or ""),
            "state": state.tolist(),
            "state_shape": list(state.shape),
            "image_keys": list(images.keys()),
            "image_shapes": {str(k): list(np.asarray(v).shape) for k, v in images.items()},
            "subgoal_image_keys": list(subgoal_images.keys()),
            "subgoal_image_shapes": {str(k): list(np.asarray(v).shape) for k, v in subgoal_images.items()},
            "frame_index": None if frame_index is None else int(frame_index),
            "subgoal_frame_index": None if frame_index is None else int(frame_index),
            "subgoal_image_paths": {},
            "subgoal_video_path": str(self._subgoal_video_path) if subgoal_images else "",
            "subgoal_video_frames": int(self._subgoal_frames),
            "raw_image_shapes": {str(k): list(np.asarray(v).shape) for k, v in raw_images.items()},
            "runtime": {
                "task": runtime_snapshot.get("task", ""),
                "subtask_key": runtime_snapshot.get("subtask_key"),
                "subtask_label": runtime_snapshot.get("subtask_label", ""),
                "traj_text": runtime_snapshot.get("traj_text", ""),
                "manual_traj_override": bool(runtime_snapshot.get("manual_traj_override", False)),
                "waiting_for_subtask_input": bool(runtime_snapshot.get("waiting_for_subtask_input", False)),
            },
            "config": {
                "action_horizon": int(cfg.action_horizon),
                "max_hz": float(cfg.max_hz),
                "video_fps": float(cfg.video_fps),
                "json_step_frequency": 1,
            },
        }
        line = json.dumps(entry, ensure_ascii=False, sort_keys=True)
        with self._lock:
            if self._closed or self._fh is None:
                return
            self._fh.write(line + "\n")
            self._fh.flush()

    def _compose_subgoal_frame(self, subgoal_images: Dict[str, Any]) -> Optional[np.ndarray]:
        frames: Dict[str, np.ndarray] = {}
        for cam, image in subgoal_images.items():
            arr = self._to_hwc_rgb(image)
            if arr.ndim != 3 or arr.shape[2] != 3:
                continue
            frames[str(cam)] = arr
        if not frames:
            return None

        if not self._subgoal_camera_order:
            self._initialize_subgoal_layout(frames)

        assert self._subgoal_canvas_shape is not None
        ordered_frames: list[np.ndarray] = []
        for cam in self._subgoal_camera_order:
            tile_h, tile_w = self._subgoal_tile_sizes[cam]
            frame = frames.get(cam)
            if frame is None:
                tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            else:
                tile = self._fit_frame_to_tile(frame, (tile_h, tile_w))
            ordered_frames.append(tile)

        canvas_h, canvas_w = self._subgoal_canvas_shape
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        for cam, tile in zip(self._subgoal_camera_order, ordered_frames):
            x0, y0 = self._subgoal_tile_offsets[cam]
            tile_h, tile_w = self._subgoal_tile_sizes[cam]
            canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile
        return canvas

    def _initialize_subgoal_layout(self, frames: Dict[str, np.ndarray]) -> None:
        self._subgoal_camera_order = sorted(frames.keys())
        self._subgoal_tile_sizes = {
            cam: frames[cam].shape[:2] for cam in self._subgoal_camera_order
        }
        canvas_h = max(h for h, _ in self._subgoal_tile_sizes.values())
        canvas_w = sum(w for _, w in self._subgoal_tile_sizes.values())
        self._subgoal_canvas_shape = (canvas_h, canvas_w)
        self._subgoal_tile_offsets = {}
        x0 = 0
        for cam in self._subgoal_camera_order:
            tile_h, tile_w = self._subgoal_tile_sizes[cam]
            y0 = (canvas_h - tile_h) // 2
            self._subgoal_tile_offsets[cam] = (x0, y0)
            x0 += tile_w

    @staticmethod
    def _to_hwc_rgb(image: Any) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        return arr

    @staticmethod
    def _fit_frame_to_tile(frame: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        target_h, target_w = target_shape
        src_h, src_w = frame.shape[:2]
        if (src_h, src_w) == (target_h, target_w):
            return frame
        out = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        copy_h = min(src_h, target_h)
        copy_w = min(src_w, target_w)
        src_y0 = max(0, (src_h - target_h) // 2)
        src_x0 = max(0, (src_w - target_w) // 2)
        dst_y0 = max(0, (target_h - src_h) // 2)
        dst_x0 = max(0, (target_w - src_w) // 2)
        out[dst_y0 : dst_y0 + copy_h, dst_x0 : dst_x0 + copy_w] = frame[
            src_y0 : src_y0 + copy_h,
            src_x0 : src_x0 + copy_w,
        ]
        return out


class _BrokerAdapter:
    """Adapts ``ModeHandler.build_obs`` -> ``BasePolicy.infer`` for the broker.

    Also intercepts the obs payload to support the "Dump model inputs"
    feature: when the runner sets ``_pending_dump`` we save a copy of the
    *exact* dict that goes onto the wire (state, processed images,
    prompt, and any subgoal_images) into a per-step subdirectory.
    """

    def __init__(self, ws_policy, handler, handler_lock, runtime, runner) -> None:
        self._ws_policy = ws_policy
        self._handler = handler
        self._handler_lock = handler_lock
        self._runtime = runtime
        self._runner = runner

    def set_handler(self, handler) -> None:
        self._handler = handler

    def infer(self, raw_obs):
        with self._handler_lock:
            payload = self._handler.build_obs(raw_obs, self._runtime)
        # Dump-on-demand: cheap snapshot of the post-repack payload + raw frames.
        if self._runner._pending_dump:
            try:
                self._runner._dump_payload(payload, raw_obs)
            except Exception as e:                       # noqa: BLE001
                logger.warning("Input dump failed: %s", e)
            self._runner._pending_dump = False
        try:
            self._runner._cache_latest_model_input(payload, raw_obs)
        except Exception as e:                           # noqa: BLE001
            logger.warning("Input log failed: %s", e)
        return self._ws_policy.infer(payload)

    def reset(self):
        self._ws_policy.reset()


# ---------------------------------------------------------------------------
class EvalRunner:
    """Drives the eval loop. All robot/network I/O happens on the worker thread.

    Public API is **thread-safe** and intended to be called from a Qt GUI:

      * ``connect()``         - bring up the ROS env + WS policy (blocking).
      * ``set_handler(h)``    - swap the mode handler hot.
      * ``start()/pause()/resume()/stop()`` - control the inference loop.
      * ``request_reset()``   - "回零" / Return-To-Zero button. Works whether
                                the loop is running, paused, or idle.
      * ``request_dump()``    - save the next model input to disk.
      * ``latest_obs()``      - thread-safe snapshot of the most recent raw
                                observation (for the GUI camera view).
    """

    def __init__(
        self,
        cfg: RunnerConfig,
        runtime: _modes.RuntimeState,
        handler: _modes.ModeHandler,
        on_event: Optional[Callable[[str, dict], None]] = None,
    ) -> None:
        self._cfg = cfg
        self._runtime = runtime
        self._handler = handler
        self._handler_lock = threading.Lock()
        self._on_event = on_event or (lambda kind, payload: None)

        self._env = None
        self._policy = None
        self._broker: Optional[_broker.ActionChunkBroker] = None

        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._stop_evt = threading.Event()
        self._reset_evt = threading.Event()
        self._manual_control_evt = threading.Event()

        # Guarantees we never call env.reset() and env.step() concurrently.
        self._env_lock = threading.Lock()

        self._latest_obs: Optional[dict] = None
        self._obs_lock = threading.Lock()
        self._step_count = 0

        self._pending_dump = False
        self._dump_counter = 0
        self._input_log_counter = 0
        self._artifact_lock = threading.Lock()
        self._video_recorder: Optional[_MultiCameraVideoRecorder] = None
        self._input_logger: Optional[_StepInputLogger] = None
        self._latest_model_payload: Optional[Dict[str, Any]] = None
        self._latest_model_raw_obs: Optional[Dict[str, Any]] = None
        self._latest_model_inference_index = -1
        self._latest_model_frame_index: Optional[int] = None

    # ------------------------------------------------------------------ #
    def connect(self) -> dict:
        """Bring up websocket policy + ROS env. Returns the server metadata."""
        from examples.aloha_real import piper_real_env as _real

        self._on_event("info", {"msg": f"Connecting to policy server {self._cfg.host}:{self._cfg.port} ..."})
        self._policy = _ws.WebsocketClientPolicy(host=self._cfg.host, port=self._cfg.port)
        metadata = self._policy.get_server_metadata()
        self._on_event("info", {"msg": f"Server metadata: {metadata}"})

        self._on_event("info", {"msg": "Initialising Piper ROS env ..."})
        self._env = _real.make_real_env(
            init_node=True,
            reset_position=metadata.get("reset_pose"),
            img_front_topic=self._cfg.img_front_topic,
            img_left_topic=self._cfg.img_left_topic,
            img_right_topic=self._cfg.img_right_topic,
            joint_left_topic=self._cfg.joint_left_topic,
            joint_right_topic=self._cfg.joint_right_topic,
            cmd_left_topic=self._cfg.cmd_left_topic,
            cmd_right_topic=self._cfg.cmd_right_topic,
            gripper_open=self._cfg.gripper_open,
            gripper_close=self._cfg.gripper_close,
            reset_move_time=self._cfg.reset_move_time,
            dry_run=self._cfg.dry_run,
        )

        adapter = _BrokerAdapter(self._policy, self._handler, self._handler_lock,
                                 self._runtime, self)
        self._broker = _broker.ActionChunkBroker(policy=adapter, action_horizon=self._cfg.action_horizon)

        with self._obs_lock:
            self._latest_obs = self._env.get_observation()
        return metadata

    # ------------------------------------------------------------------ #
    def set_handler(self, handler: _modes.ModeHandler) -> None:
        with self._handler_lock:
            self._handler = handler
            self._handler.reset()
            if self._broker is not None:
                self._broker.reset()
                adapter = self._broker._policy
                if isinstance(adapter, _BrokerAdapter):
                    adapter.set_handler(handler)
        self._on_event("mode_changed", {"mode": handler.name})

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._running.set()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="eval-runner")
        self._thread.start()

    def pause(self) -> None:
        self._running.clear()
        self._runtime.wake_subtask_waiters()

    def resume(self) -> None:
        self._running.set()

    def stop(self) -> None:
        self._stop_evt.set()
        self._running.set()
        self._runtime.wake_subtask_waiters()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._stop_active_artifacts()
        # ---- full state clear so the next Start is a clean run ---- #
        self._step_count = 0
        with self._handler_lock:
            self._handler.reset()
        if self._broker is not None:
            self._broker.reset()
        self._manual_control_evt.clear()
        self._runtime.clear()
        with self._artifact_lock:
            self._latest_model_payload = None
            self._latest_model_raw_obs = None
            self._latest_model_inference_index = -1
            self._latest_model_frame_index = None

    # ------------------------------------------------------------------ #
    # Return-to-Zero — mirrors `scripts/eval_banana.sh --reset-only` which in
    # turn calls `environment.reset()` (see piper_main.py:111-113).
    def request_reset(self) -> None:
        if self._env is None:
            self._on_event("error", {"msg": "Cannot reset: not connected. Click Connect first."})
            return

        running = self._thread is not None and self._thread.is_alive()
        if running:
            # Defer to the run loop: drained at the top of the next iteration
            # so we don't race env.step() / broker.infer().
            self._reset_evt.set()
            self._on_event("info", {"msg": "Return-to-zero queued."})
            return

        # Idle path: do it synchronously on the GUI thread, guarded by the
        # env lock. This is the same code path as the reference CLI's
        # `--reset-only` flag.
        def _do_idle_reset() -> None:
            try:
                self._on_event("info", {"msg": "Return-to-zero (idle, synchronous) ..."})
                with self._env_lock:
                    self._env.reset()
                if self._broker is not None:
                    self._broker.reset()
                with self._obs_lock:
                    self._latest_obs = self._env.get_observation()
                self._on_event("info", {"msg": "Return-to-zero done."})
            except Exception as e:                       # noqa: BLE001
                logger.exception("idle RTZ failed")
                self._on_event("error", {"msg": f"RTZ failed: {e}"})

        # Run on a small worker so the GUI doesn't block during the
        # ``reset_move_time`` interpolation.
        threading.Thread(target=_do_idle_reset, daemon=True, name="rtz-idle").start()

    def request_debug_gripper(self, value: float = 0.06) -> None:
        """Reset to home, then send a direct continuous gripper command."""
        if self._env is None:
            self._on_event("error", {"msg": "Cannot debug gripper: not connected. Click Connect first."})
            return

        value = float(value)
        self._manual_control_evt.set()
        self._running.clear()
        self._runtime.wake_subtask_waiters()
        self._on_event(
            "info",
            {"msg": f"Debug gripper armed: paused inference, resetting, then sending {value:.4f}."},
        )

        def _do_debug_gripper() -> None:
            try:
                with self._env_lock:
                    reset_ts = self._env.reset()
                    obs = getattr(reset_ts, "observation", None) or self._env.get_observation()
                    action = np.asarray(obs.get("qpos", []), dtype=np.float32).copy()
                    if action.ndim != 1 or action.size < 14:
                        raise RuntimeError("Cannot build debug gripper command: reset observation qpos is not 14-dim")
                    action = action[:14].copy()
                    action[6] = value
                    action[13] = value
                    step_ts = self._env.step(action)
                    obs_after = getattr(step_ts, "observation", None) or self._env.get_observation()
                if self._broker is not None:
                    self._broker.reset()
                with self._obs_lock:
                    self._latest_obs = obs_after
                self._emit_gripper_status(action)
                self._on_event(
                    "info",
                    {"msg": f"Debug gripper sent continuous value {value:.4f} to both grippers after reset."},
                )
            except Exception as e:                       # noqa: BLE001
                logger.exception("debug gripper failed")
                self._on_event("error", {"msg": f"Debug gripper failed: {e}"})
            finally:
                self._manual_control_evt.clear()

        threading.Thread(target=_do_debug_gripper, daemon=True, name="debug-gripper").start()

    # ------------------------------------------------------------------ #
    def request_dump(self) -> None:
        """Save the next inference's model inputs to ``cfg.dump_dir``."""
        self._pending_dump = True
        self._on_event("info", {"msg": "Dump-inputs armed; will save on next inference."})

    def _dump_payload(self, payload: Dict[str, Any], raw_obs: Dict[str, Any]) -> None:
        """Write the post-repack obs (everything that goes on the wire) to disk."""
        import cv2

        out_root = pathlib.Path(self._cfg.dump_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        idx = self._dump_counter
        self._dump_counter += 1
        ts = time.strftime("%Y%m%d_%H%M%S")
        sub = out_root / f"step_{idx:04d}_{ts}"
        sub.mkdir(parents=True, exist_ok=True)

        # 1) The exact instruction string passed to the VLA.
        prompt = payload.get("prompt", "")
        (sub / "instruction.txt").write_text(prompt or "")

        # 2) Processed images (CHW uint8 -> HWC for visual inspection).
        for cam, chw in payload.get("images", {}).items():
            arr = np.asarray(chw)
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                hwc = np.transpose(arr, (1, 2, 0))
            else:
                hwc = arr
            bgr = cv2.cvtColor(hwc, cv2.COLOR_RGB2BGR) if hwc.ndim == 3 and hwc.shape[2] == 3 else hwc
            cv2.imwrite(str(sub / f"{cam}.jpg"), bgr)

        # 3) Subgoal images (Mode 4).
        for cam, chw in (payload.get("subgoal_images", {}) or {}).items():
            arr = np.asarray(chw)
            hwc = np.transpose(arr, (1, 2, 0)) if arr.ndim == 3 and arr.shape[0] == 3 else arr
            bgr = cv2.cvtColor(hwc, cv2.COLOR_RGB2BGR) if hwc.ndim == 3 and hwc.shape[2] == 3 else hwc
            cv2.imwrite(str(sub / f"subgoal_{cam}.jpg"), bgr)

        # 4) Raw camera frames (pre-resize) for cross-checking.
        for cam, hwc in (raw_obs.get("images", {}) or {}).items():
            arr = np.asarray(hwc)
            if arr.ndim == 3 and arr.shape[2] == 3:
                cv2.imwrite(str(sub / f"raw_{cam}.jpg"), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

        # 5) Compact JSON with state vector + meta.
        meta = {
            "step": idx,
            "timestamp": ts,
            "prompt": prompt,
            "state": np.asarray(payload.get("state", []), dtype=np.float32).tolist(),
            "image_keys": list(payload.get("images", {}).keys()),
            "subgoal_image_keys": list((payload.get("subgoal_images") or {}).keys()),
            "mode": self._handler.name,
            "step_count": self._step_count,
        }
        (sub / "meta.json").write_text(json.dumps(meta, indent=2))
        self._on_event("info", {"msg": f"Dumped model inputs -> {sub}"})

    def _cache_latest_model_input(self, payload: Dict[str, Any], raw_obs: Dict[str, Any]) -> None:
        with self._artifact_lock:
            inference_index = self._input_log_counter
            self._input_log_counter += 1
            self._latest_model_payload = payload
            self._latest_model_raw_obs = raw_obs
            self._latest_model_inference_index = inference_index
            self._latest_model_frame_index = None
            input_logger = self._input_logger
        frame_index = None
        if input_logger is not None:
            frame_index = input_logger.record_subgoal_images(payload)
        with self._artifact_lock:
            self._latest_model_frame_index = frame_index

    def _log_step_model_input(self, step: int) -> None:
        with self._artifact_lock:
            input_logger = self._input_logger
            payload = self._latest_model_payload
            raw_obs = self._latest_model_raw_obs
            inference_index = self._latest_model_inference_index
            frame_index = self._latest_model_frame_index
        if input_logger is None or payload is None or raw_obs is None:
            return
        with self._handler_lock:
            mode = self._handler.name
        input_logger.log_step(
            payload,
            raw_obs,
            step=step,
            inference_index=inference_index,
            mode=mode,
            runtime_snapshot=self._runtime.snapshot(),
            cfg=self._cfg,
            frame_index=frame_index,
        )

    # ------------------------------------------------------------------ #
    def latest_obs(self) -> Optional[dict]:
        with self._obs_lock:
            return self._latest_obs

    def _store_latest_obs(self, obs: dict) -> None:
        with self._obs_lock:
            self._latest_obs = obs

    def _recording_observation(self) -> Optional[dict]:
        if self._env is None:
            return None
        # get_observation() only snapshots the latest ROS sensor data. Keep it
        # outside _env_lock so video sampling continues during reset/step waits
        # and during blocking Doubao/ForeAct/manual annotation calls.
        return self._env.get_observation()

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def running(self) -> bool:
        return self._running.is_set() and self._thread is not None and self._thread.is_alive()

    def _emit_gripper_status(self, action_arr: np.ndarray) -> None:
        if action_arr.ndim != 1 or action_arr.size <= 13:
            return
        self._on_event(
            "gripper",
            {
                "left": float(action_arr[6]),
                "right": float(action_arr[13]),
            },
        )

    def _episode_artifact_prefix(self) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        stem = _safe_filename_part(self._cfg.video_name or "eval")
        return f"{stem}_{ts}"

    def _start_video_recorder(self, artifact_prefix: str) -> Optional[_MultiCameraVideoRecorder]:
        if not self._cfg.record_video:
            return None
        output_dir = pathlib.Path(self._cfg.video_dir)
        paths = {
            name: output_dir / f"{artifact_prefix}_video_{name}.mp4"
            for name in _CONTINUOUS_VIDEO_CAMERAS
        }
        try:
            recorder = _MultiCameraVideoRecorder(
                paths=paths,
                fps=max(1.0, float(self._cfg.video_fps)),
                obs_provider=self._recording_observation,
                obs_callback=self._store_latest_obs,
                camera_keys=_CONTINUOUS_VIDEO_CAMERAS,
            )
            recorder.start()
        except Exception as e:                       # noqa: BLE001
            logger.warning("Video recording disabled: %s", e)
            self._on_event("error", {"msg": f"Video recording disabled: {e}"})
            return None
        details = ", ".join(f"{name}: {path}" for name, path in paths.items())
        self._on_event("info", {"msg": f"Recording continuous camera videos -> {details}"})
        return recorder

    def _start_input_logger(self, artifact_prefix: str) -> Optional[_StepInputLogger]:
        if not self._cfg.log_model_inputs:
            return None
        path = pathlib.Path(self._cfg.video_dir) / f"{artifact_prefix}_log.jsonl"
        try:
            input_logger = _StepInputLogger(
                path=path,
                artifact_prefix=artifact_prefix,
                video_fps=max(1.0, float(self._cfg.video_fps)),
            )
            input_logger.start()
        except Exception as e:                       # noqa: BLE001
            logger.warning("Model-input logging disabled: %s", e)
            self._on_event("error", {"msg": f"Model-input logging disabled: {e}"})
            return None
        self._on_event("info", {"msg": f"Logging VLA inputs -> {path}"})
        return input_logger

    def _stop_active_artifacts(self) -> None:
        with self._artifact_lock:
            recorder = self._video_recorder
            input_logger = self._input_logger
            self._video_recorder = None
            self._input_logger = None

        if input_logger is not None:
            input_logger.close()
            msg = f"Saved VLA input log -> {input_logger.path}"
            if input_logger.error:
                msg += f" ({input_logger.error})"
            self._on_event("info", {"msg": msg})
            if input_logger.subgoal_frame_count > 0:
                subgoal_msg = f"Saved subgoal video -> {input_logger.subgoal_video_path}"
                if input_logger.subgoal_error:
                    subgoal_msg += f" ({input_logger.subgoal_error})"
                self._on_event("info", {"msg": subgoal_msg})

        if recorder is not None:
            dropped_by_camera = recorder.stop()
            writer_errors = recorder.writer_errors
            for name, path in recorder.paths.items():
                label = name.replace("_", " ")
                if path.exists() and path.stat().st_size > 0:
                    msg = f"Saved {label} camera video -> {path}"
                else:
                    msg = f"{label.capitalize()} camera video ended without a saved file -> {path}"
                if name in writer_errors:
                    msg += f" ({writer_errors[name]})"
                dropped = dropped_by_camera.get(name, 0)
                if dropped:
                    msg += f" ({dropped} frames dropped)"
                self._on_event("info", {"msg": msg})
            if recorder.sampler_error:
                self._on_event("info", {"msg": f"Video recording issue: {recorder.sampler_error}"})

    # ------------------------------------------------------------------ #
    def _run_loop(self) -> None:
        period = 1.0 / max(1.0, self._cfg.max_hz)
        self._input_log_counter = 0
        artifact_prefix = self._episode_artifact_prefix()
        recorder = self._start_video_recorder(artifact_prefix)
        input_logger = self._start_input_logger(artifact_prefix)
        with self._artifact_lock:
            self._video_recorder = recorder
            self._input_logger = input_logger

        episode_started = False
        try:
            try:
                with self._env_lock:
                    self._env.reset()
                episode_started = True
                self._on_event("episode_start", {})
            except Exception as e:                          # noqa: BLE001
                logger.exception("Reset failed")
                self._on_event("error", {"msg": f"Reset failed: {e}"})
                return

            while not self._stop_evt.is_set() and self._step_count < self._cfg.max_steps:
                if self._reset_evt.is_set():
                    self._reset_evt.clear()
                    try:
                        self._on_event("info", {"msg": "Return-to-zero in progress ..."})
                        with self._env_lock:
                            self._env.reset()
                        if self._broker is not None:
                            self._broker.reset()
                        self._on_event("info", {"msg": "Return-to-zero done."})
                    except Exception as e:                  # noqa: BLE001
                        self._on_event("error", {"msg": f"RTZ failed: {e}"})

                if not self._running.is_set():
                    # Still publish observations to the GUI even when paused.
                    try:
                        with self._env_lock:
                            obs_now = self._env.get_observation()
                        with self._obs_lock:
                            self._latest_obs = obs_now
                    except Exception:                        # noqa: BLE001
                        pass
                    time.sleep(0.05)
                    continue

                t0 = time.time()
                try:
                    with self._env_lock:
                        raw_obs = self._env.get_observation()
                    with self._obs_lock:
                        self._latest_obs = raw_obs

                    with self._handler_lock:
                        handler = self._handler
                    reset_broker = handler.before_control_step(
                        raw_obs,
                        self._runtime,
                        self._step_count,
                        cancel_check=lambda: self._stop_evt.is_set() or not self._running.is_set(),
                    )
                    if reset_broker and self._broker is not None:
                        self._broker.reset()
                    if self._stop_evt.is_set():
                        break
                    if not self._running.is_set():
                        continue

                    action = self._broker.infer(raw_obs)
                    if self._stop_evt.is_set():
                        break
                    if not self._running.is_set():
                        continue
                    if self._manual_control_evt.is_set():
                        continue
                    action_arr = np.asarray(action["actions"], dtype=np.float32)
                    self._emit_gripper_status(action_arr)
                    with self._env_lock:
                        self._env.step(action_arr)
                    self._step_count += 1
                    self._on_event("step", {"step": self._step_count})
                    self._log_step_model_input(self._step_count)
                except Exception as e:                       # noqa: BLE001
                    logger.exception("Inference / step failed")
                    self._on_event("error", {"msg": f"Step failed: {e}"})
                    time.sleep(0.5)
                    continue

                elapsed = time.time() - t0
                if elapsed < period:
                    time.sleep(period - elapsed)
        finally:
            self._stop_active_artifacts()
        if episode_started:
            self._on_event("episode_end", {"steps": self._step_count})
