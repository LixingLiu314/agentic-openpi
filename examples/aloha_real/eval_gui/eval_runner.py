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
    video_dir: str = "test_video"
    video_name: str = "eval"
    video_fps: float = _DEFAULT_HZ


class _VideoRecorder:
    """Asynchronous RGB frame writer for VS Code-compatible H.264 MP4."""

    def __init__(self, path: pathlib.Path, fps: float) -> None:
        self.path = path
        self._fps = float(fps)
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
        h, w = frame_shape[:2]
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
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
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
        # ---- full state clear so the next Start is a clean run ---- #
        self._step_count = 0
        with self._handler_lock:
            self._handler.reset()
        if self._broker is not None:
            self._broker.reset()
        self._manual_control_evt.clear()
        self._runtime.clear()

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

    # ------------------------------------------------------------------ #
    def latest_obs(self) -> Optional[dict]:
        with self._obs_lock:
            return self._latest_obs

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

    def _start_video_recorder(self) -> Optional[_VideoRecorder]:
        if not self._cfg.record_video:
            return None
        ts = time.strftime("%Y%m%d_%H%M%S")
        stem = self._cfg.video_name.strip() or "eval"
        path = pathlib.Path(self._cfg.video_dir) / f"{stem}_{ts}.mp4"
        try:
            recorder = _VideoRecorder(path=path, fps=max(1.0, float(self._cfg.video_fps)))
            recorder.start()
        except Exception as e:                       # noqa: BLE001
            logger.warning("Video recording disabled: %s", e)
            self._on_event("error", {"msg": f"Video recording disabled: {e}"})
            return None
        self._on_event("info", {"msg": f"Recording main camera video -> {path}"})
        return recorder

    # ------------------------------------------------------------------ #
    def _run_loop(self) -> None:
        period = 1.0 / max(1.0, self._cfg.max_hz)
        try:
            with self._env_lock:
                self._env.reset()
            self._on_event("episode_start", {})
        except Exception as e:                          # noqa: BLE001
            logger.exception("Reset failed")
            self._on_event("error", {"msg": f"Reset failed: {e}"})
            return

        recorder = self._start_video_recorder()
        try:
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
                    if recorder is not None:
                        recorder.write((raw_obs.get("images") or {}).get("cam_high"))

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
                except Exception as e:                       # noqa: BLE001
                    logger.exception("Inference / step failed")
                    self._on_event("error", {"msg": f"Step failed: {e}"})
                    time.sleep(0.5)
                    continue

                elapsed = time.time() - t0
                if elapsed < period:
                    time.sleep(period - elapsed)
        finally:
            if recorder is not None:
                dropped = recorder.stop()
                if recorder.path.exists() and recorder.path.stat().st_size > 0:
                    msg = f"Saved main camera video -> {recorder.path}"
                else:
                    msg = f"Video recording ended without a saved file -> {recorder.path}"
                if recorder.error:
                    msg += f" ({recorder.error})"
                if dropped:
                    msg += f" ({dropped} frames dropped)"
                self._on_event("info", {"msg": msg})
        self._on_event("episode_end", {"steps": self._step_count})
