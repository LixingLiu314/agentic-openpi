"""Mode handlers that translate a raw env observation + GUI runtime state
into the ``obs`` dict consumed by the openpi WebSocket policy server.

Each ``ModeHandler`` returns a dict that the server feeds into

    [data_transforms.inputs] -> Normalize -> [model_transforms.inputs]

i.e. **after** the dataset's ``repack_transforms`` would normally have run.
That means the handler must do the work of those repack transforms itself:

  * basic   -- raw {state, images}; server uses ``--default-prompt``.
  * traj    -- prompt = ``"<task>, traj: Left: Go along ... Right: Go along
               ..."``.
  * subtask -- prompt = ``"<task>, subtask: <label>"`` (mirrors
               ``_transforms.AppendSubtaskToPrompt`` exactly). Labels are
               user-editable and the keys 1..4 select among them.
  * triple_cot -- prompt = ``"Task: <task>, subtask: <label>, traj: <text>"``.
                  This mode uses the semi-block trajectory pipeline while the
                  subtask key remains a live runtime value.
  * subgoal -- adds ``subgoal_images = {"cam_high": <CHW uint8 array>}``,
               which is what ``LeRobotAlohaWithSubgoalDataConfig`` passes
               into ``AlohaWithSubgoalInputs``.

For modes 2/3/4, two execution sub-modes are supported:
  * non-blocking  -- external request fired on a background thread; the
                     control loop continues using the *most recently*
                     received external input.
  * blocking      -- every ``step_interval`` steps the control loop pauses
                     and waits for a fresh external input before continuing.
"""
from __future__ import annotations

import inspect
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import einops
import numpy as np
from openpi_client import image_tools

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Default subtask label mapping for the eggplant task.
# These keys (1-8) match the operator controls in subtask mode.
# At runtime the user can edit them in the GUI or load a JSON override.
# ---------------------------------------------------------------------------
DEFAULT_SUBTASK_LABELS: Dict[int, str] = {
    1: "reach the handle of the lid",
    2: "Grasp the handle of the lid",
    3: "Move away the lid",
    4: "reach the eggplant",
    5: "grasp the eggplant",
    6: "Move the eggplant on the box",
    7: "Release the eggplant to the box",
    8: "Put the lid on the box",
}


def to_chw_uint8(img_hwc_rgb: np.ndarray, h: int = 224, w: int = 224) -> np.ndarray:
    """HxWx3 uint8 RGB -> (3, h, w) uint8 with the same resize/pad as training."""
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img_hwc_rgb, h, w))
    return einops.rearrange(img, "h w c -> c h w")



# ---------------------------------------------------------------------------
@dataclass
class RuntimeState:
    """Mutable state shared between the GUI thread and the eval runner thread."""

    mode: str = "basic"                   # one of: basic | traj | subtask | triple_cot | subgoal
    task: str = "put banana in the green plate"

    # --- subtask config (editable from GUI / JSON file) ---------------- #
    subtask_labels: Dict[int, str] = field(
        default_factory=lambda: dict(DEFAULT_SUBTASK_LABELS)
    )
    subtask_key: int = 1                  # default to key '1' on initialisation

    # --- live displays (filled by mode handlers, read by the GUI) ------ #
    last_prompt: str = ""
    last_subtask_label: str = ""
    last_traj_text: str = ""
    last_traj_image: Optional[np.ndarray] = None       # HWC uint8 RGB with trajectory overlay
    manual_traj_override: bool = False
    last_subgoal_image: Optional[np.ndarray] = None    # HWC uint8 RGB
    waiting_for_subtask_input: bool = False

    subtask_input_seq: int = 0
    subtask_wake_seq: int = 0
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)
    _subtask_cond: threading.Condition = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._subtask_cond = threading.Condition(self._lock)

    # --- helpers -------------------------------------------------------- #
    def load_subtasks_from_json(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        # Accept ``{"1": "...", ...}`` or ``{1: "...", ...}``.
        new = {int(k): str(v) for k, v in data.items() if str(k).isdigit()}
        if not new:
            raise ValueError(f"No usable {{int_key: str_label}} entries in {path}")
        with self._subtask_cond:
            self.subtask_labels = new
            if self.subtask_key not in new:
                self.subtask_key = sorted(new.keys())[0]
            self.subtask_input_seq += 1
            self._subtask_cond.notify_all()

    def set_subtask_label(self, key: int, label: str) -> None:
        key = int(key)
        with self._subtask_cond:
            self.subtask_labels[key] = str(label)
            if key == self.subtask_key:
                self.subtask_input_seq += 1
                self._subtask_cond.notify_all()

    def add_subtask_label(self, label: Optional[str] = None) -> int:
        with self._subtask_cond:
            key = (max(self.subtask_labels) + 1) if self.subtask_labels else 1
            self.subtask_labels[key] = str(label if label is not None else f"subtask_{key}")
            if len(self.subtask_labels) == 1:
                self.subtask_key = key
                self.subtask_input_seq += 1
                self._subtask_cond.notify_all()
            return key

    def remove_subtask_label(self, key: int) -> str:
        key = int(key)
        with self._subtask_cond:
            if len(self.subtask_labels) <= 1:
                raise ValueError("At least one subtask label is required")
            if key not in self.subtask_labels:
                raise KeyError(f"subtask key {key} does not exist")
            label = self.subtask_labels.pop(key)
            if self.subtask_key == key:
                self.subtask_key = sorted(self.subtask_labels.keys())[0]
                self.subtask_input_seq += 1
                self._subtask_cond.notify_all()
            return label

    def set_subtask_key(self, key: int) -> None:
        """Select or confirm a subtask key and wake blocking Mode 3 waits."""
        with self._subtask_cond:
            if int(key) not in self.subtask_labels:
                raise KeyError(f"subtask key {key} does not exist")
            self.subtask_key = int(key)
            self.subtask_input_seq += 1
            self._subtask_cond.notify_all()

    def wait_for_subtask_input(
        self,
        after_seq: int,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Optional[int]:
        """Block until the operator provides a new subtask key/confirmation."""
        with self._subtask_cond:
            wake_seq = self.subtask_wake_seq
            self.waiting_for_subtask_input = True
            try:
                while True:
                    if self.subtask_input_seq > after_seq:
                        return self.subtask_input_seq
                    if self.subtask_wake_seq > wake_seq:
                        return None
                    if cancel_check is not None and cancel_check():
                        return None
                    self._subtask_cond.wait(timeout=0.1)
            finally:
                self.waiting_for_subtask_input = False

    def wake_subtask_waiters(self) -> None:
        """Release any blocking subtask wait, used when stopping or switching."""
        with self._subtask_cond:
            self.subtask_wake_seq += 1
            self._subtask_cond.notify_all()

    def clear(self) -> None:
        """Wipe all live-display fields. Call after Stop so the next run
        starts with a blank slate.
        """
        with self._subtask_cond:
            self.last_prompt = ""
            self.last_subtask_label = ""
            self.last_traj_text = ""
            self.last_traj_image = None
            self.last_subgoal_image = None
            self.waiting_for_subtask_input = False
            self.subtask_wake_seq += 1
            self._subtask_cond.notify_all()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "mode": self.mode,
                "task": self.task,
                "subtask_key": self.subtask_key,
                "subtask_label": self.last_subtask_label,
                "subtask_labels": dict(self.subtask_labels),
                "prompt": self.last_prompt,
                "traj_text": self.last_traj_text,
                "traj_image": self.last_traj_image,
                "manual_traj_override": self.manual_traj_override,
                "subgoal_image": self.last_subgoal_image,
                "waiting_for_subtask_input": self.waiting_for_subtask_input,
            }


# ---------------------------------------------------------------------------
class ModeHandler:
    """Strategy interface: build the obs dict for one inference call."""

    name: str = "basic"

    def reset(self) -> None:
        ...

    def build_obs(self, raw_obs: Dict[str, Any], runtime: RuntimeState) -> Dict[str, Any]:
        raise NotImplementedError

    def before_control_step(
        self,
        raw_obs: Dict[str, Any],
        runtime: RuntimeState,
        loop_step: int,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> bool:
        """Run per-control-step gating before cached actions are consumed.

        Return True when the action broker should discard any cached action
        chunk and request a fresh VLA inference for this step.
        """
        return False

    # Run-time knobs (only meaningful for traj / subtask / subgoal). The base
    # impls are no-ops so the GUI can call them blindly.
    def set_blocking(self, blocking: bool) -> None:
        ...

    def set_step_interval(self, n: int) -> None:
        ...

    def set_manual_trajectory_provider(self, provider: Optional[Callable[..., Any]]) -> None:
        ...

    def set_manual_trajectory_override(self, enabled: bool) -> None:
        ...

    @staticmethod
    def _base_obs(raw_obs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a ``PiperRealEnv.get_observation()`` dict to {state, images}."""
        images = {name: to_chw_uint8(img) for name, img in raw_obs["images"].items()}
        return {
            "state": np.asarray(raw_obs["qpos"], dtype=np.float32),
            "images": images,
        }


class BasicMode(ModeHandler):
    name = "basic"

    def build_obs(self, raw_obs, runtime):
        obs = self._base_obs(raw_obs)
        obs["prompt"] = runtime.task
        with runtime._lock:
            runtime.last_prompt = runtime.task
        return obs


# ---------------------------------------------------------------------------
class SubtaskMode(ModeHandler):
    """Mode 3: prompt = f"{task}, subtask: {label}".

    Label is selected by ``runtime.subtask_key``; the GUI's 1/2/3/4 keys
    update that field. An optional ``subtask_predictor`` hook can
    auto-suggest the next key every ``step_interval`` steps; if it
    returns ``None`` we keep the human override.
    """

    name = "subtask"

    def __init__(
        self,
        *,
        predictor: Optional[Any] = None,        # ``SubtaskPredictor`` from doubao_predictor
        blocking: bool = False,
        step_interval: int = 30,
    ) -> None:
        self._predictor = predictor
        self._blocking = bool(blocking)
        self._step_interval = max(1, int(step_interval))
        self._step = 0
        self._initialized = False   # True once first suggestion received (or predictor is None)
        self._lock = threading.Lock()
        self._inflight: Optional[threading.Thread] = None
        self._suggestion: Optional[int] = None  # last auto suggestion
        self._last_manual_seq: Optional[int] = None
        self._last_blocking_control_step: Optional[int] = None

    def reset(self) -> None:
        with self._lock:
            self._step = 0
            self._initialized = False
            self._inflight = None
            self._suggestion = None
            self._last_manual_seq = None
            self._last_blocking_control_step = None
        if self._predictor is not None and hasattr(self._predictor, "reset"):
            self._predictor.reset()

    def set_blocking(self, blocking: bool) -> None:
        self._blocking = bool(blocking)

    def set_step_interval(self, n: int) -> None:
        self._step_interval = max(1, int(n))

    # ------------------------------------------------------------------ #
    def _run_predictor(self, image: np.ndarray, task: str, labels: Dict[int, str]) -> Optional[int]:
        if self._predictor is None:
            return None
        try:
            return self._predictor.suggest(image, task, labels)
        except Exception as e:  # noqa: BLE001
            logger.warning("SubtaskPredictor.suggest failed: %s", e)
            return None

    def before_control_step(
        self,
        raw_obs,
        runtime,
        loop_step: int,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> bool:
        if not self._blocking:
            return False
        if loop_step != 0 and loop_step % self._step_interval != 0:
            return False

        with self._lock:
            if self._last_blocking_control_step == loop_step:
                return False
            last_manual_seq = self._last_manual_seq

        if self._predictor is not None:
            cam = raw_obs["images"].get("cam_high")
            if cam is None:
                return False
            with runtime._lock:
                labels = dict(runtime.subtask_labels)
                task = runtime.task
            sug = self._run_predictor(cam.copy(), task, labels)
            with self._lock:
                self._suggestion = sug
                self._initialized = True
                self._last_blocking_control_step = loop_step
            return True

        with runtime._lock:
            current_seq = runtime.subtask_input_seq
        if last_manual_seq is None:
            last_manual_seq = current_seq
            with self._lock:
                if self._last_manual_seq is None:
                    self._last_manual_seq = current_seq

        if current_seq <= last_manual_seq:
            logger.info("Blocking subtask mode waiting for subtask key input at step %d", loop_step)
            maybe_seq = runtime.wait_for_subtask_input(last_manual_seq, cancel_check=cancel_check)
            if maybe_seq is None:
                return False
            current_seq = maybe_seq

        with self._lock:
            self._last_manual_seq = current_seq
            self._initialized = True
            self._last_blocking_control_step = loop_step
        return True

    def _maybe_update_suggestion(self, raw_obs, runtime):
        with self._lock:
            first_step = not self._initialized and self._predictor is not None
            should_request = first_step or (self._step % self._step_interval == 0)
            self._step += 1
            blocking = first_step or self._blocking
            in_flight = self._inflight is not None and self._inflight.is_alive()

        if self._predictor is None:
            with self._lock:
                self._initialized = True
            return

        if not should_request:
            return
        cam = raw_obs["images"].get("cam_high")
        if cam is None:
            return
        if blocking:
            with runtime._lock:
                labels = dict(runtime.subtask_labels)
                task = runtime.task
            sug = self._run_predictor(cam.copy(), task, labels)
            with self._lock:
                self._suggestion = sug
                if sug is not None:
                    self._initialized = True
            return
        if in_flight:
            return
        img_copy = cam.copy()
        with runtime._lock:
            labels = dict(runtime.subtask_labels)
            task = runtime.task

        def _run() -> None:
            sug = self._run_predictor(img_copy, task, labels)
            with self._lock:
                self._suggestion = sug
                if sug is not None:
                    self._initialized = True

        t = threading.Thread(target=_run, daemon=True, name="subtask-predict")
        t.start()
        with self._lock:
            self._inflight = t

    # ------------------------------------------------------------------ #
    def build_obs(self, raw_obs, runtime):
        obs = self._base_obs(raw_obs)
        self._maybe_update_suggestion(raw_obs, runtime)

        with runtime._lock:
            labels = dict(runtime.subtask_labels)
            human_key = runtime.subtask_key
            task = runtime.task

        # Human override always wins over the auto suggestion. The auto
        # suggestion only applies when the human hasn't picked a key yet
        # *or* the human's key is no longer in the label table.
        with self._lock:
            auto_key = self._suggestion
        key = human_key if human_key in labels else auto_key
        if key is None or key not in labels:
            key = sorted(labels.keys())[0] if labels else 1
        label = labels.get(key, "")
        prompt = f"{task}, subtask: {label}" if label else task

        with runtime._lock:
            runtime.last_subtask_label = label
            runtime.last_prompt = prompt
        obs["prompt"] = prompt
        return obs


# ---------------------------------------------------------------------------
class TrajectoryMode(ModeHandler):
    """Mode 2: prompt = f"{task}, traj: <L/R loc-token text>"."""

    name = "traj"

    def __init__(
        self,
        predictor,
        *,
        manual_provider: Optional[Callable[..., Any]] = None,
        manual_override: bool = False,
    ) -> None:
        from .doubao_predictor import CachedTrajectoryPredictor  # local import

        self._predictor: CachedTrajectoryPredictor = predictor
        self._lock = threading.Lock()
        self._last_blocking_control_step: Optional[int] = None
        self._use_cached_once = False
        self._manual_provider = manual_provider
        self._manual_override = bool(manual_override)
        self._pipeline_predictions: Dict[int, tuple[Any, np.ndarray]] = {}
        self._pipeline_triggered_slots: set[int] = set()
        self._pipeline_active_slot: Optional[int] = None
        self._pipeline_threads: Dict[int, threading.Thread] = {}
        self._pipeline_events: Dict[int, threading.Event] = {}
        self._pipeline_generation = 0
        self._manual_request_started_callback: Optional[Callable[[], None]] = None

    def reset(self) -> None:
        self._predictor.reset()
        with self._lock:
            self._last_blocking_control_step = None
            self._use_cached_once = False
            self._pipeline_predictions.clear()
            self._pipeline_triggered_slots.clear()
            self._pipeline_active_slot = None
            self._pipeline_threads.clear()
            self._pipeline_events.clear()
            self._pipeline_generation += 1
            self._manual_request_started_callback = None

    def set_blocking(self, blocking: bool) -> None:
        self._predictor.set_blocking(blocking)

    def set_step_interval(self, n: int) -> None:
        self._predictor.set_update_every(n)

    def set_manual_trajectory_provider(self, provider: Optional[Callable[..., Any]]) -> None:
        with self._lock:
            self._manual_provider = provider

    def set_manual_trajectory_override(self, enabled: bool) -> None:
        with self._lock:
            self._manual_override = bool(enabled)

    def _set_manual_request_started_callback(
        self,
        callback: Optional[Callable[[], None]],
    ) -> None:
        with self._lock:
            self._manual_request_started_callback = callback

    def _pop_manual_request_started_callback(self) -> Optional[Callable[[], None]]:
        with self._lock:
            callback = self._manual_request_started_callback
            self._manual_request_started_callback = None
            return callback

    @staticmethod
    def _call_manual_provider(
        manual_provider: Callable[..., Any],
        image: np.ndarray,
        task: str,
        cancel_check: Optional[Callable[[], bool]],
        request_started: Optional[Callable[[], None]],
    ):
        if request_started is None:
            return manual_provider(image, task, cancel_check)
        try:
            params = inspect.signature(manual_provider).parameters
            accepts_hook = (
                "request_started" in params
                or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            )
        except (TypeError, ValueError):
            accepts_hook = False
        if accepts_hook:
            return manual_provider(
                image,
                task,
                cancel_check,
                request_started=request_started,
            )
        request_started()
        return manual_provider(image, task, cancel_check)

    def _predict_trajectory_once(
        self,
        image: np.ndarray,
        task: str,
        cancel_check: Optional[Callable[[], bool]] = None,
        *,
        require_valid: bool = False,
    ):
        with self._lock:
            manual_provider = self._manual_provider
            manual_override = self._manual_override

        while True:
            if cancel_check is not None and cancel_check():
                return None
            if manual_override and manual_provider is not None:
                request_started = self._pop_manual_request_started_callback()
                pred = self._call_manual_provider(
                    manual_provider,
                    image.copy(),
                    task,
                    cancel_check,
                    request_started,
                )
            else:
                predictor_impl = getattr(self._predictor, "_predictor", None)
                if predictor_impl is None:
                    pred = self._predictor.refresh(image.copy(), task)
                else:
                    pred = predictor_impl.predict(image.copy(), task)
            if pred is not None and not pred.is_empty():
                return pred
            if not require_valid:
                return None
            logger.info("Trajectory request returned no prediction; waiting for a finished annotation.")
            time.sleep(0.1)

    def _store_pipeline_prediction(
        self,
        slot: int,
        image: np.ndarray,
        pred,
        generation: Optional[int] = None,
    ) -> bool:
        if pred is None or pred.is_empty():
            return False
        with self._lock:
            if generation is not None and generation != self._pipeline_generation:
                return False
            self._pipeline_predictions[int(slot)] = (pred, image.copy())
        return True

    def _spawn_pipeline_request(
        self,
        slot: int,
        image: np.ndarray,
        task: str,
        cancel_check: Optional[Callable[[], bool]],
    ) -> None:
        with self._lock:
            if slot in self._pipeline_triggered_slots:
                return
            self._pipeline_triggered_slots.add(slot)
            generation = self._pipeline_generation
            done_evt = threading.Event()
            self._pipeline_events[slot] = done_evt

        img_copy = image.copy()

        def _run() -> None:
            try:
                pred = self._predict_trajectory_once(
                    img_copy,
                    task,
                    cancel_check=cancel_check,
                    require_valid=False,
                )
                if self._store_pipeline_prediction(slot, img_copy, pred, generation):
                    logger.info("Trajectory annotation for slot %d finished.", slot)
                else:
                    logger.info("Trajectory annotation for slot %d did not produce a prediction.", slot)
            except Exception:
                logger.exception("Trajectory annotation worker failed for slot %d", slot)
            finally:
                done_evt.set()

        t = threading.Thread(target=_run, daemon=True, name=f"trajectory-annotate-{slot}")
        with self._lock:
            self._pipeline_threads[slot] = t
        t.start()

    def _wait_for_pipeline_slot(
        self,
        slot: int,
        fallback_image: np.ndarray,
        task: str,
        cancel_check: Optional[Callable[[], bool]],
    ) -> bool:
        with self._lock:
            event = self._pipeline_events.get(slot)
            thread = self._pipeline_threads.get(slot)
            already_ready = slot in self._pipeline_predictions
        if already_ready:
            return True

        if event is not None:
            logger.info("Trajectory pipeline waiting for annotation slot %d.", slot)
            while not event.wait(timeout=0.1):
                if cancel_check is not None and cancel_check():
                    return False
            if thread is not None:
                thread.join(timeout=0.0)

        with self._lock:
            if slot in self._pipeline_predictions:
                return True

        logger.info(
            "Trajectory annotation slot %d is missing or invalid; requesting it synchronously.",
            slot,
        )
        pred = self._predict_trajectory_once(
            fallback_image,
            task,
            cancel_check=cancel_check,
            require_valid=True,
        )
        return self._store_pipeline_prediction(slot, fallback_image, pred)

    def _apply_pipeline_slot(self, desired_slot: int, runtime: RuntimeState, task: str) -> bool:
        with self._lock:
            if self._pipeline_active_slot == desired_slot:
                return False
            if desired_slot not in self._pipeline_predictions:
                return False
            pred, image = self._pipeline_predictions[desired_slot]
            self._pipeline_active_slot = desired_slot
            self._use_cached_once = True

        self._predictor.set_cached(pred, image)
        self._store_prediction_visualization(runtime, image, pred, task)
        return True

    @staticmethod
    def _desired_pipeline_slot(execution_slot: int) -> int:
        if execution_slot <= 1:
            return 0
        return execution_slot - 1

    def before_control_step(
        self,
        raw_obs,
        runtime,
        loop_step: int,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> bool:
        if not self._predictor.blocking:
            return False
        update_every = self._predictor.update_every
        if loop_step != 0 and loop_step % update_every != 0:
            return False

        with self._lock:
            if self._last_blocking_control_step == loop_step:
                return False

        if cancel_check is not None and cancel_check():
            return False
        cam_high = raw_obs["images"].get("cam_high")
        if cam_high is None:
            return False

        with runtime._lock:
            task = runtime.task

        slot = loop_step // update_every
        applied_changed = False
        if slot == 0:
            logger.info("Trajectory pipeline waiting for initial annotation at step 0")
            pred = self._predict_trajectory_once(
                cam_high,
                task,
                cancel_check=cancel_check,
                require_valid=True,
            )
            if pred is None or pred.is_empty():
                return False
            self._store_pipeline_prediction(0, cam_high, pred)
            applied_changed = self._apply_pipeline_slot(0, runtime, task)
        elif slot == 1:
            logger.info(
                "Trajectory pipeline triggering annotation slot 1 at step %d without blocking execution.",
                loop_step,
            )
            self._spawn_pipeline_request(1, cam_high, task, cancel_check)
        else:
            desired_slot = self._desired_pipeline_slot(slot)
            logger.info(
                "Trajectory pipeline boundary step %d waiting for annotation slot %d.",
                loop_step,
                desired_slot,
            )
            if not self._wait_for_pipeline_slot(desired_slot, cam_high, task, cancel_check):
                return False
            applied_changed = self._apply_pipeline_slot(desired_slot, runtime, task)
            logger.info(
                "Trajectory pipeline triggering annotation slot %d at step %d without blocking execution.",
                slot,
                loop_step,
            )
            self._spawn_pipeline_request(slot, cam_high, task, cancel_check)
        with self._lock:
            self._last_blocking_control_step = loop_step
        return applied_changed

    def _render_trajectory_image(self, image: np.ndarray, traj_text: str) -> Optional[np.ndarray]:
        try:
            from .trajectory_visualizer import render_trajectory_overlay

            return render_trajectory_overlay(image, traj_text)
        except Exception as e:  # noqa: BLE001
            logger.warning("Trajectory visualization failed: %s", e)
            return None

    def _store_prediction_visualization(
        self,
        runtime: RuntimeState,
        image: np.ndarray,
        pred,
        task: str,
    ) -> None:
        if pred is None or pred.is_empty():
            return
        prompt_traj_text = pred.to_loc_token_text()
        traj_image = self._render_trajectory_image(image, prompt_traj_text)
        with runtime._lock:
            runtime.last_traj_text = prompt_traj_text
            runtime.last_traj_image = traj_image
            runtime.last_prompt = f"{task}, traj: {prompt_traj_text}"

    def build_obs(self, raw_obs, runtime):
        obs = self._base_obs(raw_obs)

        cam_high = raw_obs["images"]["cam_high"]
        with runtime._lock:
            task = runtime.task

        self._predictor.set_update_callback(
            lambda pred, image, runtime=runtime, task=task: self._store_prediction_visualization(
                runtime,
                image,
                pred,
                task,
            )
        )
        with self._lock:
            use_cached_once = self._use_cached_once
            self._use_cached_once = False

        if self._predictor.blocking:
            pred = self._predictor.last
            if pred is None or pred.is_empty():
                pred = self._predictor.refresh(cam_high, task)
        elif use_cached_once:
            pred = self._predictor.last
        else:
            pred = self._predictor.step(cam_high, task)

        if pred is None or pred.is_empty():
            prompt_traj_text = ""
            prompt = task
            traj_image = None
        else:
            prompt_traj_text = pred.to_loc_token_text()
            prompt = f"{task}, traj: {prompt_traj_text}"
            traj_image = self._render_trajectory_image(cam_high, prompt_traj_text)

        with runtime._lock:
            runtime.last_traj_text = prompt_traj_text
            runtime.last_traj_image = traj_image
            runtime.last_prompt = prompt
        obs["prompt"] = prompt
        return obs


# ---------------------------------------------------------------------------
class TripleCotMode(TrajectoryMode):
    """Prompt mode combining task, live subtask label, and trajectory CoT.

    The trajectory leg intentionally reuses ``TrajectoryMode``'s semi-block
    pipeline path, but ``build_obs`` reads ``runtime.subtask_key`` at inference
    time so GUI digit-key changes made during a manual/Doubao trajectory wait
    are reflected in the next model prompt.
    """

    name = "triple_cot"

    def __init__(
        self,
        predictor,
        *,
        manual_provider: Optional[Callable[..., Any]] = None,
        manual_override: bool = False,
        subgoal_handler: Optional[Any] = None,
        foreact_timeout_s: float = 3.0,
    ) -> None:
        super().__init__(
            predictor,
            manual_provider=manual_provider,
            manual_override=manual_override,
        )
        self._predictor.set_blocking(True)
        self._subgoal_handler = subgoal_handler
        self._foreact_timeout_s = max(0.1, float(foreact_timeout_s))
        self._foreact_lock = threading.Lock()
        self._foreact_start_times: Dict[int, float] = {}
        self._foreact_timed_out_steps: set[int] = set()

    def reset(self) -> None:
        super().reset()
        if self._subgoal_handler is not None:
            self._subgoal_handler.reset()
        with self._foreact_lock:
            self._foreact_start_times.clear()
            self._foreact_timed_out_steps.clear()

    def set_blocking(self, blocking: bool) -> None:
        # Triple-CoT is defined around the semi-block pipeline. Ignore attempts to relax it.
        self._predictor.set_blocking(True)

    def set_step_interval(self, n: int) -> None:
        super().set_step_interval(n)
        if self._subgoal_handler is not None:
            self._subgoal_handler.set_step_interval(n)

    def _should_generate_foreact(self, loop_step: int) -> bool:
        if self._subgoal_handler is None:
            return False
        should_generate = getattr(self._subgoal_handler, "_should_generate_subgoal", None)
        if callable(should_generate):
            return bool(should_generate(loop_step))
        update_every = self._predictor.update_every
        if loop_step == 0:
            return True
        if loop_step == update_every:
            return False
        return loop_step > update_every and loop_step % update_every == 0

    def _has_subgoal_cache(self) -> bool:
        if self._subgoal_handler is None:
            return False
        with self._subgoal_handler._lock:
            return bool(self._subgoal_handler._cache)

    def _ensure_subgoal_fallback(self, raw_obs, runtime) -> bool:
        """Keep Triple-CoT inference unblocked when ForeAct is unavailable.

        If a previous subgoal exists, we keep it. Otherwise we install a blank
        image with the same camera shape so the policy input schema remains
        stable even when the ForeAct server is down.
        """
        if self._subgoal_handler is None:
            return False
        with self._subgoal_handler._lock:
            if self._subgoal_handler._cache:
                return False
            cameras = list(self._subgoal_handler._cameras)

        raw_images = raw_obs.get("images") or {}
        installed = False
        for cam in cameras:
            img = raw_images.get(cam)
            if img is None:
                continue
            blank = np.zeros_like(img, dtype=np.uint8)
            self._subgoal_handler._store_subgoal(cam, blank, runtime)
            installed = True
        if installed:
            logger.warning("ForeAct unavailable; using a blank Triple-CoT subgoal placeholder.")
        return installed

    def _start_foreact_async(self, raw_obs, runtime, loop_step: int) -> bool:
        if self._subgoal_handler is None or not self._should_generate_foreact(loop_step):
            return False

        raw_images = raw_obs.get("images") or {}
        with self._subgoal_handler._lock:
            cameras = list(self._subgoal_handler._cameras)
        if not any(cam in raw_images for cam in cameras):
            return False

        with runtime._lock:
            task = runtime.task

        try:
            started = self._subgoal_handler._spawn_request(
                raw_obs,
                task,
                runtime,
                request_step=loop_step,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("ForeAct async request could not be started at step %d: %s", loop_step, e)
            self._ensure_subgoal_fallback(raw_obs, runtime)
            return False

        if started:
            with self._foreact_lock:
                self._foreact_start_times[loop_step] = time.monotonic()
            logger.info("Triple-CoT started ForeAct asynchronously at step %d.", loop_step)
        return started

    def _foreact_thread_for_step(self, loop_step: int) -> Optional[threading.Thread]:
        if self._subgoal_handler is None:
            return None
        with self._subgoal_handler._lock:
            if self._subgoal_handler._last_request_control_step != loop_step:
                return None
            return self._subgoal_handler._inflight

    def _finish_foreact_with_timeout(
        self,
        raw_obs,
        runtime,
        loop_step: int,
        cancel_check: Optional[Callable[[], bool]],
    ) -> bool:
        if self._subgoal_handler is None or not self._should_generate_foreact(loop_step):
            return False

        with self._foreact_lock:
            started_at = self._foreact_start_times.get(loop_step)
        thread = self._foreact_thread_for_step(loop_step)

        if started_at is not None and thread is not None and thread.is_alive():
            deadline = started_at + self._foreact_timeout_s
            while thread.is_alive():
                if cancel_check is not None and cancel_check():
                    return False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                thread.join(timeout=min(0.05, remaining))

        if thread is not None and thread.is_alive():
            with self._foreact_lock:
                first_timeout = loop_step not in self._foreact_timed_out_steps
                self._foreact_timed_out_steps.add(loop_step)
            if first_timeout:
                logger.warning(
                    "ForeAct did not finish within %.1fs at step %d; continuing with cached/blank subgoal.",
                    self._foreact_timeout_s,
                    loop_step,
                )
            return self._ensure_subgoal_fallback(raw_obs, runtime)

        if not self._has_subgoal_cache():
            logger.warning(
                "ForeAct produced no subgoal at step %d; continuing with a blank placeholder.",
                loop_step,
            )
            return self._ensure_subgoal_fallback(raw_obs, runtime)

        with self._subgoal_handler._lock:
            self._subgoal_handler._last_blocking_control_step = loop_step
            self._subgoal_handler._use_cached_once = True
        return True

    def before_control_step(
        self,
        raw_obs,
        runtime,
        loop_step: int,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> bool:
        # Step 0 owns the only synchronous "open GUI, then wait" trajectory
        # request. Start ForeAct immediately after the GUI request is emitted
        # so the subgoal request runs while the human draws the trajectory.
        if loop_step == 0 and self._should_generate_foreact(loop_step):
            self._set_manual_request_started_callback(
                lambda: self._start_foreact_async(raw_obs, runtime, loop_step)
            )
        else:
            # At later k*N boundaries the relevant trajectory GUI was already
            # launched N steps earlier; start ForeAct before waiting for that
            # annotation to finish.
            self._start_foreact_async(raw_obs, runtime, loop_step)
        traj_changed = super().before_control_step(
            raw_obs,
            runtime,
            loop_step,
            cancel_check=cancel_check,
        )
        if self._should_generate_foreact(loop_step):
            # Defensive fallback for non-GUI/manual providers that don't invoke
            # the request-started hook.
            self._start_foreact_async(raw_obs, runtime, loop_step)
        subgoal_ready = self._finish_foreact_with_timeout(
            raw_obs,
            runtime,
            loop_step,
            cancel_check=cancel_check,
        )
        return subgoal_ready or traj_changed

    @staticmethod
    def _subtask_label(runtime: RuntimeState) -> str:
        with runtime._lock:
            labels = dict(runtime.subtask_labels)
            key = runtime.subtask_key
        if key not in labels and labels:
            key = sorted(labels.keys())[0]
        return labels.get(key, "")

    @staticmethod
    def _triple_prompt(task: str, subtask: str, traj_text: str) -> str:
        return f"Task: {task}, subtask: {subtask}, traj: {traj_text}"

    def _store_prediction_visualization(
        self,
        runtime: RuntimeState,
        image: np.ndarray,
        pred,
        task: str,
    ) -> None:
        if pred is None or pred.is_empty():
            return
        prompt_traj_text = pred.to_loc_token_text()
        traj_image = self._render_trajectory_image(image, prompt_traj_text)
        subtask = self._subtask_label(runtime)
        with runtime._lock:
            runtime.last_subtask_label = subtask
            runtime.last_traj_text = prompt_traj_text
            runtime.last_traj_image = traj_image
            runtime.last_prompt = self._triple_prompt(task, subtask, prompt_traj_text)

    def build_obs(self, raw_obs, runtime):
        obs = self._base_obs(raw_obs)

        cam_high = raw_obs["images"]["cam_high"]
        with runtime._lock:
            task = runtime.task

        self._predictor.set_update_callback(
            lambda pred, image, runtime=runtime, task=task: self._store_prediction_visualization(
                runtime,
                image,
                pred,
                task,
            )
        )
        with self._lock:
            use_cached_once = self._use_cached_once
            self._use_cached_once = False

        pred = self._predictor.last
        if pred is None or pred.is_empty():
            pred = self._predictor.refresh(cam_high, task)
        elif not use_cached_once and not self._predictor.blocking:
            # Defensive fallback; set_blocking() above keeps this mode blocking.
            pred = self._predictor.step(cam_high, task)

        if pred is None or pred.is_empty():
            prompt_traj_text = ""
            traj_image = None
        else:
            prompt_traj_text = pred.to_loc_token_text()
            traj_image = self._render_trajectory_image(cam_high, prompt_traj_text)

        subtask = self._subtask_label(runtime)
        prompt = self._triple_prompt(task, subtask, prompt_traj_text)

        with runtime._lock:
            runtime.last_subtask_label = subtask
            runtime.last_traj_text = prompt_traj_text
            runtime.last_traj_image = traj_image
            runtime.last_prompt = prompt
        if self._subgoal_handler is not None:
            self._subgoal_handler.attach_cached_subgoal_images(obs)
        obs["prompt"] = prompt
        return obs


# ---------------------------------------------------------------------------
class SubgoalMode(ModeHandler):
    """Mode 4: ForeAct generates the subgoal image; we add it under ``subgoal_images``.

    The training-time repack maps::

        "subgoal_images": {"cam_high": "observation.images.cam_high_subgoal"}

    so at inference we send ``subgoal_images`` keyed by the SOURCE camera name;
    ``AlohaWithSubgoalInputs`` then places it under ``image["subgoal_base_0_rgb"]``
    according to the config's ``subgoal_camera_map``.

    Two execution sub-modes:
      * non-blocking -- request fired on a background thread; control loop
                        keeps stepping with the cached subgoal.
      * blocking     -- every ``step_interval`` steps the loop pauses and
                        synchronously waits for a fresh subgoal.
    """

    name = "subgoal"

    def __init__(
        self,
        client,
        *,
        update_every: int = 30,
        cameras: Optional[List[str]] = None,
        blocking: bool = False,
    ) -> None:
        from .foreact_client import ForeactClient  # local import

        self._client: ForeactClient = client
        self._update_every = max(1, int(update_every))
        self._cameras = list(cameras or ["cam_high"])
        self._blocking = bool(blocking)
        self._step = 0
        self._initialized = False   # True once first subgoal image received
        self._lock = threading.Lock()
        self._cache: Dict[str, np.ndarray] = {}     # camera_name -> HWC uint8
        self._inflight: Optional[threading.Thread] = None
        self._last_blocking_control_step: Optional[int] = None
        self._use_cached_once = False
        self._latest_control_step = -1
        self._latest_task = ""
        self._latest_images: Dict[str, np.ndarray] = {}
        self._last_request_control_step = -1
        self._request_generation = 0

    def reset(self) -> None:
        with self._lock:
            self._step = 0
            self._initialized = False
            self._cache.clear()
            self._inflight = None
            self._last_blocking_control_step = None
            self._use_cached_once = False
            self._latest_control_step = -1
            self._latest_task = ""
            self._latest_images = {}
            self._last_request_control_step = -1
            self._request_generation += 1

    def set_blocking(self, blocking: bool) -> None:
        self._blocking = bool(blocking)

    def set_step_interval(self, n: int) -> None:
        self._update_every = max(1, int(n))

    def _should_generate_subgoal(self, loop_step: int) -> bool:
        if loop_step == 0:
            return True
        if loop_step == self._update_every:
            return False
        return loop_step > self._update_every and loop_step % self._update_every == 0

    def before_control_step(
        self,
        raw_obs,
        runtime,
        loop_step: int,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> bool:
        raw_images = raw_obs.get("images") or {}
        latest_images = {
            cam: raw_images[cam].copy()
            for cam in self._cameras
            if cam in raw_images
        }
        if not latest_images:
            return False

        with runtime._lock:
            task = runtime.task

        if not self._should_generate_subgoal(loop_step):
            return False

        with self._lock:
            if self._last_blocking_control_step == loop_step:
                return False

        if cancel_check is not None and cancel_check():
            return False

        logger.info("Subgoal mode synchronously requesting ForeAct subgoal at step %d", loop_step)
        self._do_request(raw_obs, task, runtime)
        with self._lock:
            self._last_blocking_control_step = loop_step
            self._use_cached_once = True
        return True

    # ------------------------------------------------------------------ #
    def _predict_subgoal(self, cam: str, img: np.ndarray, task: str) -> Optional[np.ndarray]:
        try:
            return self._client.predict_subgoal(img, task)
        except Exception as e:  # noqa: BLE001
            logger.warning("ForeAct request failed for %s; continuing without fresh subgoal: %s", cam, e)
            if hasattr(self._client, "close"):
                try:
                    self._client.close()
                except Exception:  # noqa: BLE001
                    pass
            return None

    def _store_subgoal(self, cam: str, sg: np.ndarray, runtime: RuntimeState) -> None:
        with self._lock:
            self._cache[cam] = sg
            self._initialized = True
        if cam == "cam_high":
            with runtime._lock:
                runtime.last_subgoal_image = sg

    def _do_request(self, raw_obs: Dict[str, Any], task: str, runtime: RuntimeState) -> bool:
        """Synchronous ForeAct call; updates cache + runtime display."""
        updated = False
        for cam in self._cameras:
            img = raw_obs["images"].get(cam)
            if img is None:
                continue
            sg = self._predict_subgoal(cam, img, task)
            if sg is None:
                continue
            self._store_subgoal(cam, sg, runtime)
            updated = True
        if not updated:
            with self._lock:
                self._initialized = True
        return updated

    def attach_cached_subgoal_images(self, obs: Dict[str, Any]) -> None:
        with self._lock:
            cache_snapshot = dict(self._cache)
        if cache_snapshot:
            obs["subgoal_images"] = {
                cam: to_chw_uint8(img) for cam, img in cache_snapshot.items()
            }

    def _spawn_request(
        self,
        raw_obs: Dict[str, Any],
        task: str,
        runtime: RuntimeState,
        *,
        request_step: int,
    ) -> bool:
        snapshots = {
            cam: raw_obs["images"][cam].copy()
            for cam in self._cameras
            if cam in raw_obs["images"]
        }
        if not snapshots:
            return False

        generation = 0
        def _run() -> None:
            try:
                try:
                    updated = False
                    for cam, img in snapshots.items():
                        with self._lock:
                            if generation != self._request_generation:
                                return
                        sg = self._predict_subgoal(cam, img, task)
                        if sg is None:
                            continue
                        with self._lock:
                            if generation != self._request_generation:
                                return
                        self._store_subgoal(cam, sg, runtime)
                        updated = True
                    if not updated:
                        with self._lock:
                            self._initialized = True
                except Exception:  # noqa: BLE001
                    logger.exception("ForeAct async request worker crashed")
            finally:
                next_request = None
                with self._lock:
                    if self._inflight is threading.current_thread():
                        self._inflight = None
                    if (
                        not self._blocking
                        and self._latest_control_step != 0
                        and self._should_generate_subgoal(self._latest_control_step)
                        and self._latest_images
                        and generation == self._request_generation
                        and self._latest_control_step > self._last_request_control_step
                    ):
                        next_request = (
                            {"images": {cam: img.copy() for cam, img in self._latest_images.items()}},
                            self._latest_task,
                            self._latest_control_step,
                        )
                if next_request is not None:
                    next_obs, next_task, next_step = next_request
                    self._spawn_request(next_obs, next_task, runtime, request_step=next_step)

        t = threading.Thread(target=_run, daemon=True, name="foreact-predict")
        with self._lock:
            if self._inflight is not None and not self._inflight.is_alive():
                logger.warning("Recovering stale ForeAct async request state")
                self._inflight = None
            if self._inflight is not None:
                return False
            if request_step <= self._last_request_control_step:
                return False
            generation = self._request_generation
            self._last_request_control_step = int(request_step)
            self._inflight = t
        try:
            t.start()
        except Exception:
            with self._lock:
                if self._inflight is t:
                    self._inflight = None
                if self._last_request_control_step == int(request_step):
                    self._last_request_control_step = int(request_step) - 1
            raise
        return True

    # ------------------------------------------------------------------ #
    def build_obs(self, raw_obs, runtime):
        obs = self._base_obs(raw_obs)
        obs["prompt"] = runtime.task
        with runtime._lock:
            runtime.last_prompt = runtime.task

        with self._lock:
            first_step = not self._initialized
            self._step += 1
            cache_snapshot = dict(self._cache)
            use_cached_once = self._use_cached_once
            self._use_cached_once = False

        if self._blocking:
            if not use_cached_once and not cache_snapshot:
                # Fallback for direct callers that bypass before_control_step().
                self._do_request(raw_obs, runtime.task, runtime)
                with self._lock:
                    cache_snapshot = dict(self._cache)
            elif use_cached_once:
                # The fresh subgoal was already fetched synchronously in
                # before_control_step(); just consume the cached result.
                pass
        elif first_step and not cache_snapshot:
            # Keep the initial blocking request behavior for step 0.
            self._do_request(raw_obs, runtime.task, runtime)
            with self._lock:
                cache_snapshot = dict(self._cache)

        self.attach_cached_subgoal_images(obs)
        return obs


# ---------------------------------------------------------------------------
class _ManualOnlyTrajectoryPredictor:
    """Placeholder predictor used when trajectory CoT comes from the GUI."""

    def predict(self, image: np.ndarray, task_description: str) -> None:
        return None


# ---------------------------------------------------------------------------
def make_handler(
    mode: str,
    *,
    foreact_host: str = "10.1.119.68",
    foreact_port: int = 5100,
    traj_step_interval: int = 6,
    subgoal_step_interval: int = 30,
    subtask_step_interval: int = 30,
    subgoal_cameras: Optional[List[str]] = None,
    blocking: bool = False,
    doubao_api_key: Optional[str] = None,
    foreact_client: Optional[Any] = None,
    doubao_predictor: Optional[Any] = None,
    subtask_predictor: Optional[Any] = None,
    manual_traj_provider: Optional[Callable[..., Any]] = None,
    manual_traj_override: bool = False,
) -> ModeHandler:
    """Factory used by ``EvalRunner`` and CLI wrappers."""
    mode = mode.lower().replace("-", "_")
    if mode == "basic":
        return BasicMode()
    if mode == "subtask":
        return SubtaskMode(
            predictor=subtask_predictor,
            blocking=blocking,
            step_interval=subtask_step_interval,
        )
    if mode == "traj":
        from .doubao_predictor import CachedTrajectoryPredictor, DoubaoTrajectoryPredictor

        if doubao_predictor is None:
            if manual_traj_provider is not None:
                logger.info("Starting trajectory mode with manual annotation only.")
                manual_traj_override = True
                predictor_impl = _ManualOnlyTrajectoryPredictor()
            else:
                try:
                    predictor_impl = DoubaoTrajectoryPredictor(api_key=doubao_api_key)
                except Exception:
                    if not (manual_traj_override and manual_traj_provider is not None):
                        raise
                    logger.info(
                        "Doubao predictor is unavailable; starting trajectory mode with manual override only."
                    )
                    predictor_impl = _ManualOnlyTrajectoryPredictor()
            doubao_predictor = CachedTrajectoryPredictor(
                predictor=predictor_impl,
                update_every=traj_step_interval,
                blocking=blocking,
            )
        else:
            doubao_predictor.set_blocking(blocking)
            doubao_predictor.set_update_every(traj_step_interval)
        return TrajectoryMode(
            predictor=doubao_predictor,
            manual_provider=manual_traj_provider,
            manual_override=manual_traj_override,
        )
    if mode == "triple_cot":
        from .doubao_predictor import CachedTrajectoryPredictor
        from .foreact_client import ForeactClient

        if doubao_predictor is None:
            if manual_traj_provider is None:
                raise RuntimeError(
                    "Triple-CoT requires a manual trajectory provider; Doubao is not used by default."
                )
            logger.info("Starting Triple-CoT mode with manual trajectory annotation only.")
            manual_traj_override = True
            predictor_impl = _ManualOnlyTrajectoryPredictor()
            doubao_predictor = CachedTrajectoryPredictor(
                predictor=predictor_impl,
                update_every=traj_step_interval,
                blocking=True,
            )
        else:
            doubao_predictor.set_blocking(True)
            doubao_predictor.set_update_every(traj_step_interval)
        foreact_timeout_s = 3.0
        if foreact_client is None:
            foreact_client = ForeactClient(
                host=foreact_host,
                port=foreact_port,
                connect_timeout=foreact_timeout_s,
                request_timeout=foreact_timeout_s,
            )
        subgoal_handler = SubgoalMode(
            client=foreact_client,
            update_every=traj_step_interval,
            cameras=subgoal_cameras,
            blocking=True,
        )
        return TripleCotMode(
            predictor=doubao_predictor,
            manual_provider=manual_traj_provider,
            manual_override=manual_traj_override,
            subgoal_handler=subgoal_handler,
            foreact_timeout_s=foreact_timeout_s,
        )
    if mode == "subgoal":
        from .foreact_client import ForeactClient

        if foreact_client is None:
            foreact_client = ForeactClient(host=foreact_host, port=foreact_port)
        return SubgoalMode(
            client=foreact_client,
            update_every=subgoal_step_interval,
            cameras=subgoal_cameras,
            blocking=blocking,
        )
    raise ValueError(f"Unknown mode: {mode!r}")
