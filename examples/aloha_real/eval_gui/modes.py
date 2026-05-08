"""Mode handlers that translate a raw env observation + GUI runtime state
into the ``obs`` dict consumed by the openpi WebSocket policy server.

Each ``ModeHandler`` returns a dict that the server feeds into

    [data_transforms.inputs] -> Normalize -> [model_transforms.inputs]

i.e. **after** the dataset's ``repack_transforms`` would normally have run.
That means the handler must do the work of those repack transforms itself:

  * basic   -- raw {state, images}; server uses ``--default-prompt``.
  * traj    -- prompt = ``"<task>, traj: Left: Go along ... Right: Go along
               <br/>  ..."`` (mirrors ``_transforms.AppendTrajCotToPrompt``
               + the canonical ``cot_text_prompts.json`` format).
  * subtask -- prompt = ``"<task>, subtask: <label>"`` (mirrors
               ``_transforms.AppendSubtaskToPrompt`` exactly). Labels are
               user-editable and the keys 1..4 select among them.
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

import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import einops
import numpy as np
from openpi_client import image_tools

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default subtask label mapping for the banana two-tasks dataset.
# These keys (1-4) match the labels written by ``scripts/preprocess_subtask.py``.
# At runtime the user can edit them in the GUI or load a JSON override.
# ---------------------------------------------------------------------------
DEFAULT_SUBTASK_LABELS: Dict[int, str] = {
    1: "reach the banana",
    2: "grasp the banana",
    3: "move the banana to the green plate",
    4: "place the banana in the green plate",
}


def to_chw_uint8(img_hwc_rgb: np.ndarray, h: int = 224, w: int = 224) -> np.ndarray:
    """HxWx3 uint8 RGB -> (3, h, w) uint8 with the same resize/pad as training."""
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img_hwc_rgb, h, w))
    return einops.rearrange(img, "h w c -> c h w")


# ---------------------------------------------------------------------------
@dataclass
class RuntimeState:
    """Mutable state shared between the GUI thread and the eval runner thread."""

    mode: str = "basic"                   # one of: basic | traj | subtask | subgoal
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
    """Mode 2: prompt = f"{task}, traj: <strict L/R loc-token text>"."""

    name = "traj"

    def __init__(self, predictor) -> None:
        from .doubao_predictor import CachedTrajectoryPredictor  # local import

        self._predictor: CachedTrajectoryPredictor = predictor

    def reset(self) -> None:
        self._predictor.reset()

    def set_blocking(self, blocking: bool) -> None:
        self._predictor.set_blocking(blocking)

    def set_step_interval(self, n: int) -> None:
        self._predictor.set_update_every(n)

    def build_obs(self, raw_obs, runtime):
        obs = self._base_obs(raw_obs)

        cam_high = raw_obs["images"]["cam_high"]
        pred = self._predictor.step(cam_high, runtime.task)

        if pred is None or pred.is_empty():
            traj_text = ""
            prompt = runtime.task
        else:
            traj_text = pred.to_loc_token_text()
            prompt = f"{runtime.task}, traj: {traj_text}"

        with runtime._lock:
            runtime.last_traj_text = traj_text
            runtime.last_prompt = prompt
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

    def reset(self) -> None:
        with self._lock:
            self._step = 0
            self._initialized = False
            self._cache.clear()
            self._inflight = None

    def set_blocking(self, blocking: bool) -> None:
        self._blocking = bool(blocking)

    def set_step_interval(self, n: int) -> None:
        self._update_every = max(1, int(n))

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

    def _do_request(self, raw_obs: Dict[str, Any], task: str, runtime: RuntimeState) -> None:
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

    def _spawn_request(self, raw_obs: Dict[str, Any], task: str, runtime: RuntimeState) -> None:
        snapshots = {cam: raw_obs["images"][cam].copy() for cam in self._cameras
                     if cam in raw_obs["images"]}

        def _run() -> None:
            updated = False
            for cam, img in snapshots.items():
                sg = self._predict_subgoal(cam, img, task)
                if sg is None:
                    continue
                self._store_subgoal(cam, sg, runtime)
                updated = True
            if not updated:
                with self._lock:
                    self._initialized = True

        t = threading.Thread(target=_run, daemon=True, name="foreact-predict")
        t.start()
        with self._lock:
            self._inflight = t

    # ------------------------------------------------------------------ #
    def build_obs(self, raw_obs, runtime):
        obs = self._base_obs(raw_obs)
        obs["prompt"] = runtime.task
        with runtime._lock:
            runtime.last_prompt = runtime.task

        with self._lock:
            first_step = not self._initialized
            should_request = first_step or (self._step % self._update_every == 0)
            self._step += 1
            blocking = first_step or self._blocking   # force-block on step 0
            in_flight = self._inflight is not None and self._inflight.is_alive()
            cache_snapshot = dict(self._cache)

        if should_request:
            if blocking:
                # Synchronous fetch: callers pause here until the new
                # subgoal arrives (or the request fails / times out).
                # On first step this guarantees the robot never moves
                # without a valid subgoal image.
                self._do_request(raw_obs, runtime.task, runtime)
                with self._lock:
                    cache_snapshot = dict(self._cache)
            elif not in_flight:
                self._spawn_request(raw_obs, runtime.task, runtime)

        if cache_snapshot:
            obs["subgoal_images"] = {
                cam: to_chw_uint8(img) for cam, img in cache_snapshot.items()
            }
        return obs


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
) -> ModeHandler:
    """Factory used by ``EvalRunner`` and CLI wrappers."""
    mode = mode.lower()
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
            doubao_predictor = CachedTrajectoryPredictor(
                predictor=DoubaoTrajectoryPredictor(api_key=doubao_api_key),
                update_every=traj_step_interval,
                blocking=blocking,
            )
        else:
            doubao_predictor.set_blocking(blocking)
            doubao_predictor.set_update_every(traj_step_interval)
        return TrajectoryMode(predictor=doubao_predictor)
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
