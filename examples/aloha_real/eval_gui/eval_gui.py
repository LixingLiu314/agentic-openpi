"""PyQt5 GUI for the Aloha real-robot evaluation pipeline.

Layout (left = controls; right = camera views)
    [Mode v]   [help text]
    Task: [.................................]
    Policy host:[.....] port:[..]  [Connect]
    ForeAct:[.....] port:[..]  chunk:[..]  max_steps:[..]
    Sub-mode: ( ) blocking  ( ) non-blocking      step interval: [..]
    [Start] [Pause] [Stop]   [回零]   [Dump inputs]

    Subtask labels (editable):
       1: [reach_the_banana_end                  ]
       2: [grasp_the_banana_end                  ]
       3: [move_the_banana_to_the_green_plate_end]
       4: [place_the_banana_in_the_green_plate_end]
       [Add] [Remove] [Load JSON ...]    Doubao auto-suggest: [ ]

    Runtime info: prompt / subtask / traj / step
    Log pane

Keyboard:
    1-9           -> subtask key (only mode 3; use row buttons for 10+)
    Space         -> pause / resume
    H             -> 回零
    D             -> dump model inputs
    Q             -> quit
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import re
import shutil
import socket
import subprocess
import sys
import time
from typing import Optional

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from . import modes as _modes
from .eval_runner import EvalRunner, RunnerConfig

logger = logging.getLogger(__name__)


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

_MODE_DESCRIPTIONS = {
    "basic":   "Basic — VLA inference with task prompt only.",
    "traj":    "Add Trajectory — Doubao predicts L/R waypoints (loc-tokens).",
    "subtask": "Add Subtasks — keys 1/2/3/4 override the subtask label.",
    "subgoal": "Add Subgoal Images — ForeAct generates cam_high subgoal.",
}

_POLICY_PRESETS = {
    "basic": {
        "label": "basic - banana baseline",
        "config": "pi05_aloha_banana",
        "dir": "checkpoints/pi05_aloha_banana/banana_baseline/{step}",
    },
    "traj": {
        "label": "traj - Doubao trajectory",
        "config": "pi05_aloha_banana_traj",
        "dir": "checkpoints/pi05_aloha_banana_traj/banana_traj_lr5e5/{step}",
    },
    "subtask": {
        "label": "subtask - segment labels",
        "config": "pi05_aloha_banana_subtask_segment",
        "dir": "checkpoints/pi05_aloha_banana_subtask_segment/banana_subtask_seg_lr5e5/{step}",
    },
    "subgoal": {
        "label": "subgoal - base camera",
        "config": "pi05_aloha_banana_subgoal_base",
        "dir": "checkpoints/pi05_aloha_banana_subgoal_base/banana_subgoal_base_lr5e5/{step}",
    },
}

_CHECKPOINT_HISTORY_LIMIT = 30
_CHECKPOINT_HISTORY_ENV = "AGENTIC_OPENPI_EVAL_GUI_HISTORY"


def _np_to_qpixmap(img_hwc_rgb: Optional[np.ndarray], target_w: int, target_h: int) -> QtGui.QPixmap:
    if img_hwc_rgb is None:
        pm = QtGui.QPixmap(target_w, target_h)
        pm.fill(QtGui.QColor("#222"))
        return pm
    if img_hwc_rgb.dtype != np.uint8:
        img_hwc_rgb = np.clip(img_hwc_rgb, 0, 255).astype(np.uint8)
    h, w, _ = img_hwc_rgb.shape
    qimg = QtGui.QImage(img_hwc_rgb.tobytes(), w, h, 3 * w, QtGui.QImage.Format_RGB888)
    pm = QtGui.QPixmap.fromImage(qimg)
    return pm.scaled(target_w, target_h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)


class _EventBridge(QtCore.QObject):
    """Marshals events from the runner thread onto the Qt main thread."""

    info = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)
    step = QtCore.pyqtSignal(int)
    gripper = QtCore.pyqtSignal(float, float)
    mode_changed = QtCore.pyqtSignal(str)
    episode_start = QtCore.pyqtSignal()
    episode_end = QtCore.pyqtSignal(int)


# ---------------------------------------------------------------------------
class EvalGUI(QtWidgets.QMainWindow):
    def __init__(self, cfg: RunnerConfig, runtime: _modes.RuntimeState, parent=None) -> None:
        super().__init__(parent)
        self._cfg = cfg
        self._runtime = runtime
        self._runner: Optional[EvalRunner] = None
        self._handler: Optional[_modes.ModeHandler] = None
        self._policy_proc: Optional[subprocess.Popen] = None
        self._policy_spec: Optional[tuple[str, str, int]] = None
        self._blocking = False
        self._step_interval = 6  # default for traj mode
        self._checkpoint_history_path = self._checkpoint_history_file()
        self._checkpoint_history: dict[str, list[str]] = {key: [] for key in _POLICY_PRESETS}
        self._last_checkpoint_by_mode: dict[str, str] = {}
        self._current_checkpoint_default = ""
        self._load_checkpoint_history()

        self._bridge = _EventBridge()
        self._bridge.info.connect(self._log_info)
        self._bridge.error.connect(self._log_error)
        self._bridge.step.connect(self._on_step)
        self._bridge.gripper.connect(self._on_gripper)
        self._bridge.mode_changed.connect(lambda m: self._log_info(f"Mode -> {m}"))
        self._bridge.episode_start.connect(lambda: self._log_info("Episode started"))
        self._bridge.episode_end.connect(lambda n: self._log_info(f"Episode ended after {n} steps"))

        self.setWindowTitle("Aloha Eval — agentic-openpi")
        self._build_ui()

        self._ui_timer = QtCore.QTimer(self)
        self._ui_timer.setInterval(33)         # ~30 Hz refresh
        self._ui_timer.timeout.connect(self._refresh_ui)
        self._ui_timer.start()

    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        # ---------------------- LEFT (controls) -------------------- #
        left = QtWidgets.QVBoxLayout()
        root.addLayout(left, 3)

        # Mode + help
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("Mode:"))
        self.cb_mode = QtWidgets.QComboBox()
        for m in ("basic", "traj", "subtask", "subgoal"):
            self.cb_mode.addItem(m)
        self.cb_mode.currentTextChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.cb_mode)
        self.lbl_mode_help = QtWidgets.QLabel(_MODE_DESCRIPTIONS[self._runtime.mode])
        self.lbl_mode_help.setStyleSheet("color:#888;")
        mode_row.addWidget(self.lbl_mode_help, 1)
        left.addLayout(mode_row)

        # Task
        task_row = QtWidgets.QHBoxLayout()
        task_row.addWidget(QtWidgets.QLabel("Task:"))
        self.le_task = QtWidgets.QLineEdit(self._runtime.task)
        self.le_task.editingFinished.connect(self._on_task_edited)
        task_row.addWidget(self.le_task, 1)
        left.addLayout(task_row)

        # Policy checkpoint selection. The selected eval mode owns the
        # policy config; checkpoint paths are free-form and remembered.
        policy_row = QtWidgets.QHBoxLayout()
        policy_row.addWidget(QtWidgets.QLabel("policy.config:"))
        self.le_policy_config = QtWidgets.QLineEdit()
        self.le_policy_config.setReadOnly(True)
        policy_row.addWidget(self.le_policy_config, 1)
        policy_row.addWidget(QtWidgets.QLabel("default step:"))
        self.sp_ckpt_step = QtWidgets.QSpinBox()
        self.sp_ckpt_step.setRange(1, 10_000_000)
        self.sp_ckpt_step.setSingleStep(500)
        self.sp_ckpt_step.setValue(5000)
        self.sp_ckpt_step.valueChanged.connect(self._on_policy_step_changed)
        policy_row.addWidget(self.sp_ckpt_step)
        self.cb_local_policy = QtWidgets.QCheckBox("start local server")
        self.cb_local_policy.setChecked(True)
        policy_row.addWidget(self.cb_local_policy)
        left.addLayout(policy_row)

        checkpoint_row = QtWidgets.QHBoxLayout()
        checkpoint_row.addWidget(QtWidgets.QLabel("checkpoint:"))
        self.cb_checkpoint = QtWidgets.QComboBox()
        self.cb_checkpoint.setEditable(True)
        self.cb_checkpoint.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.cb_checkpoint.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.cb_checkpoint.lineEdit().editingFinished.connect(self._on_checkpoint_edited)
        checkpoint_row.addWidget(self.cb_checkpoint, 1)
        self.btn_browse_checkpoint = QtWidgets.QPushButton("Browse...")
        self.btn_browse_checkpoint.clicked.connect(self._on_browse_checkpoint)
        checkpoint_row.addWidget(self.btn_browse_checkpoint)
        left.addLayout(checkpoint_row)

        # Server connection
        srv_row = QtWidgets.QHBoxLayout()
        srv_row.addWidget(QtWidgets.QLabel("Policy host:"))
        self.le_host = QtWidgets.QLineEdit(self._cfg.host)
        srv_row.addWidget(self.le_host)
        srv_row.addWidget(QtWidgets.QLabel("port:"))
        self.le_port = QtWidgets.QLineEdit(str(self._cfg.port))
        self.le_port.setFixedWidth(60)
        srv_row.addWidget(self.le_port)
        self.btn_connect = QtWidgets.QPushButton("Connect")
        self.btn_connect.clicked.connect(self._on_connect)
        srv_row.addWidget(self.btn_connect)
        left.addLayout(srv_row)

        # ForeAct + chunk + max_steps
        cfg_row = QtWidgets.QHBoxLayout()
        cfg_row.addWidget(QtWidgets.QLabel("ForeAct:"))
        self.le_fhost = QtWidgets.QLineEdit("10.1.119.68")
        self.le_fhost.setFixedWidth(120)
        cfg_row.addWidget(self.le_fhost)
        self.le_fport = QtWidgets.QLineEdit("5100")
        self.le_fport.setFixedWidth(60)
        cfg_row.addWidget(self.le_fport)
        cfg_row.addWidget(QtWidgets.QLabel("chunk:"))
        self.le_chunk = QtWidgets.QLineEdit(str(self._cfg.action_horizon))
        self.le_chunk.setFixedWidth(40)
        cfg_row.addWidget(self.le_chunk)
        cfg_row.addWidget(QtWidgets.QLabel("max_steps:"))
        self.le_max = QtWidgets.QLineEdit(str(self._cfg.max_steps))
        self.le_max.setFixedWidth(60)
        cfg_row.addWidget(self.le_max)
        left.addLayout(cfg_row)

        # Sub-mode (blocking / non-blocking) + step interval
        sm_row = QtWidgets.QHBoxLayout()
        sm_row.addWidget(QtWidgets.QLabel("Execution:"))
        self.bg_submode = QtWidgets.QButtonGroup(self)
        self.rb_nonblock = QtWidgets.QRadioButton("non-blocking (async)")
        self.rb_nonblock.setChecked(True)
        self.rb_block = QtWidgets.QRadioButton("blocking (every N steps)")
        self.bg_submode.addButton(self.rb_nonblock)
        self.bg_submode.addButton(self.rb_block)
        self.rb_nonblock.toggled.connect(self._on_submode_changed)
        sm_row.addWidget(self.rb_nonblock)
        sm_row.addWidget(self.rb_block)
        sm_row.addStretch(1)
        sm_row.addWidget(QtWidgets.QLabel("step interval N:"))
        self.sp_interval = QtWidgets.QSpinBox()
        self.sp_interval.setRange(1, 1000)
        self.sp_interval.setValue(self._step_interval)
        self.sp_interval.valueChanged.connect(self._on_interval_changed)
        sm_row.addWidget(self.sp_interval)
        left.addLayout(sm_row)

        # Run buttons
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("▶ Start")
        self.btn_start.clicked.connect(self._on_start)
        btn_row.addWidget(self.btn_start)
        self.btn_pause = QtWidgets.QPushButton("❙❙ Pause")
        self.btn_pause.clicked.connect(self._on_pause_toggle)
        btn_row.addWidget(self.btn_pause)
        self.btn_stop = QtWidgets.QPushButton("■ Stop")
        self.btn_stop.clicked.connect(self._on_stop)
        btn_row.addWidget(self.btn_stop)
        self.btn_rtz = QtWidgets.QPushButton("⟲ 回零")
        self.btn_rtz.setStyleSheet("background:#c0392b;color:#fff;font-weight:bold;padding:6px;")
        self.btn_rtz.clicked.connect(self._on_rtz)
        btn_row.addWidget(self.btn_rtz)
        self.btn_dump = QtWidgets.QPushButton("⬇ Dump inputs")
        self.btn_dump.setToolTip("Save the next inference's images + prompt to ./debug_inputs/")
        self.btn_dump.clicked.connect(self._on_dump)
        btn_row.addWidget(self.btn_dump)
        left.addLayout(btn_row)

        # Subtask label editor (Mode 3)
        self.gb_subtask = QtWidgets.QGroupBox("Subtask labels (editable)")
        sub_layout = QtWidgets.QVBoxLayout(self.gb_subtask)
        self.subtask_inputs: dict[int, QtWidgets.QLineEdit] = {}

        self.subtask_scroll = QtWidgets.QScrollArea()
        self.subtask_scroll.setWidgetResizable(True)
        self.subtask_scroll.setMinimumHeight(112)
        self.subtask_scroll.setMaximumHeight(168)
        self.subtask_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.subtask_scroll_widget = QtWidgets.QWidget()
        self.subtask_form_layout = QtWidgets.QFormLayout(self.subtask_scroll_widget)
        self.subtask_form_layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.subtask_form_layout.setContentsMargins(6, 6, 6, 6)
        self.subtask_scroll.setWidget(self.subtask_scroll_widget)
        sub_layout.addWidget(self.subtask_scroll)

        # Active key indicator + add / remove / load
        sub_btn_row = QtWidgets.QHBoxLayout()
        self.lbl_active_key = QtWidgets.QLabel(f"active: {self._runtime.subtask_key}")
        sub_btn_row.addWidget(self.lbl_active_key)
        sub_btn_row.addStretch(1)
        self.btn_add_subtask = QtWidgets.QPushButton("Add")
        self.btn_add_subtask.clicked.connect(self._on_add_subtask)
        sub_btn_row.addWidget(self.btn_add_subtask)
        self.btn_remove_subtask = QtWidgets.QPushButton("Remove")
        self.btn_remove_subtask.setToolTip("Remove the highest-numbered subtask label")
        self.btn_remove_subtask.clicked.connect(self._on_remove_subtask)
        sub_btn_row.addWidget(self.btn_remove_subtask)
        self.btn_load_json = QtWidgets.QPushButton("Load JSON…")
        self.btn_load_json.clicked.connect(self._on_load_subtask_json)
        sub_btn_row.addWidget(self.btn_load_json)
        self.cb_doubao_auto = QtWidgets.QCheckBox("Doubao auto-suggest (placeholder)")
        self.cb_doubao_auto.setEnabled(False)  # hook reserved; actual VLM is a TODO
        self.cb_doubao_auto.setToolTip(
            "Reserved hook: enable to let a Doubao SubtaskPredictor pick the\n"
            "subtask key automatically. The current implementation is a no-op\n"
            "placeholder — see doubao_predictor.SubtaskPredictor."
        )
        sub_btn_row.addWidget(self.cb_doubao_auto)
        sub_layout.addLayout(sub_btn_row)
        left.addWidget(self.gb_subtask)
        self._refresh_subtask_inputs()

        # Live status
        self.gb_status = QtWidgets.QGroupBox("Runtime info")
        st = QtWidgets.QFormLayout(self.gb_status)
        self.lbl_prompt = QtWidgets.QLabel("(idle)")
        self.lbl_prompt.setWordWrap(True)
        self.lbl_subtask = QtWidgets.QLabel("-")
        self.lbl_traj = QtWidgets.QLabel("-")
        self.lbl_traj.setWordWrap(True)
        self.lbl_gripper = QtWidgets.QLabel("L: -   R: -")
        self.lbl_step = QtWidgets.QLabel("0")
        st.addRow("Prompt:", self.lbl_prompt)
        st.addRow("Subtask:", self.lbl_subtask)
        st.addRow("Trajectory:", self.lbl_traj)
        st.addRow("Gripper:", self.lbl_gripper)
        st.addRow("Step:", self.lbl_step)
        left.addWidget(self.gb_status)

        # Log pane
        self.txt_log = QtWidgets.QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumBlockCount(2000)
        self.txt_log.setStyleSheet("font-family:monospace;font-size:11px;")
        left.addWidget(self.txt_log, 1)

        # ---------------------- RIGHT (cameras) -------------------- #
        right = QtWidgets.QVBoxLayout()
        root.addLayout(right, 2)

        right.addWidget(QtWidgets.QLabel("cam_high (live)"))
        self.lbl_cam = QtWidgets.QLabel()
        self.lbl_cam.setMinimumSize(480, 360)
        self.lbl_cam.setStyleSheet("background:#111;")
        self.lbl_cam.setAlignment(QtCore.Qt.AlignCenter)
        right.addWidget(self.lbl_cam)

        right.addWidget(QtWidgets.QLabel("Subgoal image (Mode 4)"))
        self.lbl_subgoal = QtWidgets.QLabel()
        self.lbl_subgoal.setMinimumSize(480, 360)
        self.lbl_subgoal.setStyleSheet("background:#111;")
        self.lbl_subgoal.setAlignment(QtCore.Qt.AlignCenter)
        right.addWidget(self.lbl_subgoal)

        right.addWidget(QtWidgets.QLabel("cam_left_wrist / cam_right_wrist"))
        wrist_row = QtWidgets.QHBoxLayout()
        self.lbl_left = QtWidgets.QLabel()
        self.lbl_left.setMinimumSize(220, 165)
        self.lbl_left.setStyleSheet("background:#111;")
        self.lbl_right = QtWidgets.QLabel()
        self.lbl_right.setMinimumSize(220, 165)
        self.lbl_right.setStyleSheet("background:#111;")
        wrist_row.addWidget(self.lbl_left)
        wrist_row.addWidget(self.lbl_right)
        right.addLayout(wrist_row)
        right.addStretch(1)

        old = self.cb_mode.blockSignals(True)
        self.cb_mode.setCurrentText(self._runtime.mode)
        self.cb_mode.blockSignals(old)
        self._refresh_policy_fields_for_mode(self._runtime.mode, prefer_history=True)

    # ------------------------------------------------------------------ #
    # Logging
    def _log(self, level: str, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.txt_log.appendPlainText(f"[{ts}] [{level}] {msg}")

    @QtCore.pyqtSlot(str)
    def _log_info(self, msg: str) -> None:
        self._log("info", msg)

    @QtCore.pyqtSlot(str)
    def _log_error(self, msg: str) -> None:
        self._log("ERR ", msg)

    # ------------------------------------------------------------------ #
    def _on_event(self, kind: str, payload: dict) -> None:
        if kind == "info":
            self._bridge.info.emit(payload.get("msg", ""))
        elif kind == "error":
            self._bridge.error.emit(payload.get("msg", ""))
        elif kind == "step":
            self._bridge.step.emit(payload.get("step", 0))
        elif kind == "gripper":
            self._bridge.gripper.emit(
                float(payload.get("left", 0.0)),
                float(payload.get("right", 0.0)),
            )
        elif kind == "mode_changed":
            self._bridge.mode_changed.emit(payload.get("mode", ""))
        elif kind == "episode_start":
            self._bridge.episode_start.emit()
        elif kind == "episode_end":
            self._bridge.episode_end.emit(payload.get("steps", 0))

    @QtCore.pyqtSlot(int)
    def _on_step(self, n: int) -> None:
        self.lbl_step.setText(str(n))

    @QtCore.pyqtSlot(float, float)
    def _on_gripper(self, left: float, right: float) -> None:
        self.lbl_gripper.setText(f"L: {left:.4f}   R: {right:.4f}")

    # ------------------------------------------------------------------ #
    @staticmethod
    def _checkpoint_history_file() -> pathlib.Path:
        override = os.environ.get(_CHECKPOINT_HISTORY_ENV)
        if override:
            return pathlib.Path(override).expanduser()
        return pathlib.Path.home() / ".cache" / "agentic-openpi" / "eval_gui_checkpoints.json"

    def _load_checkpoint_history(self) -> None:
        path = self._checkpoint_history_path
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
        except Exception as e:                       # noqa: BLE001
            logger.warning("Could not load checkpoint history from %s: %s", path, e)
            return

        raw_history = data.get("history", data) if isinstance(data, dict) else {}
        if isinstance(raw_history, dict):
            for mode in _POLICY_PRESETS:
                values = raw_history.get(mode, [])
                if isinstance(values, str):
                    values = [values]
                if not isinstance(values, list):
                    continue
                cleaned = []
                for value in values:
                    text = str(value).strip()
                    if text and text not in cleaned:
                        cleaned.append(text)
                self._checkpoint_history[mode] = cleaned[:_CHECKPOINT_HISTORY_LIMIT]

        raw_last = data.get("last", {}) if isinstance(data, dict) else {}
        if isinstance(raw_last, dict):
            self._last_checkpoint_by_mode = {
                mode: str(value).strip()
                for mode, value in raw_last.items()
                if mode in _POLICY_PRESETS and str(value).strip()
            }

    def _save_checkpoint_history(self) -> None:
        payload = {
            "history": self._checkpoint_history,
            "last": self._last_checkpoint_by_mode,
        }
        try:
            self._checkpoint_history_path.parent.mkdir(parents=True, exist_ok=True)
            self._checkpoint_history_path.write_text(json.dumps(payload, indent=2))
        except Exception as e:                       # noqa: BLE001
            logger.warning("Could not save checkpoint history to %s: %s", self._checkpoint_history_path, e)

    def _checkpoint_text(self) -> str:
        return self.cb_checkpoint.currentText().strip()

    @staticmethod
    def _sanitize_video_stem(text: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
        cleaned = cleaned.strip("._-")
        return (cleaned or "checkpoint")[:120]

    def _checkpoint_video_stem(self) -> str:
        checkpoint = self._checkpoint_text().rstrip("/\\")
        if not checkpoint:
            checkpoint = self._default_checkpoint_for_mode(self._runtime.mode)
        parts = [p for p in checkpoint.replace("\\", "/").split("/") if p and p != "gs:"]
        name = parts[-1] if parts else "checkpoint"
        parent = parts[-2] if len(parts) >= 2 else ""
        if name.isdigit() and parent:
            name = f"{parent}_{name}"
        return self._sanitize_video_stem(name)

    def _configure_video_recording(self) -> None:
        self._cfg.record_video = True
        self._cfg.video_dir = str(_REPO_ROOT / "test_video")
        self._cfg.video_name = self._checkpoint_video_stem()
        self._cfg.video_fps = max(1.0, float(self._cfg.max_hz))

    def _default_checkpoint_for_mode(self, mode: str) -> str:
        preset = _POLICY_PRESETS.get(mode, _POLICY_PRESETS["basic"])
        return preset["dir"].format(step=int(self.sp_ckpt_step.value()))

    def _checkpoint_items_for_mode(self, mode: str) -> list[str]:
        default = self._default_checkpoint_for_mode(mode)
        items = []
        candidates = [
            self._last_checkpoint_by_mode.get(mode, ""),
            default,
            *self._checkpoint_history.get(mode, []),
        ]
        for value in candidates:
            value = value.strip()
            if value and value not in items:
                items.append(value)
        return items

    def _refresh_policy_fields_for_mode(
        self,
        mode: str,
        *,
        select: Optional[str] = None,
        prefer_history: bool = False,
    ) -> None:
        preset = _POLICY_PRESETS.get(mode, _POLICY_PRESETS["basic"])
        self.le_policy_config.setText(preset["config"])

        previous_default = self._current_checkpoint_default
        default = self._default_checkpoint_for_mode(mode)
        current = self._checkpoint_text() if hasattr(self, "cb_checkpoint") else ""
        if select is None:
            if prefer_history:
                select = (
                    self._last_checkpoint_by_mode.get(mode)
                    or next(iter(self._checkpoint_history.get(mode, [])), "")
                    or default
                )
            elif not current or current == previous_default:
                select = default
            else:
                select = current

        old = self.cb_checkpoint.blockSignals(True)
        self.cb_checkpoint.clear()
        self.cb_checkpoint.addItems(self._checkpoint_items_for_mode(mode))
        self.cb_checkpoint.setEditText(select)
        self.cb_checkpoint.blockSignals(old)
        self._current_checkpoint_default = default

    def _on_policy_step_changed(self, *_) -> None:
        current = self._checkpoint_text()
        select = None
        if not current or current == self._current_checkpoint_default:
            select = self._default_checkpoint_for_mode(self._runtime.mode)
        self._refresh_policy_fields_for_mode(self._runtime.mode, select=select)

    def _remember_checkpoint(self, mode: Optional[str] = None, path: Optional[str] = None) -> None:
        mode = mode or self._runtime.mode
        path = (path if path is not None else self._checkpoint_text()).strip()
        if not path:
            return
        entries = [p for p in self._checkpoint_history.get(mode, []) if p != path]
        entries.insert(0, path)
        self._checkpoint_history[mode] = entries[:_CHECKPOINT_HISTORY_LIMIT]
        self._last_checkpoint_by_mode[mode] = path
        self._save_checkpoint_history()
        self._refresh_policy_fields_for_mode(mode, select=path)

    def _on_checkpoint_edited(self) -> None:
        self._remember_checkpoint()

    def _on_browse_checkpoint(self) -> None:
        current = self._checkpoint_text()
        start_dir = _REPO_ROOT
        if current and not current.startswith("gs://"):
            candidate = pathlib.Path(current).expanduser()
            if not candidate.is_absolute():
                candidate = _REPO_ROOT / candidate
            if candidate.exists():
                start_dir = candidate if candidate.is_dir() else candidate.parent
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select checkpoint directory", str(start_dir))
        if not path:
            return
        self.cb_checkpoint.setEditText(path)
        self._remember_checkpoint(path=path)

    def _policy_python_cmd(self) -> list[str]:
        if sys.version_info >= (3, 11):
            return [sys.executable]
        uv = shutil.which("uv")
        if uv is not None:
            return [uv, "run", "python"]
        return [sys.executable]

    @staticmethod
    def _port_is_open(host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=0.25):
                return True
        except OSError:
            return False

    def _start_policy_server_if_needed(self) -> None:
        if not self.cb_local_policy.isChecked():
            return

        mode = self._runtime.mode
        config = _POLICY_PRESETS.get(mode, _POLICY_PRESETS["basic"])["config"]
        self.le_policy_config.setText(config)
        policy_dir = self._checkpoint_text()
        if not config or not policy_dir:
            raise ValueError("policy.config and policy.dir are required when starting a local policy server")

        port = int(self.le_port.text())
        host = "127.0.0.1"
        self.le_host.setText(host)
        policy_dir_arg = policy_dir

        if not policy_dir.startswith("gs://"):
            checkpoint_path = pathlib.Path(policy_dir).expanduser()
            resolved = checkpoint_path if checkpoint_path.is_absolute() else _REPO_ROOT / checkpoint_path
            if not resolved.exists():
                raise FileNotFoundError(f"checkpoint path does not exist: {resolved}")
            if policy_dir.startswith("~"):
                policy_dir_arg = str(resolved)
        spec = (config, policy_dir_arg, port)

        if self._policy_proc is not None and self._policy_proc.poll() is None:
            if self._policy_spec == spec:
                return
            self._stop_policy_server()
        elif self._policy_proc is not None:
            self._policy_proc = None
            self._policy_spec = None

        if self._port_is_open(host, port):
            raise RuntimeError(
                f"Port {port} is already in use. Stop the existing policy server, "
                "choose another port, or uncheck 'start local server'."
            )

        env = os.environ.copy()
        env["JAX_PLATFORMS"] = "cpu"
        extra_pythonpath = [
            str(_REPO_ROOT / "src"),
            str(_REPO_ROOT / "packages" / "openpi-client" / "src"),
        ]
        if env.get("PYTHONPATH"):
            extra_pythonpath.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(extra_pythonpath)

        cmd = [
            *self._policy_python_cmd(),
            "scripts/serve_policy_pytorch.py",
            "--env",
            "ALOHA",
            "--default-prompt",
            self.le_task.text().strip() or self._runtime.task,
            "--port",
            str(port),
            "policy:checkpoint",
            "--policy.config",
            config,
            "--policy.dir",
            policy_dir_arg,
        ]
        self._remember_checkpoint(mode=mode, path=policy_dir)
        self._log_info(f"Starting local policy server: {config} -> {policy_dir_arg}")
        self._policy_proc = subprocess.Popen(cmd, cwd=str(_REPO_ROOT), env=env)
        self._policy_spec = spec

    def _stop_policy_server(self) -> None:
        proc = self._policy_proc
        if proc is None:
            return
        if proc.poll() is None:
            self._log_info("Stopping local policy server ...")
            proc.terminate()
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)
        self._policy_proc = None
        self._policy_spec = None

    # ------------------------------------------------------------------ #
    def _on_task_edited(self) -> None:
        with self._runtime._lock:
            self._runtime.task = self.le_task.text().strip()
        self._log_info(f"Task -> {self._runtime.task!r}")

    def _make_subtask_row(self, key: int, label: str) -> QtWidgets.QWidget:
        row = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)

        le = QtWidgets.QLineEdit(label)
        le.editingFinished.connect(lambda kk=key: self._on_subtask_edited(kk))
        row_layout.addWidget(le, 1)
        self.subtask_inputs[key] = le

        btn_use = QtWidgets.QPushButton("Use")
        btn_use.setFixedWidth(52)
        btn_use.clicked.connect(lambda _=False, kk=key: self._set_subtask(kk))
        row_layout.addWidget(btn_use)
        return row

    def _on_subtask_edited(self, key: int) -> None:
        if key not in self.subtask_inputs:
            return
        new_label = self.subtask_inputs[key].text().strip()
        self._runtime.set_subtask_label(key, new_label)
        self._log_info(f"subtask[{key}] -> {new_label!r}")

    def _on_add_subtask(self) -> None:
        key = self._runtime.add_subtask_label()
        self._refresh_subtask_inputs()
        QtCore.QTimer.singleShot(
            0,
            lambda: self.subtask_scroll.verticalScrollBar().setValue(
                self.subtask_scroll.verticalScrollBar().maximum()
            ),
        )
        self._log_info(f"Added subtask key={key}")

    def _on_remove_subtask(self) -> None:
        with self._runtime._lock:
            labels = dict(self._runtime.subtask_labels)
        if len(labels) <= 1:
            self._log_error("Cannot remove the last subtask label.")
            return
        key = max(labels.keys())
        try:
            label = self._runtime.remove_subtask_label(key)
        except Exception as e:                       # noqa: BLE001
            self._log_error(f"Remove subtask failed: {e}")
            return
        self._refresh_subtask_inputs()
        self._log_info(f"Removed subtask key={key} ({label})")

    def _on_load_subtask_json(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load subtask JSON", "", "JSON files (*.json)")
        if not path:
            return
        try:
            self._runtime.load_subtasks_from_json(path)
        except Exception as e:                       # noqa: BLE001
            self._log_error(f"Load JSON failed: {e}")
            return
        # Rebuild input fields.
        self._refresh_subtask_inputs()
        self._log_info(f"Loaded subtask labels from {path}")

    def _refresh_subtask_inputs(self) -> None:
        snap = self._runtime.snapshot()
        labels = snap["subtask_labels"]
        layout = self.subtask_form_layout
        while layout.rowCount():
            layout.removeRow(0)
        self.subtask_inputs = {}
        for k in sorted(labels.keys()):
            layout.addRow(f"key {k}:", self._make_subtask_row(k, labels[k]))
        self.btn_remove_subtask.setEnabled(len(labels) > 1)

    def _on_mode_changed(self, mode: str) -> None:
        if self._runner is not None and self.cb_local_policy.isChecked():
            self._log_error(
                "Stop the run before changing mode while using the local policy server, "
                "so the matching checkpoint can be loaded."
            )
            old = self.cb_mode.blockSignals(True)
            self.cb_mode.setCurrentText(self._runtime.mode)
            self.cb_mode.blockSignals(old)
            return

        self.lbl_mode_help.setText(_MODE_DESCRIPTIONS.get(mode, ""))
        with self._runtime._lock:
            self._runtime.mode = mode
        self._refresh_policy_fields_for_mode(mode, prefer_history=True)
        # Sensible default step interval per mode.
        defaults = {"traj": 6, "subgoal": 30, "subtask": 30}
        if mode in defaults:
            self.sp_interval.setValue(defaults[mode])
        if self._runner is None:
            return
        try:
            handler = self._build_handler(mode)
        except Exception as e:                       # noqa: BLE001
            self._log_error(f"Mode switch failed: {e}")
            return
        self._handler = handler
        self._runtime.wake_subtask_waiters()
        self._runner.set_handler(handler)

    def _build_handler(self, mode: str) -> _modes.ModeHandler:
        kwargs = dict(
            foreact_host=self.le_fhost.text().strip(),
            foreact_port=int(self.le_fport.text()),
            blocking=self._blocking,
        )
        n = int(self.sp_interval.value())
        if mode == "traj":
            kwargs["traj_step_interval"] = n
        elif mode == "subgoal":
            kwargs["subgoal_step_interval"] = n
        elif mode == "subtask":
            kwargs["subtask_step_interval"] = n
        return _modes.make_handler(mode, **kwargs)

    def _on_submode_changed(self, *_) -> None:
        self._blocking = self.rb_block.isChecked()
        if self._handler is not None:
            self._handler.set_blocking(self._blocking)
        self._log_info(f"sub-mode -> {'blocking' if self._blocking else 'non-blocking'}")

    def _on_interval_changed(self, n: int) -> None:
        self._step_interval = int(n)
        if self._handler is not None:
            self._handler.set_step_interval(n)

    def _set_subtask(self, key: int) -> None:
        with self._runtime._lock:
            labels = dict(self._runtime.subtask_labels)
        if key not in labels:
            self._log_error(f"key {key} not in subtask_labels")
            return
        self._runtime.set_subtask_key(key)
        label = labels[key]
        self._log_info(f"subtask key={key} ({label})")

    def _on_connect(self) -> None:
        if self._runner is not None:
            self._log_info("Already connected.")
            return
        try:
            self._cfg.host = self.le_host.text().strip() or "127.0.0.1"
            self._cfg.port = int(self.le_port.text())
            self._cfg.action_horizon = int(self.le_chunk.text())
            self._cfg.max_steps = int(self.le_max.text())
            if self.cb_local_policy.isChecked():
                self._start_policy_server_if_needed()
                self._cfg.host = "127.0.0.1"
                self.le_host.setText(self._cfg.host)
            self._handler = self._build_handler(self._runtime.mode)
            self._runner = EvalRunner(self._cfg, self._runtime, self._handler, on_event=self._on_event)
            self._runner.connect()
            self._log_info("Connected.")
        except Exception as e:                           # noqa: BLE001
            logger.exception("connect failed")
            self._log_error(f"connect failed: {e}")
            self._runner = None

    def _on_start(self) -> None:
        if self._runner is None:
            self._on_connect()
        if self._runner is not None:
            self._configure_video_recording()
            self._runner.start()
            self._log_info("Run loop started.")

    def _on_pause_toggle(self) -> None:
        if self._runner is None:
            return
        if self._runner.running:
            self._runner.pause()
            self.btn_pause.setText("▶ Resume")
            self._log_info("Paused.")
        else:
            self._runner.resume()
            self.btn_pause.setText("❙❙ Pause")
            self._log_info("Resumed.")

    def _on_stop(self) -> None:
        if self._runner is None:
            return
        self._runner.stop()
        self._runner = None
        self._handler = None
        # Clear all UI display fields so stale info is not shown.
        self.lbl_prompt.setText("(idle)")
        self.lbl_subtask.setText("-")
        self.lbl_traj.setText("-")
        self.lbl_gripper.setText("L: -   R: -")
        self.lbl_step.setText("0")
        blank_pm = __import__('PyQt5.QtGui', fromlist=['QPixmap']).QPixmap(
            self.lbl_subgoal.width(), self.lbl_subgoal.height())
        blank_pm.fill(__import__('PyQt5.QtGui', fromlist=['QColor']).QColor("#111"))
        self.lbl_subgoal.setPixmap(blank_pm)
        self.btn_pause.setText("\u2759\u2759 Pause")
        self._log_info("Run loop stopped. State cleared.")

    def _on_rtz(self) -> None:
        if self._runner is None:
            self._log_error("Not connected — cannot return to zero. Click Connect first.")
            return
        self._runner.request_reset()

    def _on_dump(self) -> None:
        if self._runner is None:
            self._log_error("Not connected — cannot dump inputs.")
            return
        self._runner.request_dump()

    # ------------------------------------------------------------------ #
    def _refresh_ui(self) -> None:
        if self._runner is not None:
            obs = self._runner.latest_obs()
            if obs is not None:
                images = obs.get("images", {})
                self.lbl_cam.setPixmap(_np_to_qpixmap(images.get("cam_high"), 480, 360))
                self.lbl_left.setPixmap(_np_to_qpixmap(images.get("cam_left_wrist"), 220, 165))
                self.lbl_right.setPixmap(_np_to_qpixmap(images.get("cam_right_wrist"), 220, 165))

        snap = self._runtime.snapshot()
        self.lbl_prompt.setText(snap["prompt"] or "(empty)")
        self.lbl_subtask.setText(snap["subtask_label"] or "-")
        traj = snap["traj_text"] or "-"
        if len(traj) > 220:
            traj = traj[:220] + "..."
        self.lbl_traj.setText(traj)
        sg = snap["subgoal_image"]
        if sg is not None:
            self.lbl_subgoal.setPixmap(_np_to_qpixmap(sg, 480, 360))

        self.gb_subtask.setEnabled(snap["mode"] == "subtask")
        active = f"active: {snap['subtask_key']}"
        if snap.get("waiting_for_subtask_input"):
            active += " (waiting)"
        self.lbl_active_key.setText(active)

    # ------------------------------------------------------------------ #
    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:        # noqa: N802
        k = event.key()
        text = event.text()
        if text.isdigit() and text != "0":
            self._set_subtask(int(text))
        elif k == QtCore.Qt.Key_Space:
            self._on_pause_toggle()
        elif k == QtCore.Qt.Key_H:
            self._on_rtz()
        elif k == QtCore.Qt.Key_D:
            self._on_dump()
        elif k == QtCore.Qt.Key_Q:
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:                             # noqa: N802
        try:
            if self._runner is not None:
                self._runner.stop()
            self._stop_policy_server()
        finally:
            super().closeEvent(event)


# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Aloha eval GUI")
    p.add_argument("--mode", default="basic", choices=["basic", "traj", "subtask", "subgoal"])
    p.add_argument("--task", default="put banana in the green plate")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--action_horizon", type=int, default=25)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--dump_dir", default="debug_inputs")
    p.add_argument("--log", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log.upper(), force=True,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    cfg = RunnerConfig(
        host=args.host,
        port=args.port,
        action_horizon=args.action_horizon,
        max_steps=args.max_steps,
        dry_run=args.dry_run,
        dump_dir=args.dump_dir,
    )
    runtime = _modes.RuntimeState(mode=args.mode, task=args.task)

    app = QtWidgets.QApplication(sys.argv)
    win = EvalGUI(cfg, runtime)
    win.resize(1280, 900)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
