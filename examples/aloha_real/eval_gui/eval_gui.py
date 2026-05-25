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
import threading
import time
from typing import Optional

import numpy as np
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets

from . import modes as _modes
from .eval_runner import EvalRunner
from .eval_runner import RunnerConfig
from .manual_trajectory import build_manual_prediction
from .manual_trajectory import build_prediction_from_text
from .manual_trajectory import pixel_to_loc_tokens
from .manual_trajectory import trajectory_text_to_arm_items
from .trajectory_retrieval import TrajectoryReferenceRetriever
from .trajectory_retrieval import default_cache_path

logger = logging.getLogger(__name__)


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

_MODE_DESCRIPTIONS = {
    "basic":   "Basic — VLA inference with task prompt only.",
    "traj":    "Add Trajectory — manual L/R waypoint annotation by default.",
    "subtask": "Add Subtasks — keys 1-8 override the subtask label.",
    "triple_cot": "Triple-CoT — semi-block task + live subtask + trajectory prompt.",
    "subgoal": "Add Subgoal Images — retrieval, ForeAct, GPT, or video reference generates cam_high subgoal.",
}

_POLICY_PRESETS = {
    "basic": {
        "label": "basic - aloha_food baseline",
        "config": "pi05_aloha_food_baseline",
        "dir": "/home/agilex/agentic-openpi/food/pi05_aloha_food_baseline/{step}",
    },
    "traj": {
        "label": "traj - aloha_food trajectory cot",
        "config": "pi05_aloha_food_traj",
        "dir": "/home/agilex/agentic-openpi/food/pi05_aloha_food_traj/{step}",
    },
    "subtask": {
        "label": "subtask - aloha_food labels",
        "config": "pi05_aloha_food_subtask",
        "dir": "/home/agilex/agentic-openpi/food/pi05_aloha_food_subtask/{step}",
    },
    "triple_cot": {
        "label": "triple-cot - aloha_food all cot",
        "config": "pi05_aloha_food_all_cot",
        "dir": "/home/agilex/agentic-openpi/food/pi05_aloha_food_all_cot/{step}",
    },
    "subgoal": {
        "label": "subgoal - aloha_food base camera",
        "config": "pi05_aloha_food_subgoal",
        "dir": "/home/agilex/agentic-openpi/food/pi05_aloha_food_subgoal/{step}",
    },
}

_SUBTASK_PROMPT_MODES = {"subtask", "triple_cot"}
_TRAJECTORY_SOURCE_MODES = {"traj", "triple_cot"}
_TRAJECTORY_ARM_CONFIG_MODES = {"traj", "triple_cot"}
_BLOCKING_ONLY_MODES = {"triple_cot"}
_TRAJ_ANNOTATION_ARMS = {"both", "left", "right"}
_TRAJ_MODE_DEFAULT_INACTIVE_PIXEL = (201.0, 666.0)

_CHECKPOINT_HISTORY_LIMIT = 30
_CHECKPOINT_HISTORY_ENV = "AGENTIC_OPENPI_EVAL_GUI_HISTORY"
_TRAJ_RETRIEVAL_CACHE_ENV = "AGENTIC_OPENPI_TRAJ_RETRIEVAL_CACHE"
_TRAJ_RETRIEVAL_DATASET_ENV = "AGENTIC_OPENPI_TRAJ_RETRIEVAL_DATASET"
_DEFAULT_TRAJ_RETRIEVAL_DATASET = _REPO_ROOT / "playground" / "Datasets" / "food"


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


class _ManualTrajectoryRequest:
    """Thread handoff object used by the runner to request a GUI annotation."""

    def __init__(
        self,
        image: np.ndarray,
        task: str,
        *,
        annotate_arm: str = "both",
        right_only: bool = False,
        default_left_pixel: Optional[tuple[float, float]] = None,
        default_right_pixel: Optional[tuple[float, float]] = None,
        reference_prediction=None,
        reference_metadata: Optional[dict] = None,
    ) -> None:
        self.image = image
        self.task = task
        if right_only:
            annotate_arm = "right"
        self.annotate_arm = annotate_arm if annotate_arm in _TRAJ_ANNOTATION_ARMS else "both"
        self.right_only = self.annotate_arm == "right"
        self.left_only = self.annotate_arm == "left"
        self.default_left_pixel = default_left_pixel
        self.default_right_pixel = default_right_pixel
        self.reference_prediction = reference_prediction
        self.reference_metadata = dict(reference_metadata or {})
        self.done = threading.Event()
        self.prediction = None
        self.cancel_requested = False

    def set_prediction(self, prediction) -> None:
        if self.done.is_set():
            return
        self.prediction = prediction
        self.done.set()

    def cancel(self) -> None:
        self.cancel_requested = True
        self.done.set()


class _TrajectoryImageLabel(QtWidgets.QLabel):
    image_clicked = QtCore.pyqtSignal(float, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._source_image: Optional[np.ndarray] = None
        self.setMinimumSize(720, 540)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet("background:#111;")
        self.setMouseTracking(True)

    def set_source_image(self, image: np.ndarray) -> None:
        self._source_image = image
        self._refresh_pixmap()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh_pixmap()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() != QtCore.Qt.LeftButton or self._source_image is None:
            return
        rect = self._image_rect()
        if rect is None or not rect.contains(event.pos()):
            return
        h, w = self._source_image.shape[:2]
        x = (event.x() - rect.x()) / max(1.0, rect.width() - 1.0) * max(0, w - 1)
        y = (event.y() - rect.y()) / max(1.0, rect.height() - 1.0) * max(0, h - 1)
        self.image_clicked.emit(float(x), float(y))

    def _image_rect(self) -> Optional[QtCore.QRectF]:
        if self._source_image is None:
            return None
        h, w = self._source_image.shape[:2]
        if h <= 0 or w <= 0 or self.width() <= 0 or self.height() <= 0:
            return None
        scale = min(self.width() / float(w), self.height() / float(h))
        disp_w = w * scale
        disp_h = h * scale
        x0 = (self.width() - disp_w) / 2.0
        y0 = (self.height() - disp_h) / 2.0
        return QtCore.QRectF(x0, y0, disp_w, disp_h)

    def _refresh_pixmap(self) -> None:
        if self._source_image is None or self.width() <= 0 or self.height() <= 0:
            return
        self.setPixmap(_np_to_qpixmap(self._source_image, self.width(), self.height()))


class ManualTrajectoryDialog(QtWidgets.QDialog):
    """Modal trajectory annotation dialog shown on the Qt main thread."""

    def __init__(
        self,
        image: np.ndarray,
        task: str,
        parent=None,
        *,
        annotate_arm: str = "both",
        right_only: bool = False,
        default_left_pixel: Optional[tuple[float, float]] = None,
        default_right_pixel: Optional[tuple[float, float]] = None,
        reference_prediction=None,
        reference_metadata: Optional[dict] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Manual Trajectory Annotation")
        self.setWindowFlags(
            QtCore.Qt.Window
            | QtCore.Qt.WindowTitleHint
            | QtCore.Qt.WindowCloseButtonHint
            | QtCore.Qt.WindowMinMaxButtonsHint
        )
        self.setModal(True)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.resize(980, 760)

        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        self._base_image = arr.copy()
        if right_only:
            annotate_arm = "right"
        self._annotate_arm = annotate_arm if annotate_arm in _TRAJ_ANNOTATION_ARMS else "both"
        self._single_arm = self._annotate_arm if self._annotate_arm in {"left", "right"} else None
        self._right_only = self._annotate_arm == "right"
        self._left_only = self._annotate_arm == "left"
        self._active_arm = self._single_arm or "left"
        self._items: dict[str, list[str]] = {"left": [], "right": []}
        self._history: list[tuple[str, str]] = []
        self._prediction = None
        self._force_close = False
        self._aborted = False
        self._updating_text = False
        self._keyboard_filter_installed = False
        self._shortcuts: list[QtWidgets.QShortcut] = []
        self._reference_prediction = reference_prediction
        self._reference_metadata = dict(reference_metadata or {})
        self._default_left_pixel = default_left_pixel
        self._default_right_pixel = default_right_pixel
        self._reference_text = (
            reference_prediction.to_loc_token_text()
            if reference_prediction is not None and not reference_prediction.is_empty()
            else ""
        )
        if self._reference_text:
            try:
                parsed = trajectory_text_to_arm_items(self._reference_text)
                self._items["left"].extend(parsed.get("left", []))
                self._items["right"].extend(parsed.get("right", []))
            except Exception as e:
                logger.warning("Could not parse reference trajectory into dialog items: %s", e)
        self._apply_inactive_defaults()

        root = QtWidgets.QVBoxLayout(self)
        self.lbl_task = QtWidgets.QLabel(f"Task: {task}")
        self.lbl_task.setWordWrap(True)
        root.addWidget(self.lbl_task)

        self.image_label = _TrajectoryImageLabel(self)
        self.image_label.set_source_image(self._base_image)
        self.image_label.image_clicked.connect(self._on_image_clicked)
        root.addWidget(self.image_label, 1)

        text_header = QtWidgets.QHBoxLayout()
        text_header.addWidget(QtWidgets.QLabel("Trajectory text:"))
        self.lbl_reference = QtWidgets.QLabel(self._reference_summary_text())
        self.lbl_reference.setStyleSheet("color:#888;")
        text_header.addWidget(self.lbl_reference, 1)
        root.addLayout(text_header)

        self.txt_trajectory = QtWidgets.QPlainTextEdit()
        self.txt_trajectory.setMaximumHeight(88)
        self.txt_trajectory.setReadOnly(True)
        self.txt_trajectory.setFocusPolicy(QtCore.Qt.NoFocus)
        self.txt_trajectory.setPlaceholderText("Left: Go along <loc....><loc....>. Right: Go along ...")
        self.txt_trajectory.setToolTip("Read-only trajectory preview. Use image clicks and buttons to annotate.")
        self.txt_trajectory.textChanged.connect(self._on_trajectory_text_changed)
        root.addWidget(self.txt_trajectory)
        if self._reference_text:
            self._set_trajectory_text(self._reference_text)

        btn_row = QtWidgets.QHBoxLayout()
        self.bg_arm = QtWidgets.QButtonGroup(self)
        self.btn_left = QtWidgets.QPushButton("L (Left Arm)")
        self.btn_left.setToolTip("Shortcut: A")
        self.btn_left.setCheckable(True)
        self.btn_left.setChecked(self._active_arm == "left")
        self.btn_left.setEnabled(self._arm_selectable("left"))
        self.btn_left.setVisible(self._arm_selectable("left"))
        self.btn_right = QtWidgets.QPushButton("R (Right Arm)")
        self.btn_right.setToolTip("Shortcut: D")
        self.btn_right.setCheckable(True)
        self.btn_right.setChecked(self._active_arm == "right")
        self.btn_right.setEnabled(self._arm_selectable("right"))
        self.btn_right.setVisible(self._arm_selectable("right"))
        self.bg_arm.addButton(self.btn_left)
        self.bg_arm.addButton(self.btn_right)
        self.btn_left.clicked.connect(lambda: self._set_active_arm("left"))
        self.btn_right.clicked.connect(lambda: self._set_active_arm("right"))
        btn_row.addWidget(self.btn_left)
        btn_row.addWidget(self.btn_right)

        self.btn_open = QtWidgets.QPushButton("Open Gripper")
        self.btn_open.setToolTip("Shortcut: Q")
        self.btn_open.clicked.connect(lambda: self._append_item("open gripper"))
        btn_row.addWidget(self.btn_open)
        self.btn_close = QtWidgets.QPushButton("Close Gripper")
        self.btn_close.setToolTip("Shortcut: W")
        self.btn_close.clicked.connect(lambda: self._append_item("close gripper"))
        btn_row.addWidget(self.btn_close)
        self.btn_undo = QtWidgets.QPushButton("Undo")
        self.btn_undo.setToolTip("Remove the most recent point or gripper action. Shortcut: Z")
        self.btn_undo.clicked.connect(self._undo_last)
        btn_row.addWidget(self.btn_undo)
        self.btn_clear = QtWidgets.QPushButton("Clear")
        self.btn_clear.setToolTip("Discard the current/reference trajectory text and points.")
        self.btn_clear.clicked.connect(self._clear_all)
        btn_row.addWidget(self.btn_clear)
        btn_row.addStretch(1)
        self.btn_abort = QtWidgets.QPushButton("Emergency Stop")
        self.btn_abort.setToolTip("Discard this annotation, pause inference, and queue return-to-zero.")
        self.btn_abort.setStyleSheet(
            "background:#b00020;color:#fff;font-weight:bold;padding:8px;"
        )
        self.btn_abort.clicked.connect(self._abort)
        btn_row.addWidget(self.btn_abort)
        self.btn_finish = QtWidgets.QPushButton("Finish")
        self.btn_finish.clicked.connect(self._finish)
        btn_row.addWidget(self.btn_finish)
        root.addLayout(btn_row)

        self.lbl_sequence = QtWidgets.QLabel("")
        self.lbl_sequence.setWordWrap(True)
        self.lbl_sequence.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        root.addWidget(self.lbl_sequence)
        self._install_keyboard_shortcuts()
        self._refresh_sequence()
        self._refresh_preview()

    @property
    def prediction(self):
        return self._prediction

    @property
    def aborted(self) -> bool:
        return self._aborted

    def reject(self) -> None:
        if self._force_close:
            super().reject()
            return
        QtWidgets.QApplication.beep()

    def done(self, result: int) -> None:
        self._uninstall_keyboard_filter()
        super().done(result)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._install_keyboard_filter()
        QtCore.QTimer.singleShot(0, self._activate_keyboard_focus)
        QtCore.QTimer.singleShot(80, self._activate_keyboard_focus)

    def hideEvent(self, event) -> None:  # noqa: N802
        self._uninstall_keyboard_filter()
        super().hideEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.result() == QtWidgets.QDialog.Accepted or self._force_close:
            self._uninstall_keyboard_filter()
            super().closeEvent(event)
            return
        event.ignore()
        QtWidgets.QApplication.beep()

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if (
            self.isVisible()
            and event.type() == QtCore.QEvent.KeyPress
            and self._handle_shortcut_key(event.key())
        ):
            event.accept()
            return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        if self._handle_shortcut_key(event.key()):
            event.accept()
            return
        super().keyPressEvent(event)

    def _install_keyboard_shortcuts(self) -> None:
        def add_shortcut(sequence: str, callback) -> None:
            shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(sequence), self)
            shortcut.setContext(QtCore.Qt.ApplicationShortcut)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)

        add_shortcut("Q", lambda: self._click_button(self.btn_open))
        add_shortcut("W", lambda: self._click_button(self.btn_close))
        add_shortcut("A", lambda: self._select_arm_from_shortcut("left"))
        add_shortcut("D", lambda: self._select_arm_from_shortcut("right"))
        add_shortcut("Z", lambda: self._click_button(self.btn_undo))

    def _install_keyboard_filter(self) -> None:
        if self._keyboard_filter_installed:
            return
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        app.installEventFilter(self)
        self._keyboard_filter_installed = True

    def _uninstall_keyboard_filter(self) -> None:
        if not self._keyboard_filter_installed:
            return
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        self._keyboard_filter_installed = False

    def _activate_keyboard_focus(self) -> None:
        if not self.isVisible():
            return
        self.raise_()
        self.activateWindow()
        self.setFocus(QtCore.Qt.ActiveWindowFocusReason)

    @staticmethod
    def _click_button(button: QtWidgets.QPushButton) -> None:
        if button.isVisible() and button.isEnabled():
            button.click()

    def _select_arm_from_shortcut(self, arm: str) -> None:
        if not self._arm_selectable(arm):
            return
        self._click_button(self.btn_left if arm == "left" else self.btn_right)

    def _handle_shortcut_key(self, key: int) -> bool:
        if key == QtCore.Qt.Key_Q:
            self._click_button(self.btn_open)
            return True
        if key == QtCore.Qt.Key_W:
            self._click_button(self.btn_close)
            return True
        if key == QtCore.Qt.Key_A:
            self._select_arm_from_shortcut("left")
            return True
        if key == QtCore.Qt.Key_D:
            self._select_arm_from_shortcut("right")
            return True
        if key == QtCore.Qt.Key_Z:
            self._click_button(self.btn_undo)
            return True
        return False

    def cancel_from_runner(self) -> None:
        self._uninstall_keyboard_filter()
        self._force_close = True
        super().reject()

    def _abort(self) -> None:
        self._uninstall_keyboard_filter()
        self._prediction = None
        self._aborted = True
        self._force_close = True
        super().reject()

    def _reference_summary_text(self) -> str:
        if not self._reference_text:
            return ""
        ep = self._reference_metadata.get("episode_index")
        frame = self._reference_metadata.get("frame_index")
        score = self._reference_metadata.get("score")
        search_ms = self._reference_metadata.get("search_ms")
        parts = ["reference loaded"]
        if ep is not None and frame is not None:
            parts.append(f"episode {int(ep):06d} frame {int(frame)}")
        if score is not None:
            parts.append(f"score {float(score):.3f}")
        if search_ms is not None:
            parts.append(f"{float(search_ms):.1f} ms")
        return " | ".join(parts)

    @staticmethod
    def _default_position_token(pixel: tuple[float, float]) -> str:
        x, y = pixel
        loc_x = max(0, min(int(round(float(x))), 1000))
        loc_y = max(0, min(int(round(float(y))), 1000))
        return f"<loc{loc_x:04d}><loc{loc_y:04d}>"

    def _apply_inactive_defaults(self) -> None:
        if self._right_only and self._default_left_pixel is not None and not self._items["left"]:
            self._items["left"].append(self._default_position_token(self._default_left_pixel))
        if self._left_only and self._default_right_pixel is not None and not self._items["right"]:
            self._items["right"].append(self._default_position_token(self._default_right_pixel))

    def _set_trajectory_text(self, text: str) -> None:
        self._updating_text = True
        try:
            self.txt_trajectory.setPlainText(text)
        finally:
            self._updating_text = False

    def _on_trajectory_text_changed(self) -> None:
        if self._updating_text:
            return
        self._refresh_preview()

    def _text_prediction(self):
        text = self.txt_trajectory.toPlainText().strip()
        if not text:
            return None
        pred = build_prediction_from_text(text)
        return None if pred.is_empty() else pred

    def _arm_selectable(self, arm: str) -> bool:
        return self._single_arm is None or self._single_arm == arm

    def _set_active_arm(self, arm: str) -> None:
        if not self._arm_selectable(arm):
            return
        self._active_arm = arm
        self.btn_left.setChecked(arm == "left")
        self.btn_right.setChecked(arm == "right")
        self._refresh_sequence()

    def _on_image_clicked(self, x: float, y: float) -> None:
        h, w = self._base_image.shape[:2]
        token = pixel_to_loc_tokens(x, y, w, h)
        self._append_item(token)

    def _append_item(self, item: str) -> None:
        arm = self._active_arm
        self._items[arm].append(item)
        self._history.append((arm, item))
        self._refresh_text_from_items()
        self._refresh_preview()
        self._refresh_sequence()

    def _undo_last(self) -> None:
        if not self._history:
            return
        arm, item = self._history.pop()
        if self._items[arm] and self._items[arm][-1] == item:
            self._items[arm].pop()
        elif item in self._items[arm]:
            self._items[arm].remove(item)
        self._refresh_text_from_items()
        self._refresh_preview()
        self._refresh_sequence()

    def _finish(self) -> None:
        text_pred = self._text_prediction()
        if text_pred is not None:
            self._prediction = text_pred
            self.accept()
            return
        if not self._items["left"] or not self._items["right"]:
            if self._single_arm is not None:
                message = (
                    f"Add at least one point or gripper action for the {self._single_arm} "
                    "arm before finishing."
                )
            else:
                message = "Add at least one point or gripper action for both arms before finishing."
            QtWidgets.QMessageBox.warning(
                self,
                "Missing Trajectory",
                message,
            )
            return
        self._prediction = build_manual_prediction(self._items["left"], self._items["right"])
        self.accept()

    def _trajectory_text(self) -> str:
        text_pred = self._text_prediction()
        if text_pred is not None:
            return text_pred.to_loc_token_text()
        if not self._items["left"] and not self._items["right"]:
            return ""
        return build_manual_prediction(self._items["left"], self._items["right"]).to_loc_token_text()

    def _refresh_text_from_items(self) -> None:
        if not self._items["left"] and not self._items["right"]:
            self._set_trajectory_text("")
            return
        self._set_trajectory_text(
            build_manual_prediction(self._items["left"], self._items["right"]).to_loc_token_text()
        )

    def _clear_all(self) -> None:
        self._items = {"left": [], "right": []}
        self._history = []
        self._apply_inactive_defaults()
        self._set_trajectory_text("")
        self._refresh_preview()
        self._refresh_sequence()

    def _refresh_preview(self) -> None:
        traj_text = self._trajectory_text()
        if not traj_text.strip():
            self.image_label.set_source_image(self._base_image)
            return
        try:
            from .trajectory_visualizer import render_trajectory_overlay

            self.image_label.set_source_image(render_trajectory_overlay(self._base_image, traj_text))
        except Exception as e:
            logger.warning("Manual trajectory preview failed: %s", e)
            self.image_label.set_source_image(self._base_image)

    def _refresh_sequence(self) -> None:
        left = ", ".join(self._items["left"]) or "-"
        right = ", ".join(self._items["right"]) or "-"
        self.lbl_sequence.setText(
            f"Active: {self._active_arm.upper()}    Left: {left}    Right: {right}"
        )


class _EventBridge(QtCore.QObject):
    """Marshals events from the runner thread onto the Qt main thread."""

    info = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)
    step = QtCore.pyqtSignal(int)
    gripper = QtCore.pyqtSignal(float, float)
    mode_changed = QtCore.pyqtSignal(str)
    episode_start = QtCore.pyqtSignal()
    episode_end = QtCore.pyqtSignal(int)
    manual_trajectory = QtCore.pyqtSignal(object)
    manual_trajectory_cancel = QtCore.pyqtSignal(object)
    connect_succeeded = QtCore.pyqtSignal(object, int)
    connect_failed = QtCore.pyqtSignal(str, int)


# ---------------------------------------------------------------------------
class EvalGUI(QtWidgets.QMainWindow):
    def __init__(self, cfg: RunnerConfig, runtime: _modes.RuntimeState, parent=None) -> None:
        super().__init__(parent)
        self._cfg = cfg
        self._runtime = runtime
        self._runner: Optional[EvalRunner] = None
        self._handler: Optional[_modes.ModeHandler] = None
        self._connecting = False
        self._connect_thread: Optional[threading.Thread] = None
        self._connect_generation = 0
        self._start_after_connect = False
        self._policy_proc: Optional[subprocess.Popen] = None
        self._policy_spec: Optional[tuple[str, str, int]] = None
        self._blocking = False
        self._step_interval = 60
        self._manual_traj_dialog: Optional[ManualTrajectoryDialog] = None
        self._manual_traj_request: Optional[_ManualTrajectoryRequest] = None
        self._traj_retriever: Optional[TrajectoryReferenceRetriever] = None
        self._traj_retriever_path = ""
        self._traj_retriever_warned_paths: set[str] = set()
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
        self._bridge.manual_trajectory.connect(self._on_manual_trajectory_requested)
        self._bridge.manual_trajectory_cancel.connect(self._on_manual_trajectory_cancel)
        self._bridge.connect_succeeded.connect(self._on_connect_succeeded)
        self._bridge.connect_failed.connect(self._on_connect_failed)

        self.setWindowTitle("Aloha Eval — agentic-openpi")
        self._build_ui()

        self._ui_timer = QtCore.QTimer(self)
        self._ui_timer.setInterval(33)         # ~30 Hz refresh
        self._ui_timer.timeout.connect(self._refresh_ui)
        self._ui_timer.start()
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

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
        for m in ("basic", "traj", "subtask", "triple_cot", "subgoal"):
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
        self.btn_clear_cache = QtWidgets.QPushButton("Clear Cache")
        self.btn_clear_cache.clicked.connect(self._on_clear_cache)
        checkpoint_row.addWidget(self.btn_clear_cache)
        left.addLayout(checkpoint_row)

        output_row = QtWidgets.QHBoxLayout()
        output_row.addWidget(QtWidgets.QLabel("output dir:"))
        initial_output_dir = str(pathlib.Path(self._cfg.video_dir).expanduser()) if self._cfg.video_dir else ""
        self.le_output_dir = QtWidgets.QLineEdit(initial_output_dir)
        self.le_output_dir.setPlaceholderText("Choose output directory before Start")
        output_row.addWidget(self.le_output_dir, 1)
        self.btn_browse_output_dir = QtWidgets.QPushButton("Browse...")
        self.btn_browse_output_dir.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DirOpenIcon))
        self.btn_browse_output_dir.clicked.connect(self._on_browse_output_dir)
        output_row.addWidget(self.btn_browse_output_dir)
        left.addLayout(output_row)

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
        cfg_row.addWidget(QtWidgets.QLabel("ROS wait:"))
        self.sp_ros_timeout = QtWidgets.QSpinBox()
        self.sp_ros_timeout.setRange(5, 600)
        self.sp_ros_timeout.setSingleStep(10)
        self.sp_ros_timeout.setValue(int(self._cfg.topic_wait_timeout))
        self.sp_ros_timeout.setSuffix("s")
        self.sp_ros_timeout.setToolTip("How long Connect waits for all required ROS camera/joint topics.")
        cfg_row.addWidget(self.sp_ros_timeout)
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
        self.rb_block.toggled.connect(self._on_submode_changed)
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

        self.gb_traj_source = QtWidgets.QGroupBox("Trajectory annotation")
        traj_source_layout = QtWidgets.QVBoxLayout(self.gb_traj_source)
        traj_source_row = QtWidgets.QHBoxLayout()
        traj_source_row.addWidget(QtWidgets.QLabel("During blocking steps:"))
        self.bg_traj_source = QtWidgets.QButtonGroup(self)
        self.rb_traj_doubao = QtWidgets.QRadioButton("Doubao API")
        self.rb_traj_manual = QtWidgets.QRadioButton("Manual Annotation")
        self.rb_traj_manual.setChecked(True)
        with self._runtime._lock:
            self._runtime.manual_traj_override = True
        self.bg_traj_source.addButton(self.rb_traj_doubao)
        self.bg_traj_source.addButton(self.rb_traj_manual)
        self.rb_traj_manual.toggled.connect(self._on_manual_traj_override_changed)
        traj_source_row.addWidget(self.rb_traj_doubao)
        traj_source_row.addWidget(self.rb_traj_manual)
        traj_source_row.addStretch(1)
        self.lbl_traj_source_note = QtWidgets.QLabel(
            "Manual Annotation is the default and does not require VOLCENKEY/ARK_API_KEY."
        )
        self.lbl_traj_source_note.setStyleSheet("color:#888;")
        traj_source_row.addWidget(self.lbl_traj_source_note)
        traj_source_layout.addLayout(traj_source_row)

        traj_arm_row = QtWidgets.QHBoxLayout()
        self.lbl_traj_annotate = QtWidgets.QLabel("Annotate:")
        traj_arm_row.addWidget(self.lbl_traj_annotate)
        self.cb_traj_annotate_arm = QtWidgets.QComboBox()
        self.cb_traj_annotate_arm.addItem("Right arm only", "right")
        self.cb_traj_annotate_arm.addItem("Left arm only", "left")
        self.cb_traj_annotate_arm.addItem("Both arms", "both")
        self.cb_traj_annotate_arm.setToolTip(
            "For single-arm annotation, the inactive arm uses the default loc position below."
        )
        traj_arm_row.addWidget(self.cb_traj_annotate_arm)
        self.lbl_traj_inactive_default = QtWidgets.QLabel("inactive default loc:")
        traj_arm_row.addWidget(self.lbl_traj_inactive_default)
        self.sp_traj_default_x = QtWidgets.QSpinBox()
        self.sp_traj_default_x.setRange(0, 1000)
        self.sp_traj_default_x.setValue(int(_TRAJ_MODE_DEFAULT_INACTIVE_PIXEL[0]))
        self.sp_traj_default_x.setPrefix("x=")
        self.sp_traj_default_y = QtWidgets.QSpinBox()
        self.sp_traj_default_y.setRange(0, 1000)
        self.sp_traj_default_y.setValue(int(_TRAJ_MODE_DEFAULT_INACTIVE_PIXEL[1]))
        self.sp_traj_default_y.setPrefix("y=")
        traj_arm_row.addWidget(self.sp_traj_default_x)
        traj_arm_row.addWidget(self.sp_traj_default_y)
        traj_arm_row.addStretch(1)
        traj_source_layout.addLayout(traj_arm_row)

        traj_ref_row = QtWidgets.QHBoxLayout()
        self.cb_traj_reference = QtWidgets.QCheckBox("Auto-suggest reference")
        self.cb_traj_reference.setChecked(True)
        self.cb_traj_reference.setToolTip(
            "Use a precomputed dataset image-embedding cache to pre-fill the trajectory dialog."
        )
        traj_ref_row.addWidget(self.cb_traj_reference)
        traj_ref_row.addWidget(QtWidgets.QLabel("cache:"))
        self.le_traj_reference_cache = QtWidgets.QLineEdit(self._default_traj_retrieval_cache_text())
        self.le_traj_reference_cache.setPlaceholderText(
            "$AGENTIC_OPENPI_TRAJ_RETRIEVAL_CACHE or <dataset>/trajectory_data/"
            "cam_high_traj_reference_cache.pt"
        )
        self.le_traj_reference_cache.editingFinished.connect(self._invalidate_trajectory_retriever)
        traj_ref_row.addWidget(self.le_traj_reference_cache, 1)
        self.btn_browse_traj_reference_cache = QtWidgets.QPushButton("Browse...")
        self.btn_browse_traj_reference_cache.clicked.connect(self._on_browse_traj_reference_cache)
        traj_ref_row.addWidget(self.btn_browse_traj_reference_cache)
        traj_source_layout.addLayout(traj_ref_row)
        left.addWidget(self.gb_traj_source)

        # Subgoal source selection (Mode 4)
        self.gb_subgoal_source = QtWidgets.QGroupBox("Subgoal source (Mode 4)")
        subgoal_source_layout = QtWidgets.QVBoxLayout(self.gb_subgoal_source)
        subgoal_src_row = QtWidgets.QHBoxLayout()
        subgoal_src_row.addWidget(QtWidgets.QLabel("Source:"))
        self.bg_subgoal_source = QtWidgets.QButtonGroup(self)
        self.rb_subgoal_foreact = QtWidgets.QRadioButton("ForeAct Server")
        self.rb_subgoal_gpt = QtWidgets.QRadioButton("GPT Generation")
        self.rb_subgoal_retrieval = QtWidgets.QRadioButton("Reference Retrieval")
        self.rb_subgoal_retrieval.setChecked(True)
        self.rb_subgoal_reference = QtWidgets.QRadioButton("Reference Video")
        self.bg_subgoal_source.addButton(self.rb_subgoal_foreact)
        self.bg_subgoal_source.addButton(self.rb_subgoal_gpt)
        self.bg_subgoal_source.addButton(self.rb_subgoal_retrieval)
        self.bg_subgoal_source.addButton(self.rb_subgoal_reference)
        self.rb_subgoal_gpt.toggled.connect(self._on_subgoal_source_changed)
        self.rb_subgoal_retrieval.toggled.connect(self._on_subgoal_source_changed)
        self.rb_subgoal_reference.toggled.connect(self._on_subgoal_source_changed)
        subgoal_src_row.addWidget(self.rb_subgoal_foreact)
        subgoal_src_row.addWidget(self.rb_subgoal_gpt)
        subgoal_src_row.addWidget(self.rb_subgoal_retrieval)
        subgoal_src_row.addWidget(self.rb_subgoal_reference)
        subgoal_src_row.addStretch(1)
        subgoal_source_layout.addLayout(subgoal_src_row)

        retrieval_row = QtWidgets.QHBoxLayout()
        retrieval_row.addWidget(QtWidgets.QLabel("Retrieval lookahead:"))
        self.sp_subgoal_retrieval_lookahead = QtWidgets.QSpinBox()
        self.sp_subgoal_retrieval_lookahead.setRange(0, 10_000)
        self.sp_subgoal_retrieval_lookahead.setSingleStep(10)
        self.sp_subgoal_retrieval_lookahead.setValue(60)
        self.sp_subgoal_retrieval_lookahead.setSuffix(" frames")
        self.sp_subgoal_retrieval_lookahead.setToolTip(
            "After matching the current observation to dataset frame n, "
            "use frame n + this value as the subgoal image."
        )
        retrieval_row.addWidget(self.sp_subgoal_retrieval_lookahead)
        retrieval_row.addStretch(1)
        subgoal_source_layout.addLayout(retrieval_row)

        gpt_key_row = QtWidgets.QHBoxLayout()
        gpt_key_row.addWidget(QtWidgets.QLabel("GPT API Key:"))
        self.le_gpt_api_key = QtWidgets.QLineEdit()
        self.le_gpt_api_key.setEchoMode(QtWidgets.QLineEdit.Password)
        self.le_gpt_api_key.setPlaceholderText("sk-xxxxx (or set GPT_SUBGOAL_API_KEY env)")
        env_key = os.environ.get("GPT_SUBGOAL_API_KEY", "")
        if env_key:
            self.le_gpt_api_key.setText(env_key)
        gpt_key_row.addWidget(self.le_gpt_api_key, 1)
        subgoal_source_layout.addLayout(gpt_key_row)

        gpt_prompt_row = QtWidgets.QHBoxLayout()
        gpt_prompt_row.addWidget(QtWidgets.QLabel("GPT Prompt:"))
        self.le_gpt_prompt = QtWidgets.QLineEdit()
        self.le_gpt_prompt.setPlaceholderText(
            "Custom prompt (leave empty for default). Use {task} as placeholder."
        )
        gpt_prompt_row.addWidget(self.le_gpt_prompt, 1)
        subgoal_source_layout.addLayout(gpt_prompt_row)

        ref_video_row = QtWidgets.QHBoxLayout()
        ref_video_row.addWidget(QtWidgets.QLabel("Video:"))
        self.le_ref_video_path = QtWidgets.QLineEdit()
        self.le_ref_video_path.setPlaceholderText("Select *_video_base.mp4")
        self.le_ref_video_path.setEnabled(False)
        ref_video_row.addWidget(self.le_ref_video_path, 1)
        self.btn_browse_ref_video = QtWidgets.QPushButton("Browse...")
        self.btn_browse_ref_video.setEnabled(False)
        self.btn_browse_ref_video.clicked.connect(self._on_browse_ref_video)
        ref_video_row.addWidget(self.btn_browse_ref_video)
        subgoal_source_layout.addLayout(ref_video_row)

        ref_jsonl_row = QtWidgets.QHBoxLayout()
        ref_jsonl_row.addWidget(QtWidgets.QLabel("Log:"))
        self.le_ref_jsonl_path = QtWidgets.QLineEdit()
        self.le_ref_jsonl_path.setPlaceholderText("Auto-detected from video path")
        self.le_ref_jsonl_path.setReadOnly(True)
        self.le_ref_jsonl_path.setStyleSheet("background:#2a2a2a; color:#aaa;")
        ref_jsonl_row.addWidget(self.le_ref_jsonl_path, 1)
        subgoal_source_layout.addLayout(ref_jsonl_row)

        ref_offset_row = QtWidgets.QHBoxLayout()
        ref_offset_row.addWidget(QtWidgets.QLabel("Video Sync Offset (s):"))
        self.sp_ref_video_offset = QtWidgets.QDoubleSpinBox()
        self.sp_ref_video_offset.setRange(0.0, 999.0)
        self.sp_ref_video_offset.setSingleStep(0.5)
        self.sp_ref_video_offset.setDecimals(1)
        self.sp_ref_video_offset.setValue(0.0)
        self.sp_ref_video_offset.setToolTip(
            "Seconds the video was recording before Step 1. "
            "Positive = shift lookup forward (later frames)."
        )
        self.sp_ref_video_offset.setEnabled(False)
        ref_offset_row.addWidget(self.sp_ref_video_offset)
        ref_offset_row.addWidget(QtWidgets.QLabel("  Control Hz:"))
        self.sp_ref_control_hz = QtWidgets.QDoubleSpinBox()
        self.sp_ref_control_hz.setRange(1.0, 200.0)
        self.sp_ref_control_hz.setSingleStep(5.0)
        self.sp_ref_control_hz.setDecimals(1)
        self.sp_ref_control_hz.setValue(30.0)
        self.sp_ref_control_hz.setToolTip(
            "Policy control frequency in Hz. Used when no JSON log is available."
        )
        self.sp_ref_control_hz.setEnabled(False)
        ref_offset_row.addWidget(self.sp_ref_control_hz)
        ref_offset_row.addStretch(1)
        subgoal_source_layout.addLayout(ref_offset_row)

        self.lbl_subgoal_source_note = QtWidgets.QLabel(
            "ForeAct uses the configured host/port above. "
            "GPT requires a valid API key. "
            "Reference Retrieval uses the trajectory cache and returns the matched frame + lookahead. "
            "Reference Video pre-extracts frames from a successful evaluation."
        )
        self.lbl_subgoal_source_note.setStyleSheet("color:#888;")
        subgoal_source_layout.addWidget(self.lbl_subgoal_source_note)
        left.addWidget(self.gb_subgoal_source)

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
        self.btn_debug_gripper = QtWidgets.QPushButton("Debug Gripper")
        self.btn_debug_gripper.setToolTip(
            "Pause inference, return to zero, then send the selected continuous value to both grippers."
        )
        self.btn_debug_gripper.clicked.connect(self._on_debug_gripper)
        btn_row.addWidget(self.btn_debug_gripper)
        self.sp_debug_gripper = QtWidgets.QDoubleSpinBox()
        self.sp_debug_gripper.setRange(0.0, 4.0)
        self.sp_debug_gripper.setDecimals(4)
        self.sp_debug_gripper.setSingleStep(0.01)
        self.sp_debug_gripper.setValue(0.06)
        self.sp_debug_gripper.setToolTip("Continuous gripper debug value sent after return-to-zero.")
        btn_row.addWidget(self.sp_debug_gripper)
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
        self.cb_doubao_auto = QtWidgets.QCheckBox("Auto-suggest (placeholder)")
        self.cb_doubao_auto.setEnabled(False)  # hook reserved; actual predictor is a TODO
        self.cb_doubao_auto.setToolTip(
            "Reserved hook: enable a future SubtaskPredictor to pick the\n"
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

        self.lbl_cam_title = QtWidgets.QLabel("cam_high (live)")
        right.addWidget(self.lbl_cam_title)
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
        self._sync_execution_controls_for_mode(self._runtime.mode)
        self._update_traj_source_controls()
        self._update_subgoal_source_controls()

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

    def _on_clear_cache(self) -> None:
        """Delete the checkpoint history cache and refresh the UI."""
        path = self._checkpoint_history_path
        if path.exists():
            try:
                path.unlink()
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not delete cache file %s: %s", path, e)
                return
        self._checkpoint_history.clear()
        self._last_checkpoint_by_mode.clear()
        self._refresh_policy_fields_for_mode(self._runtime.mode)
        self._log_info(f"Cache cleared: {path}")

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

    def _configure_video_recording(self) -> bool:
        output_dir = self._selected_output_dir()
        if output_dir is None:
            return False
        self._cfg.record_video = True
        self._cfg.video_dir = str(output_dir)
        self._cfg.video_name = self._checkpoint_video_stem()
        self._cfg.video_fps = max(1.0, float(self._cfg.max_hz))
        self._log_info(f"Artifacts will be saved to {output_dir}")
        return True

    def _default_checkpoint_for_mode(self, mode: str) -> str:
        preset = _POLICY_PRESETS.get(mode, _POLICY_PRESETS["basic"])
        return preset["dir"].format(step=int(self.sp_ckpt_step.value()))

    @staticmethod
    def _checkpoint_task_markers(text: str) -> set[str]:
        lowered = text.lower()
        markers = {
            name
            for name in (
                "three_object",
                "aloha_shape",
                "aloha_letter",
                "matching_holes",
                "matching holes",
                "shape",
                "aloha_object",
                "correct_object",
                "banana",
                "cube",
                "eggplant",
                "obstacle",
                "aloha_food",
                "food",
                "non-food",
                "non_food",
                "plate",
                "place_all_the_food_into_the_plate",
                "place_all_the_non-food_items_into_the_plate",
                "eai",
                "letter",
                "stick",
            )
            if name in lowered
        }
        if markers & {"three_object", "aloha_shape", "matching_holes", "matching holes", "shape"}:
            markers.update({"three_object", "aloha_shape", "matching_holes", "shape"})
        if markers & {"aloha_letter", "eai", "letter", "stick"}:
            markers.update({"aloha_letter", "eai", "letter", "stick"})
        if markers & {
            "aloha_food",
            "food",
            "non-food",
            "non_food",
            "plate",
            "place_all_the_food_into_the_plate",
            "place_all_the_non-food_items_into_the_plate",
        }:
            markers.update({"aloha_food", "food", "non-food", "non_food", "plate"})
        return markers

    @staticmethod
    def _local_checkpoint_exists(path: str) -> bool:
        if path.startswith("gs://"):
            return True
        candidate = pathlib.Path(path).expanduser()
        resolved = candidate if candidate.is_absolute() else _REPO_ROOT / candidate
        return resolved.exists()

    def _checkpoint_matches_mode(self, mode: str, path: str) -> bool:
        preset = _POLICY_PRESETS.get(mode, _POLICY_PRESETS["basic"])
        preset_markers = self._checkpoint_task_markers(
            f"{preset['config']} {preset['dir']}"
        )
        path_markers = self._checkpoint_task_markers(path)
        return not preset_markers or not path_markers or bool(preset_markers & path_markers)

    def _checkpoint_history_candidates_for_mode(self, mode: str) -> list[str]:
        items = []
        candidates = [
            self._last_checkpoint_by_mode.get(mode, ""),
            *self._checkpoint_history.get(mode, []),
        ]
        for value in candidates:
            value = value.strip()
            if (
                value
                and value not in items
                and self._checkpoint_matches_mode(mode, value)
                and self._local_checkpoint_exists(value)
            ):
                items.append(value)
        return items

    def _checkpoint_items_for_mode(self, mode: str) -> list[str]:
        default = self._default_checkpoint_for_mode(mode)
        items = []
        candidates = [*self._checkpoint_history_candidates_for_mode(mode), default]
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
                history_candidates = self._checkpoint_history_candidates_for_mode(mode)
                select = next(iter(history_candidates), "") or default
            elif (
                not current
                or current == previous_default
                or not self._checkpoint_matches_mode(mode, current)
                or not self._local_checkpoint_exists(current)
            ):
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

    def _resolve_output_dir(self, text: str) -> pathlib.Path:
        path = pathlib.Path(text).expanduser()
        if not path.is_absolute():
            path = pathlib.Path.cwd() / path
        return path

    def _output_browse_start_dir(self) -> pathlib.Path:
        text = self.le_output_dir.text().strip()
        if text:
            candidate = self._resolve_output_dir(text)
            if candidate.exists():
                return candidate if candidate.is_dir() else candidate.parent
            parent = candidate.parent
            if parent.exists():
                return parent
        return pathlib.Path.cwd()

    def _on_browse_output_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select output directory", str(self._output_browse_start_dir()))
        if not path:
            return
        self.le_output_dir.setText(path)

    @staticmethod
    def _default_traj_retrieval_cache_text() -> str:
        explicit = os.environ.get(_TRAJ_RETRIEVAL_CACHE_ENV, "").strip()
        if explicit:
            return explicit
        dataset = os.environ.get(_TRAJ_RETRIEVAL_DATASET_ENV, "").strip()
        if dataset:
            return str(default_cache_path(dataset))
        if _DEFAULT_TRAJ_RETRIEVAL_DATASET.exists():
            return str(default_cache_path(_DEFAULT_TRAJ_RETRIEVAL_DATASET))
        return ""

    def _trajectory_retrieval_cache_path(self) -> Optional[pathlib.Path]:
        text = self.le_traj_reference_cache.text().strip()
        if not text:
            return None
        path = pathlib.Path(text).expanduser()
        if not path.is_absolute():
            path = _REPO_ROOT / path
        return path

    def _invalidate_trajectory_retriever(self) -> None:
        self._traj_retriever = None
        self._traj_retriever_path = ""

    def _on_browse_traj_reference_cache(self) -> None:
        current = self._trajectory_retrieval_cache_path()
        start_dir = current.parent if current is not None and current.parent.exists() else _REPO_ROOT
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select trajectory retrieval cache",
            str(start_dir),
            "PyTorch cache (*.pt *.pth);;All files (*)",
        )
        if not path:
            return
        self.le_traj_reference_cache.setText(path)
        self._invalidate_trajectory_retriever()

    def _prepare_trajectory_retriever(self, *, warn_missing: bool = True) -> Optional[TrajectoryReferenceRetriever]:
        if getattr(self, "cb_traj_reference", None) is None or not self.cb_traj_reference.isChecked():
            return None
        path = self._trajectory_retrieval_cache_path()
        if path is None:
            return None
        path_text = str(path)
        if self._traj_retriever is not None and self._traj_retriever_path == path_text:
            return self._traj_retriever
        if not path.exists():
            if warn_missing and path_text not in self._traj_retriever_warned_paths:
                self._traj_retriever_warned_paths.add(path_text)
                self._bridge.error.emit(
                    "Trajectory reference cache not found. Build it with "
                    f"tools/trajectory/build_trajectory_retrieval_cache.py: {path}"
                )
            self._traj_retriever = None
            self._traj_retriever_path = ""
            return None
        try:
            retriever = TrajectoryReferenceRetriever(path)
        except Exception as e:
            if path_text not in self._traj_retriever_warned_paths:
                self._traj_retriever_warned_paths.add(path_text)
                self._bridge.error.emit(f"Could not load trajectory reference cache {path}: {e}")
            self._traj_retriever = None
            self._traj_retriever_path = ""
            return None
        self._traj_retriever = retriever
        self._traj_retriever_path = path_text
        self._bridge.info.emit(f"Loaded trajectory reference cache: {retriever.count} frames from {path}")
        return retriever

    def _suggest_reference_trajectory(self, image: np.ndarray):
        retriever = self._prepare_trajectory_retriever(warn_missing=True)
        if retriever is None:
            return None, {}
        try:
            result = retriever.suggest(image)
        except Exception as e:
            logger.warning("Trajectory reference retrieval failed: %s", e)
            self._bridge.error.emit(f"Trajectory reference retrieval failed: {e}")
            return None, {}
        if result is None:
            return None, {}
        pred = build_prediction_from_text(result.trajectory_text)
        if pred.is_empty():
            return None, {}
        metadata = {
            "episode_index": result.episode_index,
            "frame_index": result.frame_index,
            "score": result.score,
            "search_ms": result.search_ms,
            "cache_path": result.cache_path,
        }
        self._bridge.info.emit(
            "Reference trajectory suggested from "
            f"episode {result.episode_index:06d} frame {result.frame_index} "
            f"(score={result.score:.3f}, search={result.search_ms:.1f} ms)."
        )
        return pred, metadata

    def _selected_output_dir(self) -> Optional[pathlib.Path]:
        text = self.le_output_dir.text().strip()
        if not text:
            self._on_browse_output_dir()
            text = self.le_output_dir.text().strip()
            if not text:
                self._log_error("Start canceled: choose an output directory.")
                return None

        path = self._resolve_output_dir(text)
        try:
            path.mkdir(parents=True, exist_ok=True)
            path = path.resolve()
        except Exception as e:                       # noqa: BLE001
            self._log_error(f"Could not create output directory {path}: {e}")
            QtWidgets.QMessageBox.critical(
                self,
                "Output Directory Error",
                f"Could not create output directory:\n{path}\n\n{e}",
            )
            return None
        if not path.is_dir():
            self._log_error(f"Output path is not a directory: {path}")
            QtWidgets.QMessageBox.critical(
                self,
                "Output Directory Error",
                f"Output path is not a directory:\n{path}",
            )
            return None
        self.le_output_dir.setText(str(path))
        return path

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

    @staticmethod
    def _canonical_mode(mode: str) -> str:
        return (mode or "").lower().replace("-", "_")

    def _sync_execution_controls_for_mode(self, mode: str) -> None:
        requires_blocking = self._canonical_mode(mode) in _BLOCKING_ONLY_MODES
        if requires_blocking:
            self._blocking = True
            old_block = self.rb_block.blockSignals(True)
            old_nonblock = self.rb_nonblock.blockSignals(True)
            try:
                self.rb_block.setChecked(True)
                self.rb_nonblock.setChecked(False)
            finally:
                self.rb_block.blockSignals(old_block)
                self.rb_nonblock.blockSignals(old_nonblock)

        self.rb_nonblock.setEnabled(not requires_blocking)
        self.rb_block.setEnabled(True)
        note = "Triple-CoT uses the semi-block trajectory pipeline." if requires_blocking else ""
        self.rb_nonblock.setToolTip(note)
        self.rb_block.setToolTip(note)

    def _on_mode_changed(self, mode: str) -> None:
        mode = self._canonical_mode(mode)
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
        self._sync_execution_controls_for_mode(mode)
        self._refresh_policy_fields_for_mode(mode, prefer_history=True)
        self._update_traj_source_controls()
        self._update_subgoal_source_controls()
        # Sensible default step interval per mode.
        defaults = {"traj": 60, "subgoal": 60, "subtask": 60, "triple_cot": 60}
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
        mode = self._canonical_mode(mode)
        kwargs = dict(
            foreact_host=self.le_fhost.text().strip(),
            foreact_port=int(self.le_fport.text()),
            blocking=True if mode in _BLOCKING_ONLY_MODES else self._blocking,
        )
        n = int(self.sp_interval.value())
        if mode == "traj":
            kwargs["traj_step_interval"] = n
            kwargs["manual_traj_provider"] = self._request_manual_trajectory
            kwargs["manual_traj_override"] = self._manual_traj_override_enabled()
        elif mode == "triple_cot":
            kwargs["traj_step_interval"] = n
            kwargs["manual_traj_provider"] = self._request_manual_trajectory
            kwargs["manual_traj_override"] = self._manual_traj_override_enabled()
            if self._subgoal_use_reference():
                kwargs["foreact_client"] = self._build_reference_video_client()
            elif self._subgoal_use_retrieval():
                kwargs["foreact_client"] = self._build_reference_retrieval_subgoal_client()
            elif self._subgoal_use_gpt():
                kwargs["foreact_client"] = self._build_gpt_subgoal_client()
        elif mode == "subgoal":
            kwargs["subgoal_step_interval"] = n
            if self._subgoal_use_reference():
                kwargs["foreact_client"] = self._build_reference_video_client()
            elif self._subgoal_use_retrieval():
                kwargs["foreact_client"] = self._build_reference_retrieval_subgoal_client()
            elif self._subgoal_use_gpt():
                kwargs["foreact_client"] = self._build_gpt_subgoal_client()
        elif mode == "subtask":
            kwargs["subtask_step_interval"] = n
        return _modes.make_handler(mode, **kwargs)

    def _subgoal_use_gpt(self) -> bool:
        return bool(getattr(self, "rb_subgoal_gpt", None) and self.rb_subgoal_gpt.isChecked())

    def _subgoal_use_reference(self) -> bool:
        return bool(getattr(self, "rb_subgoal_reference", None) and self.rb_subgoal_reference.isChecked())

    def _subgoal_use_retrieval(self) -> bool:
        return bool(getattr(self, "rb_subgoal_retrieval", None) and self.rb_subgoal_retrieval.isChecked())

    def _build_gpt_subgoal_client(self):
        from .gpt_subgoal_client import GptSubgoalClient

        api_key = self.le_gpt_api_key.text().strip()
        if not api_key:
            api_key = os.environ.get("GPT_SUBGOAL_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GPT subgoal generation requires an API key. "
                "Set it in the GUI or via GPT_SUBGOAL_API_KEY env variable."
            )
        custom_prompt = self.le_gpt_prompt.text().strip() or None
        client = GptSubgoalClient(api_key=api_key)
        if custom_prompt:
            client._custom_prompt = custom_prompt
        return client

    def _build_reference_retrieval_subgoal_client(self):
        from .trajectory_retrieval import TrajectoryReferenceSubgoalClient

        path = self._trajectory_retrieval_cache_path()
        if path is None:
            raise ValueError(
                "Reference Retrieval subgoal requires a trajectory retrieval cache. "
                "Use the Trajectory annotation cache field or AGENTIC_OPENPI_TRAJ_RETRIEVAL_CACHE."
            )
        if not path.exists():
            raise FileNotFoundError(
                f"Reference Retrieval subgoal cache not found: {path}. "
                "Build it with tools/trajectory/build_trajectory_retrieval_cache.py."
            )
        lookahead = int(self.sp_subgoal_retrieval_lookahead.value())
        return TrajectoryReferenceSubgoalClient(path, lookahead_frames=lookahead)

    def _build_reference_video_client(self):
        from .reference_video_provider import ReferenceVideoProvider

        video_path = self.le_ref_video_path.text().strip()
        jsonl_path = self.le_ref_jsonl_path.text().strip() or None
        if not video_path:
            raise ValueError("Reference Video requires a video file.")
        step_interval = int(self.sp_interval.value())
        video_offset_s = float(self.sp_ref_video_offset.value())
        control_hz = float(self.sp_ref_control_hz.value())
        return ReferenceVideoProvider(
            video_path, jsonl_path, step_interval,
            video_offset_s=video_offset_s, control_hz=control_hz,
        )

    def _on_browse_ref_video(self) -> None:
        start_dir = str(_REPO_ROOT)
        current = self.le_ref_video_path.text().strip()
        if current:
            p = pathlib.Path(current)
            if p.parent.exists():
                start_dir = str(p.parent)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select reference video", start_dir, "Video (*.mp4);;All files (*)"
        )
        if not path:
            return
        self.le_ref_video_path.setText(path)
        self._auto_detect_ref_jsonl(path)

    def _auto_detect_ref_jsonl(self, video_path: str) -> None:
        p = pathlib.Path(video_path)
        stem = p.stem
        for suffix in ("_video_base", "_video_left_wrist", "_video_right_wrist"):
            stem = stem.replace(suffix, "")
        candidate = p.parent / f"{stem}_log.jsonl"
        if candidate.exists():
            self.le_ref_jsonl_path.setText(str(candidate))
        else:
            self.le_ref_jsonl_path.setText("")

    def _manual_traj_override_enabled(self) -> bool:
        return bool(getattr(self, "rb_traj_manual", None) and self.rb_traj_manual.isChecked())

    def _traj_annotation_arm(self) -> str:
        combo = getattr(self, "cb_traj_annotate_arm", None)
        if combo is None:
            return "right"
        value = combo.currentData()
        return value if value in _TRAJ_ANNOTATION_ARMS else "right"

    def _traj_inactive_default_pixel(self) -> tuple[float, float]:
        spin_x = getattr(self, "sp_traj_default_x", None)
        spin_y = getattr(self, "sp_traj_default_y", None)
        if spin_x is None or spin_y is None:
            return _TRAJ_MODE_DEFAULT_INACTIVE_PIXEL
        return float(spin_x.value()), float(spin_y.value())

    def _request_manual_trajectory(
        self,
        image: np.ndarray,
        task: str,
        cancel_check=None,
        *,
        request_started=None,
    ):
        with self._runtime._lock:
            mode = self._canonical_mode(self._runtime.mode)
        annotate_arm = self._traj_annotation_arm() if mode in _TRAJECTORY_ARM_CONFIG_MODES else "both"
        inactive_default = self._traj_inactive_default_pixel()
        reference_prediction, reference_metadata = self._suggest_reference_trajectory(image)
        req = _ManualTrajectoryRequest(
            image.copy(),
            task,
            annotate_arm=annotate_arm,
            default_left_pixel=inactive_default if annotate_arm == "right" else None,
            default_right_pixel=inactive_default if annotate_arm == "left" else None,
            reference_prediction=reference_prediction,
            reference_metadata=reference_metadata,
        )
        self._bridge.manual_trajectory.emit(req)
        if request_started is not None:
            request_started()
        while not req.done.wait(timeout=0.1):
            if cancel_check is not None and cancel_check():
                req.cancel_requested = True
                self._bridge.manual_trajectory_cancel.emit(req)
                req.cancel()
                return None
        if req.cancel_requested:
            return None
        return req.prediction

    @QtCore.pyqtSlot(object)
    def _on_manual_trajectory_requested(self, req: _ManualTrajectoryRequest) -> None:
        if req.cancel_requested:
            req.cancel()
            return

        dialog = ManualTrajectoryDialog(
            req.image,
            req.task,
            self,
            annotate_arm=req.annotate_arm,
            default_left_pixel=req.default_left_pixel,
            default_right_pixel=req.default_right_pixel,
            reference_prediction=req.reference_prediction,
            reference_metadata=req.reference_metadata,
        )
        self._manual_traj_request = req
        self._manual_traj_dialog = dialog
        if req.annotate_arm == "right":
            x, y = req.default_left_pixel or _TRAJ_MODE_DEFAULT_INACTIVE_PIXEL
            self._log_info(
                "Manual trajectory annotation requested: right arm only; "
                f"left arm defaults to loc ({int(round(x))}, {int(round(y))})."
            )
        elif req.annotate_arm == "left":
            x, y = req.default_right_pixel or _TRAJ_MODE_DEFAULT_INACTIVE_PIXEL
            self._log_info(
                "Manual trajectory annotation requested: left arm only; "
                f"right arm defaults to loc ({int(round(x))}, {int(round(y))})."
            )
        else:
            self._log_info("Manual trajectory annotation requested.")
        if req.reference_prediction is not None and not req.reference_prediction.is_empty():
            meta = req.reference_metadata
            ep = meta.get("episode_index")
            frame = meta.get("frame_index")
            score = meta.get("score")
            search_ms = meta.get("search_ms")
            detail = f"episode={ep}, frame={frame}"
            if score is not None:
                detail += f", score={float(score):.3f}"
            if search_ms is not None:
                detail += f", search={float(search_ms):.1f} ms"
            self._log_info(f"Reference trajectory pre-filled ({detail}).")
        try:
            result = dialog.exec_()
            if req.cancel_requested:
                req.cancel()
            elif result == QtWidgets.QDialog.Accepted and dialog.prediction is not None:
                req.set_prediction(dialog.prediction)
                self._log_info("Manual trajectory annotation finished.")
            elif dialog.aborted:
                req.cancel()
                self._handle_manual_trajectory_abort()
            else:
                req.cancel()
        finally:
            if self._manual_traj_request is req:
                self._manual_traj_request = None
                self._manual_traj_dialog = None

    @QtCore.pyqtSlot(object)
    def _on_manual_trajectory_cancel(self, req: _ManualTrajectoryRequest) -> None:
        if self._manual_traj_request is not req or self._manual_traj_dialog is None:
            return
        self._manual_traj_dialog.cancel_from_runner()

    def _handle_manual_trajectory_abort(self) -> None:
        runner = self._runner
        if runner is None:
            return
        runner.pause()
        runner.request_reset()
        self.btn_pause.setText("▶ Resume")
        self._log_error("Manual trajectory emergency stop: annotation discarded, inference paused, RTZ queued.")

    def _on_submode_changed(self, *_) -> None:
        if self._runtime.mode in _BLOCKING_ONLY_MODES:
            self._sync_execution_controls_for_mode(self._runtime.mode)
            if self._handler is not None:
                self._handler.set_blocking(True)
            self._update_traj_source_controls()
            return
        self._blocking = self.rb_block.isChecked()
        if self._handler is not None:
            self._handler.set_blocking(self._blocking)
        self._update_traj_source_controls()
        self._log_info(f"sub-mode -> {'blocking' if self._blocking else 'non-blocking'}")

    def _on_manual_traj_override_changed(self, enabled: bool) -> None:
        manual_enabled = bool(enabled)
        with self._runtime._lock:
            self._runtime.manual_traj_override = manual_enabled
        if self._handler is not None:
            self._handler.set_manual_trajectory_provider(self._request_manual_trajectory)
            self._handler.set_manual_trajectory_override(manual_enabled)
        state = "enabled" if manual_enabled else "disabled"
        self._log_info(f"Manual trajectory override {state}.")

    def _update_traj_source_controls(self) -> None:
        mode = self._canonical_mode(self._runtime.mode)
        enabled = mode in _TRAJECTORY_SOURCE_MODES
        force_manual = mode in _TRAJECTORY_SOURCE_MODES
        if force_manual:
            old_manual = self.rb_traj_manual.blockSignals(True)
            old_doubao = self.rb_traj_doubao.blockSignals(True)
            try:
                self.rb_traj_manual.setChecked(True)
                self.rb_traj_doubao.setChecked(False)
            finally:
                self.rb_traj_manual.blockSignals(old_manual)
                self.rb_traj_doubao.blockSignals(old_doubao)
            with self._runtime._lock:
                self._runtime.manual_traj_override = True
            if self._handler is not None:
                self._handler.set_manual_trajectory_provider(self._request_manual_trajectory)
                self._handler.set_manual_trajectory_override(True)

        self.gb_traj_source.setVisible(enabled)
        self.gb_traj_source.setEnabled(enabled)
        self.rb_traj_doubao.setVisible(False)
        self.rb_traj_doubao.setEnabled(False)
        self.rb_traj_manual.setVisible(enabled)
        self.rb_traj_manual.setEnabled(enabled)
        self.lbl_traj_source_note.setVisible(enabled)
        self.lbl_traj_source_note.setEnabled(enabled)
        arm_config_enabled = enabled and mode in _TRAJECTORY_ARM_CONFIG_MODES
        for widget in (
            self.lbl_traj_annotate,
            self.cb_traj_annotate_arm,
            self.lbl_traj_inactive_default,
            self.sp_traj_default_x,
            self.sp_traj_default_y,
        ):
            widget.setVisible(arm_config_enabled)
            widget.setEnabled(arm_config_enabled)
        if mode == "triple_cot":
            self.lbl_traj_source_note.setText(
                "Triple-CoT uses manual trajectory annotation; choose which arm to annotate below. "
                "No VOLCENKEY/ARK_API_KEY is required."
            )
        else:
            self.lbl_traj_source_note.setText(
                "Trajectory annotation uses the manual GUI; choose which arm to annotate below."
            )

    def _update_subgoal_source_controls(self) -> None:
        mode = self._canonical_mode(self._runtime.mode)
        visible = mode in ("subgoal", "triple_cot")
        self.gb_subgoal_source.setVisible(visible)
        self.gb_subgoal_source.setEnabled(visible)
        if visible:
            gpt_selected = self.rb_subgoal_gpt.isChecked()
            retrieval_selected = self._subgoal_use_retrieval()
            ref_selected = self._subgoal_use_reference()
            self.le_gpt_api_key.setEnabled(gpt_selected)
            self.le_gpt_prompt.setEnabled(gpt_selected)
            self.sp_subgoal_retrieval_lookahead.setEnabled(retrieval_selected)
            self.le_ref_video_path.setEnabled(ref_selected)
            self.btn_browse_ref_video.setEnabled(ref_selected)
            self.sp_ref_video_offset.setEnabled(ref_selected)
            self.sp_ref_control_hz.setEnabled(ref_selected)

    def _on_subgoal_source_changed(self, _toggled: bool) -> None:
        gpt_selected = self.rb_subgoal_gpt.isChecked()
        retrieval_selected = self._subgoal_use_retrieval()
        ref_selected = self._subgoal_use_reference()
        self.le_gpt_api_key.setEnabled(gpt_selected)
        self.le_gpt_prompt.setEnabled(gpt_selected)
        self.sp_subgoal_retrieval_lookahead.setEnabled(retrieval_selected)
        self.le_ref_video_path.setEnabled(ref_selected)
        self.btn_browse_ref_video.setEnabled(ref_selected)
        self.sp_ref_video_offset.setEnabled(ref_selected)
        self.sp_ref_control_hz.setEnabled(ref_selected)
        if ref_selected:
            source = "Reference Video"
        elif retrieval_selected:
            source = "Reference Retrieval"
        elif gpt_selected:
            source = "GPT Generation"
        else:
            source = "ForeAct Server"
        self._log_info(f"Subgoal source -> {source}")
        if self._runner is not None:
            try:
                handler = self._build_handler(self._runtime.mode)
            except Exception as e:
                self._log_error(f"Subgoal source switch failed: {e}")
                return
            self._handler = handler
            self._runner.set_handler(handler)

    def _on_interval_changed(self, n: int) -> None:
        self._step_interval = int(n)
        if self._handler is not None:
            self._handler.set_step_interval(n)

    def _set_connect_button_state(self, state: str) -> None:
        if state == "connecting":
            self.btn_connect.setEnabled(False)
            self.btn_connect.setText("Connecting...")
        elif state == "connected":
            self.btn_connect.setEnabled(False)
            self.btn_connect.setText("Connected")
        elif state == "retry":
            self.btn_connect.setEnabled(True)
            self.btn_connect.setText("Retry Connect")
        else:
            self.btn_connect.setEnabled(True)
            self.btn_connect.setText("Connect")

    def _set_subtask(self, key: int) -> None:
        with self._runtime._lock:
            labels = dict(self._runtime.subtask_labels)
        if key not in labels:
            self._log_error(f"key {key} not in subtask_labels")
            return
        self._runtime.set_subtask_key(key)
        label = labels[key]
        if self._runtime.mode in _SUBTASK_PROMPT_MODES and self._runner is not None:
            self._runner.request_replan()
        self._log_info(f"subtask key={key} ({label})")

    def _on_connect(self) -> None:
        if self._runner is not None:
            self._log_info("Already connected.")
            return
        if self._connecting:
            self._log_info("Connection already in progress.")
            return
        try:
            self._on_task_edited()
            self._cfg.host = self.le_host.text().strip() or "127.0.0.1"
            self._cfg.port = int(self.le_port.text())
            self._cfg.action_horizon = int(self.le_chunk.text())
            self._cfg.max_steps = int(self.le_max.text())
            self._cfg.topic_wait_timeout = float(self.sp_ros_timeout.value())
            if self.cb_local_policy.isChecked():
                self._start_policy_server_if_needed()
                self._cfg.host = "127.0.0.1"
                self.le_host.setText(self._cfg.host)
            self._handler = self._build_handler(self._runtime.mode)
        except Exception as e:                           # noqa: BLE001
            logger.exception("connect setup failed")
            self._log_error(f"connect setup failed: {e}")
            self._start_after_connect = False
            self._runner = None
            self._handler = None
            self._set_connect_button_state("retry")
            return

        runner = EvalRunner(self._cfg, self._runtime, self._handler, on_event=self._on_event)
        self._connecting = True
        self._connect_generation += 1
        generation = self._connect_generation
        self._set_connect_button_state("connecting")
        self._log_info(
            f"Connecting in background; ROS topic wait timeout={self._cfg.topic_wait_timeout:.0f}s."
        )

        def _connect_worker() -> None:
            try:
                runner.connect()
            except Exception as e:                       # noqa: BLE001
                logger.exception("connect failed")
                try:
                    runner.stop()
                except Exception as stop_err:            # noqa: BLE001
                    logger.warning("cleanup after connect failure failed: %s", stop_err)
                self._bridge.connect_failed.emit(str(e), generation)
            else:
                self._bridge.connect_succeeded.emit(runner, generation)

        self._connect_thread = threading.Thread(
            target=_connect_worker,
            daemon=True,
            name="eval-connect",
        )
        self._connect_thread.start()

    @QtCore.pyqtSlot(object, int)
    def _on_connect_succeeded(self, runner: EvalRunner, generation: int) -> None:
        if generation != self._connect_generation:
            try:
                runner.stop()
            except Exception as e:                       # noqa: BLE001
                logger.warning("stale runner cleanup failed: %s", e)
            return
        self._runner = runner
        self._connecting = False
        self._connect_thread = None
        self._set_connect_button_state("connected")
        self._log_info("Connected.")
        if self._start_after_connect:
            self._start_after_connect = False
            self._start_runner()

    @QtCore.pyqtSlot(str, int)
    def _on_connect_failed(self, message: str, generation: int) -> None:
        if generation != self._connect_generation:
            return
        self._runner = None
        self._handler = None
        self._connecting = False
        self._connect_thread = None
        self._start_after_connect = False
        self._set_connect_button_state("retry")
        self._log_error(f"connect failed: {message}")

    def _start_runner(self) -> None:
        if self._runner is None:
            return
        self._runner.start()
        self._log_info("Run loop started.")

    def _on_start(self) -> None:
        if not self._configure_video_recording():
            return
        if self._runtime.mode in _TRAJECTORY_SOURCE_MODES:
            self._prepare_trajectory_retriever(warn_missing=True)
        if self._runner is None:
            self._start_after_connect = True
            self._on_connect()
            if self._connecting:
                self._log_info("Start queued; run loop will begin after connection succeeds.")
            return
        self._start_runner()

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
        self._set_connect_button_state("idle")
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

    def _on_debug_gripper(self) -> None:
        if self._runner is None:
            self._log_error("Not connected — cannot debug gripper. Click Connect first.")
            return
        value = float(self.sp_debug_gripper.value())
        self._runner.request_debug_gripper(value)
        self.btn_pause.setText("▶ Resume")
        self._log_info(f"Debug gripper requested with value {value:.4f}.")

    def _on_dump(self) -> None:
        if self._runner is None:
            self._log_error("Not connected — cannot dump inputs.")
            return
        self._runner.request_dump()

    # ------------------------------------------------------------------ #
    def _refresh_ui(self) -> None:
        snap = self._runtime.snapshot()
        if self._runner is not None:
            obs = self._runner.latest_obs()
            if obs is not None:
                images = obs.get("images", {})
                cam_high = images.get("cam_high")
                traj_image = snap.get("traj_image")
                if snap.get("mode") in _TRAJECTORY_SOURCE_MODES and traj_image is not None:
                    cam_high = traj_image
                    self.lbl_cam_title.setText("cam_high (trajectory overlay)")
                else:
                    self.lbl_cam_title.setText("cam_high (live)")
                self.lbl_cam.setPixmap(_np_to_qpixmap(cam_high, 480, 360))
                self.lbl_left.setPixmap(_np_to_qpixmap(images.get("cam_left_wrist"), 220, 165))
                self.lbl_right.setPixmap(_np_to_qpixmap(images.get("cam_right_wrist"), 220, 165))

        self.lbl_prompt.setText(snap["prompt"] or "(empty)")
        self.lbl_subtask.setText(snap["subtask_label"] or "-")
        traj = snap["traj_text"] or "-"
        if len(traj) > 220:
            traj = traj[:220] + "..."
        self.lbl_traj.setText(traj)
        sg = snap["subgoal_image"]
        if sg is not None:
            self.lbl_subgoal.setPixmap(_np_to_qpixmap(sg, 480, 360))

        self.gb_subtask.setEnabled(snap["mode"] in _SUBTASK_PROMPT_MODES)
        active = f"active: {snap['subtask_key']}"
        if snap.get("waiting_for_subtask_input"):
            active += " (waiting)"
        self.lbl_active_key.setText(active)

    # ------------------------------------------------------------------ #
    def eventFilter(self, obj, event) -> bool:                          # noqa: N802
        if (
            event.type() == QtCore.QEvent.KeyPress
            and self._handle_global_subtask_key(obj, event)
        ):
            return True
        return super().eventFilter(obj, event)

    def _handle_global_subtask_key(self, obj, event: QtGui.QKeyEvent) -> bool:
        if self._runtime.mode not in _SUBTASK_PROMPT_MODES:
            return False
        if event.isAutoRepeat():
            return False
        if event.modifiers() & (
            QtCore.Qt.ControlModifier | QtCore.Qt.AltModifier | QtCore.Qt.MetaModifier
        ):
            return False

        text = event.text()
        if len(text) != 1 or not text.isdigit() or text == "0":
            return False

        widget = obj if isinstance(obj, QtWidgets.QWidget) else QtWidgets.QApplication.focusWidget()
        if self._runner is None and self._subtask_shortcut_should_ignore(widget):
            return False

        self._set_subtask(int(text))
        event.accept()
        return True

    def _subtask_shortcut_should_ignore(self, widget: Optional[QtWidgets.QWidget]) -> bool:
        while widget is not None:
            if isinstance(widget, QtWidgets.QLineEdit) and not widget.isReadOnly():
                return True
            if isinstance(widget, (QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit)) and not widget.isReadOnly():
                return True
            if isinstance(widget, QtWidgets.QAbstractSpinBox):
                return True
            if isinstance(widget, QtWidgets.QComboBox) and widget.isEditable():
                return True
            widget = widget.parentWidget()
        return False

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:        # noqa: N802
        k = event.key()
        text = event.text()
        if self._runtime.mode in _SUBTASK_PROMPT_MODES and text.isdigit() and text != "0":
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
            self._connect_generation += 1
            self._connecting = False
            self._start_after_connect = False
            app = QtWidgets.QApplication.instance()
            if app is not None:
                app.removeEventFilter(self)
            if self._runner is not None:
                self._runner.stop()
            self._stop_policy_server()
        finally:
            super().closeEvent(event)


# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Aloha eval GUI")
    p.add_argument(
        "--mode",
        default="subtask",
        choices=["basic", "traj", "subtask", "triple_cot", "triple-cot", "subgoal"],
    )
    p.add_argument("--task", default=_modes.DEFAULT_TASK_PROMPT)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--action_horizon", type=int, default=25)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument(
        "--topic_wait_timeout",
        "--topic-wait-timeout",
        dest="topic_wait_timeout",
        type=float,
        default=120.0,
    )
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
        topic_wait_timeout=args.topic_wait_timeout,
        dry_run=args.dry_run,
        dump_dir=args.dump_dir,
    )
    runtime = _modes.RuntimeState(mode=EvalGUI._canonical_mode(args.mode), task=args.task)

    app = QtWidgets.QApplication(sys.argv)
    win = EvalGUI(cfg, runtime)
    win.resize(1280, 900)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
