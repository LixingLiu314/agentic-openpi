"""Small PyQt5 Piper console. Images and model outputs are display-only."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import uuid

import numpy as np
from PyQt5 import QtCore, QtGui, QtNetwork, QtWidgets

from piper_gui_support import (
    DEFAULT_CHECKPOINT, Settings, alive, checkpoint_info, client_command,
    discover_checkpoints, find_server, launch_server, process_record,
    resolve_checkpoint, server_metadata, stop_server, validate_server,
)


ROOT = Path(__file__).resolve().parents[1]
CAMERAS = {"cam_high": ("前视相机", "/camera_f/color/image_raw"),
           "cam_left_wrist": ("左腕相机", "/camera_l/color/image_raw"),
           "cam_right_wrist": ("右腕相机", "/camera_r/color/image_raw")}
EXTRA_INFO_FIELDS = [("trajectory", "轨迹预测"), ("subgoal", "子目标"), ("completion", "完成状态")]
PHASES = {"PREPARING": "准备传感器", "ENABLING": "确认使能", "RESET": "复位中",
          "INFERENCE": "推理中", "CHECK_ONLY": "检查反馈", "OBSERVE_ONLY": "观察相机", "STOPPED": "已结束"}
STYLE = """
QMainWindow, QWidget#root { background: #10171e; color: #e8eef4; }
QWidget { font-size: 13px; color: #e8eef4; }
QFrame#panel, QGroupBox { background: #19232d; border: 1px solid #2b3947; border-radius: 9px; }
QGroupBox { margin-top: 18px; padding: 12px 10px 10px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 13px; top: 0px; }
QLabel#muted { color: #96aabc; font-size: 12px; }
QLabel#title { font-size: 23px; font-weight: 650; }
QLabel#prediction { font-size: 23px; font-weight: 550; color: #77e0c4; }
QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox { background: #101922; border: 1px solid #3b4e60;
 border-radius: 5px; padding: 5px; selection-background-color: #246756; }
QComboBox QAbstractItemView { background: #18252f; selection-background-color: #246756; }
QPushButton { background: #283947; border: 1px solid #3c5266; border-radius: 6px; padding: 7px 10px; }
QPushButton:hover { background: #354b5e; }
QPushButton#primary { background: #167e67; border-color: #26aa8c; font-weight: 600; }
QPushButton#stop { background: #813b43; border-color: #af535d; font-weight: 600; }
QPushButton:disabled { background: #202b34; color: #647789; border-color: #293744; }
QProgressBar { background: #111b24; border: 0; border-radius: 4px; min-height: 15px; text-align: center; }
QProgressBar::chunk { background: #279b80; border-radius: 4px; }
QScrollArea { background: transparent; border: none; }
QScrollBar:vertical { background:#10171e; width:9px; margin:0; }
QScrollBar::handle:vertical { background:#3b4e60; border-radius:4px; min-height:24px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background:none; }
QCheckBox { spacing: 7px; }
QCheckBox::indicator { width: 16px; height: 16px; }
QStatusBar { color: #9bb0c0; }
"""


def label(text="", name=None):
    widget = QtWidgets.QLabel(text)
    widget.setWordWrap(True)
    if name:
        widget.setObjectName(name)
    return widget


class CameraFeed:
    """Independent image subscriptions keep the GUI useful before robot preflight."""
    def __init__(self):
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image
        self.lock, self.frames, self.errors = threading.Lock(), {}, {}
        self.bridge = CvBridge()
        self.subscribers = [rospy.Subscriber(topic, Image, self._callback, callback_args=name,
                                            queue_size=1, buff_size=2**23, tcp_nodelay=True)
                            for name, (_, topic) in CAMERAS.items()]

    def _callback(self, message, name):
        try:
            rgb = self.bridge.imgmsg_to_cv2(message, "rgb8")
            with self.lock:
                self.frames[name] = (np.ascontiguousarray(rgb), time.monotonic(), message.header.stamp.to_sec())
                self.errors.pop(name, None)
        except Exception as error:
            self.errors[name] = str(error)

    def snapshot(self):
        with self.lock:
            return dict(self.frames)

    def close(self):
        for subscriber in self.subscribers:
            subscriber.unregister()


class CameraCard(QtWidgets.QFrame):
    def __init__(self, title):
        super().__init__()
        self.setObjectName("panel")
        layout = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(label(title))
        self.age = label("等待画面", "muted")
        self.age.setAlignment(QtCore.Qt.AlignRight)
        row.addWidget(self.age)
        layout.addLayout(row)
        self.canvas = label("等待相机图像…", "muted")
        self.canvas.setAlignment(QtCore.Qt.AlignCenter)
        self.canvas.setMinimumSize(180, 145)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
        self.canvas.setStyleSheet("background:#091018;border-radius:5px;")
        layout.addWidget(self.canvas, 1)
        self.source, self.stamp = None, None

    def update_frame(self, frame):
        rgb, arrival, _ = frame
        age = time.monotonic() - arrival
        self.age.setText(("已过期 · " if age > 1 else "实时 · ") + "%.2f s" % age)
        self.age.setStyleSheet("color:" + ("#f4ab75" if age > 1 else "#96aabc"))
        if self.stamp != arrival:
            height, width, _ = rgb.shape
            self.source = QtGui.QImage(rgb.data, width, height, rgb.strides[0], QtGui.QImage.Format_RGB888).copy()
            self.stamp = arrival
        if self.source:
            self.canvas.setPixmap(QtGui.QPixmap.fromImage(self.source).scaled(
                self.canvas.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))


class Events(QtCore.QObject):
    result = QtCore.pyqtSignal(str, object)
    error = QtCore.pyqtSignal(str, str)
    line = QtCore.pyqtSignal(str)


class PiperWindow(QtWidgets.QMainWindow):
    def __init__(self, root=ROOT, settings_path=None, camera=None, initial=None):
        super().__init__()
        self.root = Path(root)
        self.settings = Settings(settings_path or Path.home()/".config/agentic-openpi/piper_gui.json")
        self.camera = camera
        self.client = self.client_identity = self.output = None
        self.server_process = self.server_record = self.metadata = None
        self.model_key, self.loading, self.closing = None, False, False
        self.pending_start = False
        self.token, self.latest_prediction, self.last_event = "", {}, 0.0
        self.last_phase, self.run_config, self.run_kind = "", {}, ""
        self.events = Events()
        self.events.result.connect(self._job_result)
        self.events.error.connect(self._job_error)
        self.events.line.connect(self._log)
        self.udp = QtNetwork.QUdpSocket(self)
        if not self.udp.bind(QtNetwork.QHostAddress.LocalHost, 0):
            raise RuntimeError("Cannot bind GUI telemetry socket")
        self.udp.readyRead.connect(self._read_telemetry)
        self.setWindowTitle("Piper · 推理控制台")
        self.resize(1480, 930)
        self.setMinimumSize(1120, 760)
        self._build_ui()
        self._restore(initial or {})
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(50)
        self.stop_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Esc"), self)
        self.stop_shortcut.activated.connect(self.stop_run)
        self._log("相机实时显示。选择任务和 checkpoint 后，点击开始推理执行动作。复位过程不记录。")

    def _build_ui(self):
        container = QtWidgets.QWidget()
        container.setObjectName("root")
        self.setCentralWidget(container)
        outer = QtWidgets.QVBoxLayout(container)
        outer.setContentsMargins(18, 14, 18, 12)
        header = QtWidgets.QHBoxLayout()
        title = label("PIPER  /  推理控制台", "title")
        title.setFixedHeight(38)
        header.addWidget(title)
        header.addStretch()
        self.phase = label("待机")
        self.phase.setStyleSheet("color:#77e0c4; font-size:16px;")
        header.addWidget(self.phase)
        outer.addLayout(header)
        body = QtWidgets.QHBoxLayout()
        body.setSpacing(16)
        outer.addLayout(body, 1)
        left = QtWidgets.QWidget()
        left.setObjectName("root")
        left.setFixedWidth(350)
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.viewport().setStyleSheet("background:#10171e;")
        controls = QtWidgets.QWidget()
        controls.setObjectName("root")
        control_layout = QtWidgets.QVBoxLayout(controls)
        control_layout.setContentsMargins(0, 0, 5, 0)
        scroll.setWidget(controls)
        left_layout.addWidget(scroll, 1)
        body.addWidget(left)
        self.edit_widgets = []
        task_box = QtWidgets.QGroupBox("任务")
        layout = QtWidgets.QVBoxLayout(task_box)
        self.task = QtWidgets.QComboBox()
        self.task.setEditable(True)
        self.task.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.task.currentIndexChanged.connect(self._select_task)
        layout.addWidget(self.task)
        self.prompt = QtWidgets.QPlainTextEdit()
        self.prompt.setPlaceholderText("输入发送给模型的任务指令…")
        self.prompt.setFixedHeight(85)
        layout.addWidget(self.prompt)
        self.save_task = QtWidgets.QPushButton("保存任务 / 保存修改")
        self.save_task.clicked.connect(self._save_task)
        layout.addWidget(self.save_task)
        control_layout.addWidget(task_box)
        self.edit_widgets += [task_box]

        model_box = QtWidgets.QGroupBox("模型")
        layout = QtWidgets.QVBoxLayout(model_box)
        self.checkpoint = QtWidgets.QComboBox()
        self.checkpoint.setEditable(True)
        self.checkpoint.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.checkpoint.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.checkpoint.setMinimumContentsLength(15)
        layout.addWidget(self.checkpoint)
        self.checkpoint.currentTextChanged.connect(self._checkpoint_changed)
        row = QtWidgets.QHBoxLayout()
        browse = QtWidgets.QPushButton("浏览目录…")
        browse.clicked.connect(self._browse_checkpoint)
        refresh = QtWidgets.QPushButton("刷新列表")
        refresh.clicked.connect(self._refresh_checkpoints)
        row.addWidget(browse)
        row.addWidget(refresh)
        layout.addLayout(row)
        self.checkpoint_note = label("完整 M3 / R1 增量 checkpoint", "muted")
        layout.addWidget(self.checkpoint_note)
        endpoint = QtWidgets.QHBoxLayout()
        self.host = QtWidgets.QLineEdit("127.0.0.1")
        self.host.setPlaceholderText("服务地址")
        self.port = QtWidgets.QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setValue(8000)
        self.port.setFixedWidth(92)
        self.host.textChanged.connect(self._invalidate_model)
        self.port.valueChanged.connect(self._invalidate_model)
        endpoint.addWidget(self.host)
        endpoint.addWidget(self.port)
        layout.addLayout(endpoint)
        self.model_status = label("模型未连接", "muted")
        layout.addWidget(self.model_status)
        row = QtWidgets.QHBoxLayout()
        self.connect_button = QtWidgets.QPushButton("连接 / 加载模型")
        self.connect_button.clicked.connect(lambda: self._connect_model(False))
        self.stop_model_button = QtWidgets.QPushButton("停止模型")
        self.stop_model_button.clicked.connect(self._stop_model)
        self.stop_model_button.setEnabled(False)
        row.addWidget(self.connect_button)
        row.addWidget(self.stop_model_button)
        layout.addLayout(row)
        control_layout.addWidget(model_box)
        self.model_box = model_box

        run_box = QtWidgets.QGroupBox("运行设置")
        layout = QtWidgets.QVBoxLayout(run_box)
        form = QtWidgets.QFormLayout()
        self.max_steps = QtWidgets.QSpinBox()
        self.max_steps.setRange(1, 100000)
        self.max_steps.setValue(900)
        self.chunk_steps = QtWidgets.QSpinBox()
        self.chunk_steps.setRange(1, 50)
        self.chunk_steps.setValue(25)
        form.addRow("本轮步数上限", self.max_steps)
        form.addRow("每次更新间隔（动作数）", self.chunk_steps)
        layout.addLayout(form)
        self.rtc = QtWidgets.QCheckBox("连续执行（RTC）")
        self.rtc.setChecked(True)
        self.rtc.setToolTip("执行动作时提前推理下一段，减少停顿；关闭可恢复逐段执行。RTC 下动作数为重规划间隔，最多 25。")
        self.rtc.toggled.connect(lambda enabled: self.chunk_steps.setMaximum(25 if enabled else 50))
        self.chunk_steps.setMaximum(25)
        layout.addWidget(self.rtc)
        self.reset = QtWidgets.QCheckBox("执行前复位")
        self.reset.setChecked(True)
        self.record = QtWidgets.QCheckBox("保存连续视频")
        self.record.setChecked(True)
        layout.addWidget(self.reset)
        layout.addWidget(self.record)
        layout.addWidget(label("开始推理将执行动作。复位过程不记录；复位目标：零关节 / 夹爪 0.07 m。", "muted"))
        control_layout.addWidget(run_box)
        self.edit_widgets += [run_box]
        control_layout.addStretch()
        # Run/stop controls stay visible even when settings need to scroll.
        row = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("开始推理")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(lambda: self._connect_model(True))
        self.stop_button = QtWidgets.QPushButton("停止本轮  Esc")
        self.stop_button.setObjectName("stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_run)
        row.addWidget(self.start_button)
        row.addWidget(self.stop_button)
        left_layout.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        self.check_button = QtWidgets.QPushButton("检查反馈")
        self.check_button.clicked.connect(lambda: self.start_run("check"))
        self.reset_button = QtWidgets.QPushButton("仅复位（动作）")
        self.reset_button.clicked.connect(lambda: self.start_run("reset"))
        row.addWidget(self.check_button)
        row.addWidget(self.reset_button)
        left_layout.addLayout(row)
        self.open_output = QtWidgets.QPushButton("打开本轮输出目录")
        self.open_output.setEnabled(False)
        self.open_output.clicked.connect(self._open_output)
        left_layout.addWidget(self.open_output)

        right = QtWidgets.QVBoxLayout()
        right.setSpacing(12)
        body.addLayout(right, 1)
        cameras = QtWidgets.QHBoxLayout()
        self.camera_cards = {}
        for name, (title, _) in CAMERAS.items():
            card = CameraCard(title)
            self.camera_cards[name] = card
            cameras.addWidget(card, 1)
        right.addLayout(cameras, 4)
        prediction = QtWidgets.QFrame()
        prediction.setObjectName("panel")
        layout = QtWidgets.QVBoxLayout(prediction)
        layout.addWidget(label("当前模型输出 · SUBTASK", "muted"))
        self.subtask = label("等待推理结果", "prediction")
        self.subtask.setMinimumHeight(55)
        self.subtask.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self.subtask)
        self.prediction_detail = label("显示最近一次模型返回的结果", "muted")
        layout.addWidget(self.prediction_detail)
        right.addWidget(prediction)
        metrics = QtWidgets.QHBoxLayout()
        self.metrics = {}
        for name, title in [("steps", "已处理步数"), ("queries", "推理请求"), ("latency", "往返延迟")]:
            frame = QtWidgets.QFrame()
            frame.setObjectName("panel")
            box = QtWidgets.QVBoxLayout(frame)
            box.addWidget(label(title, "muted"))
            value = label("—")
            value.setStyleSheet("font-size:21px;font-weight:600;")
            box.addWidget(value)
            self.metrics[name] = value
            metrics.addWidget(frame)
        right.addLayout(metrics)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setValue(0)
        right.addWidget(self.progress)
        extra = QtWidgets.QHBoxLayout()
        self.extra_fields = {}
        for key, title in EXTRA_INFO_FIELDS:
            frame = QtWidgets.QFrame()
            frame.setObjectName("panel")
            box = QtWidgets.QVBoxLayout(frame)
            box.addWidget(label(title, "muted"))
            value = label("预留 · 尚未接入", "muted")
            value.setMinimumHeight(40)
            box.addWidget(value)
            self.extra_fields[key] = value
            extra.addWidget(frame, 1)
        right.addLayout(extra)
        right.addWidget(label("运行信息", "muted"))
        self.console = QtWidgets.QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(200)
        self.console.setMinimumHeight(70)
        self.console.setMaximumHeight(140)
        right.addWidget(self.console, 1)
        self.statusBar().showMessage("待机 · Esc 停止本轮；停止后不会自动复位或启动下一轮")

    def _restore(self, initial):
        self.task.blockSignals(True)
        for task in self.settings.data["tasks"]:
            self.task.addItem(task["name"])
        self.task.blockSignals(False)
        self.task.setCurrentText(self.settings.data.get("task_name", self.task.itemText(0)))
        self._select_task(self.task.currentIndex())
        self._refresh_checkpoints()
        self.checkpoint.setCurrentText(initial.get("checkpoint") or self.settings.data.get("checkpoint", DEFAULT_CHECKPOINT))
        self.host.setText(initial.get("host") or self.settings.data.get("host", "127.0.0.1"))
        self.port.setValue(initial.get("port") or self.settings.data.get("port", 8000))
        self.max_steps.setValue(self.settings.data.get("max_steps", 900))
        self.rtc.setChecked(self.settings.data.get("rtc", True))
        self.chunk_steps.setValue(self.settings.data.get("chunk_steps", 25))
        if initial.get("task"):
            self.prompt.setPlainText(initial["task"])

    def _select_task(self, index):
        if not hasattr(self, "prompt"):
            return
        name = self.task.itemText(index)
        for task in self.settings.data["tasks"]:
            if task["name"] == name:
                self.prompt.setPlainText(task["prompt"])
                return

    def _save_task(self):
        try:
            name = self.task.currentText().strip()
            self.settings.save_task(name, self.prompt.toPlainText())
            if self.task.findText(name) < 0:
                self.task.addItem(name)
            self.task.setCurrentText(name)
            self._log("任务已保存：" + name)
        except Exception as error:
            self._log("保存失败：" + str(error))

    def _refresh_checkpoints(self):
        current = self.checkpoint.currentText()
        paths = [str(Path(item["path"]).relative_to(self.root)) for item in discover_checkpoints(self.root)]
        paths += self.settings.data.get("checkpoints", [])
        self.checkpoint.blockSignals(True)
        self.checkpoint.clear()
        self.checkpoint.addItems(list(dict.fromkeys(paths)))
        self.checkpoint.setCurrentText(current or DEFAULT_CHECKPOINT)
        self.checkpoint.blockSignals(False)
        self._checkpoint_changed()

    def _checkpoint_changed(self, *_):
        if not hasattr(self, "checkpoint_note"):
            return
        value = self.checkpoint.currentText()
        self.checkpoint.setToolTip(str(resolve_checkpoint(self.root, value)))
        try:
            info = checkpoint_info(resolve_checkpoint(self.root, value), root=self.root)
            if info["kind"] == "r1":
                state = "实验版本 · 未通过候选筛选" if info["experimental"] else "候选筛选通过"
                self.checkpoint_note.setText("R1 · step %s · %s" % (info["step"], state))
                self.checkpoint_note.setToolTip("M3 基座：" + info["parent_checkpoint"])
            elif info["kind"] == "recurrent_subtask":
                arm = "S 递归记忆 + reach 选臂" if info.get("label_version") == "reach_arm_v1" else "S 递归记忆" if info["arm"] == "recurrent" else "S 无状态对照"
                if info.get("experiment"):
                    arm = {"semantic_s":"S 关键语义", "semantic_s_actionrank":"S 关键语义 + 动作一致性", "decision_prefix":"选臂/选物 + 前25步", "decision_grounded":"选臂/选物 + 目标定位"}[info["experiment"]]
                self.checkpoint_note.setText("官方起点 · %s · step %s" % (arm, info["step"]))
                self.checkpoint_note.setToolTip(("B limited；reach 阶段含左右臂；" if info.get("label_version") == "reach_arm_v1" else "B 冻结；") + "A 使用原文字条件接口\nSHA256: " + info["weights_sha256"])
            elif info["kind"] in ("backbone_grad", "official_backbone_grad"):
                mode = {"frozen":"冻结 B 基线", "limited":"有限开放 B", "full":"全量更新 B"}[info["mode"]]
                if info["kind"] == "official_backbone_grad":
                    mode = "官方起点 · " + mode
                self.checkpoint_note.setText("%s · step %s · 实验模型" % (mode, info["step"]))
                self.checkpoint_note.setToolTip("完整 B/S/A 权重；尚未经真机成功率验证\nSHA256: " + info["weights_sha256"])
            else:
                self.checkpoint_note.setText("M3 · step %s · 完整模型" % info["step"])
                self.checkpoint_note.setToolTip("")
        except Exception as error:
            self.checkpoint_note.setText("请选择完整 M3 或 R1 checkpoint")
            self.checkpoint_note.setToolTip(str(error))
        self._invalidate_model()

    def _browse_checkpoint(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择 checkpoint 目录", str(self.root/"checkpoints"))
        if path:
            self.checkpoint.setCurrentText(path)

    def _invalidate_model(self, *_):
        if hasattr(self, "model_status"):
            self.model_status.setText("选择变更后，请连接 / 加载模型")
        self.model_key = None

    def config(self):
        return {"checkpoint": str(resolve_checkpoint(self.root, self.checkpoint.currentText())),
                "host": self.host.text().strip(), "port": self.port.value(),
                "prompt": self.prompt.toPlainText().strip(), "max_steps": self.max_steps.value(),
                "chunk_steps": self.chunk_steps.value(), "rtc": self.rtc.isChecked(),
                "reset": self.reset.isChecked(), "record": self.record.isChecked()}

    def _persist(self):
        config = self.config()
        self.settings.data.update({key: config[key] for key in ("checkpoint", "host", "port", "max_steps", "chunk_steps", "rtc")})
        self.settings.data["task_name"] = self.task.currentText()
        self.settings.data["checkpoints"] = list(dict.fromkeys(
            [config["checkpoint"]] + self.settings.data.get("checkpoints", [])))[:20]
        self.settings.save()

    def _job(self, name, function):
        def worker():
            try:
                self.events.result.emit(name, function())
            except Exception as error:
                self.events.error.emit(name, str(error))
        threading.Thread(target=worker, name="gui-"+name, daemon=True).start()

    def _connect_model(self, start_after=False):
        if self.loading or self.client:
            return
        try:
            config = self.config()
            info = checkpoint_info(config["checkpoint"], root=self.root)
            if start_after and not config["prompt"]:
                raise ValueError("请填写任务指令")
            self._persist()
        except Exception as error:
            self._log("配置错误：" + str(error))
            return
        self.loading = True
        self.pending_start = start_after
        self._set_busy()
        self.model_status.setText("正在连接 / 加载模型…")
        self._log("连接 %s:%d，checkpoint=%s" % (config["host"], config["port"], config["checkpoint"]))
        if info["kind"] == "r1":
            self._log("R1 解码器 + M3 基座：" + info["parent_checkpoint"])
            if info["experimental"]:
                self._log("所选 R1 为实验版本，未通过候选筛选；加载状态将保存在本轮模型信息中")

        def work():
            local = config["host"] in ("127.0.0.1", "localhost")
            try:
                metadata = server_metadata(config["host"], config["port"])
            except OSError:
                if not local:
                    raise RuntimeError("远程模型服务尚未就绪，请在目标主机启动服务")
                record = find_server(self.root, config["port"])
                log_path = None
                if record is None:
                    self.server_process, log_path = launch_server(self.root, config["checkpoint"], config["port"])
                deadline = time.monotonic() + 180
                while True:
                    if self.server_process and self.server_process.poll() is not None:
                        detail = log_path.read_text(errors="replace")[-1200:] if log_path else ""
                        raise RuntimeError("模型加载失败：%s\n%s" % (log_path or "进程已退出", detail))
                    try:
                        metadata = server_metadata(config["host"], config["port"])
                        break
                    except OSError:
                        if time.monotonic() > deadline:
                            raise RuntimeError("模型加载超过 180 秒，请检查 logs/gui 服务日志后重新连接")
                        time.sleep(0.5)
            record = find_server(self.root, config["port"]) if local else None
            return {"metadata": metadata, "record": record, "config": config, "start": start_after}
        self._job("connect", work)

    def _job_result(self, name, result):
        self.loading = False
        if name == "connect":
            start_requested = result["start"] and self.pending_start
            self.pending_start = False
            self.metadata, self.server_record = result["metadata"], result["record"]
            config = result["config"]
            try:
                validate_server(self.metadata, self.root, config["checkpoint"])
            except (ValueError, OSError, KeyError) as error:
                self.model_status.setText("端口已连接 · checkpoint 不匹配")
                self._log(str(error))
                self._set_busy()
                return
            self.model_key = (config["host"], config["port"], config["checkpoint"])
            kind = ("官方起点 · " + {"frozen":"冻结 B 基线", "limited":"有限开放 B", "full":"全量更新 B"}[self.metadata["mode"]]) if self.metadata.get("variant") == "official_pi05_backbone_v1" else ("有限开放 B 实验模型" if self.metadata.get("mode") == "limited" else "全量更新 B 实验模型") if self.metadata.get("variant") == "action_backbone_v1" else "R1 实验模型" if self.metadata.get("experimental") else (
                "R1 模型" if self.metadata.get("variant") == "r1_boundary_ce_v1" else "M3 模型")
            if self.metadata.get("variant") == "official_pi05_recurrent_s_v1":
                kind = "S 递归记忆模型" if self.metadata["arm"] == "recurrent" else "S 无状态对照模型"
            self.model_status.setText(kind + "就绪 · %s:%d" % (config["host"], config["port"]))
            self._log("模型连接成功：" + str(self.metadata.get("checkpoint")))
            self._set_busy()
            if start_requested and not self.closing:
                self.start_run("inference")
        elif name == "stop_model":
            if self.server_process:
                self.server_process.poll()
                self.server_process = None
            self.server_record = self.metadata = self.model_key = None
            self.model_status.setText("模型已停止 · 端口已释放")
            self._log("模型服务已停止，可以选择其他 checkpoint")
            self._set_busy()

    def _job_error(self, name, error):
        self.loading = False
        self.pending_start = False
        if name == "connect":
            self.model_status.setText("模型连接 / 加载失败")
        self._log(error)
        self._set_busy()

    def _stop_model(self):
        if self.client or self.loading or not self.server_record:
            return
        self.loading = True
        self._set_busy()
        self.model_status.setText("正在停止模型并清理端口…")
        record = dict(self.server_record)
        self._job("stop_model", lambda: stop_server(self.root, record))

    def start_run(self, kind="inference"):
        if self.client or self.loading or self.closing:
            return
        try:
            config = self.config()
            if kind == "inference":
                if self.model_key != (config["host"], config["port"], config["checkpoint"]):
                    raise RuntimeError("请先连接所选 checkpoint")
                validate_server(self.metadata, self.root, config["checkpoint"])
            self._persist()
            self.token = uuid.uuid4().hex
            self.output = self.root/"logs/inference"/(time.strftime("%Y%m%d_%H%M%S")+"_"+self.token[:10])
            command = client_command(self.root, config, self.output, self.udp.localPort(), self.token, kind)
            self.client = subprocess.Popen(command, cwd=str(self.root), stdin=subprocess.DEVNULL,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, bufsize=1, start_new_session=True)
            self.client_identity = process_record(self.client.pid)
            self.run_config, self.run_kind = config, kind
            self.latest_prediction, self.last_event = {}, 0.0
            self.subtask.setText("等待推理结果" if kind == "inference" else "此操作不请求模型")
            self.prediction_detail.setText("本轮尚无模型输出")
            for value in self.extra_fields.values():
                value.setText("预留 · 尚未接入")
            self.progress.setRange(0, config["max_steps"])
            self.progress.setValue(0)
            self.metrics["steps"].setText("0 / %d" % config["max_steps"])
            self.metrics["queries"].setText("0")
            self.metrics["latency"].setText("—")
            self.phase.setText("正在启动")
            self.open_output.setEnabled(False)
            stream = self.client.stdout
            def reader():
                for line in stream:
                    self.events.line.emit(line.rstrip())
                stream.close()
            threading.Thread(target=reader, name="gui-client-log", daemon=True).start()
            self._set_busy()
        except Exception as error:
            self._log("启动失败：" + str(error))

    def stop_run(self):
        if self.pending_start:
            self.pending_start = False
            self.stop_button.setEnabled(False)
            self.phase.setText("已取消本轮启动")
            self._log("本轮启动已取消；模型连接完成后也不会开始推理")
        if self.client and self.client.poll() is None and alive(self.client_identity):
            # The client owns a new process group, including shell environment
            # setup before exec. Interrupt all of that group, never ROS/model.
            os.killpg(self.client.pid, signal.SIGINT)
            self.stop_button.setEnabled(False)
            self.phase.setText("停止中")
            self._log("已请求停止本轮；进入推理后已录制的输出会完成保存，复位过程不记录")

    def _read_telemetry(self):
        while self.udp.hasPendingDatagrams():
            payload, _, _ = self.udp.readDatagram(self.udp.pendingDatagramSize())
            try:
                data = json.loads(bytes(payload))
                if not self.client or not isinstance(data, dict) or not self.token or data.get("token") != self.token:
                    continue
                if not isinstance(data.get("status"), dict) or not isinstance(data.get("prediction"), dict):
                    continue
                self._display_event(data)
            except (ValueError, TypeError, KeyError, AttributeError):
                continue

    def _display_event(self, data):
        self.last_event = time.monotonic()
        status = data.get("status", {})
        phase = PHASES.get(status.get("phase"), status.get("phase", "运行中"))
        mode = "检查" if self.run_kind == "check" else "动作"
        self.phase.setText(mode + " · " + phase)
        steps, queries = status.get("steps", 0), status.get("queries", 0)
        self.metrics["steps"].setText("%d / %d" % (steps, self.run_config.get("max_steps", 0)))
        self.metrics["queries"].setText(str(queries))
        self.progress.setValue(steps)
        prediction = data.get("prediction") or {}
        if prediction.get("query"):
            self.latest_prediction = prediction
            self.subtask.setText(str(prediction.get("subtask") or "（空输出）"))
            latency = prediction.get("roundtrip_ms")
            self.metrics["latency"].setText("%.0f ms" % latency if isinstance(latency, (int, float)) else "—")
        extras = data.get("inference_info") or {}
        for key, value in self.extra_fields.items():
            if extras.get(key) is not None:
                value.setText(str(extras[key]))
        reset = data.get("reset") or {}
        if status.get("phase") == "RESET" and reset.get("max_joint_error") is not None:
            self.statusBar().showMessage("复位中（不记录） · 最大关节误差 %.3f rad · 轨迹等待 %.2f s" %
                                        (reset["max_joint_error"], reset.get("paused_seconds") or 0))
        elif self.output and (self.output/"run.json").is_file():
            self.statusBar().showMessage(mode + " · " + phase + " · " + str(self.output))
        else:
            self.statusBar().showMessage(mode + " · " + phase + " · 尚未开始记录")

    def _tick(self):
        has_output = bool(self.output and (self.output/"run.json").is_file())
        self.open_output.setEnabled(has_output)
        if self.camera:
            for name, frame in self.camera.snapshot().items():
                if name in self.camera_cards:
                    self.camera_cards[name].update_frame(frame)
        if self.latest_prediction:
            prediction = self.latest_prediction
            age = time.monotonic() - (prediction.get("response_monotonic") or time.monotonic())
            score = prediction.get("subtask_score")
            score_text = "%.3f" % score if isinstance(score, (int, float)) else "—"
            self.prediction_detail.setText("%s · 模型分数 %s（非概率） · 结果年龄 %.1f s" %
                                           (prediction.get("subtask_status") or "—", score_text, age))
        if self.client and self.client.poll() is not None:
            code = self.client.returncode
            self.client = self.client_identity = None
            try:
                report = json.loads((self.output/"run.json").read_text())
                query = report["queries"][-1] if report["queries"] else {}
                self._display_event({"status": report["status"], "prediction": query})
                state = report["status"].get("status")
                self.phase.setText({"finished": "本轮完成", "interrupted": "已停止", "error": "运行失败"}.get(state, state))
                if report["status"].get("error"):
                    self._log(report["status"]["error"])
            except (OSError, ValueError, KeyError):
                if code == 0 and self.run_kind == "reset":
                    message = "复位完成 · 未记录"
                elif code in (130, -signal.SIGINT):
                    message = "已停止 · 未记录"
                else:
                    message = "运行失败 · 未记录 · code %s" % code
                self.phase.setText(message)
            self.statusBar().showMessage("输出：" + str(self.output) if has_output else "未生成运行记录或视频")
            self._set_busy()
            if self.closing:
                self.close()
        elif self.client and self.last_event and time.monotonic() - self.last_event > 3:
            self.statusBar().showMessage("等待客户端状态更新；停止按钮仍可用")

    def _set_busy(self):
        busy = bool(self.client) or self.loading
        for widget in self.edit_widgets:
            widget.setEnabled(not busy)
        self.model_box.setEnabled(not busy)
        self.start_button.setEnabled(not busy)
        self.check_button.setEnabled(not busy)
        self.reset_button.setEnabled(not busy)
        self.stop_button.setEnabled(bool(self.client) or self.pending_start)
        self.stop_model_button.setEnabled(not busy and bool(self.server_record) and alive(self.server_record))

    def _log(self, message):
        if message:
            self.console.appendPlainText(time.strftime("%H:%M:%S  ") + message)

    def _open_output(self):
        if self.output and (self.output/"run.json").is_file():
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.output)))

    def closeEvent(self, event):
        if self.client and self.client.poll() is None:
            self.closing = True
            self.stop_run()
            event.ignore()
            return
        self.closing = True
        self.pending_start = False
        try:
            self._persist()
        except OSError:
            pass
        if self.camera:
            self.camera.close()
        self.timer.stop()
        self.udp.close()
        # The model server stays available; only Stop model ends it.
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="Simple Piper image/inference GUI")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--checkpoint")
    parser.add_argument("--task")
    parser.add_argument("--settings", type=Path)
    args = parser.parse_args()
    import rospy
    rospy.init_node("piper_eval_gui", anonymous=True, disable_signals=True)
    app = QtWidgets.QApplication([])
    app.setFont(QtGui.QFont("Noto Sans CJK SC", 10))
    app.setStyleSheet(STYLE)
    window = PiperWindow(settings_path=args.settings, camera=CameraFeed(), initial=vars(args))
    signal.signal(signal.SIGINT, lambda *_: window.close())
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
