"""Piper ROS environment: sensors, verified enabling, reset, and action publication.

Uses the stable piper_real_env reset/step separation and ROS/native14 layout.
Checkpoint-specific gripper units, feedback checks and recording remain explicit.
"""
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time

from cv_bridge import CvBridge
import numpy as np
import rosgraph
import rospy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool

from piper_can_feedback import GRIPPERS, JOINTS, PassiveCanFeedback, check_feedback
from piper_reset import ResetPlanner

CAMERAS = {"cam_high": "/camera_f/color/image_raw",
           "cam_left_wrist": "/camera_l/color/image_raw",
           "cam_right_wrist": "/camera_r/color/image_raw"}


def require_healthy_can():
    for interface in ("can_left", "can_right"):
        result = subprocess.run(["ip", "-details", "link", "show", "dev", interface],
                                capture_output=True, text=True, timeout=2, check=True)
        match = re.search(r"can state ([A-Z-]+)", result.stdout)
        state = match.group(1) if match else "unknown"
        if state != "ERROR-ACTIVE":
            raise RuntimeError("%s CAN state is %s; fix the physical link before executing" % (interface, state))


def publish_command(publishers, action):
    command = np.asarray(action, dtype=np.float32).copy()
    if command.shape != (14,) or not np.isfinite(command).all():
        raise ValueError("Expected finite native14 command")
    command[[6, 13]] = np.clip(command[[6, 13]], 0, .09)
    for pub, position in zip(publishers, (command[:7], command[7:])):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = ["joint%d" % i for i in range(7)]
        msg.position = position.tolist()
        pub.publish(msg)


class Sensors:
    def __init__(self):
        self.lock = threading.Lock()
        self.values = {}
        self.times = {}
        self.stamps = {}
        self.last_snapshot_meta = {}
        self.bridge = CvBridge()
        self.subscribers = []
        self.can = PassiveCanFeedback()
        for name, topic in CAMERAS.items():
            self.subscribers.append(rospy.Subscriber(topic, Image, self.callback, callback_args=name,
                                                    queue_size=1, buff_size=2**23, tcp_nodelay=True))
        for name, topic in [("left", "/puppet/joint_left"), ("right", "/puppet/joint_right")]:
            self.subscribers.append(rospy.Subscriber(topic, JointState, self.callback, callback_args=name,
                                                    queue_size=1, tcp_nodelay=True))

    def callback(self, message, name):
        value = self.bridge.imgmsg_to_cv2(message, "rgb8") if name in CAMERAS else np.asarray(message.position, dtype=np.float32)
        with self.lock:
            self.values[name] = value.copy()
            self.times[name] = time.monotonic()
            self.stamps[name] = message.header.stamp.to_sec()

    def snapshot(self, copy_images=True, with_metadata=False):
        with self.lock:
            if set(self.values) != set(CAMERAS) | {"left", "right"}:
                raise RuntimeError("Missing sensor topics: " + str((set(CAMERAS) | {"left", "right"}) - set(self.values)))
            now = time.monotonic()
            ages = {name: now - stamp for name, stamp in self.times.items()}
            if max(ages.values()) > 1.0:
                raise RuntimeError("Stale sensor observation; ages_seconds=" + json.dumps(ages))
            joint_feedback_ages = {name: time.time() - self.stamps[name] for name in ("left", "right")}
            if any(not np.isfinite(age) or age > 1.0 or age < -0.1 for age in joint_feedback_ages.values()):
                raise RuntimeError("Stale CAN joint feedback; restart Piper driver after CAN recovery; ages_seconds=" + json.dumps(joint_feedback_ages))
            left, right = self.values["left"], self.values["right"]
            if left.shape != (7,) or right.shape != (7,):
                raise RuntimeError("Each Piper arm must provide exactly 7 positions")
            state = np.concatenate([left, right])
            if not np.isfinite(state).all():
                raise RuntimeError("Nonfinite measured state")
            metadata = {"snapshot_time": time.time(), "snapshot_monotonic": now,
                                       "sensor_age_seconds": ages,
                                       "joint_feedback_age_seconds": joint_feedback_ages,
                                       "sensor_ros_timestamps": dict(self.stamps)}
            self.last_snapshot_meta = metadata
            images = {name: self.values[name].copy() for name in CAMERAS} if copy_images else {}
        hardware = self.can.snapshot()
        self.last_hardware = dict(hardware, state=hardware["state"].tolist(), ros_state=state.tolist())
        check_feedback(hardware, state)
        observation = {"state": state, "images": images}
        return (observation, metadata) if with_metadata else observation

    def state(self, require_enabled=False):
        state = self.snapshot(copy_images=False)["state"]
        if require_enabled:
            check_feedback(self.can.snapshot(), state, require_enabled=True)
        return state

    def close(self):
        for subscriber in self.subscribers:
            subscriber.unregister()
        self.can.close()


class ControlRate:
    """Monotonic scheduling, with no burst of catch-up commands after a delay."""

    def __init__(self, hz):
        self.period = 1 / hz
        self.deadline = time.monotonic()

    def sleep(self):
        self.deadline += self.period
        now = time.monotonic()
        if self.deadline < now:
            self.deadline = now + self.period
        time.sleep(max(0, self.deadline - now))


class PiperRobot:
    """Only constructed for explicit execution; readonly creates no publishers."""

    def __init__(self, sensors, args, log, video=None):
        if not args.execute:
            raise ValueError("PiperRobot requires explicit --execute")
        self.sensors, self.args, self.log, self.video = sensors, args, log, video
        self.publishers, self.enable, self.lease = [], None, None
        self.ready = False
        self.last_can_check = 0.0

    def connect(self):
        self.lease = open('/tmp/openpi-piper-%d.lock' % os.getuid(), 'a')
        try:
            fcntl.flock(self.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another Piper execution client is active")
        topics = ("/master/joint_left", "/master/joint_right", "/enable_flag")
        publishers, _, _ = rosgraph.Master(rospy.get_name()).getSystemState()
        conflicts = {topic: nodes for topic, nodes in publishers if topic in topics and nodes}
        if conflicts:
            raise RuntimeError("Other robot command publishers are active: %s" % conflicts)
        require_healthy_can()
        self.sensors.state()
        self.publishers = [rospy.Publisher(topic, JointState, queue_size=1, tcp_nodelay=True)
                           for topic in topics[:2]]
        self.enable = rospy.Publisher(topics[2], Bool, queue_size=1, latch=False)
        deadline = time.monotonic() + 10
        while any(pub.get_num_connections() < 1 for pub in self.publishers) or self.enable.get_num_connections() < 2:
            if rospy.is_shutdown():
                raise KeyboardInterrupt()
            if time.monotonic() > deadline:
                raise RuntimeError("Both command and enable subscribers must be connected")
            time.sleep(0.05)

    def _check(self, enabled=True):
        if rospy.is_shutdown():
            raise KeyboardInterrupt()
        self.log.check()
        if self.video:
            self.video.check()
        now = time.monotonic()
        if now - self.last_can_check > 0.3:
            require_healthy_can()
            self.last_can_check = now
        return self.sensors.state(require_enabled=enabled)

    def prepare(self):
        self.log.status(phase="ENABLING")
        self._check(enabled=False)
        # ROS connection count is not a hardware acknowledgement. Confirm all
        # 12 motor enable bits from independent CAN frames after publishing.
        self.enable.publish(Bool(data=True))
        deadline = time.monotonic() + 5
        last_request = time.monotonic()
        time.sleep(0.3)
        while True:
            self._check(enabled=False)
            report = self.sensors.can.snapshot()
            self.log.update(enable_feedback=dict(report, state=report["state"].tolist()))
            if all(all(arm["motor_enabled"]) for arm in report["arms"].values()):
                self.ready = True
                return
            if time.monotonic() > deadline:
                raise RuntimeError("Enable timed out: all 12 motor enable bits must be confirmed")
            if time.monotonic() - last_request > 0.5:
                self.enable.publish(Bool(data=True))
                last_request = time.monotonic()
            time.sleep(0.05)

    def reset(self):
        if not self.ready:
            raise RuntimeError("Reset requires confirmed motor enabling")
        initial = self._check()
        target = np.zeros(14, dtype=np.float32)
        target[GRIPPERS] = self.args.reset_gripper
        planner = ResetPlanner(initial, target, self.args.reset_seconds, self.args.max_joint_jump)
        self.log.status(phase="RESET")
        print("Reset: at least %.1fs, feedback-paced, native zero joints, grippers %.3fm" %
              (planner.duration, self.args.reset_gripper), flush=True)
        measured, steps, last = initial, 0, time.monotonic() - 1/30
        rate = ControlRate(30)
        try:
            while not planner.done:
                measured = self._check()
                now = time.monotonic()
                command = planner.step(measured, now - last)
                last = now
                publish_command(self.publishers, command)
                steps += 1
                self.log.status(reset_steps=steps)
                self.log.update(reset=planner.report(measured))
                rate.sleep()
        except BaseException as error:
            self.log.update(reset=dict(planner.report(measured), failure=repr(error), passed=False))
            raise
        report = planner.report(measured)
        self.log.update(reset=report)
        print("Reset passed: joint error %.4frad, gripper error %.4fm, paused %.2fs" %
              (report["max_joint_error"], report["max_gripper_error"], report["paused_seconds"]), flush=True)

    def step(self, action):
        if not self.ready:
            raise RuntimeError("Action requires confirmed motor enabling")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError("Expected finite native14 action")
        measured = self._check()
        jump = float(np.max(np.abs(action[JOINTS] - measured[JOINTS])))
        if jump > self.args.max_joint_jump:
            raise RuntimeError("Joint target differs from measured state by %.3f rad (limit %.3f)" %
                               (jump, self.args.max_joint_jump))
        publish_command(self.publishers, action)

    def close(self):
        # Do not disable motors (which could release a payload) or issue a reset.
        self.ready = False
        for publisher in self.publishers:
            publisher.unregister()
        self.publishers = []
        if self.enable:
            self.enable.unregister()
            self.enable = None
        if self.lease:
            self.lease.close()
            self.lease = None
