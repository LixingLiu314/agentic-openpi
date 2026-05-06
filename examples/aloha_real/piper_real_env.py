# ruff: noqa
"""
AgiLex Piper real-robot environment for openpi evaluation.

Drop-in replacement for examples/aloha_real/real_env.py targeting the
cobot_magic / Piper stack instead of Trossen/Interbotix VX300S.
Exposes the same make_real_env() factory and RealEnv-compatible interface
so that piper_env.py (and transitively piper_main.py) can use it unchanged.

ROS topics (cobot_magic defaults):
  cam_high         <- /camera_f/color/image_raw   (sensor_msgs/Image, rgb8)
  cam_left_wrist   <- /camera_l/color/image_raw
  cam_right_wrist  <- /camera_r/color/image_raw
  puppet_left_js   <- /puppet/joint_left    (sensor_msgs/JointState, 7-dim)
  puppet_right_js  <- /puppet/joint_right
  cmd_left         -> /master/joint_left    (sensor_msgs/JointState)
  cmd_right        -> /master/joint_right

14-dim state / action layout (matches pi05_aloha_banana training convention):
  [left_arm_joint(6), left_gripper(1), right_arm_joint(6), right_gripper(1)]

Gripper values are raw continuous radian values from the Piper driver.
The pi05_aloha_banana server pipeline (adapt_to_pi + AbsoluteActions transform)
handles all unit/space conversions server-side; this env just passes raw radians.
"""

import collections
import threading
import time
from typing import List, Optional

import dm_env
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState

# Reset trajectory control rate (Hz)
_RESET_HZ = 50
_RESET_DT = 1.0 / _RESET_HZ

# Default reset arm pose (6-DOF, radians) — matches pi05_aloha_banana policy_metadata.
DEFAULT_RESET_POSITION: List[float] = [0.0, -1.5, 1.5, 0.0, 0.0, 0.0]

# Gripper radian limits for Piper (tune to your puppet's actual range).
DEFAULT_GRIPPER_OPEN:  float = 4.0   # rad — fully open
DEFAULT_GRIPPER_CLOSE: float = 0.0   # rad — fully closed


class PiperRealEnv:
    """
    Environment for AgiLex Piper bi-manual manipulation.

    Action space:
      [left_arm_joint(6), left_gripper(1), right_arm_joint(6), right_gripper(1)]
      — absolute joint targets in radians.

    Observation space:
      {"qpos":   np.ndarray (14,) float32  — same layout as action space
       "qvel":   np.ndarray (14,) float32  — zero-padded if driver omits velocity
       "images": {"cam_high":        (H, W, 3) uint8 RGB
                  "cam_left_wrist":  (H, W, 3) uint8 RGB
                  "cam_right_wrist": (H, W, 3) uint8 RGB}}
    """

    def __init__(
        self,
        init_node: bool,
        *,
        reset_position: Optional[List[float]] = None,
        # ROS topic overrides
        img_front_topic:   str = "/camera_f/color/image_raw",
        img_left_topic:    str = "/camera_l/color/image_raw",
        img_right_topic:   str = "/camera_r/color/image_raw",
        joint_left_topic:  str = "/puppet/joint_left",
        joint_right_topic: str = "/puppet/joint_right",
        cmd_left_topic:    str = "/master/joint_left",
        cmd_right_topic:   str = "/master/joint_right",
        gripper_open:  float = DEFAULT_GRIPPER_OPEN,
        gripper_close: float = DEFAULT_GRIPPER_CLOSE,
        reset_move_time: float = 2.0,
        topic_wait_timeout: float = 30.0,
        dry_run: bool = False,
    ):
        self._reset_position  = list(reset_position[:6]) if reset_position else DEFAULT_RESET_POSITION
        self._gripper_open    = gripper_open
        self._gripper_close   = gripper_close
        self._reset_move_time = reset_move_time
        self._dry_run         = dry_run

        self._bridge = CvBridge()
        self._lock   = threading.Lock()

        self._img_high:  Optional[np.ndarray] = None
        self._img_left:  Optional[np.ndarray] = None
        self._img_right: Optional[np.ndarray] = None
        self._js_left:   Optional[JointState]  = None
        self._js_right:  Optional[JointState]  = None

        if init_node:
            rospy.init_node("piper_real_env", anonymous=True)

        rospy.Subscriber(img_front_topic,   Image,      self._cb_img_high,  queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(img_left_topic,    Image,      self._cb_img_left,  queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(img_right_topic,   Image,      self._cb_img_right, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(joint_left_topic,  JointState, self._cb_jsl,       queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(joint_right_topic, JointState, self._cb_jsr,       queue_size=1, tcp_nodelay=True)

        self._pub_left  = rospy.Publisher(cmd_left_topic,  JointState, queue_size=10)
        self._pub_right = rospy.Publisher(cmd_right_topic, JointState, queue_size=10)

        rospy.loginfo("[PiperRealEnv] Waiting for all topics (timeout=%.0fs)...", topic_wait_timeout)
        self._wait_for_topics(timeout=topic_wait_timeout)
        rospy.loginfo("[PiperRealEnv] All topics ready.")

    # ------------------------------------------------------------------ #
    # ROS callbacks
    # ------------------------------------------------------------------ #

    def _cb_img_high(self, msg: Image) -> None:
        with self._lock:
            self._img_high = self._bridge.imgmsg_to_cv2(msg, "rgb8")

    def _cb_img_left(self, msg: Image) -> None:
        with self._lock:
            self._img_left = self._bridge.imgmsg_to_cv2(msg, "rgb8")

    def _cb_img_right(self, msg: Image) -> None:
        with self._lock:
            self._img_right = self._bridge.imgmsg_to_cv2(msg, "rgb8")

    def _cb_jsl(self, msg: JointState) -> None:
        with self._lock:
            self._js_left = msg

    def _cb_jsr(self, msg: JointState) -> None:
        with self._lock:
            self._js_right = msg

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _all_ready(self) -> bool:
        return all(x is not None for x in (
            self._img_high, self._img_left, self._img_right,
            self._js_left, self._js_right,
        ))

    def _wait_for_topics(self, timeout: float = 30.0) -> None:
        deadline = time.time() + timeout
        while not rospy.is_shutdown():
            if self._all_ready():
                return
            if time.time() > deadline:
                raise RuntimeError(
                    "[PiperRealEnv] Timed out waiting for ROS topics. "
                    "Check that the robot stack is running: "
                    "bash examples/Aloha/eval_files/start_robot_stack.sh --validate-only"
                )
            time.sleep(0.05)

    def _snapshot(self):
        """Atomically copy all sensor data to avoid cross-field timestamp skew."""
        with self._lock:
            high  = self._img_high.copy()
            left  = self._img_left.copy()
            right = self._img_right.copy()
            jsl_pos = list(self._js_left.position)
            jsr_pos = list(self._js_right.position)
            jsl_vel = list(self._js_left.velocity)  if self._js_left.velocity  else []
            jsr_vel = list(self._js_right.velocity) if self._js_right.velocity else []
        return high, left, right, jsl_pos, jsr_pos, jsl_vel, jsr_vel

    def _publish_cmd(self, left_cmd: np.ndarray, right_cmd: np.ndarray) -> None:
        """Publish 7-dim (arm×6 + gripper×1) JointState to both arms."""
        for pub, cmd in ((self._pub_left, left_cmd), (self._pub_right, right_cmd)):
            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.position = cmd.tolist()
            pub.publish(msg)

    @staticmethod
    def _build_state14(pos_l: list, pos_r: list) -> np.ndarray:
        """Pack left/right JointState positions into 14-dim state vector."""
        left_arm  = pos_l[:6]
        left_grip = [pos_l[6]] if len(pos_l) > 6 else [0.0]
        right_arm  = pos_r[:6]
        right_grip = [pos_r[6]] if len(pos_r) > 6 else [0.0]
        return np.array(left_arm + left_grip + right_arm + right_grip, dtype=np.float32)

    @staticmethod
    def _build_vel14(vel_l: list, vel_r: list) -> np.ndarray:
        """Pack velocities into 14-dim vector; zero-fills missing dims."""
        vl = (vel_l + [0.0] * 7)[:7]
        vr = (vel_r + [0.0] * 7)[:7]
        return np.array(vl[:6] + [vl[6]] + vr[:6] + [vr[6]], dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Observation / reward
    # ------------------------------------------------------------------ #

    def get_observation(self) -> dict:
        high, left, right, jsl_pos, jsr_pos, jsl_vel, jsr_vel = self._snapshot()
        obs = collections.OrderedDict()
        obs["qpos"]   = self._build_state14(jsl_pos, jsr_pos)
        obs["qvel"]   = self._build_vel14(jsl_vel, jsr_vel)
        obs["images"] = {
            "cam_high":        high,
            "cam_left_wrist":  left,
            "cam_right_wrist": right,
        }
        return obs

    def get_reward(self) -> float:
        return 0.0

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #

    def _move_to_position(
        self,
        target_arm: List[float],
        gripper_val: float,
        move_time: float,
    ) -> None:
        """Linearly interpolate both arms from current qpos to target."""
        _, _, _, jsl_pos, jsr_pos, _, _ = self._snapshot()
        curr_qpos = self._build_state14(jsl_pos, jsr_pos)

        curr_left_arm   = curr_qpos[0:6]
        curr_left_grip  = float(curr_qpos[6])
        curr_right_arm  = curr_qpos[7:13]
        curr_right_grip = float(curr_qpos[13])

        n = max(1, int(move_time * _RESET_HZ))
        target = np.asarray(target_arm, dtype=np.float32)

        left_arm_traj   = np.linspace(curr_left_arm,   target, n)
        right_arm_traj  = np.linspace(curr_right_arm,  target, n)
        left_grip_traj  = np.linspace(curr_left_grip,  gripper_val, n)
        right_grip_traj = np.linspace(curr_right_grip, gripper_val, n)

        for i in range(n):
            left_cmd  = np.concatenate([left_arm_traj[i],  [left_grip_traj[i]]]).astype(np.float32)
            right_cmd = np.concatenate([right_arm_traj[i], [right_grip_traj[i]]]).astype(np.float32)
            self._publish_cmd(left_cmd, right_cmd)
            time.sleep(_RESET_DT)

    def reset(self, *, fake: bool = False) -> dm_env.TimeStep:
        if not fake:
            self._move_to_position(
                self._reset_position,
                self._gripper_open,
                self._reset_move_time,
            )
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation(),
        )

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        """
        Apply a 14-dim absolute joint action and return the next timestep.

        action layout: [left_arm(6), left_grip(1), right_arm(6), right_grip(1)]

        The openpi server-side AbsoluteActions transform has already converted
        delta actions to absolute targets before this is called.
        """
        action = np.asarray(action, dtype=np.float32)
        if not self._dry_run:
            self._publish_cmd(action[0:7], action[7:14])
        else:
            rospy.logdebug("[PiperRealEnv] dry_run: suppressed command publish.")
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation(),
        )


def make_real_env(
    init_node: bool,
    *,
    reset_position: Optional[List[float]] = None,
    **kwargs,
) -> PiperRealEnv:
    """Factory matching the signature used by AlohaRealEnvironment in env.py."""
    return PiperRealEnv(init_node, reset_position=reset_position, **kwargs)
