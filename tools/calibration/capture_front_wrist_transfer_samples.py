#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState

DATA_DIR = Path(__file__).resolve().parent / "data"


class FrontWristTransferCollector:
    def __init__(
        self,
        front_topic: str,
        wrist_topic: str,
        joint_topic: str,
        front_dir: Path,
        wrist_dir: Path,
        joints_path: Path,
        manifest_path: Path,
        prefix: str,
        show: bool,
        arm_dof: int,
    ) -> None:
        self.front_topic = front_topic
        self.wrist_topic = wrist_topic
        self.joint_topic = joint_topic
        self.front_dir = front_dir
        self.wrist_dir = wrist_dir
        self.joints_path = joints_path
        self.manifest_path = manifest_path
        self.prefix = prefix
        self.show = show
        self.arm_dof = arm_dof

        self.bridge = CvBridge()
        self.lock = threading.Lock()

        self.front_frame = None
        self.front_stamp = None
        self.front_wall_time = None
        self.wrist_frame = None
        self.wrist_stamp = None
        self.wrist_wall_time = None
        self.joint = None
        self.joint_stamp = None
        self.joint_wall_time = None

        self.samples: list[dict] = []
        self.preview_thread = None

        self.front_dir.mkdir(parents=True, exist_ok=True)
        self.wrist_dir.mkdir(parents=True, exist_ok=True)
        self.joints_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        rospy.init_node("front_wrist_transfer_collector", anonymous=True)
        rospy.Subscriber(self.front_topic, Image, self._front_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(self.wrist_topic, Image, self._wrist_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(self.joint_topic, JointState, self._joint_callback, queue_size=1, tcp_nodelay=True)

        if self.show:
            self.preview_thread = threading.Thread(target=self._preview_loop, daemon=True)

    def _front_callback(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self.lock:
            self.front_frame = frame
            self.front_stamp = msg.header.stamp.to_sec() if msg.header.stamp else None
            self.front_wall_time = time.time()

    def _wrist_callback(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self.lock:
            self.wrist_frame = frame
            self.wrist_stamp = msg.header.stamp.to_sec() if msg.header.stamp else None
            self.wrist_wall_time = time.time()

    def _joint_callback(self, msg: JointState) -> None:
        if len(msg.position) < self.arm_dof:
            return
        joint = np.asarray(msg.position[: self.arm_dof], dtype=np.float64)
        with self.lock:
            self.joint = joint
            self.joint_stamp = msg.header.stamp.to_sec() if msg.header.stamp else None
            self.joint_wall_time = time.time()

    def _preview_loop(self) -> None:
        window_name = "front_wrist_transfer_preview"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        rate = rospy.Rate(15)
        while not rospy.is_shutdown():
            with self.lock:
                front = None if self.front_frame is None else self.front_frame.copy()
                wrist = None if self.wrist_frame is None else self.wrist_frame.copy()
                front_stamp = self.front_stamp
                wrist_stamp = self.wrist_stamp
                joint_stamp = self.joint_stamp
                count = len(self.samples)

            if front is not None and wrist is not None:
                front = cv2.resize(front, (640, 480))
                wrist = cv2.resize(wrist, (640, 480))
                cv2.putText(front, "front", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(wrist, "wrist", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                tile = np.hstack([front, wrist])

                lines = [
                    f"saved: {count}  enter=capture  q=quit",
                    f"front topic: {self.front_topic}",
                    f"wrist topic: {self.wrist_topic}",
                    f"joint topic: {self.joint_topic}",
                ]
                if front_stamp is not None and wrist_stamp is not None:
                    lines.append(f"|front - wrist| = {abs(front_stamp - wrist_stamp):.3f}s")
                if front_stamp is not None and joint_stamp is not None:
                    lines.append(f"|front - joint| = {abs(front_stamp - joint_stamp):.3f}s")

                panel = np.zeros((180, tile.shape[1], 3), dtype=np.uint8)
                y = 30
                for line in lines:
                    cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                    y += 28

                canvas = np.vstack([tile, panel])
                cv2.imshow(window_name, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    rospy.signal_shutdown("preview window quit")
                    break

            rate.sleep()
        cv2.destroyAllWindows()

    def start_preview(self) -> None:
        if self.preview_thread is not None:
            self.preview_thread.start()

    def wait_for_first_data(self, timeout_sec: float) -> None:
        deadline = time.time() + timeout_sec
        while not rospy.is_shutdown() and time.time() < deadline:
            with self.lock:
                ready = self.front_frame is not None and self.wrist_frame is not None and self.joint is not None
            if ready:
                return
            time.sleep(0.05)
        raise TimeoutError("Did not receive front image, wrist image, and joint data in time.")

    def capture(self, max_data_age_sec: float) -> dict:
        with self.lock:
            if self.front_frame is None or self.wrist_frame is None or self.joint is None:
                raise RuntimeError("Missing data stream.")

            now = time.time()
            ages = {
                "front_age_sec": now - self.front_wall_time if self.front_wall_time is not None else 1e9,
                "wrist_age_sec": now - self.wrist_wall_time if self.wrist_wall_time is not None else 1e9,
                "joint_age_sec": now - self.joint_wall_time if self.joint_wall_time is not None else 1e9,
            }
            if any(age > max_data_age_sec for age in ages.values()):
                raise RuntimeError(f"Stale data: {ages}")

            front = self.front_frame.copy()
            wrist = self.wrist_frame.copy()
            joint = self.joint.copy()
            front_stamp = self.front_stamp
            wrist_stamp = self.wrist_stamp
            joint_stamp = self.joint_stamp
            idx = len(self.samples)

        stem = f"{self.prefix}_{idx:03d}"
        front_name = f"{stem}.png"
        wrist_name = f"{stem}.png"
        front_path = self.front_dir / front_name
        wrist_path = self.wrist_dir / wrist_name
        if not cv2.imwrite(str(front_path), front):
            raise IOError(f"Failed to write {front_path}")
        if not cv2.imwrite(str(wrist_path), wrist):
            raise IOError(f"Failed to write {wrist_path}")

        sample = {
            "index": idx,
            "front_image": front_name,
            "wrist_image": wrist_name,
            "joints_rad": joint.tolist(),
            "front_stamp_sec": front_stamp,
            "wrist_stamp_sec": wrist_stamp,
            "joint_stamp_sec": joint_stamp,
            "front_wrist_delta_sec": None if front_stamp is None or wrist_stamp is None else abs(front_stamp - wrist_stamp),
            "front_joint_delta_sec": None if front_stamp is None or joint_stamp is None else abs(front_stamp - joint_stamp),
            "saved_wall_time_sec": time.time(),
        }
        self.samples.append(sample)
        self._flush_metadata()
        return sample

    def _flush_metadata(self) -> None:
        joints = np.asarray([sample["joints_rad"] for sample in self.samples], dtype=np.float64)
        np.save(self.joints_path, joints)
        manifest = {
            "front_topic": self.front_topic,
            "wrist_topic": self.wrist_topic,
            "joint_topic": self.joint_topic,
            "front_dir": str(self.front_dir),
            "wrist_dir": str(self.wrist_dir),
            "joints_path": str(self.joints_path),
            "num_samples": len(self.samples),
            "samples": self.samples,
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture front+wrist transfer samples with Enter.")
    parser.add_argument("--arm", choices=["left", "right"], default="left")
    parser.add_argument("--front-topic", default="/camera_f/color/image_raw")
    parser.add_argument("--wrist-topic-left", default="/camera_l/color/image_raw")
    parser.add_argument("--wrist-topic-right", default="/camera_r/color/image_raw")
    parser.add_argument("--joint-topic-left", default="/puppet/joint_left")
    parser.add_argument("--joint-topic-right", default="/puppet/joint_right")
    default_left_front_dir = DATA_DIR / "transfer_front_left" / "front_imgs"
    default_left_wrist_dir = DATA_DIR / "transfer_front_left" / "wrist_imgs"
    default_left_joints_path = DATA_DIR / "transfer_front_left" / "joints.npy"
    default_left_manifest_path = DATA_DIR / "transfer_front_left" / "manifest.json"
    parser.add_argument("--front-dir", type=Path, default=default_left_front_dir)
    parser.add_argument("--wrist-dir", type=Path, default=default_left_wrist_dir)
    parser.add_argument("--joints-path", type=Path, default=default_left_joints_path)
    parser.add_argument("--manifest-path", type=Path, default=default_left_manifest_path)
    parser.add_argument("--prefix", default="transfer")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max-data-age", type=float, default=0.5)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--arm-dof", type=int, default=6)
    args = parser.parse_args()

    if args.arm == "right":
        if args.front_dir == default_left_front_dir:
            args.front_dir = DATA_DIR / "transfer_front_right" / "front_imgs"
        if args.wrist_dir == default_left_wrist_dir:
            args.wrist_dir = DATA_DIR / "transfer_front_right" / "wrist_imgs"
        if args.joints_path == default_left_joints_path:
            args.joints_path = DATA_DIR / "transfer_front_right" / "joints.npy"
        if args.manifest_path == default_left_manifest_path:
            args.manifest_path = DATA_DIR / "transfer_front_right" / "manifest.json"

    wrist_topic = args.wrist_topic_left if args.arm == "left" else args.wrist_topic_right
    joint_topic = args.joint_topic_left if args.arm == "left" else args.joint_topic_right
    collector = FrontWristTransferCollector(
        front_topic=args.front_topic,
        wrist_topic=wrist_topic,
        joint_topic=joint_topic,
        front_dir=args.front_dir,
        wrist_dir=args.wrist_dir,
        joints_path=args.joints_path,
        manifest_path=args.manifest_path,
        prefix=args.prefix,
        show=args.show,
        arm_dof=args.arm_dof,
    )
    collector.start_preview()

    print(f"front topic: {args.front_topic}")
    print(f"wrist topic: {wrist_topic}")
    print(f"joint topic: {joint_topic}")
    print(f"front dir: {args.front_dir}")
    print(f"wrist dir: {args.wrist_dir}")
    print(f"joints file: {args.joints_path}")
    print("waiting for front + wrist + joints...")
    collector.wait_for_first_data(args.timeout)
    print("ready")
    print("keep robot still, make sure both cameras see the board, then press Enter to save; input q then Enter to quit")

    while not rospy.is_shutdown():
        try:
            user_input = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nexit")
            break

        if user_input == "q":
            print("exit")
            break

        try:
            sample = collector.capture(args.max_data_age)
        except Exception as exc:
            print(f"capture failed: {exc}")
            continue
        print(
            f"saved sample {sample['index']:03d}  "
            f"front_wrist_delta={sample['front_wrist_delta_sec']}  "
            f"front_joint_delta={sample['front_joint_delta_sec']}"
        )


if __name__ == "__main__":
    main()
