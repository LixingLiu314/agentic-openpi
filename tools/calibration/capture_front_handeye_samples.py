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


class FrontHandeyeCollector:
    def __init__(
        self,
        image_topic: str,
        joint_topic: str,
        output_dir: Path,
        joints_path: Path,
        manifest_path: Path,
        prefix: str,
        show: bool,
        arm_dof: int,
    ) -> None:
        self.image_topic = image_topic
        self.joint_topic = joint_topic
        self.output_dir = output_dir
        self.joints_path = joints_path
        self.manifest_path = manifest_path
        self.prefix = prefix
        self.show = show
        self.arm_dof = arm_dof

        self.bridge = CvBridge()
        self.lock = threading.Lock()

        self.latest_frame = None
        self.latest_frame_ros_stamp = None
        self.latest_frame_wall_time = None
        self.latest_joint = None
        self.latest_joint_ros_stamp = None
        self.latest_joint_wall_time = None

        self.samples: list[dict] = []
        self.preview_thread = None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.joints_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        rospy.init_node("front_handeye_collector", anonymous=True)
        rospy.Subscriber(self.image_topic, Image, self._image_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(self.joint_topic, JointState, self._joint_callback, queue_size=1, tcp_nodelay=True)

        if self.show:
            self.preview_thread = threading.Thread(target=self._preview_loop, daemon=True)

    def _image_callback(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self.lock:
            self.latest_frame = frame
            self.latest_frame_ros_stamp = msg.header.stamp.to_sec() if msg.header.stamp else None
            self.latest_frame_wall_time = time.time()

    def _joint_callback(self, msg: JointState) -> None:
        if len(msg.position) < self.arm_dof:
            return
        joint = np.asarray(msg.position[: self.arm_dof], dtype=np.float64)
        with self.lock:
            self.latest_joint = joint
            self.latest_joint_ros_stamp = msg.header.stamp.to_sec() if msg.header.stamp else None
            self.latest_joint_wall_time = time.time()

    def _preview_loop(self) -> None:
        window_name = "front_handeye_preview"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 720)
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            with self.lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()
                frame_stamp = self.latest_frame_ros_stamp
                joint = None if self.latest_joint is None else self.latest_joint.copy()
                joint_stamp = self.latest_joint_ros_stamp
                sample_count = len(self.samples)

            if frame is not None:
                joint_text = "joint: ready" if joint is not None else "joint: waiting"
                stamp_delta = None
                if frame_stamp is not None and joint_stamp is not None:
                    stamp_delta = abs(frame_stamp - joint_stamp)
                overlay = [
                    f"saved: {sample_count}  enter=capture  q=quit",
                    joint_text,
                    f"image_topic: {self.image_topic}",
                    f"joint_topic: {self.joint_topic}",
                ]
                if stamp_delta is not None:
                    overlay.append(f"|image_stamp - joint_stamp| = {stamp_delta:.3f}s")

                y = 30
                for line in overlay:
                    cv2.putText(
                        frame,
                        line,
                        (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    y += 28

                cv2.imshow(window_name, frame)
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
                if self.latest_frame is not None and self.latest_joint is not None:
                    return
            time.sleep(0.05)
        raise TimeoutError(
            f"Did not receive both image and joint data within {timeout_sec:.1f}s "
            f"(image_topic={self.image_topic}, joint_topic={self.joint_topic})"
        )

    def capture(self, max_data_age_sec: float) -> tuple[Path, dict]:
        with self.lock:
            if self.latest_frame is None:
                raise RuntimeError("No image received yet.")
            if self.latest_joint is None:
                raise RuntimeError("No joint state received yet.")

            now = time.time()
            frame_age = now - self.latest_frame_wall_time if self.latest_frame_wall_time is not None else 1e9
            joint_age = now - self.latest_joint_wall_time if self.latest_joint_wall_time is not None else 1e9
            if frame_age > max_data_age_sec:
                raise RuntimeError(f"Latest image is stale: {frame_age:.3f}s > {max_data_age_sec:.3f}s")
            if joint_age > max_data_age_sec:
                raise RuntimeError(f"Latest joint state is stale: {joint_age:.3f}s > {max_data_age_sec:.3f}s")

            frame = self.latest_frame.copy()
            joint = self.latest_joint.copy()
            image_stamp = self.latest_frame_ros_stamp
            joint_stamp = self.latest_joint_ros_stamp
            sample_idx = len(self.samples)

        image_name = f"{self.prefix}_{sample_idx:03d}.png"
        image_path = self.output_dir / image_name
        ok = cv2.imwrite(str(image_path), frame)
        if not ok:
            raise IOError(f"Failed to save image to {image_path}")

        sample = {
            "index": sample_idx,
            "image": image_name,
            "joints_rad": joint.tolist(),
            "image_stamp_sec": image_stamp,
            "joint_stamp_sec": joint_stamp,
            "stamp_delta_sec": None if image_stamp is None or joint_stamp is None else abs(image_stamp - joint_stamp),
            "saved_wall_time_sec": time.time(),
        }
        self.samples.append(sample)
        self._flush_metadata()
        return image_path, sample

    def _flush_metadata(self) -> None:
        joints = np.asarray([sample["joints_rad"] for sample in self.samples], dtype=np.float64)
        np.save(self.joints_path, joints)

        manifest = {
            "image_topic": self.image_topic,
            "joint_topic": self.joint_topic,
            "num_samples": len(self.samples),
            "joints_path": str(self.joints_path),
            "image_dir": str(self.output_dir),
            "samples": self.samples,
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture front-camera hand-eye samples with Enter.")
    parser.add_argument("--topic", default="/camera_f/color/image_raw", help="Front camera topic.")
    parser.add_argument("--arm", choices=["left", "right"], default="left")
    parser.add_argument("--joint-topic-left", default="/puppet/joint_left")
    parser.add_argument("--joint-topic-right", default="/puppet/joint_right")
    default_left_output_dir = DATA_DIR / "handeye_front_left_imgs"
    default_left_joints_path = DATA_DIR / "handeye_joints_left.npy"
    default_left_manifest_path = DATA_DIR / "handeye_front_left_manifest.json"
    parser.add_argument("--output-dir", type=Path, default=default_left_output_dir)
    parser.add_argument("--joints-path", type=Path, default=default_left_joints_path)
    parser.add_argument("--manifest-path", type=Path, default=default_left_manifest_path)
    parser.add_argument("--prefix", default="front_handeye")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max-data-age", type=float, default=0.5, help="Reject capture if latest image/joint is too old.")
    parser.add_argument("--show", action="store_true", help="Show a live preview window.")
    parser.add_argument("--arm-dof", type=int, default=6, help="How many arm joints to save from JointState.position.")
    args = parser.parse_args()

    if args.arm == "right":
        default_output_dir = DATA_DIR / "handeye_front_right_imgs"
        default_joints_path = DATA_DIR / "handeye_joints_right.npy"
        default_manifest_path = DATA_DIR / "handeye_front_right_manifest.json"
        if args.output_dir == default_left_output_dir:
            args.output_dir = default_output_dir
        if args.joints_path == default_left_joints_path:
            args.joints_path = default_joints_path
        if args.manifest_path == default_left_manifest_path:
            args.manifest_path = default_manifest_path

    joint_topic = args.joint_topic_left if args.arm == "left" else args.joint_topic_right
    collector = FrontHandeyeCollector(
        image_topic=args.topic,
        joint_topic=joint_topic,
        output_dir=args.output_dir,
        joints_path=args.joints_path,
        manifest_path=args.manifest_path,
        prefix=args.prefix,
        show=args.show,
        arm_dof=args.arm_dof,
    )
    collector.start_preview()

    print(f"image topic: {args.topic}")
    print(f"joint topic: {joint_topic}")
    print(f"image dir: {args.output_dir}")
    print(f"joints file: {args.joints_path}")
    print(f"manifest: {args.manifest_path}")
    print("waiting for first image + joint...")
    collector.wait_for_first_data(args.timeout)
    print("ready")
    print("move robot to a new pose, hold still, then press Enter to save; input q then Enter to quit")

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
            image_path, sample = collector.capture(max_data_age_sec=args.max_data_age)
        except Exception as exc:
            print(f"capture failed: {exc}")
            continue

        delta = sample["stamp_delta_sec"]
        if delta is None:
            delta_text = "n/a"
        else:
            delta_text = f"{delta:.3f}s"
        print(f"saved: {image_path}  joints={sample['joints_rad']}  stamp_delta={delta_text}")


if __name__ == "__main__":
    main()
