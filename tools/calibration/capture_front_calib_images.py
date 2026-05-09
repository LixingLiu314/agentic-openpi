#!/usr/bin/env python3
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

DATA_DIR = Path(__file__).resolve().parent / "data"


class FrontImageCollector:
    def __init__(self, image_topic: str, output_dir: Path, prefix: str, show: bool) -> None:
        self.image_topic = image_topic
        self.output_dir = output_dir
        self.prefix = prefix
        self.show = show

        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_stamp = None
        self.frame_count = 0
        self.preview_thread = None

        self.output_dir.mkdir(parents=True, exist_ok=True)

        rospy.init_node("front_calib_image_collector", anonymous=True)
        rospy.Subscriber(self.image_topic, Image, self._image_callback, queue_size=1, tcp_nodelay=True)

        if self.show:
            self.preview_thread = threading.Thread(target=self._preview_loop, daemon=True)

    def _image_callback(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self.lock:
            self.latest_frame = frame
            self.latest_stamp = msg.header.stamp.to_sec() if msg.header.stamp else time.time()

    def _preview_loop(self) -> None:
        window_name = "front_camera_preview"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 720)
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            with self.lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()
                frame_count = self.frame_count
            if frame is not None:
                cv2.putText(
                    frame,
                    f"saved: {frame_count}  enter=capture  q=quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
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

    def wait_for_first_frame(self, timeout_sec: float) -> None:
        deadline = time.time() + timeout_sec
        while not rospy.is_shutdown() and time.time() < deadline:
            with self.lock:
                if self.latest_frame is not None:
                    return
            time.sleep(0.05)
        raise TimeoutError(f"No image received from topic {self.image_topic} within {timeout_sec:.1f}s")

    def save_current_frame(self) -> Path:
        with self.lock:
            if self.latest_frame is None:
                raise RuntimeError("No image has been received yet.")
            frame = self.latest_frame.copy()
            stamp = self.latest_stamp if self.latest_stamp is not None else time.time()
            index = self.frame_count
            self.frame_count += 1

        filename = f"{self.prefix}_{index:03d}_{int(stamp * 1000)}.png"
        path = self.output_dir / filename
        ok = cv2.imwrite(str(path), frame)
        if not ok:
            raise IOError(f"Failed to save image to {path}")
        return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture front-camera calibration images by pressing Enter.")
    parser.add_argument("--topic", default="/camera_f/color/image_raw", help="ROS image topic for the front camera.")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "front_intrinsic_imgs")
    parser.add_argument("--prefix", default="front")
    parser.add_argument("--timeout", type=float, default=10.0, help="Seconds to wait for the first image.")
    parser.add_argument("--show", action="store_true", help="Show a live preview window.")
    args = parser.parse_args()

    collector = FrontImageCollector(
        image_topic=args.topic,
        output_dir=args.output_dir,
        prefix=args.prefix,
        show=args.show,
    )
    collector.start_preview()

    print(f"subscribing to: {args.topic}")
    print(f"output dir: {args.output_dir}")
    print("waiting for first frame...")
    collector.wait_for_first_frame(args.timeout)
    print("ready")
    print("press Enter to save the current frame, input q then Enter to quit")

    while not rospy.is_shutdown():
        try:
            user_input = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nexit")
            break

        if user_input == "q":
            print("exit")
            break

        path = collector.save_current_frame()
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
