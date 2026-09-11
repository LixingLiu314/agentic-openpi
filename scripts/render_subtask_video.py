"""Render exact archived inference inputs with their generated subtask labels."""

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


def render(input_dir, output, title):
    records = [json.loads(line) for line in (input_dir / "queries.jsonl").read_text().splitlines()]
    assert records, "No inference records"
    output.parent.mkdir(parents=True, exist_ok=True)
    fps, width, height = 10, 1440, 576
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
               "-pix_fmt", "bgr24", "-s", "%dx%d" % (width, height), "-r", str(fps), "-i", "-",
               "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    start = records[0]["observation"]["snapshot_time"]
    frame_count = 0
    try:
        for index, record in enumerate(records):
            archive = input_dir / record["input_archive"]
            with np.load(archive, allow_pickle=False) as data:
                missing = set(["cam_high", "cam_left_wrist", "cam_right_wrist"]) - set(data.files)
                if missing:
                    raise ValueError("Cannot reconstruct video: inference images were not recorded: " + str(archive))
                frame = np.full((height, width, 3), (24, 20, 16), dtype=np.uint8)
                for col, (name, label) in enumerate([("cam_high", "FRONT / cam_high"),
                                                     ("cam_left_wrist", "LEFT WRIST"),
                                                     ("cam_right_wrist", "RIGHT WRIST")]):
                    image = cv2.cvtColor(data[name], cv2.COLOR_RGB2BGR)
                    frame[92:452, col*480:(col+1)*480] = cv2.resize(image, (480, 360), interpolation=cv2.INTER_AREA)
                    cv2.putText(frame, label, (col*480+16, 79), cv2.FONT_HERSHEY_SIMPLEX, .65, (210,210,210), 1, cv2.LINE_AA)
            timestamp = record["observation"]["snapshot_time"]
            end = records[index+1]["observation"]["snapshot_time"] if index+1 < len(records) else timestamp+.7
            cv2.putText(frame, title, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, .72, (230,230,230), 2, cv2.LINE_AA)
            line = "INPUT t=%.2fs | query %d/%d | infer %.1fms | status=%s" % (
                timestamp-start, index+1, len(records), record["policy_timing"]["infer_ms"], record["subtask_status"])
            cv2.putText(frame, line, (16, 489), cv2.FONT_HERSHEY_SIMPLEX, .67, (190,210,210), 1, cv2.LINE_AA)
            cv2.putText(frame, "SUBTASK: " + str(record["subtask"]), (16, 527),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80,220,250), 2, cv2.LINE_AA)
            cv2.putText(frame, "Exact model input snapshots; held until next query. No invented intermediate frames.",
                        (16, 558), cv2.FONT_HERSHEY_SIMPLEX, .53, (160,160,160), 1, cv2.LINE_AA)
            if index == 0:
                cv2.imwrite(str(output.with_suffix(".png")), frame)
            target_frame_count = max(frame_count+1, round((end-start)*fps))
            for _ in range(target_frame_count-frame_count):
                process.stdin.write(frame.tobytes())
            frame_count = target_frame_count
    finally:
        process.stdin.close()
        result = process.wait()
    if result:
        raise RuntimeError("Video encoding failed")
    report = {"input_dir": str(input_dir), "video": str(output), "queries": len(records),
              "duration_seconds": frame_count/fps, "video_fps": fps,
              "input_times_seconds": [x["observation"]["snapshot_time"]-start for x in records],
              "executing_actions": any(x["executing"] for x in records),
              "scope": "Sampled inference input video, not a continuous camera recording"}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--title", default="M3 seed42 - readonly preview - NO MOTOR COMMANDS")
    args = parser.parse_args()
    render(args.input_dir, args.output, args.title)
