"""Continuous real-time cameras. All frame metadata is returned to run.json."""
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
import time
import cv2
import numpy as np


class LiveVideo:
    def __init__(self, sensors, output, run_status, fps=30):
        self.sensors, self.output, self.run_status = sensors, Path(output), run_status
        self.fps, self.prediction = fps, None
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.error, self.frames = None, []
        self.pending = queue.Queue(maxsize=16)
        self.encoded_frames = 0
        self.start_wall, self.start_monotonic = None, None
        self.stderr = tempfile.TemporaryFile(mode="w+b")
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                   "-pix_fmt", "bgr24", "-s", "1440x576", "-r", str(fps), "-i", "-", "-an",
                   "-c:v", "libx264", "-threads", "2", "-preset", "veryfast", "-crf", "22",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.output / "video.mp4")]
        try:
            self.encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=self.stderr)
        except BaseException:
            self.stderr.close()
            raise
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.encoder_thread = threading.Thread(target=self._encode, daemon=True)

    def start(self):
        self.encoder_thread.start()
        self.thread.start()
        deadline = time.monotonic() + 5
        while not self.encoded_frames:
            self.check()
            if time.monotonic() > deadline:
                raise RuntimeError("Video recorder did not produce its first frame")
            time.sleep(.01)

    def update_prediction(self, entry):
        with self.lock:
            self.prediction = dict(entry)

    def check(self):
        if self.error is not None:
            raise RuntimeError("Video recorder failed: " + str(self.error))

    def _capture(self):
        names = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
        with self.sensors.lock:
            mono, wall = time.monotonic(), time.time()
            images = {name: self.sensors.values[name].copy() for name in names}
            stamps = {name: self.sensors.stamps[name] for name in names}
            ages = {name: mono-self.sensors.times[name] for name in names}
        with self.lock:
            prediction = self.prediction
        phase = self.run_status.get("phase", "PREPARING")
        canvas = np.full((576, 1440, 3), (24, 20, 16), dtype=np.uint8)
        for col, (name, label) in enumerate(zip(names, ["FRONT", "LEFT WRIST", "RIGHT WRIST"])):
            rgb = cv2.resize(images[name], (480, 360), interpolation=cv2.INTER_AREA)
            canvas[92:452, col*480:(col+1)*480] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(canvas, "%s | age %.3fs" % (label, ages[name]), (col*480+12, 79),
                        cv2.FONT_HERSHEY_SIMPLEX, .62, (210,210,210), 1, cv2.LINE_AA)
        mode = "EXECUTE" if self.run_status.get("executing") else "READ ONLY"
        cv2.putText(canvas, "M3 | %s | %s | t=%.2fs" % (mode, phase, mono-self.start_monotonic),
                    (16, 32), cv2.FONT_HERSHEY_SIMPLEX, .79, (230,230,230), 2, cv2.LINE_AA)
        if prediction:
            input_time = prediction["observation"]["snapshot_monotonic"]-self.start_monotonic
            text = "LAST SUBTASK: " + str(prediction["subtask"])
            details = "query=%d | input t=%.2fs | result age=%.2fs | status=%s" % (
                prediction["query"], input_time, mono-prediction["response_monotonic"],
                prediction["subtask_status"])
        else:
            text = "No policy prediction yet"
            details = "No subtask is inferred during reset / camera-only recording"
        cv2.putText(canvas, details, (16, 487), cv2.FONT_HERSHEY_SIMPLEX, .66, (190,210,210), 1, cv2.LINE_AA)
        cv2.putText(canvas, text, (16, 528), cv2.FONT_HERSHEY_SIMPLEX, .98, (80,220,250), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Real-time cameras + latest returned prediction. Input timestamps and nearest frame: run.json",
                    (16, 560), cv2.FONT_HERSHEY_SIMPLEX, .49, (175,175,175), 1, cv2.LINE_AA)
        record = {"time": wall, "capture_monotonic": mono, "phase": phase,
                  "sensor_ros_timestamps": stamps, "sensor_age_seconds": ages,
                  "query": prediction["query"] if prediction else None}
        return canvas.tobytes(), record

    def _write(self, pixels, captured, repeated=False):
        index = len(self.frames)
        # Camera sampling must not wait for encoder startup or pipe backpressure.
        self.pending.put_nowait(pixels)
        self.frames.append(dict(captured, frame=index, video_time=index/self.fps,
                                repeated_for_timing=repeated))

    def _encode(self):
        try:
            while True:
                pixels = self.pending.get()
                if pixels is None:
                    break
                self.encoder.stdin.write(pixels)
                self.encoded_frames += 1
        except BaseException as error:
            self.error = repr(error)
            self.stopping.set()
        finally:
            try:
                self.encoder.stdin.close()
            except (BrokenPipeError, OSError):
                pass

    def _run(self):
        self.start_monotonic, self.start_wall = time.monotonic(), time.time()
        previous = None
        try:
            while not self.stopping.is_set():
                deadline = self.start_monotonic+len(self.frames)/self.fps
                if self.stopping.wait(max(0, deadline-time.monotonic())):
                    break
                pixels, captured = self._capture()
                slot = int((captured["capture_monotonic"]-self.start_monotonic)*self.fps)
                missing = max(0, slot-len(self.frames))
                if missing > self.fps:
                    raise RuntimeError("Video capture fell more than 1 second behind real time")
                # Preserve elapsed time after missed slots; never speed up later
                # samples to catch up. Repeats are counted, not interpolated/hidden.
                if previous:
                    for _ in range(missing):
                        self._write(*previous, repeated=True)
                self._write(pixels, captured)
                previous = (pixels, captured)
        except BaseException as error:
            self.error = repr(error)
        finally:
            try:
                self.pending.put(None, timeout=1)
            except queue.Full:
                self.error = self.error or "Video encoder queue did not drain"

    def stop(self):
        self.stopping.set()
        self.thread.join(timeout=5)
        self.encoder_thread.join(timeout=5)
        if self.thread.is_alive() or self.encoder_thread.is_alive():
            self.encoder.terminate()
            self.error = self.error or "Recorder thread did not finish"
        try:
            code = self.encoder.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.encoder.kill()
            code = self.encoder.wait()
            self.error = self.error or "Video encoder timed out"
        self.thread.join(timeout=2)
        self.encoder_thread.join(timeout=2)
        self.stderr.seek(0)
        encoder_error = self.stderr.read(8192).decode("utf-8", errors="replace")
        self.stderr.close()
        if code:
            self.error = self.error or "ffmpeg exited with code %d" % code
        real = [x for x in self.frames if not x["repeated_for_timing"]]
        gaps = np.diff([x["capture_monotonic"] for x in real])
        repeats = sum(x["repeated_for_timing"] for x in self.frames)
        span = (real[-1]["capture_monotonic"]-self.start_monotonic) if real else 0
        return {"file": "video.mp4", "fps": self.fps, "frame_count": len(self.frames),
                "encoded_seconds": len(self.frames)/self.fps, "capture_span_seconds": span,
                "encoder_frames_written": self.encoded_frames,
                "start_time": self.start_wall, "start_monotonic": self.start_monotonic,
                "max_capture_gap_seconds": float(max(gaps)) if len(gaps) else None,
                "p99_capture_gap_seconds": float(np.percentile(gaps, 99)) if len(gaps) else None,
                "repeated_timing_frames": repeats,
                "timing_warning": "Capture missed slots; inspect repeats" if repeats else None,
                "encoder_returncode": code, "error": self.error, "encoder_stderr": encoder_error,
                "scope": "Continuous real-time cameras; latest returned subtask, not framewise ground truth",
                "frames": self.frames}
