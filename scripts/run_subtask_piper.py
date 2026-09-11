"""M3 Piper entry point: prepare -> reset -> observe/infer -> bounded action chunk."""
import argparse
from pathlib import Path
import time

import numpy as np
import rospy

from piper_robot import CAMERAS, ControlRate, PiperRobot, Sensors, require_healthy_can
from piper_run_log import RunLog

RUN_STATUS = {"steps": 0, "queries": 0}
RUN_LOG = VIDEO = ROBOT = SENSORS = None


def wait_post_reset_images(sensors):
    """Exclude camera frames still buffered from reset, with a bounded wait."""
    boundary = time.time()
    deadline = time.monotonic() + 1.0
    while not rospy.is_shutdown():
        sensors.snapshot(copy_images=False)
        stamps = sensors.last_snapshot_meta["sensor_ros_timestamps"]
        if all(stamps[name] >= boundary for name in CAMERAS):
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("No fresh post-reset images; recording has not started")
        time.sleep(.01)
    raise KeyboardInterrupt()


def start_recording(args, sensors, phase, video=True):
    global VIDEO
    observation = sensors.snapshot()
    RUN_LOG.start_recording(phase,
        initial_observation=dict(sensors.last_snapshot_meta, state=observation["state"].tolist()),
        hardware_preflight=sensors.last_hardware)
    print("Run output: " + str(args.output.resolve()), flush=True)
    if video and args.record_continuous:
        from subtask_live_video import LiveVideo
        VIDEO = LiveVideo(sensors, args.output, RUN_STATUS)
        if ROBOT is not None:
            ROBOT.video = VIDEO
        VIDEO.start()


def connect_policy(url, wait_seconds, connect):
    """Wait only at startup; never reconnect/replay after robot execution begins."""
    deadline = time.monotonic() + wait_seconds
    announced = False
    while not rospy.is_shutdown():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("Policy server was not ready within %.0fs at %s" % (wait_seconds, url))
        try:
            return connect(url, compression=None, max_size=None, open_timeout=min(5, remaining))
        except (OSError, TimeoutError):
            if not announced:
                print("Waiting for policy server at %s (up to %.0fs)..." % (url, wait_seconds), flush=True)
                announced = True
            time.sleep(min(0.25, remaining))
    raise KeyboardInterrupt()


def main():
    global RUN_LOG, VIDEO, ROBOT, SENSORS
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--expected-checkpoint", type=Path, help="Require the server to advertise this checkpoint")
    parser.add_argument("--telemetry-port", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--telemetry-token", default="", help=argparse.SUPPRESS)
    parser.add_argument("--server-wait-seconds", type=float, default=60, help="Bounded startup wait before enabling/reset")
    parser.add_argument("--prompt", default="Put the eggplant into the box")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observe-only", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Validate ROS/CAN feedback without enabling, model requests or recording")
    parser.add_argument("--observe-seconds", type=float, default=5.0)
    parser.add_argument("--execute", action="store_true")
    reset_mode = parser.add_mutually_exclusive_group()
    reset_mode.add_argument("--reset-before", dest="reset_before", action="store_true")
    reset_mode.add_argument("--no-reset", dest="reset_before", action="store_false")
    parser.set_defaults(reset_before=None)
    parser.add_argument("--reset-only", action="store_true")
    parser.add_argument("--reset-seconds", type=float, default=6.0)
    parser.add_argument("--reset-gripper", type=float, default=.07)
    video_mode = parser.add_mutually_exclusive_group()
    video_mode.add_argument("--record-continuous", dest="record_continuous", action="store_true")
    video_mode.add_argument("--no-record-continuous", dest="record_continuous", action="store_false")
    parser.set_defaults(record_continuous=True)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--chunk-steps", type=int, default=25)
    parser.add_argument("--max-joint-jump", type=float, default=0.35)
    parser.add_argument("--log-every", type=int, default=10, help="Print every N queries, and on subtask changes")
    rtc_mode = parser.add_mutually_exclusive_group()
    rtc_mode.add_argument("--rtc", dest="rtc", action="store_true", help="Continuous latency-aligned action chunking (default)")
    rtc_mode.add_argument("--no-rtc", dest="rtc", action="store_false", help="Synchronous chunk execution")
    parser.set_defaults(rtc=True)
    args = parser.parse_args()
    if args.rtc and args.chunk_steps > 25 and not (args.check_only or args.observe_only or args.reset_only):
        parser.error("RTC requires --chunk-steps between 1 and 25; use --no-rtc for longer chunks")
    if not 0 <= args.telemetry_port <= 65535:
        parser.error("Invalid GUI telemetry port")
    if args.reset_before is None:
        args.reset_before = args.execute
    if not 1 <= args.chunk_steps <= 50 or args.max_steps < 1 or args.max_joint_jump <= 0:
        parser.error("Invalid execution budget")
    if args.check_only and (args.execute or args.reset_before or args.reset_only):
        parser.error("--check-only cannot execute or reset")
    if args.observe_only and args.execute:
        parser.error("Observation capture cannot execute actions")
    if not (args.execute or args.observe_only or args.check_only):
        parser.error("Inference requires --execute; inference without action execution has been removed")
    if (args.reset_before or args.reset_only) and (not args.execute or args.observe_only):
        parser.error("Reset requires --execute and cannot be combined with observation-only mode")
    if not np.isfinite([args.reset_seconds, args.reset_gripper, args.observe_seconds, args.max_joint_jump, args.server_wait_seconds]).all():
        parser.error("Control limits and durations must be finite")
    if args.reset_seconds < 2 or not 0 <= args.reset_gripper <= .09:
        parser.error("Invalid reset duration or native gripper target")
    if args.observe_seconds <= 0 or args.log_every < 1:
        parser.error("Observation duration and log interval must be positive")
    if args.server_wait_seconds <= 0:
        parser.error("Server startup wait must be positive")
    if args.output.exists():
        parser.error("Output directory already exists: " + str(args.output))
    RUN_STATUS.update(executing=args.execute, status="running", phase="PREPARING", recording=False)
    RUN_LOG = RunLog(args.output, args, RUN_STATUS, deferred=True)
    if args.execute:
        require_healthy_can()
    rospy.init_node("m3_piper_client", anonymous=True)
    sensors = SENSORS = Sensors()
    deadline = time.monotonic() + 20
    while not rospy.is_shutdown():
        try:
            observation = sensors.snapshot()
            break
        except RuntimeError:
            if time.monotonic() > deadline:
                raise
            time.sleep(.05)
    if rospy.is_shutdown():
        raise KeyboardInterrupt()
    if args.check_only:
        require_healthy_can()
        start_recording(args, sensors, "CHECK_ONLY", video=False)
        deadline = time.monotonic() + args.observe_seconds
        while time.monotonic() < deadline:
            if rospy.is_shutdown():
                raise KeyboardInterrupt()
            sensors.state()
            RUN_LOG.update(hardware_preflight=sensors.last_hardware)
            time.sleep(0.05)
        print("Preflight passed: fresh ROS/CAN feedback agrees for both arms; no commands published", flush=True)
        return
    if args.observe_only:
        start_recording(args, sensors, "OBSERVE_ONLY")
        deadline = time.monotonic() + args.observe_seconds
        while time.monotonic() < deadline:
            if rospy.is_shutdown():
                raise KeyboardInterrupt()
            sensors.snapshot()
            RUN_LOG.check()
            if VIDEO:
                VIDEO.check()
            time.sleep(1/30)
        return
    if args.execute:
        ROBOT = PiperRobot(sensors, args, RUN_LOG, VIDEO)
        ROBOT.connect()
    if args.reset_only:
        ROBOT.prepare()
        ROBOT.reset()
        print("Reset completed; no video or run record created", flush=True)
        return
    from openpi_client import msgpack_numpy
    from websockets.sync.client import connect
    steps, queries = 0, 0
    packer = msgpack_numpy.Packer()
    with connect_policy("ws://%s:%d" % (args.host, args.port), args.server_wait_seconds, connect) as websocket:
        metadata = msgpack_numpy.unpackb(websocket.recv(timeout=10))
        from piper_checkpoint import require_native_policy_contract
        try:
            require_native_policy_contract(metadata)
        except ValueError as error:
            raise RuntimeError(str(error)) from error
        if args.expected_checkpoint is not None:
            advertised = metadata.get("checkpoint")
            if not advertised or Path(advertised).expanduser().resolve() != args.expected_checkpoint.expanduser().resolve():
                raise RuntimeError("Server checkpoint does not match the selected checkpoint: %s" % advertised)
        if args.rtc and metadata.get("rtc_protocol") != "piper_rtc_v1":
            raise RuntimeError("模型服务尚未启用 RTC，请停止并重新加载模型服务，或关闭 RTC 后运行")
        RUN_LOG.update(server=metadata)
        if ROBOT:
            ROBOT.prepare()
            if args.reset_before:
                ROBOT.reset()
                wait_post_reset_images(sensors)
        start_recording(args, sensors, "INFERENCE")
        if args.rtc:
            from piper_rtc_runtime import run_rtc_client
            run_rtc_client(args, sensors, ROBOT, RUN_LOG, VIDEO, websocket, stopped=rospy.is_shutdown)
            return
        last_subtask = None
        while steps < args.max_steps and not rospy.is_shutdown():
            RUN_LOG.check()
            if VIDEO:
                VIDEO.check()
            if args.execute:
                require_healthy_can()
            observation = sensors.snapshot()
            observation_metadata = dict(sensors.last_snapshot_meta)
            observation["prompt"] = args.prompt
            begun = time.monotonic()
            websocket.send(packer.pack(observation))
            message = websocket.recv(timeout=5)
            if isinstance(message, str):
                raise RuntimeError(message)
            result = msgpack_numpy.unpackb(message)
            response_time = time.time()
            response_monotonic = time.monotonic()
            roundtrip_ms = (time.monotonic() - begun) * 1000
            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.shape != (50, 14) or not np.isfinite(actions).all():
                raise RuntimeError("Invalid action chunk")
            queries += 1
            RUN_LOG.status(queries=queries)
            entry = {"query": queries, "time": response_time, "executing": args.execute,
                     "response_monotonic": response_monotonic,
                     "observation": observation_metadata,
                     "state": observation["state"].tolist(), "raw_actions": actions.tolist(),
                     "action_step_times": [],
                     "command_transform": "Native absolute joints, grippers clipped to [0, 0.09] metres",
                     "subtask": result.get("subtask"), "subtask_status": result.get("subtask_status"),
                     "subtask_score": result.get("subtask_score"),
                     "roundtrip_ms": roundtrip_ms,
                     "policy_timing": result.get("policy_timing"), "steps_before": steps,
                     "gripper_clipped_values": int(np.sum((actions[:, [6,13]] < 0) | (actions[:, [6,13]] > .09)))}
            RUN_LOG.add_query(entry)
            if VIDEO:
                VIDEO.update_prediction(entry)
            if queries == 1 or queries % args.log_every == 0 or entry["subtask"] != last_subtask:
                print("q=%d steps=%d subtask=%r status=%s %.0fms" % (
                    queries, steps, entry["subtask"], entry["subtask_status"], roundtrip_ms), flush=True)
            last_subtask = entry["subtask"]
            rate = ControlRate(30)
            for action in actions[:min(args.chunk_steps, args.max_steps - steps)]:
                if rospy.is_shutdown():
                    break
                if VIDEO:
                    VIDEO.check()
                ROBOT.step(action)
                RUN_LOG.action_sent(entry, time.time())
                steps += 1
                RUN_LOG.status(steps=steps)
                rate.sleep()
    if rospy.is_shutdown():
        raise KeyboardInterrupt()
    print("Finished bounded execution: %d steps, %d queries" % (steps, queries), flush=True)


def run():
    exit_code = 0
    try:
        main()
        if RUN_LOG:
            RUN_LOG.status(status="finished")
    except (KeyboardInterrupt, rospy.ROSInterruptException):
        if RUN_LOG:
            RUN_LOG.status(status="interrupted")
        print("Stopping; saving recorded inference output..." if RUN_LOG and RUN_LOG.recording
              else "Stopping; no run recording was started", flush=True)
        exit_code = 130
    except SystemExit:
        raise
    except BaseException as error:
        if RUN_LOG:
            RUN_LOG.status(status="error", error=repr(error))
        print("Stopped: " + str(error), flush=True)
        exit_code = 1
    finally:
        # Revoke this client's command publishers before waiting for video/JSON.
        if ROBOT is not None:
            ROBOT.close()
        if SENSORS is not None and hasattr(SENSORS, "last_hardware") and RUN_LOG:
            RUN_LOG.update(hardware_final=SENSORS.last_hardware)
        video_report = None
        if VIDEO is not None:
            try:
                video_report = VIDEO.stop()
                if video_report["error"]:
                    raise RuntimeError(video_report["error"])
                print("Video: %.2fs, %d frames, %d missed slots" % (
                    video_report["encoded_seconds"], video_report["frame_count"],
                    video_report["repeated_timing_frames"]), flush=True)
            except Exception as error:
                RUN_LOG.status(status="error", video_error=repr(error))
                print("Video error: " + str(error), flush=True)
                exit_code = 1
        if SENSORS is not None:
            SENSORS.close()
        if RUN_LOG is not None:
            RUN_LOG.status(ended=time.time(), phase="STOPPED")
            try:
                RUN_LOG.close(video_report)
                if RUN_LOG.recording:
                    print("Saved: " + str(RUN_LOG.path.resolve()), flush=True)
            except Exception as error:
                print("Failed to finalize run.json: " + str(error), flush=True)
                exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run())
