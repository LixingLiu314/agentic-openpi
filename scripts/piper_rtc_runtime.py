"""Connect the tested RTC scheduler to existing native controls and evidence."""
import time

import numpy as np

from piper_rtc import execute_rtc


def run_rtc_client(args, sensors, robot, run_log, video, websocket, *, stopped):
    from openpi_client import msgpack_numpy
    packer = msgpack_numpy.Packer()
    last_subtask = [None]

    def snapshot():
        observation, metadata = sensors.snapshot(with_metadata=True)
        observation["prompt"] = args.prompt
        return observation, metadata

    def request(job):
        observation = dict(job["observation"], rtc=job["rtc"])
        begun = time.monotonic()
        websocket.send(packer.pack(observation))
        message = websocket.recv(timeout=5)
        if isinstance(message, str):
            raise RuntimeError(message)
        result = msgpack_numpy.unpackb(message)
        result["client_received_time"] = time.time()
        result["client_received_monotonic"] = time.monotonic()
        result["client_roundtrip_ms"] = (time.monotonic()-begun)*1000
        return result

    def record(job, result, skipped, total):
        actions = np.asarray(result["actions"], dtype=np.float32)
        entry = dict(query=job["rtc"]["query_id"], time=result["client_received_time"],
                     executing=True, response_monotonic=result["client_received_monotonic"],
                     observation=job["observation_metadata"], state=job["observation"]["state"].tolist(),
                     raw_actions=actions.tolist(), action_step_times=[], action_step_indices=[],
                     command_transform="Native absolute joints, grippers clipped to [0, 0.09] metres",
                     subtask=result.get("subtask"), subtask_status=result.get("subtask_status"),
                     subtask_score=result.get("subtask_score"), roundtrip_ms=result["client_roundtrip_ms"],
                     policy_timing=result.get("policy_timing"), steps_before=job["start_step"],
                     gripper_clipped_values=int(np.sum((actions[:, [6,13]]<0)|(actions[:, [6,13]]>.09))),
                     rtc=dict(result["rtc"], skipped_elapsed_actions=skipped, adopted_at_step=total))
        run_log.add_query(entry)
        run_log.status(queries=entry["query"], rtc=True, rtc_skipped_actions=skipped)
        if video:
            video.update_prediction(entry)
        if entry["query"]==1 or entry["query"]%args.log_every==0 or entry["subtask"]!=last_subtask[0]:
            print("RTC q=%d steps=%d skip=%d subtask=%r %.0fms" %
                  (entry["query"], total, skipped, entry["subtask"], entry["roundtrip_ms"]), flush=True)
        last_subtask[0] = entry["subtask"]
        return entry

    def action_sent(entry, index, total):
        with run_log.lock:
            entry["action_step_times"].append(time.time())
            entry["action_step_indices"].append(index)
        run_log.status(steps=total+1)

    def check():
        run_log.check()
        if video:
            video.check()

    report = execute_rtc(request=request, snapshot=snapshot, publish=robot.step,
                         record=record, action_sent=action_sent, check=check, stopped=stopped,
                         max_steps=args.max_steps, chunk_steps=args.chunk_steps,
                         on_shutdown=robot.close)
    run_log.update(rtc_summary=report)
    print("Finished RTC: %d steps, %d queries" % (report["steps"], report["queries"]), flush=True)
