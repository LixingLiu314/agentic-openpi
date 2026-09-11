"""Bounded 30 Hz RTC client; only the caller's control thread publishes actions.

No ROS dependency: the same loop is exercised with fake hardware and a real
WebSocket in the no-motion tests. One request at a time preserves S chronology.
"""
from collections import deque
import math
import queue
import threading
import time

import numpy as np

PROTOCOL = "piper_rtc_v1"


class InferenceWorker:
    def __init__(self, request):
        self.request = request
        self.jobs, self.results = queue.Queue(1), queue.Queue(1)
        self.stopping = threading.Event()
        self.busy = False
        self.thread = threading.Thread(target=self._run, name="rtc-inference", daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stopping.is_set():
            job = self.jobs.get()
            if job is None or self.stopping.is_set():
                return
            try:
                value = self.request(job)
                result = (job, value, None)
            except BaseException as error:
                result = (job, None, error)
            if not self.stopping.is_set():
                self.results.put_nowait(result)

    def submit(self, job):
        if self.busy or self.stopping.is_set():
            raise RuntimeError("RTC permits only one outstanding inference")
        self.busy = True
        self.jobs.put_nowait(job)

    def poll(self):
        try:
            job, result, error = self.results.get_nowait()
        except queue.Empty:
            return None
        self.busy = False
        if error is not None:
            raise error
        return job, result

    def cancel(self):
        # The worker cannot publish. The socket owner closes its connection
        # after command publishers are revoked; late results are never adopted.
        self.stopping.set()
        try:
            self.jobs.put_nowait(None)
        except queue.Full:
            pass


class Timeline:
    def __init__(self):
        self.total = 0
        self.origin = 0
        self.query_id = 0
        self.actions = None
        self.entry = None

    @property
    def index(self):
        return self.total - self.origin

    def adopt(self, job, result):
        rtc = result.get("rtc", {})
        expected = job["rtc"]
        if any(rtc.get(k) != expected[k] for k in expected):
            raise RuntimeError("RTC server response does not match the pending request")
        if expected["query_id"] != self.query_id + 1:
            raise RuntimeError("Stale or out-of-order RTC result")
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.shape != (50, 14) or not np.isfinite(actions).all():
            raise RuntimeError("Invalid RTC action chunk")
        skipped = self.total - job["start_step"]
        if not 0 <= skipped < 50:
            raise RuntimeError("RTC result has already expired")
        self.origin = job["start_step"]
        self.query_id = expected["query_id"]
        self.actions = actions.copy()
        self.actions[:, [6, 13]] = np.clip(self.actions[:, [6, 13]], 0, .09)
        return skipped

    def next_action(self):
        if self.actions is None or self.index >= len(self.actions):
            raise RuntimeError("RTC action queue exhausted; stopping instead of replaying stale commands")
        return self.actions[self.index].copy(), self.index


def execute_rtc(*, request, snapshot, publish, record, action_sent, check, stopped,
                max_steps, chunk_steps=25, clock=time.monotonic, sleep=time.sleep,
                on_shutdown=lambda: None):
    if not 1 <= chunk_steps <= 25 or max_steps < 1:
        raise ValueError("RTC replanning interval must be between 1 and 25 actions")
    period = 1.0 / 30
    timeline = Timeline()
    worker = InferenceWorker(request)
    delays = deque([12], maxlen=10)
    pending = None
    last_publish = None
    next_tick = None
    peak_gap = 0.0

    def submit():
        nonlocal pending
        consumed = timeline.index if timeline.actions is not None else 0
        delay = min(max(delays), 49-consumed) if timeline.actions is not None else 0
        if timeline.actions is not None and (consumed >= 49 or max(delays) >= 50-consumed):
            raise RuntimeError("RTC latency leaves insufficient overlap; use synchronous mode")
        observation, observation_meta = snapshot()
        pending = dict(observation=observation, observation_metadata=observation_meta,
                       start_step=timeline.total, started=clock(),
                       rtc=dict(protocol=PROTOCOL, query_id=timeline.query_id+1,
                                previous_query_id=timeline.query_id if timeline.actions is not None else None,
                                consumed_steps=consumed, delay_steps=delay))
        worker.submit(pending)

    try:
        submit()
        while timeline.total < max_steps:
            if stopped():
                raise KeyboardInterrupt()
            check()
            now = clock()
            if last_publish is not None and now-last_publish > .15:
                raise RuntimeError("RTC control deadline missed by more than 150ms; stopping")
            available = worker.poll()
            if available is not None:
                job, result = available
                if job is not pending:
                    raise RuntimeError("Unexpected RTC worker result")
                elapsed = now-job["started"]
                if timeline.actions is not None:
                    executed = timeline.total-job["start_step"]
                    if elapsed-executed*period > .10:
                        raise RuntimeError("RTC observation/action timeline drift exceeded 100ms")
                    delays.append(max(1, math.ceil(elapsed/period)+1))
                skipped = timeline.adopt(job, result)
                timeline.entry = record(job, result, skipped, timeline.total)
                pending = None
                if next_tick is None:
                    next_tick = clock()
            if timeline.actions is None:
                if pending is not None and now-pending["started"] > 5:
                    raise TimeoutError("Initial RTC inference timed out")
                sleep(.005)
                continue
            if pending is not None and now-pending["started"] > 5:
                raise TimeoutError("RTC inference timed out")
            # Infer after s_min executed time slots, continuing the previous
            # tail while the model inpaints the overlapping trajectory.
            if pending is None and timeline.index >= max(chunk_steps, max(delays)) and max_steps-timeline.total > 1:
                submit()
            if clock() < next_tick:
                sleep(min(.005, next_tick-clock()))
                continue
            if stopped():
                raise KeyboardInterrupt()
            action, index = timeline.next_action()
            publish(action)
            published = clock()
            if last_publish is not None:
                peak_gap = max(peak_gap, published-last_publish)
            last_publish = published
            action_sent(timeline.entry, index, timeline.total)
            timeline.total += 1
            # Never issue a catch-up burst after a slow callback.
            next_tick = max(next_tick+period, published+period)
        return dict(steps=timeline.total, queries=timeline.query_id, rtc=True,
                    maximum_publish_gap_ms=peak_gap*1000, delay_estimate_steps=max(delays))
    finally:
        worker.cancel()
        on_shutdown()
