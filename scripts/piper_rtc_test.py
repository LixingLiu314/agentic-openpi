"""No-motion RTC timing, ordering, expiry, cancellation and sampler contracts."""
import copy
from pathlib import Path
import threading
import time
import unittest

import numpy as np

from piper_rtc import PROTOCOL, Timeline, execute_rtc


def response(job):
    actions = np.repeat((np.arange(50)+job["start_step"])[:, None], 14, axis=1).astype(np.float32)/1000
    return {"actions": actions, "rtc": dict(job["rtc"]), "subtask": "reach"}


class RTCTests(unittest.TestCase):
    def run_loop(self, request, *, maximum=65, stop=None):
        self.commands, self.records, self.timestamps, self.indices = [], [], [], []
        self.closed = False
        self.request_thread = None
        control = threading.get_ident()
        def infer(job):
            self.request_thread = threading.get_ident()
            return request(job)
        def publish(action):
            self.assertEqual(threading.get_ident(), control)
            self.commands.append(action.copy());self.timestamps.append(time.monotonic())
        def record(job, result, skipped, total):
            item=dict(query=job["rtc"]["query_id"],skipped=skipped,total=total)
            self.records.append(item);return item
        return execute_rtc(request=infer, snapshot=lambda: ({"state":np.zeros(14)},{}),
                           publish=publish, record=record,
                           action_sent=lambda entry,index,total:self.indices.append((entry["query"],index,total)),
                           check=lambda:None, stopped=stop or (lambda:False), max_steps=maximum,
                           on_shutdown=lambda:setattr(self,"closed",True))

    def test_real_async_keeps_30hz_and_skips_elapsed_actions(self):
        def request(job):
            time.sleep(.015 if job["start_step"]==0 else .12)
            return response(job)
        result=self.run_loop(request)
        self.assertEqual(len(self.commands),65)
        self.assertNotEqual(self.request_thread,threading.get_ident())
        self.assertTrue(self.closed)
        self.assertGreaterEqual(len(self.records),3)
        self.assertTrue(all(x["skipped"]>=3 for x in self.records[1:]))
        np.testing.assert_allclose(np.asarray(self.commands)[:,0],np.arange(65)/1000,atol=1e-7)
        intervals=np.diff(self.timestamps)
        self.assertGreater(intervals.min(),.025)
        self.assertLess(intervals.max(),.095)
        self.assertTrue(result["rtc"])

    def test_expired_queue_stops_without_repeating_commands(self):
        def request(job):
            if job["start_step"]:time.sleep(2)
            return response(job)
        with self.assertRaisesRegex(RuntimeError,"queue exhausted"):
            self.run_loop(request,maximum=70)
        self.assertEqual(len(self.commands),50)
        before=len(self.commands);time.sleep(.02)
        self.assertEqual(len(self.commands),before)
        self.assertTrue(self.closed)

    def test_stop_during_pending_request_never_publishes_late_result(self):
        def request(job):
            if job["start_step"]:time.sleep(.5)
            return response(job)
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop(request,stop=lambda:len(self.commands)>=30)
        self.assertEqual(len(self.commands),30)
        time.sleep(.6)
        self.assertEqual(len(self.commands),30)
        self.assertTrue(self.closed)

    def test_worker_error_propagates(self):
        def request(job):raise ConnectionError("lost model connection")
        with self.assertRaisesRegex(ConnectionError,"lost model"):
            self.run_loop(request)
        self.assertEqual(self.commands,[])
        self.assertTrue(self.closed)

    def test_response_identity_rejected_before_motion(self):
        def request(job):
            out=response(job);out["rtc"]["query_id"]+=1;return out
        with self.assertRaisesRegex(RuntimeError,"does not match"):
            self.run_loop(request)
        self.assertEqual(self.commands,[])

    def test_invalid_actions_rejected_before_motion(self):
        def request(job):
            out=response(job);out["actions"][0,0]=np.nan;return out
        with self.assertRaisesRegex(RuntimeError,"Invalid RTC action"):
            self.run_loop(request)
        self.assertEqual(self.commands,[])

    def test_timeline_never_replays_elapsed_prefix(self):
        t=Timeline();job=dict(start_step=0,rtc=dict(protocol=PROTOCOL,query_id=1,previous_query_id=None,consumed_steps=0,delay_steps=0))
        t.adopt(job,response(job));t.total=29
        next_job=dict(start_step=25,rtc=dict(protocol=PROTOCOL,query_id=2,previous_query_id=1,consumed_steps=25,delay_steps=12))
        self.assertEqual(t.adopt(next_job,response(next_job)),4)
        action,index=t.next_action();self.assertEqual(index,4);self.assertAlmostEqual(action[0],.029)
        with self.assertRaisesRegex(RuntimeError,"out-of-order"):
            t.adopt(next_job,response(next_job))
        expired=copy.deepcopy(next_job);expired["rtc"]["query_id"]=3;t.total=75
        with self.assertRaisesRegex(RuntimeError,"expired"):
            t.adopt(expired,response(expired))


if __name__=="__main__":unittest.main()
