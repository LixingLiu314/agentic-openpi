import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

import piper_recording_test as recording
from piper_gui_support import client_command
from piper_rtc import PROTOCOL


class LegacyRecordingTests(recording.RecordingTests):
    def run_client(self,*args):
        return super().run_client("--no-rtc",*args)


class RTCRuntimeTests(recording.RecordingTests):
    # The inherited lifecycle cases explicitly exercise synchronous fallback.
    def run_client(self,*args):
        return super().run_client("--no-rtc",*args)

    def snapshot(self,**kwargs):
        observation=super().snapshot(**kwargs)
        return (observation,dict(self.sensors.last_snapshot_meta)) if kwargs.get("with_metadata") else observation

    def test_rtc_default_executes_and_records_actual_action_indices(self):
        first={"protocol":PROTOCOL,"query_id":1,"previous_query_id":None,"consumed_steps":0,"delay_steps":0}
        self.websocket.recv.side_effect=[
            self.pack.pack({"stage":"m3","state_dim":14,"action_horizon":50,"rtc_protocol":PROTOCOL}),
            self.pack.pack({"actions":np.zeros((50,14),np.float32),"subtask":"reach","rtc":first})]
        self.assertEqual(recording.RecordingTests.run_client(self,"--execute"),0)
        report=json.loads((self.output/"run.json").read_text())
        self.assertTrue(report["config"]["rtc"])
        self.assertEqual(report["config"]["chunk_steps"],25)
        self.assertEqual(report["queries"][0]["action_step_indices"],[0])
        self.assertEqual(report["rtc_summary"]["steps"],1)
        self.robot.step.assert_called_once()
        self.assertNotIn("reset",report)

    def test_rtc_capability_mismatch_rejects_before_enable_or_reset(self):
        self.assertEqual(recording.RecordingTests.run_client(self,"--execute"),1)
        self.robot.prepare.assert_not_called()
        self.robot.reset.assert_not_called()
        self.robot.step.assert_not_called()
        self.assert_no_recording()


class GUIRTCTests(unittest.TestCase):
    def test_cli_on_and_off_are_explicit(self):
        from piper_gui_test import config
        for value,flag in [(True,"--rtc"),(False,"--no-rtc")]:
            cmd=client_command("/repo",config(rtc=value),"/out",123,"token")
            self.assertIn(flag,cmd)
        self.assertIn("--rtc",client_command("/repo",config(),"/out",123,"token"))

    def test_default_persistence_and_running_lock(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
        from PyQt5 import QtWidgets
        from piper_eval_gui import PiperWindow
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"settings.json"
            window=PiperWindow(root=Path(root),settings_path=path)
            self.assertTrue(window.rtc.isChecked())
            self.assertEqual(window.chunk_steps.value(),25)
            window.rtc.setChecked(False);window.chunk_steps.setValue(40);window._persist()
            window.close()
            second=PiperWindow(root=Path(root),settings_path=path)
            self.assertFalse(second.rtc.isChecked());self.assertEqual(second.chunk_steps.value(),40)
            second.rtc.setChecked(True);self.assertEqual(second.chunk_steps.value(),25)
            second.client=Mock();second._set_busy();self.assertFalse(second.rtc.isEnabled())
            second.client=None;second.close()


if __name__=="__main__":unittest.main()
