"""Lease ownership survives either job finishing; no GPU operations."""

import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import gpu_reservation as guard


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    leases = root / "concurrent_jobs"
    leases.mkdir()
    (root / "control.json").write_text(json.dumps({"job_pid":1,"job_created":10}))
    (leases / "ours.json").write_text(json.dumps({"job_pid":2,"job_created":20,
                                                "supervisor_pid":3,"supervisor_created":30}))
    for live, expected in [({(1,10),(2,20)},True), ({(1,10)},True), ({(2,20)},True),
                           ({(3,30)},True), (set(),False)]:
        with patch.object(guard, "alive", side_effect=lambda pid, created:(pid,created) in live):
            assert guard.active_job(root) is expected
    (leases / "broken.json").write_text("{")
    with patch.object(guard, "alive", return_value=False):
        assert not guard.active_job(root)
    print(json.dumps({"passed":True,"checks":6,"scope":"exclusive/concurrent/supervisor identity and stale lease"}))
