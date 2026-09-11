"""Detached, identity-recorded launch after CPU and reservation compatibility gates."""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

import psutil

import gpu_reservation


root = Path.cwd()
logs = root / "logs/pi05_piper_backbone_grad"
target = logs / "pair_seed42.process.json"
with (logs / "launch.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if target.exists():
        previous = json.loads(target.read_text())
        if gpu_reservation.alive(previous["pid"], previous["created"]):
            raise RuntimeError("An existing pair is already active")
        raise FileExistsError("Preserve the previous launch record and choose a reviewed recovery path")
    for name in ("limited_cpu_gate.json", "full_cpu_gate.json", "reservation_concurrent_upgrade.json"):
        if not json.loads((logs / name).read_text())["passed"]:
            raise RuntimeError(f"Failed prerequisite: {name}")
    finished = [json.loads(line) for line in (logs / "limited_cpu_training_gate/metrics.jsonl").read_text().splitlines()]
    if not any(row.get("event") == "complete" and row.get("completed_steps") == 2 for row in finished):
        raise RuntimeError("CPU trainer save/resume gate incomplete")
    command = [sys.executable, "scripts/run_backbone_gradient_pair.py", "--concurrent"]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", JAX_PLATFORMS="cpu",
                       HF_HUB_OFFLINE="1", OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    with (logs / "pair_seed42.launch.log").open("x") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                    start_new_session=True, env=environment)
    record = {"pid":process.pid, "created":psutil.Process(process.pid).create_time(),
              "command":command, "mode":"concurrent with preserved RobotWin task", "seed":42,
              "log":str(logs / "pair_seed42/managed.log")}
    gpu_reservation.write_json(target, record)
    print(json.dumps(record), flush=True)
