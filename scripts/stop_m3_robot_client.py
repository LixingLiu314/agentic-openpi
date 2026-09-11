"""Stop only this deployment's recorded action client, preserving arm holding."""

import json
import os
from pathlib import Path
import signal


root = Path.home() / "agentic-openpi"
record_path = root / "logs/robot_m3_seed42_20260907/robot_client.process.json"
if not record_path.exists():
    raise SystemExit("No managed robot client has been started")
record = json.loads(record_path.read_text())
proc = Path("/proc") / str(record["pid"])
if not proc.exists():
    raise SystemExit("The recorded robot client has already stopped")
arguments = proc.joinpath("cmdline").read_bytes().split(b"\0")
ticks = proc.joinpath("stat").read_text().rsplit(")", 1)[1].split()[19]
if ticks != record["start_ticks"] or not any(x.endswith(b"/run_subtask_piper.py") for x in arguments):
    raise SystemExit("Process identity changed; no signal sent")
os.kill(record["pid"], signal.SIGINT)
print("Stopped M3 action client PID %d; model service and arm driver remain available" % record["pid"])
