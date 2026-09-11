"""Adopt concurrent leases by restarting only identity-verified guard workers."""

import json
from pathlib import Path
import signal
import time

import psutil

import gpu_reservation as guard


def main():
    root = Path.cwd()
    directory = root / "logs/pi05_subtask_stage1/gpu_reservation"
    output = root / "logs/pi05_piper_backbone_grad/reservation_concurrent_upgrade.json"
    original_control = json.loads((directory / "control.json").read_text())
    manager = json.loads((directory / "manager.json").read_text())
    if not guard.alive(manager["pid"], manager["created"]):
        raise RuntimeError("Existing manager not alive")
    transitions = []
    for gpu in manager["gpus"]:
        path = directory / f"gpu_{gpu}.json"
        old = json.loads(path.read_text())
        if old.get("revision", 0) >= 3 and guard.alive(old["pid"], old["created"]):
            continue
        process = psutil.Process(old["pid"])
        command = process.cmdline()
        if not guard.alive(old["pid"], old["created"]) or "worker" not in command or "--gpu" not in command:
            raise RuntimeError("Guard worker identity mismatch")
        if not any(Path(token).name == "gpu_reservation.py" for token in command):
            raise RuntimeError("Unexpected worker executable")
        if command[command.index("--gpu") + 1] != str(gpu) or Path(process.cwd()).resolve() != root:
            raise RuntimeError("Unexpected worker project/device")
        process.send_signal(signal.SIGTERM)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            current = json.loads(path.read_text())
            if current.get("revision", 0) >= 3 and guard.alive(current["pid"], current["created"]):
                break
            time.sleep(.5)
        else:
            raise TimeoutError(f"Manager did not adopt new worker for GPU {gpu}")
        transitions.append({"gpu":gpu, "old_pid":old["pid"], "new_pid":current["pid"], "revision":current["revision"]})
        print(json.dumps(transitions[-1]), flush=True)
    current_control = json.loads((directory / "control.json").read_text())
    result = {"passed":True, "transitions":transitions, "initial_exclusive_control":original_control,
              "current_exclusive_control":current_control, "robotwin_stop_signals_sent":False,
              "scope":"Only the eight project reservation workers were restarted; existing manager/job preserved"}
    guard.write_json(output, result)


if __name__ == "__main__":
    main()
