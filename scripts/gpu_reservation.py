"""Keep per-GPU leases across managed training exits, including failures.

Use `start` for the persistent manager and `run -- ...` for each training job.
Workers hold a small buffer during a live job. Otherwise they reserve 85% of
VRAM and continuously execute matrix multiplies, targeting >=80% GPU utilization
and >=80% memory occupancy. The manager records measured utilization in health.json.
The control file includes PID creation time, so a recycled PID cannot suppress
the fallback reservation. This program only manages its own workers/jobs.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

WORKER_REVISION = 3


def write_json(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def alive(pid, created):
    try:
        process = psutil.Process(pid)
        return abs(process.create_time() - created) < 0.01 and process.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def active_job(directory):
    states = [directory / "control.json", *sorted((directory / "concurrent_jobs").glob("*.json"))]
    for path in states:
        try:
            state = json.loads(path.read_text())
            if alive(state["job_pid"], state["job_created"]):
                return True
            if state.get("supervisor_pid") and alive(state["supervisor_pid"], state["supervisor_created"]):
                return True
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            continue
    return False


def legacy_active_job(directory):
    """Identity of the exclusive job, retained separately for shared-run records."""
    try:
        state = json.loads((directory / "control.json").read_text())
        return alive(state["job_pid"], state["job_created"])
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return False


def run_concurrent(directory, logfile, command):
    """User-authorized shared GPU job; never overwrite the exclusive job's control."""
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("A command is required")
    manager_state = json.loads((directory / "manager.json").read_text())
    if not alive(manager_state["pid"], manager_state["created"]):
        raise RuntimeError("Reservation manager is not alive")
    for gpu in manager_state["gpus"]:
        state = json.loads((directory / f"gpu_{gpu}.json").read_text())
        if state.get("revision", 0) < 3 or not alive(state["pid"], state["created"]):
            raise RuntimeError("All reservation workers must support concurrent leases before joining")
    leases = directory / "concurrent_jobs"
    leases.mkdir(exist_ok=True)
    lease = leases / f"{os.getpid()}_{time.time_ns()}.json"
    record = {"job_pid":os.getpid(), "job_created":psutil.Process().create_time(),
              "supervisor_pid":os.getpid(), "supervisor_created":psutil.Process().create_time(),
              "command":command, "log":str(logfile), "phase":"handoff", "started":time.time()}
    logfile.parent.mkdir(parents=True, exist_ok=True)
    process = None
    old_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        with logfile.open("x") as stream:
            write_json(lease, record)
            deadline = time.monotonic() + 60
            while True:
                states = [json.loads((directory / f"gpu_{gpu}.json").read_text()) for gpu in manager_state["gpus"]]
                if all(state.get("mode") == "training" and state.get("reserved_gib", 100) < 1 for state in states):
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("Concurrent lease did not release fallback buffers")
                time.sleep(.25)
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            record.update(job_pid=process.pid, job_created=psutil.Process(process.pid).create_time(), phase="training")
            write_json(lease, record)
            code = process.wait()
            write_json(directory / f"concurrent_exit_{os.getpid()}.json", {**record, "exit_code":code, "ended":time.time()})
            return code
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if lease.exists():
            lease.unlink()
        signal.signal(signal.SIGTERM, old_handler)


def worker(directory, gpu, fraction):
    import torch

    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    total = torch.cuda.get_device_properties(0).total_memory
    base = torch.empty(64 * 1024**2, device="cuda", dtype=torch.uint8)
    compute = None
    reserve = []
    reserved_bytes = 0
    last_report = 0
    previous_mode = None
    mode_since = time.time()
    while True:
        mode = "training" if active_job(directory) else "holding"
        if mode != previous_mode:
            mode_since = time.time()
        if mode == "training" and (reserve or compute is not None):
            # Finish queued kernels before acknowledging the handoff. No large
            # compute buffers or background compute compete with a managed job.
            torch.cuda.synchronize()
            compute = None
            reserve.clear()
            reserved_bytes = 0
            torch.cuda.empty_cache()
        if mode == "holding":
            if compute is None:
                compute = (
                    torch.full((8192, 8192), 0.001, device="cuda", dtype=torch.float16),
                    torch.full((8192, 8192), 0.001, device="cuda", dtype=torch.float16),
                    torch.empty((8192, 8192), device="cuda", dtype=torch.float16),
                )
            nonreserve_bytes = torch.cuda.memory_allocated() - reserved_bytes
            target = max(0, int(total * fraction) - nonreserve_bytes)
            free, _ = torch.cuda.mem_get_info()
            # Leave 2 GiB free for CUDA context setup and handoff; grow as the old job releases memory.
            remaining = min(target - reserved_bytes, max(0, free - 2 * 1024**3))
            while remaining >= 256 * 1024**2:
                size = min(remaining, 1024**3)
                try:
                    reserve.append(torch.empty(size, device="cuda", dtype=torch.uint8))
                    reserved_bytes += size
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    break
                remaining -= size
            # A bounded batch keeps the GPU busy while allowing prompt handoff.
            # Reuse an output buffer; no growing graph or accumulation of results.
            for _ in range(8):
                torch.mm(compute[0], compute[1], out=compute[2])
            torch.cuda.synchronize()
        now = time.time()
        if now - last_report > 2 or mode != previous_mode:
            write_json(
                directory / f"gpu_{gpu}.json",
                {
                    "pid": os.getpid(),
                    "created": psutil.Process().create_time(),
                    "gpu": gpu,
                    "mode": mode,
                    "mode_since": mode_since,
                    "revision": WORKER_REVISION,
                    "compute_mode": "continuous_8192_fp16_matmul" if compute is not None else "released",
                    "reserved_gib": torch.cuda.memory_allocated() / 2**30,
                    "reservation_buffer_gib": (base.numel() + reserved_bytes) / 2**30,
                    "memory_target_fraction": fraction,
                    "updated": now,
                },
            )
            last_report = now
            previous_mode = mode
        if mode == "training":
            time.sleep(0.25)


def record_health(directory, gpus):
    """Measure both requested thresholds without interrupting worker compute."""
    now = time.time()
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        devices = {}
        for line in output.splitlines():
            index, total, used, utilization, temperature, power = [value.strip() for value in line.split(",")]
            if int(index) not in gpus:
                continue
            state = json.loads((directory / f"gpu_{index}.json").read_text())
            memory_percent = 100 * float(used) / float(total)
            holding = state.get("mode") == "holding"
            stable = holding and now - state.get("mode_since", now) >= 10 and now - state["updated"] < 10
            devices[index] = {
                "mode": state.get("mode"),
                "worker_revision": state.get("revision"),
                "utilization_gpu_percent": float(utilization),
                "memory_used_mib": float(used),
                "memory_total_mib": float(total),
                "memory_used_percent": memory_percent,
                "temperature_c": float(temperature),
                "power_w": float(power),
                "worker_heartbeat_age_seconds": max(0, time.time() - state["updated"]),
                "holding_stable": stable,
                "meets_holding_targets": float(utilization) >= 80 and memory_percent >= 80 if stable else None,
            }
        write_json(
            directory / "health.json",
            {
                "updated": now,
                "gpu_utilization_min_percent": 80,
                "memory_occupancy_min_percent": 80,
                "scope": "holding mode; training releases both large buffers and compute",
                "devices": devices,
            },
        )
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        write_json(directory / "health.json", {"updated": now, "error": str(error)})


def manager(directory, gpus, fraction):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "manager.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(
            directory / "manager.json",
            {"pid": os.getpid(), "created": psutil.Process().create_time(), "gpus": gpus, "revision": WORKER_REVISION},
        )
        last_health = 0
        while True:
            for gpu in gpus:
                status_file = directory / f"gpu_{gpu}.json"
                try:
                    status = json.loads(status_file.read_text())
                    if alive(status["pid"], status["created"]):
                        continue
                except (FileNotFoundError, KeyError, json.JSONDecodeError):
                    pass
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=str(gpu), JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1")
                with (directory / f"gpu_{gpu}.log").open("a") as stream:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            __file__,
                            "worker",
                            "--directory",
                            str(directory),
                            "--gpu",
                            str(gpu),
                            "--fraction",
                            str(fraction),
                        ],
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                write_json(
                    status_file,
                    {
                        "pid": process.pid,
                        "created": psutil.Process(process.pid).create_time(),
                        "gpu": gpu,
                        "mode": "starting",
                        "updated": time.time(),
                    },
                )
            if time.monotonic() - last_health >= 2:
                record_health(directory, gpus)
                last_health = time.monotonic()
            time.sleep(1)


def start(directory, gpus, fraction):
    directory.mkdir(parents=True, exist_ok=True)
    try:
        current = json.loads((directory / "manager.json").read_text())
        if alive(current["pid"], current["created"]):
            print(json.dumps(current))
            return
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        pass
    with (directory / "manager.log").open("a") as stream:
        process = subprocess.Popen(
            [
                sys.executable,
                __file__,
                "manager",
                "--directory",
                str(directory),
                "--gpus",
                ",".join(map(str, gpus)),
                "--fraction",
                str(fraction),
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(json.dumps({"manager_pid": process.pid, "directory": str(directory)}))


def run(directory, logfile, command):
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("A training command is required after --")
    manager_state = json.loads((directory / "manager.json").read_text())
    if not alive(manager_state["pid"], manager_state["created"]):
        raise RuntimeError("Start the GPU reservation manager before launching training")
    with (directory / "job.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if active_job(directory):
            raise RuntimeError("Another managed job is still active")
        # The supervisor is alive during the handoff, so workers release large buffers first.
        write_json(
            directory / "control.json",
            {
                "job_pid": os.getpid(),
                "job_created": psutil.Process().create_time(),
                "phase": "handoff",
                "command": command,
            },
        )
        try:
            deadline = time.monotonic() + 60
            while True:
                states = [json.loads((directory / f"gpu_{gpu}.json").read_text()) for gpu in manager_state["gpus"]]
                if all(state.get("mode") == "training" and state.get("reserved_gib", 100) < 1 for state in states):
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("GPU workers did not release their large reservations")
                time.sleep(0.25)
            logfile.parent.mkdir(parents=True, exist_ok=True)
            with logfile.open("x") as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                write_json(
                    directory / "control.json",
                    {
                        "job_pid": process.pid,
                        "job_created": psutil.Process(process.pid).create_time(),
                        "phase": "training",
                        "command": command,
                        "log": str(logfile),
                    },
                )
                write_json(
                    directory / "last_job.json",
                    {"pid": process.pid, "command": command, "log": str(logfile), "started": time.time()},
                )
                exit_code = process.wait()
            write_json(
                directory / "last_exit.json",
                {"pid": process.pid, "exit_code": exit_code, "ended": time.time(), "log": str(logfile)},
            )
            return exit_code
        finally:
            write_json(directory / "control.json", {"phase": "holding", "updated": time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["start", "manager", "worker", "run", "run-concurrent"])
    parser.add_argument("--directory", type=Path, default=Path("logs/pi05_subtask_stage1/gpu_reservation"))
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--fraction", type=float, default=0.85)
    parser.add_argument("--log", type=Path)
    args, command = parser.parse_known_args()
    if not 0.8 <= args.fraction <= 0.95:
        raise ValueError("GPU reservation fraction must be between 0.80 and 0.95")
    directory = args.directory.resolve()
    if args.mode == "worker":
        worker(directory, args.gpu, args.fraction)
    elif args.mode == "manager":
        manager(directory, [int(value) for value in args.gpus.split(",")], args.fraction)
    elif args.mode == "start":
        start(directory, [int(value) for value in args.gpus.split(",")], args.fraction)
    else:
        if args.log is None:
            raise ValueError("--log is required for a managed run")
        runner = run_concurrent if args.mode == "run-concurrent" else run
        sys.exit(runner(directory, args.log, command))


if __name__ == "__main__":
    main()
