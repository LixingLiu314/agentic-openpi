"""Real eight-GPU reservation checks: sustained targets and normal/failing handoff."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil


def snapshot(directory):
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
        timeout=5,
    )
    rows = []
    for line in output.splitlines():
        gpu, total, used, utilization = [float(value.strip()) for value in line.split(",")]
        state = json.loads((directory / f"gpu_{int(gpu)}.json").read_text())
        process = psutil.Process(state["pid"])
        identity_ok = abs(process.create_time() - state["created"]) < 0.01 and process.status() != psutil.STATUS_ZOMBIE
        rows.append(
            {
                "gpu": int(gpu),
                "memory_percent": 100 * used / total,
                "utilization_percent": utilization,
                "mode": state["mode"],
                "reserved_gib": state.get("reserved_gib"),
                "revision": state.get("revision"),
                "identity_ok": identity_ok,
                "heartbeat_age_seconds": max(0, time.time() - state["updated"]),
            }
        )
    return {"time": time.time(), "gpus": rows}


def meets_targets(sample):
    return len(sample["gpus"]) == 8 and all(
        row["mode"] == "holding"
        and row["revision"] == 2
        and row["identity_ok"]
        and row["heartbeat_age_seconds"] < 5
        and row["memory_percent"] >= 80
        and row["utilization_percent"] >= 80
        for row in sample["gpus"]
    )


def sustained(directory, count, samples_path):
    samples = []
    for index in range(count):
        begun = time.monotonic()
        sample = snapshot(directory)
        samples.append(sample)
        with samples_path.open("a") as stream:
            stream.write(json.dumps(sample) + "\n")
        if not meets_targets(sample):
            raise AssertionError(f"Holding targets not met: {sample}")
        if (index + 1) % 10 == 0:
            print(json.dumps({"event": "sustained", "samples": index + 1, "target": count}), flush=True)
        time.sleep(max(0, 1 - (time.monotonic() - begun)))
    return {
        "samples": count,
        "gpu_observations": count * 8,
        "min_gpu_utilization": min(row["utilization_percent"] for sample in samples for row in sample["gpus"]),
        "min_memory_percent": min(row["memory_percent"] for sample in samples for row in sample["gpus"]),
        "duration_seconds": samples[-1]["time"] - samples[0]["time"] if count > 1 else 0,
    }


def probe_job(args):
    import torch
    import torch.distributed as dist

    rank = int(os.environ["LOCAL_RANK"])
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group("nccl", device_id=device)
    state = json.loads((args.directory / f"gpu_{rank}.json").read_text())
    if state["mode"] != "training" or state["compute_mode"] != "released" or state["reserved_gib"] > 0.1:
        raise AssertionError(f"Guard did not release GPU {rank}: {state}")
    buffer = torch.empty(512 * 1024**2, dtype=torch.uint8, device=device)
    matrix = torch.full((4096, 4096), 0.001, device=device, dtype=torch.float16)
    result = torch.empty_like(matrix)
    for _ in range(32):
        torch.mm(matrix, matrix, out=result)
    torch.cuda.synchronize()
    assert buffer.numel() > 0
    assert torch.isfinite(result).all()
    payload = {
        "rank": rank,
        "guard": state,
        "probe_allocated_gib": torch.cuda.memory_allocated() / 2**30,
        "finite": True,
    }
    (args.output / f"probe_{'failure' if args.fail else 'normal'}_rank_{rank:03d}.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    dist.barrier()
    dist.destroy_process_group()
    if args.fail:
        raise RuntimeError("Intentional managed-job failure to verify automatic reservation recovery")


def audit(args):
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"before": sustained(args.directory, 60, args.output / "before.jsonl"), "handoffs": []}
    for fail in [False, True]:
        kind = "failure" if fail else "normal"
        command = [
            sys.executable,
            "scripts/gpu_reservation.py",
            "run",
            "--directory",
            str(args.directory),
            "--log",
            str(args.output / f"{kind}.log"),
            "--",
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=8",
            __file__,
            "--probe-job",
            "--directory",
            str(args.directory),
            "--output",
            str(args.output),
            *(["--fail"] if fail else []),
        ]
        with (args.output / f"{kind}.supervisor.log").open("x") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            print(json.dumps({"event": "handoff_started", "kind": kind, "pid": process.pid}), flush=True)
            exit_code = process.wait(timeout=180)
        if (exit_code != 0) != fail:
            raise AssertionError(f"Unexpected {kind} exit code {exit_code}")
        if len(list(args.output.glob(f"probe_{kind}_rank_*.json"))) != 8:
            raise AssertionError("Probe did not verify release on every GPU")
        ended = json.loads((args.directory / "last_exit.json").read_text())["ended"]
        deadline, recovery = time.monotonic() + 15, []
        while True:
            sample = snapshot(args.directory)
            recovery.append(sample)
            if meets_targets(sample):
                break
            if time.monotonic() > deadline:
                raise AssertionError(f"Failed to restore targets after {kind}: {sample}")
            time.sleep(0.25)
        record = {
            "kind": kind,
            "exit_code": exit_code,
            "both_targets_observed_after_exit_seconds": sample["time"] - ended,
            "recovery_samples": recovery,
        }
        print(
            json.dumps(
                {"event": "recovered", **{key: value for key, value in record.items() if key != "recovery_samples"}}
            ),
            flush=True,
        )
        record["sustained_after"] = sustained(args.directory, 10, args.output / f"after_{kind}.jsonl")
        report["handoffs"].append(record)
    report["passed"] = True
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"event": "complete", "report": str(args.output / "report.json")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("logs/pi05_subtask_stage1/gpu_reservation"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-job", action="store_true")
    parser.add_argument("--fail", action="store_true")
    args = parser.parse_args()
    probe_job(args) if args.probe_job else audit(args)
