"""Finite, gated sequence: finish M3 engineering gates, then run the M2/M3 pilot.

Every GPU child uses the reservation supervisor. Existing runs/logs are never
overwritten. A failure stops this sequence and leaves reservation recovery to
the persistent manager. This is a one-shot continuation, not a recurring task.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path("checkpoints/pi05_piper_stage1")
LOGS = Path("logs/pi05_subtask_stage1")


def require_event(run, event, step):
    rows = [json.loads(line) for line in (ROOT / run / "metrics.jsonl").read_text().splitlines()]
    if not any(row.get("event") == event and row.get("completed_steps") == step for row in rows):
        raise RuntimeError(f"Missing {run} {event} at {step}")


def job(name, arguments):
    log = LOGS / f"{name}.log"
    if log.exists() or (LOGS / f"{name}.sequence_command.json").exists():
        raise FileExistsError(f"Refusing to overwrite prior job records: {name}")
    command = [
        sys.executable,
        "scripts/gpu_reservation.py",
        "run",
        "--log",
        str(log),
        "--",
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        "scripts/train_subtask_hierarchy.py",
        *arguments,
    ]
    record = {"event": "launch", "name": name, "time": time.time(), "command": command}
    (LOGS / f"{name}.sequence_command.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    subprocess.run(command, check=True)
    print(json.dumps({"event": "job_finished", "name": name, "time": time.time()}), flush=True)


def verify(parent, child, report):
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
    subprocess.run(
        [
            sys.executable,
            "scripts/verify_hierarchy_freeze.py",
            "--parent",
            str(ROOT / parent),
            "--child",
            str(ROOT / child),
            "--output",
            str(LOGS / report),
        ],
        env=environment,
        check=True,
    )


def main():
    require_event("m2_trainer_ddp8_smoke_seed42", "complete", 2)
    verify("m1_pilot_seed42/step_001900", "m2_trainer_ddp8_smoke_seed42/step_000002", "m2_checkpoint_freeze_ddp8.json")
    common = ["--batch-size", "2", "--accumulation", "2", "--workers", "2", "--cpu-threads", "4"]
    smoke = [
        "--stage",
        "m3",
        "--initialize-from",
        str(ROOT / "m2_trainer_ddp8_smoke_seed42/step_000002"),
        "--output",
        str(ROOT / "m3_trainer_ddp8_smoke_seed42"),
        "--steps",
        "8",
        "--m3-planned-steps",
        "8",
        *common,
        "--eval-samples",
        "8",
        "--eval-draws",
        "1",
        "--eval-every",
        "2",
        "--checkpoint-every",
        "2",
        "--warmup-subtask",
        "1",
        "--warmup-action",
        "1",
        "--engineering-smoke",
    ]
    job("m3_trainer_ddp8_smoke_start", [*smoke, "--stop-after", "4"])
    require_event("m3_trainer_ddp8_smoke_seed42", "stopped_at_checkpoint", 4)
    job("m3_trainer_ddp8_smoke_resume", [*smoke, "--resume"])
    require_event("m3_trainer_ddp8_smoke_seed42", "complete", 8)
    metadata = json.loads((ROOT / "m3_trainer_ddp8_smoke_seed42/step_000008/metadata.json").read_text())
    if metadata["counters"]["action"] != 10 or metadata["best"]["step"] != 8:
        raise RuntimeError("M3 engineering action budget/late-checkpoint gate failed")
    verify(
        "m2_trainer_ddp8_smoke_seed42/step_000002",
        "m3_trainer_ddp8_smoke_seed42/step_000008",
        "m3_checkpoint_freeze_ddp8.json",
    )
    require_event("m1_pilot_seed42", "complete", 2000)
    research = [
        *common,
        "--m3-planned-steps",
        "3500",
        "--eval-samples",
        "128",
        "--eval-draws",
        "2",
        "--eval-every",
        "100",
        "--checkpoint-every",
        "100",
        "--warmup-subtask",
        "200",
        "--warmup-action",
        "200",
    ]
    job(
        "m2_pilot_seed42",
        [
            "--stage",
            "m2",
            "--initialize-from",
            str(ROOT / "m1_pilot_seed42/step_001900"),
            "--output",
            str(ROOT / "m2_pilot_seed42"),
            "--steps",
            "500",
            *research,
        ],
    )
    require_event("m2_pilot_seed42", "complete", 500)
    verify("m1_pilot_seed42/step_001900", "m2_pilot_seed42/step_000500", "m2_research_checkpoint_freeze.json")
    job(
        "m3_pilot_seed42",
        [
            "--stage",
            "m3",
            "--initialize-from",
            str(ROOT / "m2_pilot_seed42/step_000500"),
            "--output",
            str(ROOT / "m3_pilot_seed42"),
            "--steps",
            "3500",
            *research,
        ],
    )
    require_event("m3_pilot_seed42", "complete", 3500)
    verify("m2_pilot_seed42/step_000500", "m3_pilot_seed42/step_003500", "m3_research_checkpoint_freeze.json")
    print(json.dumps({"event": "sequence_complete", "time": time.time()}), flush=True)


if __name__ == "__main__":
    main()
