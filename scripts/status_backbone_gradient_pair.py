"""Read-only launch/config status; deliberately no GPU telemetry."""

import json
from pathlib import Path

import psutil


def alive(record):
    try:
        p = psutil.Process(record["pid"])
        return abs(p.create_time() - record["created"]) < .01 and p.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


root = Path.cwd()
logs = root / "logs/pi05_piper_backbone_grad"
identity = json.loads((logs / "pair_seed42.process.json").read_text())
result = {"supervisor_alive":alive(identity), "supervisor":identity,
          "engineering":json.loads((logs / "pair_seed42/engineering_passed.json").read_text()), "runs":{}}
for mode in ("limited", "full"):
    output = root / "checkpoints/pi05_piper_backbone_grad" / f"{mode}_seed42"
    if not (output / "run_config.json").exists():
        result["runs"][mode] = {"status":"scheduled_after_limited", "output":str(output)}
        continue
    cfg = json.loads((output / "run_config.json").read_text())
    expected = {"steps":5000, "global_batch":256, "seed":42, "checkpoint_every":500,
                "warmup":500, "peak_lr":2.5e-5, "decay_lr":2.5e-6, "world_size":8,
                "memory_fraction":.55, "engineering_smoke":False}
    for key, value in expected.items():
        if cfg[key] != value:
            raise AssertionError(f"Unexpected formal {key}: {cfg[key]}")
    assert cfg["batch_size"] * cfg["accumulation"] * cfg["world_size"] == 256
    records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()] if (output / "metrics.jsonl").exists() else []
    updates = [row for row in records if row.get("event") == "train"]
    result["runs"][mode] = {"status":"training" if updates else "initializing", "output":str(output),
                           "verified_recipe":expected, "last_update":updates[-1] if updates else None,
                           "first_update_visualization":(output / "first_update/manifest.json").exists()}
control = json.loads((root / "logs/pi05_subtask_stage1/gpu_reservation/control.json").read_text())
result["exclusive_job_control"] = control
if "job_pid" in control:
    result["exclusive_job_alive"] = alive({"pid":control["job_pid"], "created":control["job_created"]})
print(json.dumps(result, indent=2), flush=True)
