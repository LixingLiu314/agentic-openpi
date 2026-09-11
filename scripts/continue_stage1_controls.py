"""Finite pilot sequence: final M3 evaluation, matched B1/O gates and controls."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import psutil
from safetensors import safe_open
import torch

from openpi.training.subtask_evaluation import native_action_metrics

ROOT = Path("checkpoints/pi05_piper_stage1")
LOGS = Path("logs/pi05_subtask_stage1")
C0 = ROOT / "m0_pilot_seed42/step_001000"


def require_complete(run, steps):
    rows = [json.loads(line) for line in (ROOT / run / "metrics.jsonl").read_text().splitlines()]
    if not any(row.get("event") == "complete" and row.get("completed_steps") == steps for row in rows):
        raise RuntimeError(f"Run {run} did not complete {steps} steps")


def job(name, script, arguments):
    log = LOGS / f"{name}.log"
    record_path = LOGS / f"{name}.sequence_command.json"
    if log.exists() or record_path.exists():
        raise FileExistsError(f"Prior job records exist: {name}")
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
        script,
        *map(str, arguments),
    ]
    record = {"event": "launch", "name": name, "time": time.time(), "command": command}
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    subprocess.run(command, check=True)
    print(json.dumps({"event": "job_finished", "name": name, "time": time.time()}), flush=True)


def verify_control(parent, child, name):
    parent_metadata = json.loads((parent / "metadata.json").read_text())
    groups = {key: {"tensors": 0, "changed": 0} for key in ["frozen", "action", "subtask"]}
    with (
        safe_open(parent / "model.safetensors", framework="pt", device="cpu") as before,
        safe_open(child / "model.safetensors", framework="pt", device="cpu") as after,
    ):
        previous = set(before.keys())
        for key in after.keys():  # noqa: SIM118 - safetensors exposes keys(), not iteration
            source = key.removeprefix("base.") if parent_metadata["stage"] == "m0" else key
            if key.startswith("decoder.") and parent_metadata["stage"] == "m0":
                continue
            if source not in previous:
                raise AssertionError(f"Unexpected control tensor {key}")
            group = (
                "subtask"
                if key.startswith("decoder.")
                else "action"
                if key.startswith(("base.paligemma_with_expert.gemma_expert.model.", "base.action_", "base.time_mlp_"))
                else "frozen"
            )
            changed = not torch.equal(before.get_tensor(source), after.get_tensor(key))
            groups[group]["tensors"] += 1
            groups[group]["changed"] += int(changed)
            if group != "action" and changed:
                raise AssertionError(f"Frozen control tensor changed: {key}")
        mapped = {f"base.{key}" if parent_metadata["stage"] == "m0" else key for key in previous}
        assert mapped <= set(after.keys())
    assert groups["frozen"]["tensors"] == 604
    assert groups["action"]["changed"] > 0
    report = {"parent": str(parent), "child": str(child), "groups": groups}
    with (LOGS / name).open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({"event": "control_freeze_verified", **report}), flush=True)


def verify_archives(output, expected):
    count = 0
    identities = set()
    for path in sorted(output.glob("native_rank_*.npz")):
        rank = path.stem.rsplit("_", 1)[1]
        rows = json.loads((output / f"actions_rank_{rank}.json").read_text())
        with np.load(path, allow_pickle=False) as arrays:
            assert arrays["predicted"].shape == arrays["target"].shape == (len(rows), 50, 14)
            assert np.isfinite(arrays["predicted"]).all()
            assert np.isfinite(arrays["target"]).all()
            for i, row in enumerate(rows):
                identity = (row["index"], row["draw"], row["condition_mode"])
                assert identity not in identities
                identities.add(identity)
                assert identity == (int(arrays["index"][i]), int(arrays["draw"][i]), str(arrays["condition"][i]))
                assert row["valid_horizon"] == int(arrays["valid_horizon"][i])
                metrics = native_action_metrics(
                    arrays["predicted"][i], arrays["target"][i], valid_horizon=row["valid_horizon"]
                )
                for key, value in metrics.items():
                    np.testing.assert_allclose(value, row[key], rtol=1e-12, atol=1e-12)
                count += 1
    assert count == expected, (count, expected)
    report = {
        "rows": count,
        "unique_identities": len(identities),
        "finite_native_shapes": True,
        "saved_metrics_recomputed": True,
    }
    with (output / "archive_verification.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({"event": "archive_verified", "output": str(output), **report}), flush=True)


def main(args):
    torch.set_num_threads(4)
    for run in ["b1_trainer_cpu_smoke_seed42", "o_trainer_cpu_smoke_seed42"]:
        require_complete(run, 5)
    print(json.dumps({"event": "waiting_for_main_m3", "pid": args.wait_pid, "created": args.wait_created}), flush=True)
    deadline = time.monotonic() + 7200
    while True:
        try:
            process = psutil.Process(args.wait_pid)
            alive = abs(process.create_time() - args.wait_created) < 0.01 and process.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            alive = False
        if not alive:
            break
        if time.monotonic() > deadline:
            raise TimeoutError("M3 sequence did not exit within two hours")
        time.sleep(10)
    require_complete("m3_pilot_seed42", 3500)
    freeze = json.loads((LOGS / "m3_research_checkpoint_freeze.json").read_text())
    assert freeze["groups"]["frozen"]["changed_tensors"] == 0
    m3 = ROOT / "m3_pilot_seed42/step_003500"
    metadata = json.loads((m3 / "metadata.json").read_text())
    assert metadata["counters"]["action"] == 4000
    assert metadata["best"]["step"] == 3500
    output = LOGS / "m3_native_dense_val"
    job(
        "m3_native_dense_val",
        "scripts/eval_subtask_offline.py",
        [
            "--checkpoint",
            m3,
            "--output",
            output,
            "--semantic-samples",
            0,
            "--action-samples",
            128,
            "--latency-samples",
            128,
            "--draws",
            2,
            "--no-image",
        ],
    )
    verify_archives(output, 128 * 2 * 4)
    output = LOGS / "c0_comparable_native_val"
    job("c0_comparable_native_val", "scripts/eval_action_control.py", ["--checkpoint", C0, "--output", output])
    verify_archives(output, 128 * 2)
    for method, condition in [("b1", "none"), ("o", "oracle")]:
        run = f"{method}_trainer_ddp8_smoke_seed42"
        common = [
            "--condition",
            condition,
            "--initialize-from",
            C0,
            "--output",
            ROOT / run,
            "--steps",
            5,
            "--warmup-phase-steps",
            1,
            "--batch-size",
            2,
            "--accumulation",
            2,
            "--workers",
            2,
            "--cpu-threads",
            4,
            "--eval-samples",
            8,
            "--eval-draws",
            1,
            "--eval-every",
            5,
            "--checkpoint-every",
            5,
            "--warmup-action",
            1,
            "--engineering-smoke",
        ]
        job(
            f"{method}_trainer_ddp8_smoke_start",
            "scripts/train_subtask_action_control.py",
            [*common, "--stop-after", 2],
        )
        job(f"{method}_trainer_ddp8_smoke_resume", "scripts/train_subtask_action_control.py", [*common, "--resume"])
        require_complete(run, 5)
        verify_control(
            ROOT / run / "step_000002", ROOT / run / "step_000005", f"{method}_control_checkpoint_freeze_ddp8.json"
        )
    for method, condition in [("b1", "none"), ("o", "oracle")]:
        run = f"{method}_pilot_seed42"
        job(
            run,
            "scripts/train_subtask_action_control.py",
            [
                "--condition",
                condition,
                "--initialize-from",
                C0,
                "--output",
                ROOT / run,
                "--steps",
                4000,
                "--warmup-phase-steps",
                500,
            ],
        )
        require_complete(run, 4000)
        checkpoint = ROOT / run / "step_004000"
        assert json.loads((ROOT / run / "best.json").read_text())["step"] == 4000
        verify_control(C0, checkpoint, f"{method}_research_checkpoint_freeze.json")
        output = LOGS / f"{method}_native_val"
        job(f"{method}_native_val", "scripts/eval_action_control.py", ["--checkpoint", checkpoint, "--output", output])
        verify_archives(output, 128 * 2)
    print(json.dumps({"event": "sequence_complete", "time": time.time()}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--wait-created", type=float, required=True)
    main(parser.parse_args())
