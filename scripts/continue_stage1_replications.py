"""Run seeds 43/44 after the pilot controls and official weight audit pass."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from continue_stage1_controls import C0
from continue_stage1_controls import LOGS
from continue_stage1_controls import ROOT
from continue_stage1_controls import job
from continue_stage1_controls import require_complete
from continue_stage1_controls import verify_archives
from continue_stage1_controls import verify_control
import psutil


def wait_record(path, deadline_seconds):
    record = json.loads(path.read_text())
    print(
        json.dumps(
            {"event": "wait_dependency", "record": str(path), "pid": record["pid"], "created": record["created"]}
        ),
        flush=True,
    )
    deadline = time.monotonic() + deadline_seconds
    while True:
        try:
            process = psutil.Process(record["pid"])
            alive = abs(process.create_time() - record["created"]) < 0.01 and process.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            alive = False
        if not alive:
            return
        if time.monotonic() > deadline:
            raise TimeoutError(f"Dependency remains active: {path}")
        time.sleep(10)


def verify_hierarchy(parent, child, name):
    subprocess.run(
        [
            sys.executable,
            "scripts/verify_hierarchy_freeze.py",
            "--parent",
            str(parent),
            "--child",
            str(child),
            "--output",
            str(LOGS / name),
        ],
        check=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu"),
    )


def main():
    protocol = json.loads(Path("assets/pi05_piper_stage1/eggplant_potato/research_protocol_v1.json").read_text())
    if protocol["training_seeds"] != [42, 43, 44]:
        raise ValueError("Unexpected seed protocol")
    wait_record(LOGS / "controls_phase_sequence.process.json", 4 * 3600)
    for method in ["b1", "o"]:
        require_complete(f"{method}_pilot_seed42", 4000)
        if not (LOGS / f"{method}_native_val/archive_verification.json").exists():
            raise ValueError(f"Pilot evaluation incomplete for {method}")
    wait_record(LOGS / "official_base_weight_audit_ranges.process.json", 4 * 3600)
    audit = json.loads((LOGS / "official_base_weight_audit.json").read_text())
    if audit["mismatched_tensors"] or audit["tensor_count"] != 812:
        raise ValueError("Official base weight audit did not pass")
    experiments = json.loads((LOGS / "pilot_comparison_manifest.json").read_text())["experiments"]
    for seed in [43, 44]:
        parents = {"m1": C0}
        for stage, steps in [("m1", 2000), ("m2", 500), ("m3", 3500)]:
            run = f"{stage}_pilot_seed{seed}"
            job(
                run,
                "scripts/train_subtask_hierarchy.py",
                [
                    "--stage",
                    stage,
                    "--initialize-from",
                    parents[stage],
                    "--output",
                    ROOT / run,
                    "--steps",
                    steps,
                    "--seed",
                    seed,
                    "--m3-planned-steps",
                    3500,
                ],
            )
            require_complete(run, steps)
            best = json.loads((ROOT / run / "best.json").read_text())
            checkpoint = ROOT / run / best["checkpoint"] if stage == "m1" else ROOT / run / f"step_{steps:06d}"
            verify_hierarchy(parents[stage], checkpoint, f"{stage}_research_checkpoint_freeze_seed{seed}.json")
            if stage == "m1":
                parents["m2"] = checkpoint
                output = LOGS / f"m1_dense_val_seed{seed}"
                job(
                    f"m1_dense_val_seed{seed}",
                    "scripts/eval_subtask_offline.py",
                    [
                        "--checkpoint",
                        checkpoint,
                        "--output",
                        output,
                        "--semantic-samples",
                        0,
                        "--action-samples",
                        0,
                        "--latency-samples",
                        0,
                        "--no-image",
                    ],
                )
            elif stage == "m2":
                parents["m3"] = checkpoint
            else:
                if best["step"] != 3500:
                    raise ValueError("Unexpected late-curriculum checkpoint selection")
                output = LOGS / f"m3_native_dense_val_seed{seed}"
                job(
                    f"m3_native_dense_val_seed{seed}",
                    "scripts/eval_subtask_offline.py",
                    [
                        "--checkpoint",
                        checkpoint,
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
        for method, condition in [("b1", "none"), ("o", "oracle")]:
            run = f"{method}_pilot_seed{seed}"
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
                    "--seed",
                    seed,
                ],
            )
            require_complete(run, 4000)
            checkpoint = ROOT / run / "step_004000"
            verify_control(C0, checkpoint, f"{method}_research_checkpoint_freeze_seed{seed}.json")
            output = LOGS / f"{method}_native_val_seed{seed}"
            job(
                f"{method}_native_val_seed{seed}",
                "scripts/eval_action_control.py",
                ["--checkpoint", checkpoint, "--output", output],
            )
            verify_archives(output, 128 * 2)
        experiments.append(
            {
                "seed": seed,
                "g": str(LOGS / f"m3_native_dense_val_seed{seed}"),
                "b0": str(LOGS / "c0_comparable_native_val"),
                "b1": str(LOGS / f"b1_native_val_seed{seed}"),
                "o": str(LOGS / f"o_native_val_seed{seed}"),
            }
        )
        with (LOGS / f"completed_seed{seed}_comparison_manifest.json").open("x") as stream:
            json.dump({"scope": "matched validation", "experiments": experiments}, stream, indent=2)
    with (LOGS / "three_seed_comparison_manifest.json").open("x") as stream:
        json.dump({"scope": "three-seed validation", "experiments": experiments}, stream, indent=2)
    print(json.dumps({"event": "sequence_complete", "time": time.time(), "seeds": [42, 43, 44]}), flush=True)


if __name__ == "__main__":
    main()
