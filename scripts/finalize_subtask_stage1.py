"""Finish validation, seal selected weights/options, and run the final test report."""

import json
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
from continue_stage1_replications import wait_record

from openpi.training.research_checkpoint import require_research_checkpoint
from openpi.training.stage1_data import sha256_file

PROTOCOL_PATH = Path("assets/pi05_piper_stage1/eggplant_potato/research_protocol_v1.json")


def checkpoint_options(hierarchical, *, semantics_only=False):
    options = {"draws": 2, "num_steps": 10, "latency_samples": 0, "device": "cuda", "cpu_threads": 4}
    if hierarchical:
        options.update(
            action_samples=0 if semantics_only else 128, semantic_samples=0, no_image=True, batch_size=4, workers=2
        )
    else:
        options["samples"] = 128
    return options


def options_argv(options):
    result = []
    for key, value in options.items():
        option = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                result.append(option)
        else:
            result.extend([option, str(value)])
    return result


def summarize(manifest, execution_frames, output):
    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_subtask_comparison.py",
            "--manifest",
            str(manifest),
            "--execution-frames",
            str(execution_frames),
            "--output",
            str(output),
        ],
        check=True,
    )


def main():
    wait_record(LOGS / "replications_sequence.process.json", 12 * 3600)
    manifest_path = LOGS / "three_seed_comparison_manifest.json"
    validation_manifest = json.loads(manifest_path.read_text())
    experiments = validation_manifest["experiments"]
    if sorted(experiment["seed"] for experiment in experiments) != [42, 43, 44]:
        raise ValueError("Three independent conditional-on-C0 training seeds must complete first")
    for seed in [42, 43, 44]:
        for stage, steps in [("m1", 2000), ("m2", 500), ("m3", 3500), ("b1", 4000), ("o", 4000)]:
            require_complete(f"{stage}_pilot_seed{seed}", steps)
    audit = json.loads((LOGS / "official_base_weight_audit.json").read_text())
    if (
        audit["mismatched_tensors"]
        or audit["verified_tensor_count"] != 811
        or audit["unverified_unused_tensors"] != ["paligemma_with_expert.gemma_expert.lm_head.weight"]
    ):
        raise ValueError("Official base provenance gate failed")
    # Select a deployment candidate on validation only. Test still reports all seeds.
    selected = min(
        experiments,
        key=lambda e: (
            json.loads((Path(e["g"]) / "report.json").read_text())["action_conditions"]["generated"]["overall"][
                "flow_native14_normalized"
            ],
            e["seed"],
        ),
    )
    selected_seed = selected["seed"]
    profiles = []
    for method, checkpoint in [
        ("g", ROOT / f"m3_pilot_seed{selected_seed}/step_003500"),
        ("b0", C0),
        ("b1", ROOT / f"b1_pilot_seed{selected_seed}/step_004000"),
    ]:
        output = LOGS / f"deployment_profile_{method}_seed{selected_seed}"
        job(output.name, "scripts/benchmark_subtask_policy.py", ["--checkpoint", checkpoint, "--output", output])
        profiles.append(json.loads((output / "report.json").read_text()))
    latency_evidence = []
    for experiment in experiments:
        for method in ["g", "b0", "b1"]:
            report = json.loads((Path(experiment[method]) / "report.json").read_text())
            value = (
                report["timing"]["p50_p95_ms"]["infer_ms"][1]
                if method == "g"
                else report["complete_policy_ms_p50_p95"][1]
            )
            latency_evidence.append(
                {"source": experiment[method], "seed": experiment["seed"], "method": method, "p95_ms": value}
            )
    latency_evidence.extend(
        {
            "source": profile["checkpoint"],
            "kind": "isolated deployment profile",
            "p95_ms": profile["policy_ms_p50_p95"][1],
        }
        for profile in profiles
    )
    protocol = json.loads(PROTOCOL_PATH.read_text())
    maximum_ms = max(item["p95_ms"] for item in latency_evidence)
    candidates = [
        frames
        for frames in protocol["execution_length_selection"]["candidates_frames"]
        if frames / 30 >= 1.2 * maximum_ms / 1000
    ]
    if not candidates:
        raise ValueError("No planned action execution period meets the measured policy timing; do not claim real-time")
    execution_frames = min(candidates)
    summarize(manifest_path, execution_frames, LOGS / "three_seed_validation_comparison.json")
    jobs = [("c0_test", C0, False, False)]
    final_experiments = []
    for seed in [42, 43, 44]:
        m1_best = json.loads((ROOT / f"m1_pilot_seed{seed}/best.json").read_text())["checkpoint"]
        jobs.extend(
            [
                (f"m1_test_seed{seed}", ROOT / f"m1_pilot_seed{seed}" / m1_best, True, True),
                (f"g_test_seed{seed}", ROOT / f"m3_pilot_seed{seed}/step_003500", True, False),
                (f"b1_test_seed{seed}", ROOT / f"b1_pilot_seed{seed}/step_004000", False, False),
                (f"o_test_seed{seed}", ROOT / f"o_pilot_seed{seed}/step_004000", False, False),
            ]
        )
        final_experiments.append(
            {
                "seed": seed,
                "g": str(LOGS / f"g_test_seed{seed}"),
                "b0": str(LOGS / "c0_test"),
                "b1": str(LOGS / f"b1_test_seed{seed}"),
                "o": str(LOGS / f"o_test_seed{seed}"),
            }
        )
    checkpoints = {}
    for _, checkpoint, hierarchical, semantics_only in jobs:
        metadata = json.loads((checkpoint / "metadata.json").read_text())
        require_research_checkpoint(checkpoint, metadata)
        checkpoints[str(checkpoint.resolve())] = {
            "weights_sha256": sha256_file(checkpoint / "model.safetensors"),
            "metadata_sha256": sha256_file(checkpoint / "metadata.json"),
            "world_size": 8,
            "evaluation_options": checkpoint_options(hierarchical, semantics_only=semantics_only),
        }
    seal = {
        "status": "sealed",
        "split": "test",
        "created_at": time.time(),
        "research_protocol": str(PROTOCOL_PATH),
        "research_protocol_sha256": sha256_file(PROTOCOL_PATH),
        "split_sha256": protocol["split_sha256"],
        "norm_sha256": protocol["norm_sha256"],
        "execution_frames": execution_frames,
        "data_fps": 30,
        "latency_evidence": latency_evidence,
        "deployment_candidate_seed": selected_seed,
        "deployment_selection": "lowest generated-condition validation flow error; test results for all three seeds retained",
        "checkpoints": checkpoints,
        "sources": {
            str(path): sha256_file(path)
            for path in [
                Path(__file__),
                Path("src/openpi/training/evaluation_protocol.py"),
                Path("src/openpi/training/research_checkpoint.py"),
                Path("scripts/eval_subtask_offline.py"),
                Path("scripts/eval_action_control.py"),
            ]
        },
    }
    time_report = LOGS / "time_baseline_val/report.json"
    seal["time_baseline"] = {
        "training_report": str(time_report),
        "training_report_sha256": sha256_file(time_report),
        "evaluation_source_sha256": sha256_file(Path("scripts/eval_subtask_time_final.py")),
    }
    seal_path = PROTOCOL_PATH.with_name("final_test_protocol.json")
    with seal_path.open("x") as stream:
        json.dump(seal, stream, indent=2)
    print(
        json.dumps(
            {
                "event": "test_protocol_sealed",
                "path": str(seal_path),
                "execution_frames": execution_frames,
                "deployment_candidate_seed": selected_seed,
            }
        ),
        flush=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/eval_subtask_time_final.py",
            "--test-protocol",
            str(seal_path),
            "--output",
            str(LOGS / "time_baseline_test"),
        ],
        check=True,
    )
    for name, checkpoint, hierarchical, semantics_only in jobs:
        output = LOGS / name
        script = "scripts/eval_subtask_offline.py" if hierarchical else "scripts/eval_action_control.py"
        options = checkpoint_options(hierarchical, semantics_only=semantics_only)
        job(
            name,
            script,
            [
                "--checkpoint",
                checkpoint,
                "--output",
                output,
                "--split",
                "test",
                "--test-protocol",
                seal_path,
                *options_argv(options),
            ],
        )
        if not semantics_only:
            verify_archives(output, 128 * 2 * (4 if hierarchical else 1))
    test_manifest_path = LOGS / "final_test_comparison_manifest.json"
    with test_manifest_path.open("x") as stream:
        json.dump(
            {"scope": "three-seed final test", "protocol": str(seal_path), "experiments": final_experiments},
            stream,
            indent=2,
        )
    summarize(test_manifest_path, execution_frames, LOGS / "three_seed_test_comparison.json")
    for split, output in [("val", "three_seed_validation_report"), ("test", "final_research_report")]:
        subprocess.run(
            [
                sys.executable,
                "scripts/write_subtask_research_report.py",
                "--split",
                split,
                "--require-complete",
                "--output",
                str(LOGS / output),
            ],
            check=True,
        )
    print(
        json.dumps(
            {
                "event": "sequence_complete",
                "time": time.time(),
                "validation_summary": str(LOGS / "three_seed_validation_comparison.json"),
                "test_summary": str(LOGS / "three_seed_test_comparison.json"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
