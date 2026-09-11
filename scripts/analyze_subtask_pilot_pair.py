"""Diagnose a completed G/B1 validation pair without changing model selection."""

import argparse
import json
from pathlib import Path

import numpy as np

from openpi.training.stage1_data import sha256_file
from openpi.training.subtask_comparison import METRICS
from openpi.training.subtask_comparison import aggregate
from openpi.training.subtask_comparison import load_windows
from openpi.training.subtask_comparison import window_values


def main(args):
    sources = [load_windows(args.generated), load_windows(args.baseline)]
    fields = ["split", "split_sha256", "norm_sha256", "flow_steps", "noise_draws"]
    if any(sources[0][0][key] != sources[1][0][key] for key in fields):
        raise ValueError("Evaluation protocols differ")
    if sources[0][0]["split"] != "val":
        raise ValueError("This exploratory diagnostic is for validation only")
    provenance = []
    metadata = []
    for method, (report, _windows) in zip(["G", "B1"], sources, strict=True):
        checkpoint = Path(report["checkpoint"])
        record = json.loads((checkpoint / "metadata.json").read_text())
        if record["stage"] != {"G": "m3", "B1": "control_b1"}[method]:
            raise ValueError("Unexpected checkpoint stage")
        if record["config"]["engineering_smoke"] or record["counters"]["action"] != 4000:
            raise ValueError("Need completed matched research budgets")
        if sha256_file(checkpoint / "model.safetensors") != report["weights_sha256"]:
            raise ValueError("Checkpoint changed")
        metadata.append(record)
        provenance.append({"method": method, "checkpoint": str(checkpoint), "sha256": report["weights_sha256"]})
    for key in ["seed", "c0_weights_sha256"]:
        if metadata[0]["config"][key] != metadata[1]["config"][key]:
            raise ValueError("Training identities differ")
    selected = [
        {(index, draw): item for (index, draw, mode), item in windows.items() if mode == condition}
        for (_report, windows), condition in zip(sources, ["generated", "none"], strict=True)
    ]
    keys = sorted(selected[0])
    if not keys or keys != sorted(selected[1]):
        raise ValueError("Frame/noise identities differ")
    for key in keys:
        left, right = selected[0][key], selected[1][key]
        for field in ["episode", "frame", "task", "valid_horizon", "crosses_subtask_boundary"]:
            if left[0][field] != right[0][field]:
                raise ValueError(f"Paired metadata differs: {field}")
        np.testing.assert_array_equal(left[2], right[2])
    anchors = sorted({key[0] for key in keys})
    rows = [selected[0][next(key for key in keys if key[0] == index)][0] for index in anchors]
    task_episodes = {
        task: sorted({row["episode"] for row in rows if row["task"] == task})
        for task in sorted({row["task"] for row in rows})
    }
    rng = np.random.default_rng(90210)
    weights = []
    for _ in range(2000):
        counts = {row["episode"]: 0 for row in rows}
        for episodes in task_episodes.values():
            for episode in rng.choice(episodes, len(episodes), replace=True):
                counts[int(episode)] += 1
        weights.append([counts[row["episode"]] for row in rows])
    subsets = {"all": np.ones(len(rows), dtype=bool)}
    subsets.update({task: np.asarray([row["task"] == task for row in rows]) for task in task_episodes})
    subsets.update(
        {
            "crosses_boundary": np.asarray([row["crosses_subtask_boundary"] for row in rows], dtype=bool),
            "within_phase": np.asarray([not row["crosses_subtask_boundary"] for row in rows], dtype=bool),
        }
    )
    horizons = {}
    for horizon in [10, 50]:
        values = np.asarray([[window_values(*method[key], horizon) for key in keys] for method in selected])
        values = np.stack(
            [values[:, [i for i, key in enumerate(keys) if key[0] == index]].mean(axis=1) for index in anchors], axis=1
        )
        point = aggregate(values)
        boot = np.asarray([aggregate(values, weight) for weight in weights])
        metrics = {
            name: {
                "G": float(point[0, i]),
                "B1": float(point[1, i]),
                "G_minus_B1": float(point[0, i] - point[1, i]),
                "paired_episode_bootstrap_ci95": np.percentile(boot[:, 0, i] - boot[:, 1, i], [2.5, 97.5]).tolist(),
            }
            for i, name in enumerate(METRICS)
        }
        strata = {}
        for name, mask in subsets.items():
            if mask.any():
                result = aggregate(values[:, mask])
                strata[name] = {
                    "frames": int(mask.sum()),
                    "metrics": {
                        metric: {
                            "G": float(result[0, i]),
                            "B1": float(result[1, i]),
                            "G_minus_B1": float(result[0, i] - result[1, i]),
                        }
                        for i, metric in enumerate(METRICS)
                    },
                }
        horizons[str(horizon)] = {"metrics": metrics, "strata_descriptive_only": strata}
    result = {
        "scope": "Exploratory single-seed validation pair; no selection or execution-length changes",
        "seed": metadata[0]["config"]["seed"],
        "frames": len(rows),
        "draws": len(keys),
        "episodes": sum(len(episodes) for episodes in task_episodes.values()),
        "bootstrap": "2000 task-stratified paired episode resamples, fixed seed 90210; conditional on two fitted checkpoints. Noise draws averaged per frame. Subgroup results are descriptive and not multiplicity-adjusted.",
        "aggregation": "Frame-weighted mean MSE then square root for RMSE; execution clips to valid episode tail. Positive G-minus-B1 is worse except gripper accuracy.",
        "horizons": horizons,
        "checkpoints": provenance,
        "evaluation_reports": {
            str(directory / "report.json"): sha256_file(directory / "report.json")
            for directory in [args.generated, args.baseline]
        },
        "sources": {
            str(path): sha256_file(path) for path in [Path(__file__), Path("src/openpi/training/subtask_comparison.py")]
        },
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "frames": len(rows), "episodes": result["episodes"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
