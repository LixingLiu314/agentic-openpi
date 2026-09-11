"""Task plus elapsed-frame diagnostic; fit median phase starts on training episodes only.

This simple clock predictor receives neither images nor state. It uses elapsed
frames since episode start, not normalized progress or the future episode length.
There is no validation-based threshold tuning and no test-set access.
"""

import argparse
from collections import defaultdict
import dataclasses
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np

from openpi.training import config
from openpi.training import data_loader
from openpi.training.stage1_data import sha256_file
from openpi.training.subtask_evaluation import semantic_metrics
from openpi.training.subtask_evaluation import transition_metrics


def scalar_rows(raw):
    table = raw.hf_dataset
    episodes = defaultdict(list)
    for index, (episode, frame, task, label) in enumerate(
        zip(table["episode_index"], table["frame_index"], table["task"], table["subtask"], strict=True)
    ):
        episodes[int(episode)].append(
            {"index": index, "episode": int(episode), "frame": int(frame), "task": task, "label": label}
        )
    result = []
    for group in episodes.values():
        group.sort(key=lambda row: row["frame"])
        if [row["frame"] for row in group] != list(range(len(group))):
            raise ValueError("Expected complete episodes for the elapsed-frame diagnostic")
        result.extend(dict(row, episode_length=len(group)) for row in group)
    return result


def fit(train_rows):
    episodes = defaultdict(list)
    for row in train_rows:
        episodes[row["episode"]].append(row)
    tasks = {}
    for group in episodes.values():
        group.sort(key=lambda row: row["frame"])
        starts = [row for i, row in enumerate(group) if i == 0 or row["label"] != group[i - 1]["label"]]
        task = group[0]["task"]
        labels = [row["label"] for row in starts]
        if task not in tasks:
            tasks[task] = {"labels": labels, "starts_by_episode": []}
        if labels != tasks[task]["labels"]:
            raise ValueError("Training phase order is not fixed; the median-clock diagnostic is inapplicable")
        tasks[task]["starts_by_episode"].append([row["frame"] for row in starts])
    return {
        task: {
            "labels": value["labels"],
            "phase_start_frames": np.median(value["starts_by_episode"], axis=0).tolist(),
            "training_episodes": len(value["starts_by_episode"]),
        }
        for task, value in tasks.items()
    }


def evaluate(args):
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = config.get_config("pi05_piper_stage1")
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    train_rows = scalar_rows(
        data_loader.create_torch_dataset(dataclasses.replace(dc, split="train"), cfg.model.action_horizon, cfg.model)
    )
    parameters = fit(train_rows)
    vocabulary = sorted({row["label"] for row in train_rows})
    task_vocabulary = {task: value["labels"] for task, value in parameters.items()}
    validation = scalar_rows(
        data_loader.create_torch_dataset(dataclasses.replace(dc, split="val"), cfg.model.action_horizon, cfg.model)
    )
    if {row["episode"] for row in train_rows} & {row["episode"] for row in validation}:
        raise AssertionError("Training and validation episodes overlap")
    rows = []
    for row in validation:
        learned = parameters[row["task"]]
        phase = int(np.searchsorted(learned["phase_start_frames"][1:], row["frame"], side="right"))
        rows.append(dict(row, prediction=learned["labels"][phase], status="ok"))
    sampled = set(np.unique(np.linspace(0, len(rows) - 1, 128, dtype=int)).tolist())
    manifest = json.loads(Path(dc.split_manifest).read_text())
    report = {
        "method": "task + elapsed frames; per-task training-median phase starts",
        "input_contract": "global task and elapsed frame index only; no state/images/future episode length",
        "split": "val",
        "split_sha256": manifest["manifest_sha256"],
        "parameters": parameters,
        "hyperparameter_selection": "none; fixed diagnostic protocol",
        "semantics": semantic_metrics(rows, vocabulary, task_vocabulary=task_vocabulary),
        "fixed128_semantics": semantic_metrics(
            [row for row in rows if row["index"] in sampled], vocabulary, task_vocabulary=task_vocabulary
        ),
        "transition_metrics": transition_metrics(rows, tolerance=10),
        "sources": {
            str(path): sha256_file(path) for path in [Path(__file__), Path("src/openpi/training/subtask_evaluation.py")]
        },
    }
    (args.output / "predictions.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "event": "complete",
                "report": str(args.output / "report.json"),
                "exact_match": report["semantics"]["exact_match"],
                "macro_f1": report["semantics"]["macro_f1"],
                "frames": len(rows),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    evaluate(parser.parse_args())
