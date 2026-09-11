"""Final task/elapsed-time diagnostic using the previously fixed training parameters."""

import argparse
import dataclasses
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from eval_subtask_time_baseline import scalar_rows
import numpy as np

from openpi.training import config
from openpi.training import data_loader
from openpi.training.stage1_data import sha256_file
from openpi.training.subtask_evaluation import semantic_metrics
from openpi.training.subtask_evaluation import transition_metrics


def main(args):
    seal = json.loads(args.test_protocol.read_text())
    if seal.get("status") != "sealed" or seal.get("split") != "test":
        raise ValueError("Final time diagnostic requires the sealed test protocol")
    selected = seal["time_baseline"]
    report_path = Path(selected["training_report"])
    if (
        sha256_file(report_path) != selected["training_report_sha256"]
        or sha256_file(Path(__file__)) != selected["evaluation_source_sha256"]
    ):
        raise ValueError("Fixed time-diagnostic parameters or evaluator changed")
    training_report = json.loads(report_path.read_text())
    parameters = training_report["parameters"]
    if training_report["split"] != "val" or training_report["split_sha256"] != seal["split_sha256"]:
        raise ValueError("Time diagnostic training provenance changed")
    cfg = config.get_config("pi05_piper_stage1")
    dc = dataclasses.replace(cfg.data.create(cfg.assets_dirs, cfg.model), split="test")
    if json.loads(Path(dc.split_manifest).read_text())["manifest_sha256"] != seal["split_sha256"]:
        raise ValueError("Final dataset split differs from the sealed protocol")
    raw = data_loader.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
    rows = []
    for row in scalar_rows(raw):
        learned = parameters[row["task"]]
        phase = int(np.searchsorted(learned["phase_start_frames"][1:], row["frame"], side="right"))
        rows.append(dict(row, prediction=learned["labels"][phase], status="ok"))
    vocabulary = sorted({label for learned in parameters.values() for label in learned["labels"]})
    task_vocabulary = {task: learned["labels"] for task, learned in parameters.items()}
    report = {
        "split": "test",
        "split_sha256": seal["split_sha256"],
        "test_protocol_sha256": sha256_file(args.test_protocol),
        "method": training_report["method"],
        "input_contract": training_report["input_contract"],
        "parameters": parameters,
        "parameter_source": str(report_path),
        "parameter_source_sha256": selected["training_report_sha256"],
        "semantics": semantic_metrics(rows, vocabulary, task_vocabulary=task_vocabulary),
        "transition_metrics": transition_metrics(rows, tolerance=10),
        "sources": {
            str(path): sha256_file(path)
            for path in [
                Path(__file__),
                Path("scripts/eval_subtask_time_baseline.py"),
                Path("src/openpi/training/subtask_evaluation.py"),
            ]
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "predictions.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "event": "complete",
                "report": str(args.output / "report.json"),
                "frames": len(rows),
                "exact_match": report["semantics"]["exact_match"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
