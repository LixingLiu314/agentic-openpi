"""Summarize matched pilot or multi-seed native action evaluations."""

import argparse
import json
from pathlib import Path

from openpi.training.research_checkpoint import require_research_checkpoint
from openpi.training.stage1_data import sha256_file
from openpi.training.subtask_comparison import clustered_summary
from openpi.training.subtask_comparison import paired_arrays


def main(args):
    manifest = json.loads(args.manifest.read_text())
    for experiment in manifest["experiments"]:
        reports = {
            name: json.loads((Path(experiment[name]) / "report.json").read_text()) for name in ["g", "b0", "b1", "o"]
        }
        stages = {"g": "m3", "b0": "m0", "b1": "control_b1", "o": "control_o"}
        for name, report in reports.items():
            checkpoint = Path(report["checkpoint"])
            metadata = json.loads((checkpoint / "metadata.json").read_text())
            if metadata["stage"] != stages[name]:
                raise ValueError("Comparison requires research checkpoints of the expected stages")
            require_research_checkpoint(checkpoint, metadata)
            if sha256_file(checkpoint / "model.safetensors") != report["weights_sha256"]:
                raise ValueError("Evaluation checkpoint changed")
            if name != "b0":
                if metadata["config"]["seed"] != experiment["seed"]:
                    raise ValueError("Declared training seed differs from checkpoint")
                if metadata["config"]["c0_weights_sha256"] != reports["b0"]["weights_sha256"]:
                    raise ValueError("Methods must share the same C0")
                if metadata["counters"]["action"] != 4000:
                    raise ValueError("Pilot methods must match the fixed 4000-update action budget")
    names, seeds, values, metadata, provenance = paired_arrays(manifest["experiments"], args.execution_frames)
    report = clustered_summary(names, seeds, values, metadata, repetitions=args.bootstrap_repetitions)
    report.update(
        execution_frames=args.execution_frames,
        data_fps=30,
        manifest=str(args.manifest),
        manifest_sha256=sha256_file(args.manifest),
        evaluations=provenance,
    )
    report["sources"] = {
        str(path): sha256_file(path)
        for path in [
            Path(__file__),
            Path("src/openpi/training/subtask_comparison.py"),
            Path("src/openpi/training/research_checkpoint.py"),
        ]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "seeds": seeds, "frames": len(metadata)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--execution-frames", type=int, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
