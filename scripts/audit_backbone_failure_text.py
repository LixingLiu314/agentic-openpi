"""Paired CPU text-only replay of a selected backbone model and its M3 parent."""

import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import jax
import numpy as np
import torch

from openpi.models.model import Observation
from openpi.policies.backbone_gradient_policy import create_backbone_gradient_policy
from openpi.policies.subtask_policy import create_subtask_policy
from openpi.training.stage1_data import sha256_file


def normalized(text):
    return " ".join(text.casefold().split())


def summarize(rows, constraints):
    forbidden = set(constraints["forbidden_final_put_queries"])
    closed = set(constraints["closed_gripper_queries"])
    allowed = {normalized(t) for t in constraints["closed_gripper_allowed_text"]}
    put = [r["query"] for r in rows if r["query"] in forbidden and normalized(r["text"]) == "put the lid on the box"]
    recognized = [r["query"] for r in rows if r["query"] in closed and normalized(r["text"]) in allowed and r["status"] == "ok"]
    invalid = [r["query"] for r in rows if r["status"] != "ok"]
    return {
        "queries": len(rows),
        "text_counts": dict(Counter(normalized(r["text"]) for r in rows)),
        "forbidden_put_queries": put,
        "closed_gripper_allowed_queries": recognized,
        "invalid_queries": invalid,
        "text_constraints_passed": (
            len(put) <= constraints["max_forbidden_final_put"]
            and len(recognized) >= constraints["required_closed_gripper_allowed_count"]
            and len(invalid) <= constraints["max_invalid_generations"]
        ),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    started = time.time()
    root = Path.cwd().resolve()
    replay = root / "assets/pi05_piper_transition/eggplant_potato/r2_failure_replay_v1"
    protocol_hash = sha256_file(replay / "protocol.json")
    assert protocol_hash == "f365cd67c9caffd9d13353029359ede2b140c80b5783cd6e9ef8d4bedb9d202a"
    protocol = json.loads((replay / "protocol.json").read_text())
    records = protocol["records"]
    assert [r["query"] for r in records] == list(range(1, 100))
    for record in records:
        assert sha256_file(replay / record["file"]) == record["sha256"]
    checkpoint = args.checkpoint.resolve()
    best = json.loads((checkpoint.parent / "best.json").read_text())
    candidate = json.loads((checkpoint.parent / "candidate.json").read_text())
    assert best["checkpoint"] == checkpoint.name == candidate["checkpoint"]
    assert candidate["policy_load_gate_passed"]
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    assert not metadata["config"]["engineering_smoke"]
    selected_name = metadata["config"]["mode"] + "_selected"
    parent = root / "checkpoints/pi05_piper_stage1/m3_pilot_seed42/step_003500"
    parent_hash = sha256_file(parent / "model.safetensors")
    assert parent_hash == metadata["config"]["parent_weights_sha256"]
    assert parent_hash == "64600bed0b60e0529ac2721edb55717ac1c80fba2a607292857668a35e7a9275"
    result = {
        "scope": "Paired CPU FP32 native text-generation path on all 99 frozen approximate failure observations; no action prediction or robot I/O",
        "seed": 42,
        "replay_protocol_sha256": protocol_hash,
        "selected_checkpoint": str(checkpoint),
        "selected_weights_sha256": metadata["weights_sha256"],
        "parent_weights_sha256": parent_hash,
        "selected_by": best,
        "model_summaries": {},
        "rows": {},
        "qualifies_boundary_improvement": False,
        "limitations": "Lossy reconstructed RGB, CPU FP32, one previously observed failure run. Text-only constraints are necessary diagnostic checks, not complete semantic/action/native gates or physical task-success evidence.",
    }
    for name, path, loader in [
        ("m3", parent, create_subtask_policy),
        (selected_name, checkpoint, create_backbone_gradient_policy),
    ]:
        print(json.dumps({"phase": "loading", "model": name}), flush=True)
        policy = loader(path, device="cpu")
        policy.model.float().eval()
        rows = []
        with (args.output / f"{name}.jsonl").open("x") as stream:
            for record in records:
                with np.load(replay / record["file"], allow_pickle=False) as saved:
                    observation = {
                        "state": saved["state"].copy(),
                        "images": {k: saved[k].copy() for k in ["cam_high", "cam_left_wrist", "cam_right_wrist"]},
                        "prompt": record["prompt"],
                    }
                before = {"state": observation["state"].copy(), **{k: v.copy() for k, v in observation["images"].items()}}
                inputs = policy.input_transform(observation)
                tensors = jax.tree.map(lambda x: torch.as_tensor(np.asarray(x), device="cpu")[None], inputs)
                context = policy.model.prepare_context(Observation.from_dict(tensors), [record["prompt"]])
                texts, statuses, generation = policy.model.generate_subtask(context)
                assert np.array_equal(before["state"], observation["state"])
                assert all(np.array_equal(before[k], v) for k, v in observation["images"].items())
                row = {
                    "query": record["query"], "timestamp": record["timestamp"],
                    "text": texts[0], "status": statuses[0],
                    "mean_token_log_probability": float(generation.mean_log_probability[0]),
                    "input_sha256": record["sha256"],
                }
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                if record["query"] % 10 == 0 or record["query"] in [94, 95, 99]:
                    print(json.dumps({"model": name, **row}), flush=True)
                del context, generation, tensors, inputs
        result["rows"][name] = rows
        result["model_summaries"][name] = summarize(rows, protocol["constraints"])
        (args.output / f"{name}_summary.json").write_text(json.dumps(result["model_summaries"][name], indent=2) + "\n")
        del policy
        gc.collect()
    result["paired_input_identity_equal"] = all(
        a["query"] == b["query"] and a["input_sha256"] == b["input_sha256"]
        for a, b in zip(result["rows"]["m3"], result["rows"][selected_name], strict=True)
    )
    sources = {Path(__file__).resolve()}
    for name, module in list(sys.modules.items()):
        if name.startswith("openpi.") and getattr(module, "__file__", None):
            source = Path(module.__file__).resolve()
            if source.suffix == ".py" and source.is_relative_to(root):
                sources.add(source)
    source_hashes = {}
    for source in sorted(sources):
        relative = source.relative_to(root)
        target = args.output / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_hashes[str(relative)] = sha256_file(source)
    result["source_hashes"] = source_hashes
    result["runtime"] = {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "jax": jax.__version__}
    result["elapsed_seconds"] = time.time() - started
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"complete": True, "model_summaries": result["model_summaries"]}), flush=True)


if __name__ == "__main__":
    main()
