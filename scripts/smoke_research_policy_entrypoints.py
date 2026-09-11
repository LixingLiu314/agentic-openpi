"""CPU verification of actual research deployment loaders and client outputs."""

import argparse
import gc
import json
import os
from pathlib import Path
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["HF_HUB_OFFLINE"] = "1"

from benchmark_subtask_policy import load_policy
from benchmark_subtask_policy import verify_chunk_broker
import numpy as np
import torch

from openpi.training import data_loader
from openpi.training.stage1_data import sha256_file


@torch.no_grad()
def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    assert not torch.cuda.is_available()
    root = Path("checkpoints/pi05_piper_stage1")
    checkpoints = [
        ("C0", root / "m0_pilot_seed42/step_001000"),
        ("G", root / "m3_pilot_seed42/step_003500"),
        ("B1", root / "b1_pilot_seed42/step_004000"),
    ]
    rows = []
    for method, checkpoint in checkpoints:
        print(json.dumps({"event": "loading", "method": method, "checkpoint": str(checkpoint)}), flush=True)
        begun = time.monotonic()
        policy, model_config, dc = load_policy(checkpoint, torch.device("cpu"), 10)
        assert dc.split == "val"
        dataset = data_loader.create_torch_dataset(dc, model_config.action_horizon, model_config)
        sample = dataset[0]
        observation = {
            "images": {
                camera: np.asarray(sample[f"observation.images.{camera}"])
                for camera in ["cam_high", "cam_left_wrist", "cam_right_wrist"]
            },
            "state": np.asarray(sample["observation.state"]),
            "prompt": sample["task"],
        }
        state = observation["state"].copy()
        images = {name: image.copy() for name, image in observation["images"].items()}
        noise = np.random.default_rng(4250).standard_normal((50, 32)).astype(np.float32)
        print(json.dumps({"event": "loaded", "method": method}), flush=True)
        first = policy.infer(observation, noise=noise)
        second = policy.infer(observation, noise=noise)
        assert first["actions"].shape == (50, 14)
        assert np.isfinite(first["actions"]).all()
        np.testing.assert_array_equal(first["actions"], second["actions"])
        np.testing.assert_array_equal(observation["state"], state)
        for name, image in images.items():
            np.testing.assert_array_equal(observation["images"][name], image)
        verify_chunk_broker(first)
        if method == "G":
            assert first["subtask"] == second["subtask"]
            assert first["subtask_status"] in {"ok", "empty", "truncated"}
            assert isinstance(first["subtask_score"], float)
            try:
                policy.infer({**observation, "subtask": "external ground truth"}, noise=noise)
            except ValueError:
                pass
            else:
                raise AssertionError("Normal G deployment accepted an external subtask")
        row = {
            "method": method,
            "checkpoint": str(checkpoint),
            "weights_sha256": sha256_file(checkpoint / "model.safetensors"),
            "metadata_sha256": sha256_file(checkpoint / "metadata.json"),
            "normal_loader_without_engineering_override": True,
            "flow_steps": 10,
            "native_shape": list(first["actions"].shape),
            "finite_actions": True,
            "fixed_noise_repeat_exact": True,
            "inputs_unchanged": True,
            "chunk_broker_50_slices_metadata_reset_replan": True,
            "subtask": first.get("subtask"),
            "subtask_status": first.get("subtask_status"),
            "external_subtask_rejected": True if method == "G" else None,
            "elapsed_seconds_including_load": time.monotonic() - begun,
        }
        rows.append(row)
        print(json.dumps({"event": "method_passed", **row}), flush=True)
        del policy, dataset, sample, first, second
        gc.collect()
    report = {
        "scope": "Actual research seed42 weights through normal C0/G/B1 deployment loaders; one real validation observation, CPU only. Interface and deterministic-output evidence, not a latency/memory benchmark or generalization result.",
        "torch_version": torch.__version__,
        "cpu_threads": 4,
        "checks": rows,
        "sources": {
            str(path): sha256_file(path)
            for path in [
                Path(__file__),
                Path("scripts/benchmark_subtask_policy.py"),
                Path("src/openpi/training/research_checkpoint.py"),
                Path("src/openpi/policies/subtask_policy.py"),
                Path("src/openpi/policies/policy.py"),
                Path("packages/openpi-client/src/openpi_client/action_chunk_broker.py"),
            ]
        },
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"event": "complete", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
