"""Validate the exact exported checkpoint through native policy inference."""

import argparse
import dataclasses
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

from openpi.policies.backbone_gradient_policy import create_backbone_gradient_policy
from openpi.training import config, data_loader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--allow-engineering", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    policy = create_backbone_gradient_policy(args.checkpoint, device=args.device,
                                             allow_engineering=args.allow_engineering)
    cfg = config.get_config("pi05_piper_stage1")
    dc = dataclasses.replace(cfg.data.create(cfg.assets_dirs, cfg.model), split="val")
    dataset = data_loader.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
    sample = dataset[0]
    observation = {"prompt":sample["task"], "state":np.asarray(sample["observation.state"]).copy(),
                   "images":{name:np.asarray(sample[f"observation.images.{name}"]).copy()
                             for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")}}
    state = observation["state"].copy()
    images = {name:value.copy() for name, value in observation["images"].items()}
    noise = np.random.default_rng(4250).standard_normal((50, 32)).astype(np.float32)
    first = policy.infer(observation, noise=noise)
    second = policy.infer(observation, noise=noise)
    assert first["actions"].shape == (50, 14) and np.isfinite(first["actions"]).all()
    np.testing.assert_array_equal(first["actions"], second["actions"])
    np.testing.assert_array_equal(state, observation["state"])
    for name, image in images.items():
        np.testing.assert_array_equal(image, observation["images"][name])
    assert first["subtask"] == second["subtask"]
    try:
        policy.infer({**observation, "subtask":"external target"})
    except ValueError:
        pass
    else:
        raise AssertionError("External subtask supervision was accepted")
    result = {"passed":True, "checkpoint":str(args.checkpoint), "metadata":policy.metadata,
              "native_shape":[50,14], "fixed_noise_repeat_equal":True, "inputs_unchanged":True,
              "external_subtask_rejected":True, "subtask":first["subtask"], "subtask_status":first["subtask_status"],
              "scope":"loadable experimental robot policy; not a physical readiness or task-success result"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
