"""Label-free real policy check, including native joint/gripper reconstruction."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
import torch

from openpi.models.model import Observation
from openpi.policies.piper_policy import CAMERA_MAP
from openpi.policies.piper_policy import JOINT_MASK
from openpi.policies.subtask_policy import create_subtask_policy
from openpi.shared import normalize
from openpi.training import config
from openpi.training import data_loader


def main(checkpoint, output):
    torch.set_num_threads(16)
    cfg = config.get_config("pi05_piper_stage1")
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    raw = data_loader.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
    sample = raw[0]
    obs = {
        "images": {camera: np.asarray(sample[f"observation.images.{camera}"]) for camera in CAMERA_MAP},
        "state": np.asarray(sample["observation.state"], dtype=np.float32),
        "prompt": sample["task"],
    }
    original_state = obs["state"].copy()
    original_images = {camera: value.copy() for camera, value in obs["images"].items()}
    policy = create_subtask_policy(checkpoint, device="cpu", allow_engineering=True)
    noise = np.random.default_rng(42).normal(size=(50, 32)).astype(np.float32)
    result = policy.infer(obs, noise=noise)
    assert result["actions"].shape == (50, 14)
    assert np.isfinite(result["actions"]).all()
    assert isinstance(result["subtask"], str)
    assert isinstance(result["subtask_score"], float)
    np.testing.assert_array_equal(obs["state"], original_state)
    for camera, expected in original_images.items():
        np.testing.assert_array_equal(obs["images"][camera], expected)
    transformed = policy.input_transform(obs)
    observation = Observation.from_dict(
        jax.tree.map(lambda value: torch.as_tensor(np.asarray(value))[None], transformed)
    )
    model_result = policy.model.infer(observation, [obs["prompt"]], noise=torch.from_numpy(noise)[None], num_steps=10)
    model_actions = model_result["actions"][0, :, :14].cpu().numpy()
    stats = normalize.load(checkpoint / "assets/eggplant_potato")["actions"]
    expected = (model_actions + 1) * 0.5 * (stats.q99 - stats.q01 + 1e-6) + stats.q01
    expected[:, np.asarray(JOINT_MASK)] += original_state[np.asarray(JOINT_MASK)]
    np.testing.assert_allclose(result["actions"], expected, rtol=2e-5, atol=2e-6)
    try:
        policy.infer({**obs, "subtask": "ground truth must be rejected"})
    except ValueError:
        pass
    else:
        raise AssertionError("Deployment policy accepted a subtask target")
    report = {
        "scope": "CPU FP32 engineering check, not a GPU latency benchmark or learned-quality evaluation",
        "checkpoint": str(checkpoint),
        "shape": list(result["actions"].shape),
        "subtask": result["subtask"],
        "subtask_status": result["subtask_status"],
        "policy_timing": result["policy_timing"],
        "input_unchanged": True,
        "label_free": True,
        "explicit_supervision_rejected": True,
        "native14_unnormalization_and_joint_only_absolute_reconstruction": True,
        "flow_steps": 10,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.checkpoint, args.output)
