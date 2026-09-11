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
import safetensors

from openpi.policies.recurrent_subtask_policy import create_recurrent_subtask_policy
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
    policy = create_recurrent_subtask_policy(args.checkpoint, device=args.device,
                                             allow_engineering=args.allow_engineering)
    metadata = json.loads((args.checkpoint / "metadata.json").read_text())
    frozen_tensors = 0
    mode = metadata["config"]["mode"]
    if mode in {"frozen", "limited"}:
        # The unmerged LoRA file keeps original B separate from its learned delta.
        filename = "training_model.safetensors" if mode == "limited" else "model.safetensors"
        with safetensors.safe_open(str(args.checkpoint / filename), framework="pt", device="cpu") as trained:
            with safetensors.safe_open("checkpoints/pi05_base_pytorch/model.safetensors", framework="pt", device="cpu") as official:
                for key in official.keys():
                    if not key.startswith("paligemma_with_expert.paligemma."):
                        continue
                    target = "base." + key
                    if target not in trained.keys():
                        target = target.rsplit(".", 1)[0] + ".original." + target.rsplit(".", 1)[1]
                    actual = trained.get_tensor(target)
                    expected = official.get_tensor(key).to(actual.dtype)
                    if not torch.equal(actual, expected):
                        raise AssertionError(f"Frozen original B changed: {key}")
                    frozen_tensors += 1
        # safetensors stores the tied language embedding / LM head only once.
        assert frozen_tensors == 603, frozen_tensors
        backbone = policy.model.base.paligemma_with_expert.paligemma
        assert backbone.lm_head.weight.data_ptr() == backbone.language_model.embed_tokens.weight.data_ptr()
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
    left, right = policy.new_session(), policy.new_session()
    first = left.infer(observation, noise=noise)
    second = right.infer(observation, noise=noise)
    next_left = left.infer(observation, noise=noise)
    next_right = right.infer(observation, noise=noise)
    np.testing.assert_array_equal(next_left["actions"], next_right["actions"])
    if policy.model.decoder.recurrent:
        assert policy._memory is None
        assert left._memory is not right._memory
        torch.testing.assert_close(left._memory, right._memory, rtol=0, atol=0)
    left.reset()
    restarted = left.infer(observation, noise=noise)
    np.testing.assert_array_equal(first["actions"], restarted["actions"])
    assert first["actions"].shape == (50, 14) and np.isfinite(first["actions"]).all()
    np.testing.assert_array_equal(first["actions"], second["actions"])
    np.testing.assert_array_equal(state, observation["state"])
    for name, image in images.items():
        np.testing.assert_array_equal(image, observation["images"][name])
    assert first["subtask"] == second["subtask"]
    gradient_contract = None
    if args.allow_engineering:
        from openpi.training.subtask_batch import SubtaskTrainingDataset, collate_subtask
        batch = collate_subtask([SubtaskTrainingDataset(dataset, dc)[0]]).to(args.device)
        model = policy.model
        context = model.prepare_context(batch.observation, batch.global_prompts)
        projected, mask, _ = model.compose_context(context)
        ce = model.decoder.loss_projected(batch.target_ids, batch.target_mask, projected, mask, model.embedding_weight)
        ce.backward()
        assert all(p.grad is None for p in model.base.parameters())
        if model.decoder.recurrent:
            assert model.decoder.memory_update.weight_hh.grad.abs().sum() > 0
        model.zero_grad(set_to_none=True)
        generated = model.generate_subtask(context)[0]
        model.action_loss_joint(batch.observation, context, generated, batch.actions).backward()
        assert all(p.grad is None for p in model.subtask_parameters())
        assert all(p.grad is None for p in model.base.paligemma_with_expert.paligemma.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.action_parameters())
        model.zero_grad(set_to_none=True)
        gradient_contract = "actual checkpoint: CE only S including recurrence; action only A; B frozen"
    try:
        policy.infer({**observation, "subtask":"external target"})
    except ValueError:
        pass
    else:
        raise AssertionError("External subtask supervision was accepted")
    result = {"passed":True, "checkpoint":str(args.checkpoint), "metadata":policy.metadata,
              "gradient_contract":gradient_contract,
              "unchanged_original_B_tensors":frozen_tensors,
              "shared_embedding_head_alias_verified":mode in {"frozen", "limited"},
              "native_shape":[50,14], "fixed_noise_repeat_equal":True, "inputs_unchanged":True,
              "external_subtask_rejected":True, "session_isolation":True, "reset_reproducible":True, "subtask":first["subtask"], "subtask_status":first["subtask_status"],
              "scope":"loadable experimental robot policy; not a physical readiness or task-success result"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
