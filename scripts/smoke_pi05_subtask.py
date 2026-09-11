"""CPU real-model gate: branch gradients, empty-condition parity and immutable KV.

This uses a smoke checkpoint and one real observation. It establishes numerical
contracts only; it is not an evaluation of learned subtask/action quality.
"""

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import safetensors.torch
import torch
from train_subtask_pytorch import collate_observation
from train_subtask_pytorch import to_device

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pi05_subtask_pytorch import Pi05SubtaskPytorch
from openpi.training import config
from openpi.training import data_loader


def smoke(checkpoint, output):
    torch.set_num_threads(16)
    torch.manual_seed(42)
    cfg = config.get_config("pi05_piper_stage1")
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, dtype="float32"))
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    raw = data_loader.create_torch_dataset(dc, 50, cfg.model)
    sample = raw[0]
    observation, actions = to_device(
        collate_observation([data_loader.transform_dataset(raw, dc)[0]]), torch.device("cpu")
    )
    base = PI0Pytorch(cfg.model)
    safetensors.torch.load_model(base, checkpoint / "model.safetensors", strict=True)
    model = Pi05SubtaskPytorch(base)
    subtask_ids = {id(parameter) for parameter in model.subtask_parameters()}
    action_ids = {id(parameter) for parameter in model.action_parameters()}
    assert not subtask_ids & action_ids
    assert id(model.embedding_weight) not in subtask_ids | action_ids
    model.train()
    assert not model.base.paligemma_with_expert.paligemma.training
    print("loaded real pi05; parameter ownership and frozen train mode passed", flush=True)
    context = model.prepare_context(observation, [sample["task"]])
    prompt_ids, prompt_mask = model.codec.prompts(context.global_prompts, context.state.numpy())
    np.testing.assert_array_equal(prompt_ids, observation.tokenized_prompt.numpy())
    np.testing.assert_array_equal(prompt_mask, observation.tokenized_prompt_mask.numpy())
    targets, target_mask = model.codec.targets([sample["subtask"]])
    loss_subtask = model.subtask_loss(context, torch.from_numpy(targets), torch.from_numpy(target_mask))
    loss_subtask.backward()
    assert all(parameter.grad is None for parameter in model.base.parameters())
    subtask_grad_norm = float(model.decoder.output_projection.weight.grad.norm())
    assert subtask_grad_norm > 0
    model.zero_grad(set_to_none=True)
    print("real prefix -> subtask CE gradient isolation passed", flush=True)
    prefix = model.action_prefix(context, [""])
    cache_before = [
        (prefix.cache[layer][0].clone(), prefix.cache[layer][1].clone()) for layer in range(len(prefix.cache))
    ]
    noise = torch.randn_like(actions)
    flow_time = torch.tensor([0.4])
    loss_action = model.action_loss(context, prefix, actions, noise=noise, time=flow_time)
    loss_action.backward()
    assert all(parameter.grad is None for parameter in model.subtask_parameters())
    assert all(parameter.grad is None for parameter in base.paligemma_with_expert.paligemma.parameters())
    assert all(not parameter.requires_grad for parameter in base.paligemma_with_expert.paligemma.parameters())
    action_grad_norm = float(base.action_out_proj.weight.grad.norm())
    assert action_grad_norm > 0
    model.zero_grad(set_to_none=True)
    print("cached prefix -> action loss gradient isolation passed", flush=True)
    model.eval()
    with torch.no_grad():
        reference_loss = base(observation, actions, noise=noise, time=flow_time).mean()
        torch.testing.assert_close(loss_action, reference_loss, rtol=2e-5, atol=2e-6)
        reference = base.sample_actions(torch.device("cpu"), observation, noise=noise, num_steps=2)
        predicted = model.sample_actions_from_prefix(context, prefix, noise=noise, num_steps=2)
        torch.testing.assert_close(predicted, reference, rtol=2e-5, atol=2e-6)
    for layer, (key, value) in enumerate(cache_before):
        torch.testing.assert_close(prefix.cache[layer][0], key, rtol=0, atol=0)
        torch.testing.assert_close(prefix.cache[layer][1], value, rtol=0, atol=0)
    print("official empty-condition parity and immutable cache passed", flush=True)
    result = model.infer(observation, [sample["task"]], noise=noise, num_steps=2)
    assert result["actions"].shape == (1, 50, 32)
    assert torch.isfinite(result["actions"]).all()
    report = {
        "scope": "engineering smoke using one real train frame; decoder untrained",
        "checkpoint": str(checkpoint),
        "subtask_parameters": sum(parameter.numel() for parameter in model.subtask_parameters()),
        "action_parameters": sum(parameter.numel() for parameter in model.action_parameters()),
        "loss_subtask": float(loss_subtask.detach()),
        "loss_action": float(loss_action.detach()),
        "subtask_output_gradient_norm": subtask_grad_norm,
        "action_output_gradient_norm": action_grad_norm,
        "gradient_ownership": "S-only CE; A-only flow; zero VLM gradients",
        "empty_condition_parity": True,
        "cache_immutable": True,
        "inference_shape": list(result["actions"].shape),
        "generated_subtasks": result["subtasks"],
        "subtask_status": result["subtask_status"],
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/pi05_piper_stage1/m0_cpu_smoke_seed42/step_000002")
    )
    parser.add_argument("--output", type=Path, default=Path("logs/pi05_subtask_stage1/hierarchy_cpu_smoke.json"))
    args = parser.parse_args()
    smoke(args.checkpoint, args.output)
