"""Prove the constructor-only expert LM head is absent from the VLA path."""

import dataclasses
import json
from pathlib import Path

import safetensors.torch
import torch

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pi05_subtask_pytorch import Pi05SubtaskPytorch
from openpi.training import config
from openpi.training import data_loader
from openpi.training.stage1_data import sha256_file
from openpi.training.subtask_batch import SubtaskTrainingDataset
from openpi.training.subtask_batch import collate_subtask


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    checkpoint = Path("checkpoints/pi05_piper_stage1/m3_pilot_seed42/step_003500")
    cfg = config.get_config("pi05_piper_stage1")
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, dtype="float32"))
    dc = dataclasses.replace(cfg.data.create(cfg.assets_dirs, cfg.model), split="val")
    raw = data_loader.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
    dataset = SubtaskTrainingDataset(raw, dc)
    batch = collate_subtask([dataset[0]])
    model = Pi05SubtaskPytorch(PI0Pytorch(cfg.model)).eval()
    safetensors.torch.load_model(model, checkpoint / "model.safetensors", strict=True)
    noise = torch.randn(batch.actions.shape, generator=torch.Generator().manual_seed(90210))
    before = model.infer(batch.observation, batch.global_prompts, noise=noise, num_steps=2)
    head = model.base.paligemma_with_expert.gemma_expert.lm_head
    assert not head.weight.requires_grad
    head.weight.fill_(float("nan"))

    def fail_if_called(*_):
        raise AssertionError("Unused expert LM head was called")

    hook = head.register_forward_pre_hook(fail_if_called)
    after = model.infer(batch.observation, batch.global_prompts, noise=noise, num_steps=2)
    context = model.prepare_context(batch.observation, batch.global_prompts)
    prefix = model.action_prefix(context, [""])
    loss = model.action_loss(context, prefix, batch.actions, noise=noise, time=torch.tensor([0.5]))
    assert torch.isfinite(loss)
    assert torch.isfinite(after["actions"]).all()
    assert torch.equal(before["actions"], after["actions"])
    for key in before:
        if key != "actions":
            assert (
                torch.equal(before[key], after[key]) if torch.is_tensor(before[key]) else before[key] == after[key]
            ), key
    hook.remove()
    report = {
        "passed": True,
        "checkpoint": str(checkpoint),
        "device": "cpu",
        "precision": "float32",
        "scope": "One real validation observation, complete generated subtask and two-step model-space action inference; exact actions/semantic metadata before and after poisoning expert lm_head with NaNs; empty-condition flow loss finite and forward hook never invoked.",
        "tensor": "paligemma_with_expert.gemma_expert.lm_head.weight",
        "sources": {
            str(path): sha256_file(path)
            for path in [
                Path(__file__),
                Path("src/openpi/models_pytorch/pi0_pytorch.py"),
                Path("src/openpi/models_pytorch/gemma_pytorch.py"),
                Path("src/openpi/models_pytorch/pi05_subtask_pytorch.py"),
            ]
        },
    }
    output = Path("logs/pi05_subtask_stage1/unused_expert_head_behavior.json")
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
