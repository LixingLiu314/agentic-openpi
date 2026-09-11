"""Actual M3 weights/observation: loss routing, joint/cache parity and merged export."""

import argparse
import dataclasses
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import safetensors.torch
import torch

from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.backbone_gradient import BackboneGradientModel, LoRALinear, cosine_lr
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pi05_subtask_pytorch import Pi05SubtaskPytorch
from openpi.models_pytorch.subtask_decoder import SubtaskDecoderConfig
from openpi.training import config, data_loader, optimizer
from openpi.training.subtask_batch import SubtaskTrainingDataset, collate_subtask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["limited", "full"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(42)
    schedule = optimizer.CosineDecaySchedule(warmup_steps=500, peak_lr=2.5e-5, decay_steps=5000, decay_lr=2.5e-6).create()
    for step in [0, 1, 499, 500, 501, 2500, 4999, 5000, 6000]:
        np.testing.assert_allclose(cosine_lr(step), float(schedule(step)), rtol=3e-6, atol=1e-12)
    parent = Path("checkpoints/pi05_piper_stage1/m3_pilot_seed42/step_003500")
    metadata = json.loads((parent / "metadata.json").read_text())
    cfg = config.get_config("pi05_piper_stage1")
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(Pi0Config(**metadata["config"]["model"]), dtype="float32"))
    dc = dataclasses.replace(cfg.data.create(cfg.assets_dirs, cfg.model), split="val")
    dataset = SubtaskTrainingDataset(data_loader.create_torch_dataset(dc, 50, cfg.model), dc)
    batch = collate_subtask([dataset[0]]).to(torch.device(args.device))
    model = BackboneGradientModel(PI0Pytorch(cfg.model).to(args.device), SubtaskDecoderConfig(**metadata["config"]["decoder"]))
    safetensors.torch.load_model(model, parent / "model.safetensors", strict=True)
    model.enable_backbone(args.mode)
    print(json.dumps({"event":"loaded", "mode":args.mode}), flush=True)
    context = model.prepare_context(batch.observation, batch.global_prompts)
    ce = model.subtask_loss(context, batch.target_ids, batch.target_mask)
    ce.backward()
    assert all(p.grad is None for p in model.base.parameters())
    assert any(p.grad is not None and bool(torch.count_nonzero(p.grad)) for p in model.subtask_parameters())
    model.zero_grad(set_to_none=True)
    generated, _, _ = model.generate_subtask(context)
    noise = torch.randn_like(batch.actions)
    flow_time = torch.tensor([0.4], device=args.device)
    model.eval()
    with torch.no_grad():
        cached = model.action_loss(context, model.action_prefix(context, generated), batch.actions,
                                   noise=noise, time=flow_time)
        joint = model.action_loss_joint(batch.observation, context, generated, batch.actions,
                                        noise=noise, time=flow_time)
    torch.testing.assert_close(joint, cached, rtol=2e-5, atol=2e-6)
    model.train()
    action = model.action_loss_joint(batch.observation, context, generated, batch.actions,
                                     noise=noise, time=flow_time)
    action.backward()
    assert all(p.grad is None for p in model.subtask_parameters())
    counts = {"B_with_nonzero_action_grad":sum(p.grad is not None and bool(torch.count_nonzero(p.grad))
                                               for p in model.backbone_parameters()),
              "A_with_nonzero_action_grad":sum(p.grad is not None and bool(torch.count_nonzero(p.grad))
                                               for p in model.action_parameters())}
    assert min(counts.values()) > 0
    vision = model.base.paligemma_with_expert.paligemma.vision_tower
    vision_nonzero = sum(p.grad is not None and bool(torch.count_nonzero(p.grad)) for p in vision.parameters())
    assert (vision_nonzero > 0) == (args.mode == "full")
    model.zero_grad(set_to_none=True)
    print(json.dumps({"event":"gradient_gate_passed", "mode":args.mode, **counts,
                      "vision_nonzero":vision_nonzero}), flush=True)
    if args.mode == "limited":
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, LoRALinear):
                    module.lora_b.normal_(0, .001)
        model.eval()
        with torch.no_grad():
            adapted = model.infer(batch.observation, batch.global_prompts, noise=noise, num_steps=1)
        exported = model.deployment_state()
        plain = Pi05SubtaskPytorch(PI0Pytorch(cfg.model).to(args.device), SubtaskDecoderConfig(**metadata["config"]["decoder"]))
        # save_model deduplicates tied embedding aliases; recreate them through the standard loader.
        temporary = args.output.with_suffix(".export.safetensors")
        safetensors.torch.save_file(exported, temporary)
        safetensors.torch.load_model(plain, temporary, strict=True)
        with torch.no_grad():
            merged = plain.infer(batch.observation, batch.global_prompts, noise=noise, num_steps=1)
        torch.testing.assert_close(adapted["actions"], merged["actions"], rtol=0, atol=0)
        assert adapted["subtasks"] == merged["subtasks"]
        temporary.unlink()  # only this script's temporary engineering export
    result = {"passed":True, "mode":args.mode, "device":args.device, "ce_to_S_only":True,
              "action_to_A_B_only":True, "joint_cache_parity":True, "vision_nonzero_gradients":vision_nonzero,
              "merged_export_exact":args.mode == "limited", "official_lr_schedule_match":True, **counts,
              "loss_subtask":float(ce.detach()), "loss_action":float(action.detach())}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
