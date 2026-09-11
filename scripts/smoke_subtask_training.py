"""Real-frame check of sidecar labels and M1/M2/M3 training forward contracts."""

import argparse
import dataclasses
import json
from pathlib import Path

import safetensors.torch
import torch

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pi05_subtask_pytorch import Pi05SubtaskPytorch
from openpi.training import config
from openpi.training import data_loader
from openpi.training.subtask_batch import SubtaskTrainingDataset
from openpi.training.subtask_batch import collate_subtask


def main(output):
    torch.set_num_threads(16)
    torch.manual_seed(42)
    cfg = config.get_config("pi05_piper_stage1")
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, dtype="float32"))
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    raw = data_loader.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
    dataset = SubtaskTrainingDataset(raw, dc)
    labels = dataset.labels_without_video()
    assert len(labels) == len(dataset)
    batch = collate_subtask([dataset[0]]).to(torch.device("cpu"))
    assert batch.labels == (labels[0],)
    assert batch.actions.shape == (1, 50, 32)
    assert not hasattr(batch.observation, "subtask")
    base = PI0Pytorch(cfg.model)
    checkpoint = Path("checkpoints/pi05_piper_stage1/m0_cpu_smoke_seed42/step_000002/model.safetensors")
    safetensors.torch.load_model(base, checkpoint, strict=True)
    model = Pi05SubtaskPytorch(base)
    report = {"scope": "real CPU FP32 engineering check; untrained subtask decoder", "stages": {}}
    for stage in ["m1", "m2", "m3"]:
        model.set_stage(stage)
        model.zero_grad(set_to_none=True)
        torch.manual_seed(19)
        output_losses = model(batch)
        loss = sum(value for name, value in output_losses.items() if name.startswith("loss_"))
        loss.backward()
        assert all(parameter.grad is None for parameter in model.base.paligemma_with_expert.paligemma.parameters())
        assert any(parameter.grad is not None for parameter in model.subtask_parameters())
        if stage == "m1":
            assert all(parameter.grad is None for parameter in model.action_parameters())
        else:
            assert any(parameter.grad is not None for parameter in model.action_parameters())
        report["stages"][stage] = {
            name: float(value.detach()) if isinstance(value, torch.Tensor) else value
            for name, value in output_losses.items()
        }
        del output_losses, loss
        print(f"{stage}: forward/backward parameter ownership passed", flush=True)
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        torch.manual_seed(21)
        expected = model(batch)["loss_action"]
        # Corrupt both supervision text and target tokens; generated action loss must be identical.
        ids, mask = model.codec.targets(["close the box lid"])
        corrupted = dataclasses.replace(
            batch, labels=("close the box lid",), target_ids=torch.from_numpy(ids), target_mask=torch.from_numpy(mask)
        )
        torch.manual_seed(21)
        actual = model(corrupted)["loss_action"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    report["predicted_action_independent_of_supervision"] = True
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
