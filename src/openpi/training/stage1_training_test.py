"""Resume tests use a small stochastic model; the full pi05 smoke is separate."""

import importlib.util
from pathlib import Path
import random

import numpy as np
import safetensors.torch
import torch


def _trainer():
    path = Path(__file__).resolve().parents[3] / "scripts/train_subtask_pytorch.py"
    spec = importlib.util.spec_from_file_location("stage1_trainer_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sampling_resume_ignores_worker_prefetch():
    module = _trainer()
    complete = list(module.StepBatchSampler(137, 3, 4, 8, 0, 42))
    resumed = list(module.StepBatchSampler(137, 3, 4, 8, 3, 42))
    assert complete[3 * 4 :] == resumed
    assert complete != list(module.StepBatchSampler(137, 3, 4, 8, 0, 43))
    assert len(complete) == 32
    assert all(0 <= index < 137 for batch in complete for index in batch)


def test_checkpoint_restores_optimizer_and_all_random_sources(tmp_path):
    module = _trainer()
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(0.3), torch.nn.Linear(8, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    def update(current_model, current_optimizer):
        current_optimizer.zero_grad(set_to_none=True)
        x = torch.randn(5, 4) + float(np.random.normal()) + random.random()
        loss = current_model(x).square().mean()
        loss.backward()
        current_optimizer.step()
        return loss.detach()

    update(model, optimizer)
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "split.json").write_text("{}")
    output = tmp_path / "run"
    output.mkdir()
    checkpoint = module.save_checkpoint(model, optimizer, output, 1, {"seed": 17}, {"step": 1}, assets)
    expected_loss = update(model, optimizer)
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    restored = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(0.3), torch.nn.Linear(8, 2))
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
    safetensors.torch.load_model(restored, checkpoint / "model.safetensors", strict=True)
    state = torch.load(checkpoint / "training.pt", weights_only=False)
    restored_optimizer.load_state_dict(state["optimizer"])
    module.restore_random_state(state["rng"])
    actual_loss = update(restored, restored_optimizer)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
