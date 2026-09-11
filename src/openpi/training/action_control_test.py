import copy
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from openpi.training.action_control import ActionOptimizer


def test_control_sampler_restarts_and_resumes_at_main_stage_boundary(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    module = importlib.import_module("train_subtask_action_control")
    args = SimpleNamespace(warmup_phase_steps=2, steps=10, batch_size=2, accumulation=2, seed=42)
    batches = list(module.MatchedSampler(109, args, 0, rank=1, world=2))
    assert len(batches) == 20
    # M3 reseeds its sampler at step zero, just as the standalone main trainer does.
    assert batches[:4] == batches[4:8]
    for start in [1, 2, 3, 9, 10]:
        assert list(module.MatchedSampler(109, args, start, rank=1, world=2)) == batches[start * 2:]


class TinyControl(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.action = torch.nn.Linear(3, 2)
        self.backbone = torch.nn.Parameter(torch.ones(3), requires_grad=False)
        self.decoder = torch.nn.Parameter(torch.ones(2), requires_grad=False)

    def action_parameters(self):
        return list(self.action.parameters())


def update(model, optimizer, values):
    optimizer.zero_grad()
    model.action(values).square().mean().backward()
    optimizer.step()


def test_control_optimizer_next_update_is_exact_after_resume():
    torch.manual_seed(81)
    model = TinyControl()
    optimizer = ActionOptimizer(model, 1e-3)
    values = torch.randn(4, 3)
    update(model, optimizer, values)
    saved_model = copy.deepcopy(model.state_dict())
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    clone = TinyControl()
    clone.load_state_dict(saved_model)
    resumed = ActionOptimizer(clone, 1e-3)
    resumed.load_state_dict(saved_optimizer)
    update(model, optimizer, values * 2)
    update(clone, resumed, values * 2)
    for name, value in model.state_dict().items():
        assert torch.equal(value, clone.state_dict()[name])
    assert torch.equal(model.decoder, saved_model["decoder"])
    assert torch.equal(model.backbone, saved_model["backbone"])
    assert not torch.equal(model.action.weight, saved_model["action.weight"])


def test_control_optimizer_rejects_an_accidentally_trainable_decoder():
    model = TinyControl()
    model.decoder.requires_grad_(requires_grad=True)
    with pytest.raises(ValueError, match="exactly the action"):
        ActionOptimizer(model, 1e-3)
