import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import safetensors.torch
import torch

from openpi.training import hierarchy_training as training


class TinyHierarchy(torch.nn.Module):
    def __init__(self, *, stage="m2"):
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4)
        self.subtask = torch.nn.Linear(4, 2)
        self.action = torch.nn.Linear(4, 2).double()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(requires_grad=False)
        for parameter in self.action.parameters():
            parameter.requires_grad_(stage != "m1")

    def subtask_parameters(self):
        return list(self.subtask.parameters())

    def action_parameters(self):
        return list(self.action.parameters())

    def forward(self, x):
        features = self.backbone(x).detach()
        return self.subtask(features).square().mean(), self.action(features.double()).square().mean()


def test_momentum_and_weight_decay_cannot_cross_branch_ownership():
    torch.manual_seed(42)
    model = TinyHierarchy()
    optimizers = training.BranchOptimizers(model, "m2", 0.01, 0.01)
    x = torch.randn(3, 4)
    sum(model(x)).backward()
    optimizers.step()
    for active, inactive in [(0, "action"), (1, "subtask")]:
        optimizers.zero_grad()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        model(x)[active].backward()
        optimizers.step()
        for key, value in model.state_dict().items():
            if key.startswith((inactive, "backbone")):
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_checkpoint_resume_and_stage_optimizer_transition(tmp_path):
    torch.manual_seed(19)
    model = TinyHierarchy(stage="m1")
    optimizers = training.BranchOptimizers(model, "m1", 0.01, 0.01)
    model(torch.randn(3, 4))[0].backward()
    optimizers.step()
    optimizers.zero_grad()
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "split.json").write_text("{}")
    output = tmp_path / "run"
    output.mkdir()
    (output / "sources").mkdir()
    (output / "sources/model.py").write_text("source snapshot")
    checkpoint = training.save_checkpoint(
        model, optimizers, output, 1, {"stage": "m1"}, None, {"subtask": 1, "action": 0}, assets
    )
    assert (checkpoint / "sources/model.py").read_text() == "source snapshot"
    expected_x = torch.randn(3, 4)
    expected_loss = model(expected_x)[0]
    expected_loss.backward()
    optimizers.step()
    expected_state = {key: value.clone() for key, value in model.state_dict().items()}
    restored = TinyHierarchy(stage="m1")
    safetensors.torch.load_model(restored, checkpoint / "model.safetensors", strict=True)
    resumed = training.BranchOptimizers(restored, "m1", 0.01, 0.01)
    state = torch.load(checkpoint / "training.pt", weights_only=False)
    resumed.load_state_dict(state["branch_optimizers"])
    training.restore_random_state(state["rng"])
    actual_loss = restored(torch.randn(3, 4))[0]
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    actual_loss.backward()
    resumed.step()
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected_state[key], rtol=0, atol=0)
    state = torch.load(checkpoint / "training.pt", weights_only=False)
    next_model = TinyHierarchy(stage="m2")
    safetensors.torch.load_model(next_model, checkpoint / "model.safetensors", strict=True)
    next_optimizers = training.BranchOptimizers(next_model, "m2", 0.01, 0.01)
    next_optimizers.load_state_dict(state["branch_optimizers"], transition=True)
    assert not next_optimizers.optimizers["action"].state
    assert all(value["step"] == 1 for value in next_optimizers.optimizers["subtask"].state.values())


def test_generation_metrics_count_unknowns_as_errors():
    rows = [
        {"label": "A", "prediction": "a", "status": "ok"},
        {"label": "B", "prediction": "unknown", "status": "ok"},
        {"label": "B", "prediction": "", "status": "truncated"},
    ]
    result = training.text_metrics(rows, ["A", "B"])
    assert result["exact_match"] == 0
    assert result["normalized_exact_match"] == pytest.approx(1 / 3)
    assert result["macro_f1"] == 0.5
    assert result["unknown_rate"] == pytest.approx(2 / 3)
    assert result["per_class"]["b"]["recall"] == 0


def test_stage_budget_and_overfit_coverage(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))
    module = importlib.import_module("train_subtask_hierarchy")
    labels = ["shared"] * 100 + ["unique"] * 20
    tasks = ["A"] * 50 + ["B"] * 50 + ["B"] * 20
    indices = module.overfit_indices(labels, tasks, 6)
    assert len(set(indices)) == 6
    assert {(tasks[index], labels[index]) for index in indices} == {("A", "shared"), ("B", "shared"), ("B", "unique")}
    (tmp_path / "metrics.jsonl").write_text('{"event":"complete","completed_steps":12}\n')
    args = SimpleNamespace(
        initialize_from=tmp_path / "checkpoint",
        stage="m2",
        engineering_smoke=False,
        overfit_samples=0,
        steps=500,
        m3_planned_steps=2000,
    )
    module.validate_transition(args, {"stage": "m1", "config": {"steps": 12}})
    args.steps = 501
    with pytest.raises(ValueError, match="20%"):
        module.validate_transition(args, {"stage": "m1", "config": {"steps": 12}})
