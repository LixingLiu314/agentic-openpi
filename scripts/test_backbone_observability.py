"""Diagnostics must not change gradients, clipping, optimizer results or RNG."""

import copy
import math
from types import SimpleNamespace
import unittest

import torch

from openpi.training.backbone_observability import GradientObserver, event_payload, log_event


class ToyModel(torch.nn.Module):
    def __init__(self, *, frozen=False):
        super().__init__()
        self.action = torch.nn.Linear(3, 2)
        self.backbone = torch.nn.Module()
        self.backbone.vision_tower = torch.nn.Linear(3, 2, bias=False)
        self.backbone.multi_modal_projector = torch.nn.Linear(2, 2, bias=False)
        self.backbone.language_model = torch.nn.Linear(3, 2, bias=False)
        self.backbone.unused = torch.nn.Parameter(torch.ones(2))
        if frozen:
            self.backbone.requires_grad_(False)
        self.base = SimpleNamespace(paligemma_with_expert=SimpleNamespace(paligemma=self.backbone))

    def action_parameters(self):
        return list(self.action.parameters())

    def backbone_parameters(self):
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def loss(self, inputs):
        result = self.action(inputs) + self.backbone.multi_modal_projector(self.backbone.vision_tower(inputs))
        return (result + self.backbone.language_model(inputs)).square().mean()


class Checks(unittest.TestCase):
    def test_readonly_and_exact_optimizer_update(self):
        torch.manual_seed(42)
        a = ToyModel()
        b = copy.deepcopy(a)
        x = torch.randn(5, 3)
        a.loss(x).backward()
        b.loss(x).backward()
        parameters = list(b.parameters())
        before = [(p.grad, p.grad.clone() if p.grad is not None else None) for p in parameters]
        rng = torch.random.get_rng_state().clone()
        observed = GradientObserver(b).measure()
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for p, (reference, value) in zip(parameters, before):
            self.assertIs(p.grad, reference)
            if value is not None:
                self.assertTrue(torch.equal(p.grad, value))
        direct_b = math.sqrt(sum(float(p.grad.double().square().sum()) for p in b.backbone_parameters() if p.grad is not None))
        self.assertAlmostEqual(observed["backbone"], direct_b, places=6)
        self.assertGreater(observed["backbone_vision"], 0)
        self.assertGreater(observed["backbone_language"], 0)
        for model in (a, b):
            params = model.action_parameters() + model.backbone_parameters()
            opt = torch.optim.AdamW(params, lr=2.5e-5, betas=(.9,.95), eps=1e-8, weight_decay=1e-10)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        for pa, pb in zip(a.parameters(), b.parameters()):
            self.assertTrue(torch.equal(pa, pb))

    def test_frozen_and_missing_gradients(self):
        model = ToyModel(frozen=True)
        model.loss(torch.ones(2,3)).backward()
        result = GradientObserver(model).measure()
        self.assertEqual(result["backbone"], 0)
        self.assertEqual(result["backbone_tensors_with_grad"], 0)
        self.assertGreater(result["action"], 0)

    def test_mixed_precision_reference(self):
        params = [torch.nn.Parameter(torch.zeros(3,dtype=d)) for d in (torch.bfloat16,torch.float32)]
        params[0].grad = torch.tensor([3.,4.,0.],dtype=torch.bfloat16)
        params[1].grad = torch.tensor([0.,0.,12.])
        self.assertEqual(GradientObserver.norm(params), 13.0)

    def test_shared_loss_preserved_and_no_invented_b_loss(self):
        row = {"event":"train", "step":10, "loss_action":.2, "loss_subtask":.1,
               "grad_norms":{"action":3., "backbone":4., "backbone_vision":2.,
                             "backbone_language":math.sqrt(12), "action_backbone":5.,
                             "backbone_tensors_with_grad":4, "backbone_trainable_tensors":5}}
        payload = event_payload(row,{"steps":5000})
        self.assertEqual(payload["trainer/step"],10)
        self.assertEqual(payload["train/action_flow_mse_all32"],.2)
        self.assertEqual(payload["optim/grad_norm_b"],4.)
        self.assertEqual(payload["optim/b_grad_tensor_fraction"],.8)
        self.assertFalse(any("b_loss" in key for key in payload))
        class Run:
            def __init__(self):self.rows=[]
            def log(self,value,step=None):self.rows.append(value)
        run=Run();log_event(run,row,{"steps":5000})
        self.assertEqual(len(run.rows),1)
        self.assertIn("train/action_flow_mse_all32",run.rows[0])
        self.assertIn("optim/grad_norm_b",run.rows[0])

    def test_validation_never_reuses_train_metrics(self):
        payload=event_payload({"event":"validation","step":500,"loss_subtask":.5,
                              "flow_generated_native14_normalized":.03},{})
        self.assertIn("val/subtask_ce",payload)
        self.assertFalse(any(k.startswith(("train/","optim/")) for k in payload))


if __name__ == "__main__":
    unittest.main()
