"""Matched frozen-backbone action controls without subtask generation/training."""

import torch

from openpi.models_pytorch.pi05_subtask_pytorch import Pi05SubtaskPytorch
from openpi.models_pytorch.pi05_subtask_pytorch import SubtaskContext
from openpi.training.hierarchy_training import distributed_context
from openpi.training.stage1_optimizer import MixedPrecisionZeroAdamW


class ActionControl(Pi05SubtaskPytorch):
    def __init__(self, base, *, condition):
        if condition not in {"none", "oracle"}:
            raise ValueError("Action control condition must be none or oracle")
        super().__init__(base)
        self.condition = condition
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(requires_grad=False)

    @torch.no_grad()
    def prepare_action_context(self, observation, global_prompts):
        images, image_masks, _, _, state = self.base._preprocess_observation(observation, train=False)  # noqa: SLF001
        features = tuple(self.base.paligemma_with_expert.embed_image(image) for image in images)
        # Neither control invokes S or needs the first language prefix.
        return SubtaskContext(state, tuple(global_prompts), features, tuple(image_masks), None, None)

    def forward(self, batch):
        context = self.prepare_action_context(batch.observation, batch.global_prompts)
        conditions = batch.labels if self.condition == "oracle" else [""] * len(batch.labels)
        prefix = self.action_prefix(context, conditions)
        return self.action_loss(context, prefix, batch.actions)


class ActionOptimizer:
    def __init__(self, model, lr):
        self.parameters = model.action_parameters()
        if {id(p) for p in self.parameters} != {id(p) for p in model.parameters() if p.requires_grad}:
            raise ValueError("Control trainable parameters must be exactly the action expert and projections")
        self.world = distributed_context()[1]
        options = {"lr": lr, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.01}
        self.optimizer = (
            MixedPrecisionZeroAdamW(self.parameters, **options)
            if self.world > 1
            else torch.optim.AdamW(self.parameters, **options)
        )

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def step(self):
        norm = float(torch.nn.utils.clip_grad_norm_(self.parameters, 1.0, error_if_nonfinite=True))
        self.optimizer.step()
        return norm

    def state_dict(self):
        return {"action": self.optimizer.local_state_dict() if self.world > 1 else self.optimizer.state_dict()}

    def load_state_dict(self, state):
        if set(state) != {"action"}:
            raise ValueError("Unexpected control optimizer ownership")
        if self.world > 1:
            self.optimizer.load_local_state_dict(state["action"])
        else:
            self.optimizer.load_state_dict(state["action"])
