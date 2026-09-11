"""Frozen pi05 backbone, independent subtask decoder, internally conditioned actions."""

import dataclasses
import math
import time as timing
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from openpi.models.subtask_tokenizer import SubtaskTextCodec
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.models_pytorch.subtask_decoder import SubtaskDecoder
from openpi.models_pytorch.subtask_decoder import SubtaskDecoderConfig
from openpi.training.subtask_batch import select_action_conditions


@dataclasses.dataclass
class SubtaskContext:
    state: torch.Tensor
    global_prompts: tuple[str, ...]
    image_features: tuple[torch.Tensor, ...]
    image_masks: tuple[torch.Tensor, ...]
    memory: torch.Tensor
    memory_mask: torch.Tensor


@dataclasses.dataclass
class ActionPrefix:
    mask: torch.Tensor
    cache: Any


class Pi05SubtaskPytorch(nn.Module):
    """Owns one pi05 model and one small decoder; no duplicate VLM weights.

    Supervision is passed only to subtask_loss. The default infer path accepts
    observations and global instructions, and obtains its action condition by
    full autoregressive decoding from BOS. Teacher-forced states never feed A.
    """

    def __init__(self, base: PI0Pytorch, decoder_config: SubtaskDecoderConfig | None = None, *, state_dim=14):
        super().__init__()
        if not base.pi05:
            raise ValueError("The hierarchical model requires pi05=True")
        self.base = base
        self.decoder = SubtaskDecoder(decoder_config).to(base.action_in_proj.weight.device)
        self.codec = SubtaskTextCodec(base.config.max_token_len, self.decoder.config.max_tokens, state_dim)
        for parameter in self.base.parameters():
            parameter.requires_grad_(requires_grad=False)
        for parameter in self.action_parameters():
            parameter.requires_grad_(requires_grad=True)
        self.base.gradient_checkpointing_disable()
        self.train()

    def set_stage(self, stage):
        if stage not in {"m1", "m2", "m3"}:
            raise ValueError("Expected hierarchy stage m1, m2 or m3")
        self.stage = stage
        for parameter in self.action_parameters():
            parameter.requires_grad_(stage != "m1")
        for parameter in self.subtask_parameters():
            parameter.requires_grad_(requires_grad=True)

    def forward(self, batch, *, use_prediction=None, drop_condition=None):
        """Training entry point for DDP, with disjoint S/A gradient ownership.

        Target token IDs enter only CE. Action conditioning uses an independent
        BOS-only generation or the explicit M2/M3 teacher curriculum sidecar.
        The two losses can be backpropagated together because B is frozen and
        there is no differentiable S-to-A connection.
        """
        stage = getattr(self, "stage", "m3")
        context = self.prepare_context(batch.observation, batch.global_prompts)
        loss_subtask = self.subtask_loss(context, batch.target_ids, batch.target_mask)
        if stage == "m1":
            return {"loss_subtask": loss_subtask}
        size = len(batch.global_prompts)
        use_prediction = tuple([stage == "m3"] * size if use_prediction is None else use_prediction)
        drop_condition = tuple([False] * size if drop_condition is None else drop_condition)
        if stage == "m2" and any(use_prediction):
            raise ValueError("M2 is the explicit GT action warmup")
        generated, statuses = [""] * size, ["unused"] * size
        if any(use_prediction):
            generated, statuses, _ = self.generate_subtask(context)
        conditions = select_action_conditions(batch.labels, generated, use_prediction, drop_condition)
        prefix = self.action_prefix(context, conditions)
        loss_action = self.action_loss(context, prefix, batch.actions)
        return {
            "loss_subtask": loss_subtask,
            "loss_action": loss_action,
            "generated_count": sum(use_prediction),
            "invalid_generation_count": sum(
                use and status != "ok" for use, status in zip(use_prediction, statuses, strict=True)
            ),
            "empty_condition_count": sum(not condition for condition in conditions),
        }

    @property
    def embedding_weight(self):
        return self.base.paligemma_with_expert.paligemma.language_model.embed_tokens.weight

    def action_parameters(self):
        modules = [
            self.base.paligemma_with_expert.gemma_expert.model,
            self.base.action_in_proj,
            self.base.action_out_proj,
            self.base.time_mlp_in,
            self.base.time_mlp_out,
        ]
        return [parameter for module in modules for parameter in module.parameters()]

    def subtask_parameters(self):
        return list(self.decoder.parameters())

    def train(self, mode=True):  # noqa: FBT002
        super().train(mode)
        # Outer train() must never re-enable VLM training/checkpointing modes.
        self.base.paligemma_with_expert.paligemma.eval()
        self.base.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.base.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        return self

    def _prefix(self, image_features, image_masks, tokens, token_masks, *, use_cache):
        model = self.base.paligemma_with_expert
        language = model.embed_language_tokens(tokens)
        language = language * math.sqrt(language.shape[-1])
        embeddings = torch.cat([*image_features, language], dim=1)
        masks = [
            mask[:, None].expand(feature.shape[:2]) for feature, mask in zip(image_features, image_masks, strict=True)
        ]
        mask = torch.cat([*masks, token_masks], dim=1)
        ar_mask = torch.zeros_like(mask)
        attention = self.base._prepare_attention_masks_4d(make_att_2d_masks(mask, ar_mask))  # noqa: SLF001
        embeddings = embeddings.to(model.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype)
        model.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        (hidden, _), cache = model(
            attention_mask=attention,
            position_ids=mask.cumsum(dim=1) - 1,
            inputs_embeds=[embeddings, None],
            past_key_values=None,
            use_cache=use_cache,
        )
        return hidden, mask, cache

    def _tokens(self, global_prompts, state, subtasks=None):
        ids, masks = self.codec.prompts(global_prompts, state.detach().cpu().numpy(), subtasks)
        return torch.as_tensor(ids, device=state.device), torch.as_tensor(masks, device=state.device)

    @torch.no_grad()
    def prepare_context(self, observation, global_prompts, *, augment=False, timings=None):
        def timestamp():
            device = self.base.action_in_proj.weight.device
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            return timing.perf_counter()

        begun = timestamp() if timings is not None else None
        images, image_masks, _, _, state = self.base._preprocess_observation(observation, train=augment)  # noqa: SLF001
        features = tuple(self.base.paligemma_with_expert.embed_image(image) for image in images)
        after_vision = timestamp() if timings is not None else None
        tokens, masks = self._tokens(global_prompts, state)
        hidden, mask, _ = self._prefix(features, image_masks, tokens, masks, use_cache=False)
        if timings is not None:
            timings["vision_ms"] = (after_vision - begun) * 1000
            timings["prefix1_ms"] = (timestamp() - after_vision) * 1000
        return SubtaskContext(state, tuple(global_prompts), features, tuple(image_masks), hidden, mask)

    def subtask_loss(self, context, target_ids, target_mask):
        return self.decoder.compute_loss(
            target_ids, target_mask, context.memory, context.memory_mask, self.embedding_weight
        )

    @torch.no_grad()
    def generate_subtask(self, context):
        generation = self.decoder.generate(context.memory, context.memory_mask, self.embedding_weight)
        texts, statuses = self.codec.decode(generation)
        return texts, statuses, generation

    @torch.no_grad()
    def action_prefix(self, context, subtasks):
        tokens, masks = self._tokens(context.global_prompts, context.state, subtasks)
        # Recompute the entire bidirectional language prefix; only images are reused.
        _, mask, cache = self._prefix(context.image_features, context.image_masks, tokens, masks, use_cache=True)
        if cache is None:
            raise RuntimeError("Frozen VLM did not produce a prefix cache")
        return ActionPrefix(mask, cache)

    def action_loss(self, context, prefix, actions, *, noise=None, time=None, reduction="mean"):
        noise = self.base.sample_noise(actions.shape, actions.device) if noise is None else noise
        time = self.base.sample_time(actions.shape[0], actions.device) if time is None else time
        noisy = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        velocity = self.base.denoise_step(context.state, prefix.mask, prefix.cache, noisy, time)
        return F.mse_loss(velocity, noise - actions, reduction=reduction)

    @torch.no_grad()
    def sample_actions_from_prefix(self, context, prefix, *, noise=None, num_steps=10):
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        batch = context.state.shape[0]
        device = context.state.device
        if noise is None:
            noise = self.base.sample_noise(
                (batch, self.base.config.action_horizon, self.base.config.action_dim), device
            )
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        actions = noise
        while time >= -dt / 2:
            velocity = self.base.denoise_step(context.state, prefix.mask, prefix.cache, actions, time.expand(batch))
            actions = actions + dt * velocity
            time += dt
        return actions

    @torch.no_grad()
    def infer(self, observation, global_prompts, *, noise=None, num_steps=10):
        """Model-space outputs; the policy layer performs native action conversion."""
        previous_training = self.training
        self.eval()
        try:
            context = self.prepare_context(observation, global_prompts)
            texts, statuses, generation = self.generate_subtask(context)
            prefix = self.action_prefix(context, texts)
            actions = self.sample_actions_from_prefix(context, prefix, noise=noise, num_steps=num_steps)
            return {
                "actions": actions,
                "subtasks": texts,
                "subtask_status": statuses,
                "subtask_sequence_score": generation.mean_log_probability,
            }
        finally:
            self.train(previous_training)
