"""Action-supervised backbone ablations; the discrete subtask bridge stays detached."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.models_pytorch.pi05_subtask_pytorch import Pi05SubtaskPytorch
from openpi.training.subtask_batch import select_action_conditions


class LoRALinear(nn.Module):
    """Materialize the effective weight so merged deployment is numerically identical."""

    def __init__(self, original, rank=16, alpha=32):
        super().__init__()
        self.original = original
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, original.in_features,
                                             device=original.weight.device, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(original.out_features, rank,
                                              device=original.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    @property
    def weight(self):
        return (self.original.weight.float() + self.scale * (self.lora_b @ self.lora_a)).to(self.original.weight.dtype)

    @property
    def bias(self):
        return self.original.bias

    def forward(self, inputs):
        return F.linear(inputs, self.weight, self.bias)


class BackboneGradientModel(Pi05SubtaskPytorch):
    def enable_backbone(self, mode, *, lora_rank=16, lora_alpha=32, last_layers=2):
        if mode not in {"limited", "full"}:
            raise ValueError(mode)
        self.backbone_mode = mode
        self.set_stage("m3")
        backbone = self.base.paligemma_with_expert.paligemma
        if mode == "full":
            for parameter in backbone.parameters():
                parameter.requires_grad_(True)
        else:
            if not 1 <= last_layers <= len(backbone.language_model.layers):
                raise ValueError("Invalid LoRA layer count")
            for layer in list(backbone.language_model.layers)[-last_layers:]:
                for owner, names in [(layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
                                     (layer.mlp, ("gate_proj", "up_proj", "down_proj"))]:
                    for name in names:
                        setattr(owner, name, LoRALinear(getattr(owner, name), lora_rank, lora_alpha))
        self.train()

    def backbone_parameters(self):
        return [p for p in self.base.paligemma_with_expert.paligemma.parameters() if p.requires_grad]

    def action_loss_joint(self, observation, context, subtasks, actions, *, noise=None, time=None):
        """Official joint prefix/suffix attention, with gradients through B into A.

        The prefix cannot attend action tokens. Unlike the frozen prefix-cache
        path this supports per-layer activation checkpointing without mutable KV.
        """
        model = self.base.paligemma_with_expert
        noise = self.base.sample_noise(actions.shape, actions.device) if noise is None else noise
        time = self.base.sample_time(actions.shape[0], actions.device) if time is None else time
        noisy = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        tokens, token_mask = self._tokens(context.global_prompts, context.state, subtasks)
        if self.backbone_mode == "full" and torch.is_grad_enabled():
            # Recompute images: prepare_context intentionally creates only S's detached view.
            images, image_masks, _, _, _ = self.base._preprocess_observation(observation, train=False)
            features = tuple(checkpoint(model.embed_image, image, use_reentrant=False) for image in images)
        else:
            features, image_masks = context.image_features, context.image_masks
        language = model.embed_language_tokens(tokens)
        prefix = torch.cat([*features, language * math.sqrt(language.shape[-1])], dim=1)
        prefix_mask = torch.cat([
            *(mask[:, None].expand(feature.shape[:2]) for feature, mask in zip(features, image_masks, strict=True)),
            token_mask,
        ], dim=1)
        suffix, suffix_mask, suffix_ar, cond = self.base.embed_suffix(context.state, noisy, time)
        dtype = model.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        masks = torch.cat([prefix_mask, suffix_mask], dim=1)
        ar = torch.cat([torch.zeros_like(prefix_mask), suffix_ar], dim=1)
        attention = self.base._prepare_attention_masks_4d(make_att_2d_masks(masks, ar))
        # The installed official joint forward checkpoints each Gemma layer.
        (_, output), _ = model(
            attention_mask=attention,
            position_ids=masks.cumsum(dim=1) - 1,
            inputs_embeds=[prefix.to(dtype), suffix.to(dtype)],
            past_key_values=None,
            use_cache=False,
            adarms_cond=[None, cond],
        )
        velocity = self.base.action_out_proj(output[:, -self.base.config.action_horizon:].float())
        return F.mse_loss(velocity, noise - actions)

    def forward(self, batch, *, drop_condition=None):
        context = self.prepare_context(batch.observation, batch.global_prompts)
        loss_subtask = self.subtask_loss(context, batch.target_ids, batch.target_mask)
        generated, statuses, _ = self.generate_subtask(context)
        size = len(generated)
        dropped = [False] * size if drop_condition is None else drop_condition
        conditions = select_action_conditions(batch.labels, generated, [True] * size, dropped)
        loss_action = self.action_loss_joint(batch.observation, context, conditions, batch.actions)
        return {
            "loss_subtask": loss_subtask,
            "loss_action": loss_action,
            "generated_count": size,
            "invalid_generation_count": sum(status != "ok" for status in statuses),
            "empty_condition_count": sum(not value for value in conditions),
        }

    @torch.no_grad()
    def deployment_state(self):
        """Ordinary Pi05SubtaskPytorch keys; no adapter dependency at deployment."""
        state = self.state_dict()
        for name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                for key in list(state):
                    if key.startswith(name + "."):
                        del state[key]
                state[name + ".weight"] = module.weight
                if module.bias is not None:
                    state[name + ".bias"] = module.bias
        # Match safetensors.save_model's removal of tied aliases.
        result, seen = {}, set()
        for name in sorted(state):
            value = state[name]
            identity = (value.device, value.data_ptr(), value.numel(), value.dtype)
            if identity not in seen:
                result[name] = value.detach().cpu().contiguous()
                seen.add(identity)
        return result


def cosine_lr(step, *, warmup=500, decay_steps=5000, peak=2.5e-5, end=2.5e-6):
    """Match optimizer.CosineDecaySchedule/Optax, including the nonzero initial LR."""
    if not 0 <= warmup < decay_steps:
        raise ValueError("Require 0 <= warmup < decay_steps")
    if step < warmup:
        initial = peak / (warmup + 1)
        return initial + (peak - initial) * step / warmup
    progress = min(max((step - warmup) / (decay_steps - warmup), 0.0), 1.0)
    return end + (peak - end) * 0.5 * (1.0 + math.cos(math.pi * progress))
