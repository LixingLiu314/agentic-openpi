"""N1: native PaliGemma text generation, isolated global-only action stream.

No extra language network or recurrent state. Native text tokens use the same
PaliGemma layer objects and tied vocabulary as the observation stream. Original
prefix/action computations are unchanged; text reads prefix, never vice versa.
"""
import dataclasses
import math
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.models.gemma import modeling_gemma

from openpi.models.subtask_tokenizer import SubtaskTextCodec
from openpi.models_pytorch.parallel_subtask import parallel_layer
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

VARIANT = "official_pi05_native_subtask_n1_v1"
SCHEMA_VERSION = 11
DISPLAY_SET = "official-pi05-native-n1-reach-arm-s42-v1"
CONTRACT = "native CE->B; flow->A only; A never reads text targets or generated subtasks"
TEXT_CUE = "\nSubtask: "


@dataclasses.dataclass
class NativeGeneration:
    token_ids: torch.Tensor
    token_mask: torch.Tensor
    ended: torch.Tensor
    mean_log_probability: torch.Tensor


def text_attention_mask(prefix_mask, current_valid, past_length=0):
    b, t = current_valid.shape
    total = past_length + t
    causal = torch.arange(total, device=prefix_mask.device)[None, :] <= (
        past_length + torch.arange(t, device=prefix_mask.device)[:, None])
    valid = torch.cat([torch.ones((b, past_length), dtype=torch.bool, device=prefix_mask.device),
                       current_valid.bool()], 1)
    return torch.cat([prefix_mask[:, None, :].expand(b, t, -1),
                      causal[None] & valid[:, None, :]], -1)


def prefix_kv(wrapper, index, hidden, positions):
    layer = wrapper.paligemma.language_model.layers[index]
    normalized, _ = layer.input_layernorm(hidden, cond=None)
    shape = (*normalized.shape[:-1], -1, layer.self_attn.head_dim)
    k = layer.self_attn.k_proj(normalized).view(shape).transpose(1, 2)
    v = layer.self_attn.v_proj(normalized).view(shape).transpose(1, 2)
    dummy = k.new_zeros((k.shape[0], k.shape[2], k.shape[-1]))
    cos, sin = wrapper.paligemma.model.language_model.rotary_emb(dummy, positions)
    _, k = modeling_gemma.apply_rotary_pos_emb(k, k, cos, sin, unsqueeze_dim=1)
    return k, v


def native_text_layer(wrapper, index, hidden, pk, pv, positions, attention, past=None):
    """Native B layer on text rows, with a private immutable text-only KV cache."""
    layer = wrapper.paligemma.language_model.layers[index]
    normalized, gate = layer.input_layernorm(hidden, cond=None)
    shape = (*normalized.shape[:-1], -1, layer.self_attn.head_dim)
    q, k, v = [proj(normalized).view(shape).transpose(1, 2) for proj in
               (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj)]
    dummy = q.new_zeros((q.shape[0], q.shape[2], q.shape[-1]))
    cos, sin = wrapper.paligemma.model.language_model.rotary_emb(dummy, positions)
    q, k = modeling_gemma.apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
    if past is not None:
        k, v = torch.cat([past[0], k], 2), torch.cat([past[1], v], 2)
    attended, _ = modeling_gemma.eager_attention_forward(layer.self_attn, q,
        torch.cat([pk, k], 2), torch.cat([pv, v], 2), attention, layer.self_attn.scaling)
    attended = attended.reshape(hidden.shape[0], hidden.shape[1], -1).to(layer.self_attn.o_proj.weight.dtype)
    out = modeling_gemma._gated_residual(hidden, layer.self_attn.o_proj(attended), gate)
    residual = out
    out, gate = layer.post_attention_layernorm(out, cond=None)
    out = layer.mlp(out.to(layer.mlp.up_proj.weight.dtype))
    return modeling_gemma._gated_residual(residual, out, gate), (k, v)


def cache_pairs(cache, depth):
    # Indexing is supported by this project's pinned transformers cache. Keep
    # its tensors read-only: native text owns separate K/V tensors, never update
    # the observation cache that the action expert will use.
    return tuple((cache[i][0], cache[i][1]) for i in range(depth))


class NativeSubtaskModel(nn.Module):
    def __init__(self, base, *, max_tokens=16):
        super().__init__()
        if not base.pi05:
            raise ValueError("N1 requires pi05")
        self.base = base
        self.max_tokens = max_tokens
        self.codec = SubtaskTextCodec(base.config.max_token_len, max_tokens, 14)
        self.cue_ids = tuple(self.codec.processor.encode(TEXT_CUE))
        if not self.cue_ids:
            raise ValueError("Missing native text task cue")
        self.stage, self.backbone_mode = "native_subtask", "action_stop"
        for p in base.parameters():
            p.requires_grad_(False)
        for p in self.backbone_parameters() + self.action_parameters():
            p.requires_grad_(True)
        base.gradient_checkpointing_disable()
        self.verify_tied_head()

    def verify_tied_head(self):
        b = self.base.paligemma_with_expert.paligemma
        if b.lm_head.weight is not b.language_model.embed_tokens.weight:
            raise ValueError("Native PaliGemma vocabulary must retain its pretrained tied output head")
        if b.lm_head.weight.shape != b.language_model.embed_tokens.weight.shape:
            raise ValueError("Native vocabulary shape mismatch")

    def backbone_parameters(self):
        return list(self.base.paligemma_with_expert.paligemma.parameters())

    def action_parameters(self):
        modules = [self.base.paligemma_with_expert.gemma_expert.model, self.base.action_in_proj,
                   self.base.action_out_proj, self.base.time_mlp_in, self.base.time_mlp_out]
        return [p for module in modules for p in module.parameters()]

    def train(self, mode=True):
        super().train(mode)
        # Explicit non-reentrant checkpoints below; never let HF silently
        # disable KV caching or alter observation behavior when training.
        self.base.paligemma_with_expert.paligemma.eval()
        return self

    def prepare_features(self, observation, prompts):
        images, image_masks, _, _, state = self.base._preprocess_observation(observation, train=False)
        wrapper = self.base.paligemma_with_expert
        features = tuple(checkpoint(wrapper.embed_image, x, use_reentrant=False) if torch.is_grad_enabled()
                         else wrapper.embed_image(x) for x in images)
        ids, valid = self.codec.prompts(prompts, state.detach().cpu().numpy())
        ids, valid = torch.as_tensor(ids, device=state.device), torch.as_tensor(valid, device=state.device)
        language = wrapper.embed_language_tokens(ids)
        prefix = torch.cat([*features, language * math.sqrt(language.shape[-1])], 1)
        mask = torch.cat([*(m[:, None].expand(f.shape[:2]) for m, f in zip(image_masks, features)), valid], 1)
        return state, prefix, mask

    def teacher_inputs(self, target_ids, target_mask):
        if target_ids.shape != target_mask.shape or target_ids.shape[1] != self.max_tokens:
            raise ValueError("Wrong native subtask target layout")
        cue = torch.tensor(self.cue_ids, device=target_ids.device).expand(target_ids.shape[0], -1)
        previous = torch.where(target_mask[:, :-1], target_ids[:, :-1], self.codec.pad_id)
        return torch.cat([cue, previous], 1), torch.cat([torch.ones_like(cue, dtype=torch.bool),
                                                       target_mask[:, :-1].bool()], 1)

    def embed_text(self, ids):
        x = self.base.paligemma_with_expert.embed_language_tokens(ids)
        return x * math.sqrt(x.shape[-1])

    def text_logits(self, hidden):
        b = self.base.paligemma_with_expert.paligemma
        hidden, _ = b.language_model.norm(hidden, cond=None)
        return b.lm_head(hidden.to(b.lm_head.weight.dtype))

    def ce(self, logits, target_ids, target_mask):
        targets = torch.where(target_mask, target_ids, self.codec.pad_id)
        loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), reduction="none").view_as(targets)
        counts = target_mask.sum(1)
        return ((loss * target_mask).sum(1) / counts.clamp_min(1)).sum() / (counts > 0).sum().clamp_min(1)

    def joint_outputs(self, observation, prompts, actions, target_ids, target_mask, *, noise, time):
        state, prefix, mask = self.prepare_features(observation, prompts)
        wrapper = self.base.paligemma_with_expert
        noisy = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        suffix, suffix_mask, suffix_ar, cond = self.base.embed_suffix(state, noisy, time)
        all_mask = torch.cat([mask, suffix_mask], 1)
        ar = torch.cat([torch.zeros_like(mask), suffix_ar], 1)
        attention = self.base._prepare_attention_masks_4d(make_att_2d_masks(all_mask, ar))
        positions = all_mask.cumsum(1) - 1
        ids, valid = self.teacher_inputs(target_ids, target_mask)
        text = self.embed_text(ids)
        text_positions = mask.sum(1)[:, None] + torch.arange(ids.shape[1], device=ids.device)[None]
        text_mask = self.base._prepare_attention_masks_4d(text_attention_mask(mask, valid))
        dtype = wrapper.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        prefix, suffix, text = prefix.to(dtype), suffix.to(dtype), text.to(dtype)
        for index in range(len(wrapper.paligemma.language_model.layers)):
            def layer(p, a, t, i=index):
                pk, pv = prefix_kv(wrapper, i, p, positions[:, :p.shape[1]])
                nt, _ = native_text_layer(wrapper, i, t, pk, pv, text_positions, text_mask)
                nprefix, naction = parallel_layer(wrapper, i, p, a, attention, positions, cond)
                return nprefix, naction, nt
            prefix, suffix, text = (checkpoint(layer, prefix, suffix, text, use_reentrant=False)
                                    if torch.is_grad_enabled() else layer(prefix, suffix, text))
        suffix, _ = wrapper.gemma_expert.model.norm(suffix, cond=cond)
        velocity = self.base.action_out_proj(suffix[:, -self.base.config.action_horizon:].float())
        logits = self.text_logits(text[:, len(self.cue_ids) - 1:])
        return logits, velocity

    def forward(self, batch, *, noise=None, time=None):
        noise = self.base.sample_noise(batch.actions.shape, batch.actions.device) if noise is None else noise
        time = self.base.sample_time(batch.actions.shape[0], batch.actions.device) if time is None else time
        logits, velocity = self.joint_outputs(batch.observation, batch.global_prompts, batch.actions,
            batch.target_ids, batch.target_mask, noise=noise, time=time)
        return dict(loss_subtask=self.ce(logits, batch.target_ids, batch.target_mask),
                    loss_action=(velocity.float() - (noise - batch.actions)).square().mean(),
                    generated_count=0, invalid_generation_count=0, empty_condition_count=0)

    @torch.no_grad()
    def prepare_context(self, observation, global_prompts, *, timings=None):
        state, prefix, mask = self.prepare_features(observation, global_prompts)
        wrapper = self.base.paligemma_with_expert
        attention = self.base._prepare_attention_masks_4d(make_att_2d_masks(mask, torch.zeros_like(mask)))
        prefix = prefix.to(wrapper.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype)
        wrapper.paligemma.language_model.config._attn_implementation = "eager"
        _, cache = wrapper(attention_mask=attention, position_ids=mask.cumsum(1) - 1,
                           inputs_embeds=[prefix, None], use_cache=True)
        return SimpleNamespace(state=state, mask=mask, cache=cache,
            pairs=cache_pairs(cache, len(wrapper.paligemma.language_model.layers)), prefix=prefix)

    def logits_cached(self, context, ids, valid=None, past=None, *, output_start=0):
        valid = torch.ones_like(ids, dtype=torch.bool) if valid is None else valid
        past_len = 0 if past is None else past[0][0].shape[2]
        positions = context.mask.sum(1)[:, None] + past_len + torch.arange(ids.shape[1], device=ids.device)[None]
        attention = self.base._prepare_attention_masks_4d(text_attention_mask(context.mask, valid, past_len))
        wrapper = self.base.paligemma_with_expert
        hidden = self.embed_text(ids).to(wrapper.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype)
        caches = []
        for i, (pk, pv) in enumerate(context.pairs):
            hidden, kv = native_text_layer(wrapper, i, hidden, pk, pv, positions, attention,
                                          None if past is None else past[i])
            caches.append(kv)
        # Select supervised rows BEFORE final norm/vocabulary projection, just
        # as joint_outputs does. Projecting the cue too changes GEMM shape and
        # can round BF16 differently even for bit-identical hidden features.
        return self.text_logits(hidden[:, output_start:]), tuple(caches)

    def teacher_logits_cached(self, context, target_ids, target_mask):
        ids, valid = self.teacher_inputs(target_ids, target_mask)
        logits, _ = self.logits_cached(context, ids, valid, output_start=len(self.cue_ids) - 1)
        return logits

    def subtask_loss(self, context, target_ids, target_mask):
        logits = self.teacher_logits_cached(context, target_ids, target_mask)
        return self.ce(logits, target_ids, target_mask)

    @torch.no_grad()
    def generate_subtask(self, context):
        b, device = context.state.shape[0], context.state.device
        ids = torch.tensor(self.cue_ids, device=device).expand(b, -1)
        output = torch.full((b, self.max_tokens), self.codec.pad_id, device=device, dtype=torch.long)
        valid = torch.zeros_like(output, dtype=torch.bool)
        ended = torch.zeros(b, device=device, dtype=torch.bool)
        scores = torch.zeros(b, device=device)
        past = None
        for position in range(self.max_tokens):
            logits, past = self.logits_cached(context, ids, past=past)
            logits = logits[:, -1].float()
            logits[:, [self.codec.bos_id, self.codec.pad_id]] = -torch.inf
            token = logits.argmax(-1)
            active = ~ended
            output[:, position] = torch.where(active, token, self.codec.pad_id)
            valid[:, position] = active
            scores += torch.where(active, logits.log_softmax(-1).gather(1, token[:, None]).squeeze(1), 0)
            ended |= active & token.eq(self.codec.eos_id)
            if ended.all():
                break
            ids = token[:, None]
        generation = NativeGeneration(output, valid, ended, scores / valid.sum(1).clamp_min(1))
        texts, statuses = self.codec.decode(generation)
        return texts, statuses, generation

    def action_prefix(self, context, subtasks=None):
        return SimpleNamespace(mask=context.mask, cache=context.cache)

    @torch.no_grad()
    def sample_actions_from_prefix(self, context, prefix, *, noise=None, num_steps=10):
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        if noise is None:
            noise = self.base.sample_noise((context.state.shape[0], self.base.config.action_horizon,
                                           self.base.config.action_dim), context.state.device)
        dt = torch.tensor(-1. / num_steps, device=context.state.device, dtype=torch.float32)
        time = torch.tensor(1., device=context.state.device, dtype=torch.float32)
        actions = noise
        while time >= -dt / 2:
            velocity = self.base.denoise_step(context.state, prefix.mask, prefix.cache, actions,
                                             time.expand(context.state.shape[0]))
            actions = actions + dt * velocity
            time += dt
        return actions

    @torch.no_grad()
    def infer(self, observation, global_prompts, *, noise=None, num_steps=10):
        previous = self.training
        self.eval()
        try:
            context = self.prepare_context(observation, global_prompts)
            texts, statuses, generation = self.generate_subtask(context)
            actions = self.sample_actions_from_prefix(context, self.action_prefix(context), noise=noise, num_steps=num_steps)
            return dict(actions=actions, subtasks=texts, subtask_status=statuses,
                        subtask_sequence_score=generation.mean_log_probability)
        finally:
            self.train(previous)

    @torch.no_grad()
    def deployment_state(self):
        result, seen = {}, set()
        for name, value in sorted(self.state_dict().items()):
            identity = (value.device, value.data_ptr(), value.numel(), value.dtype)
            if identity not in seen:
                result[name] = value.detach().cpu().contiguous()
                seen.add(identity)
        return result
