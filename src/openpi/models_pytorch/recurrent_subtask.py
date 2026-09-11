"""One observation-driven recurrent state in S; A receives ordinary text only."""
import dataclasses

import torch
from torch import nn
import torch.nn.functional as F

from openpi.models_pytorch.official_backbone_gradient import OfficialBackboneGradientModel
from openpi.models_pytorch.subtask_decoder import SubtaskDecoder, SubtaskGeneration

VARIANT = "official_pi05_recurrent_s_v1"


class RecurrentSubtaskDecoder(SubtaskDecoder):
    def __init__(self, config=None, *, recurrent=False, memory_tokens=4):
        # Common decoder construction order is identical in both arms.
        super().__init__(config)
        self.recurrent = recurrent
        self.memory_tokens = memory_tokens
        if recurrent:
            w = self.config.width
            self.initial_memory = nn.Parameter(torch.randn(memory_tokens, w) * .02)
            self.memory_type = nn.Parameter(torch.randn(1, memory_tokens, w) * .02)
            self.query_norm = nn.LayerNorm(w)
            self.observation_norm = nn.LayerNorm(w)
            self.memory_attention = nn.MultiheadAttention(w, self.config.heads, batch_first=True)
            self.memory_update = nn.GRUCell(w, w)
            self.memory_norm = nn.LayerNorm(w)

    def compose_sequence(self, memory, mask, carry=None, resets=None, *, unroll=1):
        """Flattened time-major [unroll * streams, tokens, width]. No text targets."""
        projected = self.memory_projection(memory.detach().to(self.memory_projection.weight.dtype))
        if not self.recurrent:
            return projected, mask.bool(), None
        if memory.shape[0] % unroll:
            raise ValueError("Sequence batch is not divisible by unroll")
        streams = memory.shape[0] // unroll
        initial = self.initial_memory[None].expand(streams, -1, -1)
        if carry is None:
            carry = initial
        if carry.shape != initial.shape:
            raise ValueError("Recurrent state layout changed")
        if resets is None:
            resets = torch.zeros(memory.shape[0], dtype=torch.bool, device=memory.device)
        if resets.shape != (memory.shape[0],):
            raise ValueError("Invalid sequence reset mask")
        states = []
        for t in range(unroll):
            section = slice(t * streams, (t + 1) * streams)
            carry = torch.where(resets[section, None, None], initial, carry)
            current = self.observation_norm(projected[section])
            observed, _ = self.memory_attention(
                self.query_norm(carry), current, current,
                key_padding_mask=~mask[section].bool(), need_weights=False,
            )
            carry = self.memory_update(observed.flatten(0, 1), carry.flatten(0, 1)).view_as(initial)
            states.append(self.memory_norm(carry) + self.memory_type)
        return (
            torch.cat([projected, torch.cat(states, dim=0)], dim=1),
            torch.cat([mask.bool(), torch.ones((memory.shape[0], self.memory_tokens),
                                              dtype=torch.bool, device=memory.device)], dim=1),
            carry,
        )

    def logits_projected(self, input_ids, projected_memory, memory_mask, embedding_weight):
        c = self.config
        if input_ids.ndim != 2 or not 1 <= input_ids.shape[1] <= c.max_tokens:
            raise ValueError("Invalid autoregressive token layout")
        vocabulary = embedding_weight.detach()
        hidden = self.input_projection(F.embedding(input_ids, vocabulary).to(self.input_projection.weight.dtype))
        hidden = hidden + self.position_embedding[None, :input_ids.shape[1]]
        causal = torch.ones((input_ids.shape[1], input_ids.shape[1]),
                            dtype=torch.bool, device=input_ids.device).triu(1)
        for layer in self.layers:
            hidden = layer(hidden, projected_memory, tgt_mask=causal,
                           tgt_key_padding_mask=input_ids.eq(c.pad_id),
                           memory_key_padding_mask=~memory_mask.bool())
        output = self.output_projection(self.output_norm(hidden))
        return F.linear(output.to(vocabulary.dtype), vocabulary)

    def loss_projected(self, ids, valid, projected, mask, embedding):
        inputs = torch.full_like(ids, self.config.pad_id)
        inputs[:, 0] = self.config.bos_id
        inputs[:, 1:] = torch.where(valid[:, :-1], ids[:, :-1], self.config.pad_id)
        logits = self.logits_projected(inputs, projected, mask, embedding)
        targets = torch.where(valid, ids, self.config.pad_id)
        loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), reduction="none").view_as(ids)
        counts = valid.sum(1)
        return ((loss * valid).sum(1) / counts.clamp_min(1)).sum() / (counts > 0).sum().clamp_min(1)

    @torch.no_grad()
    def generate_projected(self, projected, memory_mask, embedding):
        # Project observation tokens once per request, not once per output word.
        previous = self.training
        self.eval()
        c = self.config
        b, device = projected.shape[0], projected.device
        ids = torch.full((b, 1), c.bos_id, dtype=torch.long, device=device)
        output = torch.full((b, c.max_tokens), c.pad_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(output, dtype=torch.bool)
        ended = torch.zeros(b, dtype=torch.bool, device=device)
        scores = torch.zeros(b, device=device, dtype=torch.float32)
        try:
            for position in range(c.max_tokens):
                logits = self.logits_projected(ids, projected, memory_mask, embedding)[:, -1].float()
                logits[:, [c.bos_id, c.pad_id]] = -torch.inf
                token = logits.argmax(-1)
                active = ~ended
                output[:, position] = torch.where(active, token, c.pad_id)
                mask[:, position] = active
                scores += torch.where(active, logits.log_softmax(-1).gather(1, token[:, None]).squeeze(1), 0)
                ended |= active & token.eq(c.eos_id)
                if ended.all():
                    break
                ids = torch.cat([ids, output[:, position:position + 1]], dim=1)
            return SubtaskGeneration(output, mask, ended, scores / mask.sum(1).clamp_min(1))
        finally:
            self.train(previous)


class RecurrentSubtaskModel(OfficialBackboneGradientModel):
    def __init__(self, base, decoder_config=None, *, recurrent=False, seed=42, unroll=4):
        super().__init__(base, decoder_config)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            decoder = RecurrentSubtaskDecoder(decoder_config, recurrent=recurrent)
        self.decoder = decoder.to(base.action_in_proj.weight.device)
        self.unroll = unroll
        self.enable_backbone("frozen")

    def forward(self, batch, *, carry=None, drop_condition=None):
        context = self.prepare_context(batch.observation, batch.global_prompts)
        projected, mask, next_carry = self.decoder.compose_sequence(
            context.memory, context.memory_mask, carry, batch.reset_mask, unroll=self.unroll,
        )
        ce = self.decoder.loss_projected(batch.target_ids, batch.target_mask, projected, mask, self.embedding_weight)
        generation = self.decoder.generate_projected(projected.detach(), mask, self.embedding_weight)
        texts, statuses = self.codec.decode(generation)
        drops = [False] * len(texts) if drop_condition is None else drop_condition
        conditions = ["" if drop else text for text, drop in zip(texts, drops, strict=True)]
        # Reuse the existing official joint-prefix/action flow implementation.
        action = self.action_loss_joint(batch.observation, context, conditions, batch.actions)
        return dict(loss_subtask=ce, loss_action=action,
                    carry=None if next_carry is None else next_carry.detach(),
                    generated_count=len(texts), invalid_generation_count=sum(x != "ok" for x in statuses),
                    empty_condition_count=sum(not x for x in conditions))

    def compose_context(self, context, carry=None, resets=None):
        return self.decoder.compose_sequence(context.memory, context.memory_mask, carry, resets, unroll=1)

    def subtask_loss(self, context, target_ids, target_mask):
        projected, mask, _ = self.compose_context(context)
        return self.decoder.loss_projected(target_ids, target_mask, projected, mask, self.embedding_weight)

    @torch.no_grad()
    def generate_with_memory(self, context, carry=None, resets=None):
        projected, mask, carry = self.compose_context(context, carry, resets)
        generation = self.decoder.generate_projected(projected, mask, self.embedding_weight)
        texts, statuses = self.codec.decode(generation)
        return texts, statuses, generation, carry

    @torch.no_grad()
    def generate_subtask(self, context):
        return self.generate_with_memory(context)[:3]
