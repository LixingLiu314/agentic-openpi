"""Autoregressive subtask decoder with a frozen, externally owned VLM vocabulary.

The embedding table is passed to forward/generate instead of registered as a
second parameter. Both it and prefix memory are detached at the branch boundary;
the decoder's output projection still receives gradients through the tied head.
"""

import dataclasses

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812


@dataclasses.dataclass(frozen=True)
class SubtaskDecoderConfig:
    memory_dim: int = 2048
    embedding_dim: int = 2048
    width: int = 512
    heads: int = 8
    layers: int = 4
    mlp_dim: int = 2048
    max_tokens: int = 16
    dropout: float = 0.0
    bos_id: int = 2
    eos_id: int = 1
    pad_id: int = 0


@dataclasses.dataclass
class SubtaskGeneration:
    token_ids: torch.Tensor
    token_mask: torch.Tensor
    ended: torch.Tensor
    mean_log_probability: torch.Tensor


class SubtaskDecoder(nn.Module):
    def __init__(self, config: SubtaskDecoderConfig | None = None):
        super().__init__()
        config = config or SubtaskDecoderConfig()
        if config.width % config.heads or min(config.layers, config.max_tokens) < 1:
            raise ValueError("Invalid subtask decoder dimensions")
        if len({config.bos_id, config.eos_id, config.pad_id}) != 3:
            raise ValueError("BOS, EOS and PAD must be distinct")
        self.config = config
        self.memory_projection = nn.Linear(config.memory_dim, config.width)
        self.input_projection = nn.Linear(config.embedding_dim, config.width)
        self.position_embedding = nn.Parameter(torch.empty(config.max_tokens, config.width))
        nn.init.normal_(self.position_embedding, std=0.02)
        # Instantiate layers independently instead of cloning identical initial weights.
        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    config.width,
                    config.heads,
                    config.mlp_dim,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config.layers)
            ]
        )
        self.output_norm = nn.LayerNorm(config.width)
        self.output_projection = nn.Linear(config.width, config.embedding_dim, bias=False)

    def forward(self, input_ids, memory, memory_mask, embedding_weight):
        if input_ids.ndim != 2 or not 1 <= input_ids.shape[1] <= self.config.max_tokens:
            raise ValueError("Decoder input must be [B, T], with 1 <= T <= max_tokens")
        if memory.ndim != 3 or memory.shape[:2] != memory_mask.shape or memory.shape[0] != input_ids.shape[0]:
            raise ValueError("Prefix memory/mask dimensions do not match decoder batch")
        if not memory_mask.bool().any(dim=1).all():
            raise ValueError("Each observation needs at least one valid prefix token")
        if embedding_weight.ndim != 2 or embedding_weight.shape[1] != self.config.embedding_dim:
            raise ValueError("Unexpected VLM embedding dimensions")
        dtype = self.memory_projection.weight.dtype
        vocabulary = embedding_weight.detach()
        memory = self.memory_projection(memory.detach().to(dtype))
        hidden = self.input_projection(F.embedding(input_ids, vocabulary).to(dtype))
        hidden = hidden + self.position_embedding[None, : input_ids.shape[1]]
        causal_mask = torch.ones(
            input_ids.shape[1], input_ids.shape[1], dtype=torch.bool, device=input_ids.device
        ).triu(1)
        for layer in self.layers:
            hidden = layer(
                hidden,
                memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=input_ids.eq(self.config.pad_id),
                memory_key_padding_mask=~memory_mask.bool(),
            )
        projected = self.output_projection(self.output_norm(hidden))
        # Do not put this matmul in no_grad: its gradient must reach output_projection.
        return F.linear(projected.to(vocabulary.dtype), vocabulary)

    def compute_loss(self, target_ids, target_mask, memory, memory_mask, embedding_weight):
        """Targets exclude BOS and include EOS; missing labels have an all-false mask."""
        if target_ids.shape != target_mask.shape or target_ids.ndim != 2:
            raise ValueError("Expected matching [B, T] targets and masks")
        inputs = torch.full_like(target_ids, self.config.pad_id)
        inputs[:, 0] = self.config.bos_id
        inputs[:, 1:] = torch.where(target_mask[:, :-1], target_ids[:, :-1], self.config.pad_id)
        logits = self(inputs, memory, memory_mask, embedding_weight)
        safe_targets = torch.where(target_mask, target_ids, self.config.pad_id)
        token_loss = F.cross_entropy(logits.float().flatten(0, 1), safe_targets.flatten(), reduction="none").view_as(
            target_ids
        )
        counts = target_mask.sum(dim=1)
        per_example = (token_loss * target_mask).sum(dim=1) / counts.clamp_min(1)
        return per_example.sum() / (counts > 0).sum().clamp_min(1)

    @torch.no_grad()
    def generate(self, memory, memory_mask, embedding_weight):
        """Greedy decoding from BOS only; finished samples pad independently."""
        previous_training = self.training
        self.eval()
        batch = memory.shape[0]
        ids = torch.full((batch, 1), self.config.bos_id, dtype=torch.long, device=memory.device)
        output = torch.full((batch, self.config.max_tokens), self.config.pad_id, dtype=torch.long, device=memory.device)
        mask = torch.zeros_like(output, dtype=torch.bool)
        ended = torch.zeros(batch, dtype=torch.bool, device=memory.device)
        scores = torch.zeros(batch, dtype=torch.float32, device=memory.device)
        try:
            for position in range(self.config.max_tokens):
                logits = self(ids, memory, memory_mask, embedding_weight)[:, -1].float()
                logits[:, [self.config.bos_id, self.config.pad_id]] = -torch.inf
                next_ids = logits.argmax(dim=-1)
                active = ~ended
                output[:, position] = torch.where(active, next_ids, self.config.pad_id)
                mask[:, position] = active
                scores += torch.where(active, logits.log_softmax(dim=-1).gather(1, next_ids[:, None]).squeeze(1), 0)
                ended |= active & next_ids.eq(self.config.eos_id)
                if ended.all():
                    break
                ids = torch.cat([ids, output[:, position : position + 1]], dim=1)
            return SubtaskGeneration(output, mask, ended, scores / mask.sum(dim=1).clamp_min(1))
        finally:
            self.train(previous_training)
