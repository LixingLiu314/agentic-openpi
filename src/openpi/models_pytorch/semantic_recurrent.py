"""Training-only semantic supervision; unchanged recurrent S and native pi05 A."""
import dataclasses
import difflib
import math

import torch
import torch.nn.functional as F

from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.models_pytorch.recurrent_subtask import RecurrentSubtaskModel

VARIANT = "official_pi05_recurrent_semantic_v1"
SCHEMA_VERSION = 8
DISPLAY_SET = "official-pi05-semantic-pair-s42-v1"
EXPERIMENTS = ("semantic_s", "semantic_s_actionrank")


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a):
        nxt = [i + 1]
        for j, y in enumerate(b):
            nxt.append(min(nxt[-1] + 1, row[j + 1] + 1, row[j] + (x != y)))
        row = nxt
    return row[-1]


def semantic_rules(codec, vocabulary):
    """Mine only existing train-label alternatives; never infer task/arm rules."""
    vocabulary = sorted(set(vocabulary))
    encoded = {s: codec.processor.encode(s) for s in vocabulary}
    weights, negatives, audit = {}, {}, {}
    for text in vocabulary:
        tokens = encoded[text]
        distances = {s: edit_distance(tokens, encoded[s]) for s in vocabulary if s != text}
        nearest = min(distances.values()) if distances else 0
        # A local alternative differs by at most one more token than the closest.
        neighbors = sorted(s for s, d in distances.items() if d <= nearest + 1)
        important = set()
        for neighbor in neighbors:
            for op, a, b, _, _ in difflib.SequenceMatcher(a=tokens, b=encoded[neighbor], autojunk=False).get_opcodes():
                if op != "equal":
                    important.update(range(a, b))
                    if a == b and tokens:
                        important.add(min(a, len(tokens) - 1))
        ids, valid = codec.targets([text])
        row = [4.0 if i in important else 1.0 for i in range(codec.max_subtask_tokens)]
        weights[tuple(ids[0])] = row
        negatives[text] = sorted(s for s, d in distances.items() if d == nearest)
        audit[text] = dict(important_token_positions=sorted(important),
                           important_pieces=[codec.processor.id_to_piece(tokens[i]) for i in sorted(important)],
                           neighbors=neighbors, negative_labels=negatives[text], weights=row[:int(valid.sum())])
    return weights, negatives, audit


def eligible_pairs(labels, generated, statuses, dropped, negatives, offset=0, limit=4, stable=None):
    """Generated text remains the only positive action condition, with exact-match gating."""
    selected = []
    for n in range(len(labels)):
        i = (n + offset) % len(labels)
        if (stable is None or stable[i]) and not dropped[i] and statuses[i] == "ok" and generated[i] == labels[i] and negatives.get(labels[i]):
            choices = negatives[labels[i]]
            selected.append((i, choices[(offset + i) % len(choices)]))
            if len(selected) == limit:
                break
    return selected


def stable_training_frames(table, horizon=50, past=7):
    """Training-only eligibility sidecar. No boundary/label is a policy input."""
    episodes, frames, labels = (table[k] for k in ("episode_index", "frame_index", "subtask"))
    by_episode = {}
    for ep, frame, label in zip(episodes, frames, labels, strict=True):
        by_episode.setdefault(int(ep), []).append((int(frame), label))
    stable = set()
    for ep, values in by_episode.items():
        values.sort()
        for i, (frame, label) in enumerate(values):
            if all(other == label for _, other in values[max(0, i-past):i+horizon]):
                stable.add((ep, frame))
    return stable


def select_context(context, indices):
    return dataclasses.replace(context, state=context.state[indices],
        global_prompts=tuple(context.global_prompts[i] for i in indices),
        image_features=tuple(x[indices] for x in context.image_features),
        image_masks=tuple(x[indices] for x in context.image_masks),
        memory=context.memory[indices], memory_mask=context.memory_mask[indices])


def ranking_loss(positive, negative, margin=.01):
    # Normalized native14 flow errors, same observation/time/noise/recorded action.
    return F.relu(margin + positive - negative).mean()


class SemanticRecurrentModel(RecurrentSubtaskModel):
    def configure_semantics(self, vocabulary, experiment):
        if experiment not in EXPERIMENTS:
            raise ValueError(experiment)
        self.experiment = experiment
        self.semantic_weights, self.negative_labels, self.semantic_audit = semantic_rules(self.codec, vocabulary)

    def semantic_loss(self, ids, valid, projected, mask):
        decoder = self.decoder
        inputs = torch.full_like(ids, decoder.config.pad_id)
        inputs[:, 0] = decoder.config.bos_id
        inputs[:, 1:] = torch.where(valid[:, :-1], ids[:, :-1], decoder.config.pad_id)
        logits = decoder.logits_projected(inputs, projected, mask, self.embedding_weight)
        targets = torch.where(valid, ids, decoder.config.pad_id)
        losses = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), reduction="none").view_as(ids)
        weights = torch.tensor([self.semantic_weights[tuple(row)] for row in ids.detach().cpu().tolist()],
                               device=ids.device) * valid
        weighted = ((losses * weights).sum(1) / weights.sum(1).clamp_min(1)).mean()
        ordinary = ((losses * valid).sum(1) / valid.sum(1).clamp_min(1)).mean()
        return weighted, ordinary.detach()

    def action_errors(self, context, subtasks, actions, *, noise, time):
        """The existing joint B/A forward, preserving elementwise errors for ranking."""
        if self.backbone_mode != "limited":
            raise ValueError("This experiment supports limited B only")
        model = self.base.paligemma_with_expert
        noisy = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        tokens, token_mask = self._tokens(context.global_prompts, context.state, subtasks)
        features, image_masks = context.image_features, context.image_masks
        language = model.embed_language_tokens(tokens)
        prefix = torch.cat([*features, language * math.sqrt(language.shape[-1])], dim=1)
        prefix_mask = torch.cat([*(m[:, None].expand(f.shape[:2]) for f, m in zip(features, image_masks, strict=True)), token_mask], dim=1)
        suffix, suffix_mask, suffix_ar, cond = self.base.embed_suffix(context.state, noisy, time)
        dtype = model.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        masks = torch.cat([prefix_mask, suffix_mask], dim=1)
        ar = torch.cat([torch.zeros_like(prefix_mask), suffix_ar], dim=1)
        attention = self.base._prepare_attention_masks_4d(make_att_2d_masks(masks, ar))
        (_, output), _ = model(attention_mask=attention, position_ids=masks.cumsum(dim=1) - 1,
            inputs_embeds=[prefix.to(dtype), suffix.to(dtype)], past_key_values=None,
            use_cache=False, adarms_cond=[None, cond])
        velocity = self.base.action_out_proj(output[:, -self.base.config.action_horizon:].float())
        return (velocity - (noise - actions)).square()

    def forward(self, batch, *, carry=None, drop_condition=None, rank_offset=0, rank_stable_mask=None):
        context = self.prepare_context(batch.observation, batch.global_prompts)
        projected, mask, next_carry = self.decoder.compose_sequence(context.memory, context.memory_mask,
            carry, batch.reset_mask, unroll=self.unroll)
        semantic, ordinary = self.semantic_loss(batch.target_ids, batch.target_mask, projected, mask)
        generation = self.decoder.generate_projected(projected.detach(), mask, self.embedding_weight)
        texts, statuses = self.codec.decode(generation)
        dropped = [False] * len(texts) if drop_condition is None else drop_condition
        conditions = ["" if drop else text for text, drop in zip(texts, dropped, strict=True)]
        noise = self.base.sample_noise(batch.actions.shape, batch.actions.device)
        time = self.base.sample_time(batch.actions.shape[0], batch.actions.device)
        errors = self.action_errors(context, conditions, batch.actions, noise=noise, time=time)
        action = errors.mean()
        if self.experiment == "semantic_s_actionrank" and rank_stable_mask is None:
            raise ValueError("Action ranking requires the training-only boundary exclusion mask")
        pairs = eligible_pairs(batch.labels, texts, statuses, dropped, self.negative_labels, rank_offset, stable=rank_stable_mask)
        if self.experiment != "semantic_s_actionrank":
            pairs = []
        auxiliary = action.new_zeros(())
        if pairs:
            indices = [i for i, _ in pairs]
            negative = self.action_errors(select_context(context, indices), [s for _, s in pairs],
                batch.actions[indices], noise=noise[indices], time=time[indices])
            auxiliary = ranking_loss(errors[indices, :, :14].mean((1, 2)), negative[:, :, :14].mean((1, 2)))
        return dict(loss_subtask=semantic, loss_action=action + .1 * auxiliary,
            loss_action_main=action.detach(), loss_subtask_unweighted=ordinary,
            loss_action_rank=auxiliary.detach(), rank_pairs=len(pairs),
            carry=None if next_carry is None else next_carry.detach(), generated_count=len(texts),
            invalid_generation_count=sum(x != "ok" for x in statuses),
            empty_condition_count=sum(not x for x in conditions))
