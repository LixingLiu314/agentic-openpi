"""R1 boundary sampling and timestamp-based label-proxy event metrics.

Physical readiness is deliberately never inferred from annotation boundaries.
"""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def text_loss_per_example(decoder, target_ids, target_mask, memory, memory_mask, embedding_weight):
    if target_ids.shape != target_mask.shape or target_ids.ndim != 2:
        raise ValueError("Expected matching [B,T] targets and masks")
    inputs = torch.full_like(target_ids, decoder.config.pad_id)
    inputs[:, 0] = decoder.config.bos_id
    inputs[:, 1:] = torch.where(target_mask[:, :-1], target_ids[:, :-1], decoder.config.pad_id)
    logits = decoder(inputs, memory, memory_mask, embedding_weight)
    targets = torch.where(target_mask, target_ids, decoder.config.pad_id)
    token_loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), reduction="none").view_as(targets)
    counts = target_mask.sum(1)
    return (token_loss * target_mask).sum(1) / counts.clamp_min(1), counts > 0


def weighted_numerator(losses, valid, weights):
    if losses.shape != valid.shape or losses.shape != weights.shape:
        raise ValueError("Expected matching sample losses, validity and weights")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Weights must be finite and nonnegative")
    effective = weights * valid.to(weights.dtype)
    return (losses * effective).sum(), effective.sum()


class BoundarySampler:
    """Stateless per-update global draws, then rank/microbatch slicing.

    Boundary half: task -> transition -> episode -> pre/post -> frame.
    Overlapping windows are assigned to the nearest event during D0.
    """
    def __init__(self, rows, *, batch_size, accumulation, steps, start=0, seed=42, rank=0, world_size=1):
        self.options = batch_size, accumulation, steps, start, seed, rank, world_size
        self.stable = np.array([r["index"] for r in rows if r["boundary_event"] is None], dtype=np.int64)
        self.tree = {}
        self.by_index = {r["index"]: r for r in rows}
        for row in rows:
            if row["boundary_event"] is not None:
                self.tree.setdefault(row["task"], {}).setdefault(row["transition"], {}).setdefault(
                    row["episode"], {}).setdefault(row["boundary_side"], []).append(row["index"])
        if not len(self.stable) or not self.tree or (batch_size * accumulation * world_size) % 2:
            raise ValueError("Need stable/boundary samples and an even global batch")

    def global_indices(self, step):
        batch_size, accumulation, _, _, seed, _, world = self.options
        rng = np.random.default_rng(np.random.SeedSequence([seed, step, 42]))
        count = batch_size * accumulation * world
        indices = rng.choice(self.stable, count // 2).tolist()
        for _ in range(count // 2):
            node = self.tree
            for _ in range(4):
                keys = sorted(node)
                node = node[keys[int(rng.integers(len(keys)))]]
            indices.append(int(rng.choice(node)))
        rng.shuffle(indices)
        return np.asarray(indices).reshape(accumulation, world, batch_size)

    def __iter__(self):
        _, _, steps, start, _, rank, _ = self.options
        for step in range(start, steps):
            for micro in self.global_indices(step):
                yield micro[rank].tolist()

    def __len__(self):
        _, accumulation, steps, start, _, _, _ = self.options
        return (steps - start) * accumulation


def sampled_rows(rows, period, phase=0.0):
    """Causal floor sampling: never select a future frame."""
    if period <= 0 or phase < 0:
        raise ValueError("Invalid period/phase")
    groups = defaultdict(list)
    for row in rows:
        groups[row["episode"]].append(row)
    result = []
    for group in groups.values():
        group.sort(key=lambda r: r["timestamp"])
        times = np.asarray([r["timestamp"] for r in group])
        selected = np.searchsorted(times, np.arange(times[0] + phase, times[-1] + 1e-8, period), side="right") - 1
        for i in np.unique(selected[selected >= 0]):
            result.append(group[int(i)])
    return result


def event_metrics(predictions, boundaries, *, hold_seconds=0.3):
    """Label-proxy times only; all events retained, failed events never imputed.

    First stable new-label run inside a fixed neighboring-boundary midpoint cell.
    At least two actual observations must span hold_seconds. Legitimately short
    successor stages are separately reported and are not automatically failures.
    """
    groups = defaultdict(list)
    for row in predictions:
        groups[row["episode"]].append(row)
    for group in groups.values():
        group.sort(key=lambda r: r["timestamp"])
    events = []
    for event in boundaries:
        group = groups.get(event["episode"], [])
        runs = []
        for i, row in enumerate(group):
            if i == 0 or row["prediction"] != group[i - 1]["prediction"]:
                runs.append([i, i])
            else:
                runs[-1][1] = i
        chosen = None
        transient = 0
        eligible = []
        for begin, end in runs:
            first = group[begin]
            if first["prediction"] != event["new"] or not event["cell_start"] <= first["timestamp"] < event["cell_end"]:
                continue
            eligible.append((begin, end))
            confirmations = [i for i in range(begin + 1, end + 1)
                             if group[i]["timestamp"] - first["timestamp"] >= hold_seconds - 1e-6]
            if confirmations and chosen is None:
                chosen = (begin, confirmations[0])
            elif not confirmations:
                transient += 1
        short = event["new_end_time"] - event["t_label"] < hold_seconds
        status = "stable" if chosen else ("short_stage" if short else "missed_or_unconfirmed")
        switch = group[chosen[0]]["timestamp"] if chosen else None
        error = switch - event["t_label"] if chosen else None
        row = dict(event_id=event["event_id"], episode=event["episode"], task=event["task"],
                   transition=event["transition"], status=status, t_label=event["t_label"],
                   t_pred_obs=switch, confirm_observation_time=group[chosen[1]]["timestamp"] if chosen else None,
                   signed_label_error_seconds=error, transient_correct_runs=transient,
                   previous_prediction=group[chosen[0]-1]["prediction"] if chosen and chosen[0] else None,
                   no_observation_in_successor=not any(event["t_label"] <= r["timestamp"] < event["new_end_time"] for r in group),
                   readiness_status="unreviewed", ready_delay_seconds=None, command_time=None)
        events.append(row)
    def summary(items):
        errors = [r["signed_label_error_seconds"] for r in items if r["status"] == "stable"]
        count = len(items)
        late = [max(0, e) for e in errors]
        early = [max(0, -e) for e in errors]
        return {"events": count, "stable": len(errors), "failed_or_unconfirmed": sum(r["status"] == "missed_or_unconfirmed" for r in items),
                "short_stage": sum(r["status"] == "short_stage" for r in items),
                "stable_rate": len(errors) / max(1, count),
                "within_2s_rate": sum(e <= 2 for e in errors) / max(1, count),
                "late_over_2s_or_failed": sum(e > 2 for e in errors) + sum(r["status"] == "missed_or_unconfirmed" for r in items),
                "early_over_100ms": sum(e < -0.1 for e in errors),
                "late_p50_p95_seconds": np.percentile(late, [50,95]).tolist() if late else None,
                "early_p50_p95_seconds": np.percentile(early, [50,95]).tolist() if early else None,
                "transient_correct_runs": sum(r["transient_correct_runs"] for r in items),
                "no_observation_in_successor": sum(r["no_observation_in_successor"] for r in items)}
    result = summary(events)
    result.update(definition="label-proxy first stable arrival; observed timestamps; no physical readiness claims",
                  quantile_population="stable events only; zero late/early magnitudes included; failures separate",
                  hold_seconds=hold_seconds, events_detail=events)
    result["by_task_transition"] = {key: summary([r for r in events if r["task"]+" | "+r["transition"] == key])
                                    for key in sorted({r["task"]+" | "+r["transition"] for r in events})}
    return result


def evaluation_report(rows, boundaries):
    def accuracy(items):
        return sum(r["label"] == r["prediction"] for r in items) / max(1,len(items))
    return {"frames":len(rows), "exact_match":accuracy(rows),
            "stable_exact_match":accuracy([r for r in rows if r["boundary_event"] is None]),
            "boundary_exact_match":accuracy([r for r in rows if r["boundary_event"] is not None]),
            "dense":event_metrics(rows,boundaries),
            "low_rate":{str(phase):event_metrics(sampled_rows(rows,0.764,phase),boundaries) for phase in [0.0,0.255,0.509]},
            "physical_readiness_evaluated":False}
