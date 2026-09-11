"""Global-task-only action evaluation with the existing causal reach-arm protocol."""
import re
import numpy as np
import torch
import torch.distributed as dist

from openpi.training import hierarchy_training as training
from openpi.training.recurrent_sequence import episode_rows
from openpi.training.subtask_batch import collate_subtask


def phase_actor(text):
    text = " ".join((text or "").lower().split())
    match = re.fullmatch(r"(.+) with the (left|right) arm", text)
    return (match.group(1), match.group(2)) if match else (text, None)


def role_metrics(rows):
    reach = [r for r in rows if phase_actor(r["label"])[1] is not None]
    starts = [r for r in rows if r["frame"] == 0]
    result = {}
    for name, selected in [("reach", reach), ("initial", starts)]:
        result[name+"_actor_samples"] = len(selected)
        result[name+"_actor_accuracy"] = sum(phase_actor(r["prediction"])[1] == phase_actor(r["label"])[1]
                                                  for r in selected) / max(1,len(selected))
        result[name+"_phase_actor_accuracy"] = sum(phase_actor(r["prediction"]) == phase_actor(r["label"])
                                                        for r in selected) / max(1,len(selected))
    result["phase_only_exact_match"] = sum(phase_actor(r["prediction"])[0] == phase_actor(r["label"])[0]
                                            for r in rows) / max(1,len(rows))
    return result


@torch.no_grad()
def evaluate_recurrent(model, dataset, device, *, samples=128, draws=2, vocabulary=(),
                       interval=.7, reset_each=False, condition_controls=False, engineering=False):
    if condition_controls:
        raise ValueError('Parallel action has no subtask condition controls')
    model = training.unwrap(model)
    rank, world = training.distributed_context()
    rng, previous = training.random_state(), model.training
    model.eval()
    table = episode_rows(dataset.raw_dataset.hf_dataset)
    action_indices = set(np.unique(np.linspace(0, len(dataset) - 1, min(samples, len(dataset)), dtype=int)).tolist())
    # Every episode's initial observation and first second are represented.
    # Labels affect evaluation sampling only, never recurrent/model inputs.
    labels = dataset.labels_without_video()
    for indices, times in table.values():
        for delta in (0., .5, 1.):
            pos = min(int(np.searchsorted(times,times[0]+delta)),len(indices)-1)
            action_indices.add(int(indices[pos]))
        for pos, index in enumerate(indices):
            if phase_actor(labels[int(index)])[1] is not None and (pos == 0 or labels[int(indices[pos-1])] != labels[int(index)]):
                action_indices.add(int(index))
    episodes = sorted(table)
    if engineering:
        episodes = episodes[:min(len(episodes), world)]
    streams = []
    for ep in episodes[rank::world]:
        indices, times = table[ep]
        positions = np.unique(np.clip(np.searchsorted(times, np.arange(times[0], times[-1] + 1e-6, interval)),
                                      0, len(indices) - 1))
        chosen = set(indices[positions].tolist()) | (set(indices.tolist()) & action_indices) | {int(indices[-1])}
        ordered = sorted(chosen)
        if engineering:
            ordered = ordered[:2]
            action_indices.update(ordered)
        streams.append(dict(episode=ep, indices=ordered, offset=0, carry=None))
    totals = torch.zeros(9, dtype=torch.float64, device=device)
    rows = []
    try:
        while streams:
            selected = [item["indices"][item["offset"]] for item in streams]
            batch = collate_subtask([dataset[index] for index in selected]).to(device)
            context = model.prepare_context(batch.observation, batch.global_prompts)
            carry = None
            if model.decoder.recurrent and not reset_each:
                carry = torch.stack([model.decoder.initial_memory if item["carry"] is None else item["carry"]
                                     for item in streams])
            projected, mask, next_carry = model.compose_context(context, carry)
            ce = model.decoder.loss_projected(batch.target_ids, batch.target_mask, projected, mask, model.embedding_weight)
            texts, statuses = model.codec.decode(model.decoder.generate_projected(projected, mask, model.embedding_weight))
            totals[0] += ce * len(selected)
            totals[1] += len(selected)
            for j, (index, label, text, status) in enumerate(zip(selected, batch.labels, texts, statuses, strict=True)):
                rows.append(dict(index=index, episode=int(batch.episode_indices[j]), frame=int(batch.frame_indices[j]),
                                 label=label, prediction=text, status=status))
                if next_carry is not None:
                    streams[j]["carry"] = next_carry[j]
            # Evaluate A only on the common fixed action frames. Memory still saw
            # the entire preceding causal replay, never reference labels.
            keep = [j for j, index in enumerate(selected) if index in action_indices]
            if keep:
                for name, conditions, offset in [("generated", texts, 2)] + (
                    [("oracle", list(batch.labels), 5), ("empty", [""] * len(texts), 7)]
                    if condition_controls else []
                ):
                    prefix = model.action_prefix(context, conditions)
                    for draw in range(draws):
                        noises, times = [], []
                        for index in selected:
                            gen = torch.Generator().manual_seed(100000 + int(index) * draws + draw)
                            noises.append(torch.randn(batch.actions.shape[1:], generator=gen))
                            times.append(float(np.random.default_rng(200000 + int(index) * draws + draw).beta(1.5, 1) * .999 + .001))
                        noise = torch.stack(noises).to(device)
                        time = torch.tensor(times, device=device)
                        noisy = time[:, None, None] * noise + (1 - time[:, None, None]) * batch.actions
                        velocity = model.base.denoise_step(context.state, prefix.mask, prefix.cache, noisy, time)
                        error = (velocity.float() - (noise - batch.actions)).square()[keep]
                        totals[offset] += error[..., :14].mean((1, 2)).sum()
                        totals[offset + 1] += len(keep)
                        if name == "generated":
                            totals[4] += error.mean((1, 2)).sum()
            for item in streams:
                item["offset"] += 1
            streams = [item for item in streams if item["offset"] < len(item["indices"])]
        if world > 1:
            dist.all_reduce(totals)
            gathered = [None] * world
            dist.all_gather_object(gathered, rows)
            rows = [row for part in gathered for row in part]
        rows.sort(key=lambda row: (row["episode"], row["frame"]))
        metrics = training.text_metrics(rows, vocabulary)
        metrics.update(role_metrics(rows))
        metrics.update(loss_subtask=float(totals[0] / totals[1].clamp_min(1)),
                       flow_global_only_native14_normalized=float(totals[2] / totals[3].clamp_min(1)),
                       flow_global_only_all32=float(totals[4] / totals[3].clamp_min(1)),
                       sequence_frames=len(rows), action_frame_draws=int(totals[3]),
                       reset_each=reset_each, causal_interval_seconds=interval)
        metrics["action_frame_pool"] = len(action_indices)
        if condition_controls:
            metrics.update(flow_oracle_native14_normalized=float(totals[5] / totals[6].clamp_min(1)),
                           flow_empty_native14_normalized=float(totals[7] / totals[8].clamp_min(1)))
        # The reference sequence itself determines transitions, allowing repeats.
        transitions, missed, delays, premature = 0, 0, [], 0
        for ep in episodes:
            erows = [r for r in rows if r["episode"] == ep]
            changes = [i for i in range(1, len(erows)) if erows[i]["label"] != erows[i - 1]["label"]]
            for n, i in enumerate(changes):
                transitions += 1
                end = changes[n + 1] if n + 1 < len(changes) else len(erows)
                matches = [j for j in range(i, end) if erows[j]["prediction"] == erows[i]["label"]]
                missed += not matches
                if matches:
                    _, times = table[ep]
                    indices, _ = table[ep]
                    time_map = dict(zip(indices.tolist(), times.tolist(), strict=True))
                    delays.append(time_map[erows[matches[0]]["index"]] - time_map[erows[i]["index"]])
                premature += erows[i - 1]["prediction"] == erows[i]["label"]
        metrics.update(reference_transitions=transitions, missed_transitions=missed,
                       boundary_delay_seconds_mean=float(np.mean(delays)) if delays else None,
                       premature_at_previous_sample=premature)
        metrics["selection_score"] = (.5 * (1 - metrics["normalized_exact_match"]) +
                                      .5 * (1 - metrics["macro_f1"]) +
                                      5 * metrics["flow_global_only_native14_normalized"])
        return metrics, rows
    finally:
        model.train(previous)
        training.restore_random_state(rng)
