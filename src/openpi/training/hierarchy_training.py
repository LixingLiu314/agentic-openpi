"""Branch optimizer ownership, atomic checkpoints and hierarchy validation."""

import json
import random
import shutil

import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist

from openpi.training.stage1_optimizer import MixedPrecisionZeroAdamW
from openpi.training.subtask_batch import collate_subtask


def distributed_context():
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def unwrap(model):
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def random_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_random_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])


class BranchOptimizers:
    """Two disjoint parameter owners, independently clipped and updated."""

    def __init__(self, model, stage, subtask_lr, action_lr):
        self.parameters = {"subtask": model.subtask_parameters()}
        if stage != "m1":
            self.parameters["action"] = model.action_parameters()
        owners = [id(parameter) for group in self.parameters.values() for parameter in group]
        if len(owners) != len(set(owners)):
            raise ValueError("Optimizer parameter ownership overlaps")
        if set(owners) != {id(parameter) for parameter in model.parameters() if parameter.requires_grad}:
            raise ValueError("Trainable parameters do not exactly match optimizer ownership")
        self.optimizers = {}
        self.world_size = distributed_context()[1]
        for branch, parameters in self.parameters.items():
            options = {
                "lr": subtask_lr if branch == "subtask" else action_lr,
                "betas": (0.9, 0.95),
                "eps": 1e-8,
                "weight_decay": 0.01,
            }
            self.optimizers[branch] = (
                MixedPrecisionZeroAdamW(parameters, **options)
                if self.world_size > 1
                else torch.optim.AdamW(parameters, **options)
            )

    def zero_grad(self):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=True)

    def step(self):
        # Clip every branch before updating either branch; fail without a partial update.
        norms = {
            branch: float(torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True))
            for branch, parameters in self.parameters.items()
        }
        for optimizer in self.optimizers.values():
            optimizer.step()
        return norms

    def state_dict(self):
        return {
            branch: optimizer.local_state_dict() if self.world_size > 1 else optimizer.state_dict()
            for branch, optimizer in self.optimizers.items()
        }

    def load_state_dict(self, state, *, transition=False):
        if not transition and set(state) != set(self.optimizers):
            raise ValueError("Resume optimizer branches changed")
        if not set(state) <= set(self.optimizers):
            raise ValueError("Stage transition would discard an existing optimizer")
        for branch, saved in state.items():
            optimizer = self.optimizers[branch]
            if self.world_size > 1:
                optimizer.load_local_state_dict(saved)
            else:
                optimizer.load_state_dict(saved)


def checkpoint_state_path(checkpoint, rank, world):
    return checkpoint / (f"training_rank_{rank:03d}.pt" if world > 1 else "training.pt")


def save_checkpoint(model, optimizers, output, step, run_config, best, counters, assets):
    rank, world = distributed_context()
    target = output / f"step_{step:06d}"
    temporary = output / f".step_{step:06d}.tmp"
    if rank == 0:
        if target.exists() or temporary.exists():
            raise FileExistsError(f"Refusing to overwrite {target}")
        temporary.mkdir()
    if world > 1:
        dist.barrier()
    torch.save(
        {
            "branch_optimizers": optimizers.state_dict(),
            "rng": random_state(),
            "rank": rank,
            "world_size": world,
        },
        checkpoint_state_path(temporary, rank, world),
    )
    if rank == 0:
        safetensors.torch.save_model(unwrap(model), temporary / "model.safetensors")
        metadata = {
            "schema_version": 3,
            "stage": run_config["stage"],
            "completed_steps": step,
            "config": run_config,
            "best": best,
            "counters": counters,
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        shutil.copytree(assets, temporary / "assets" / "eggplant_potato")
        if (output / "sources").exists():
            shutil.copytree(output / "sources", temporary / "sources")
        if (output / "runtime_sources").exists():
            shutil.copytree(output / "runtime_sources", temporary / "runtime_sources")
    if world > 1:
        dist.barrier()
    if rank == 0:
        temporary.rename(target)
        pointer = output / ".latest.json.tmp"
        pointer.write_text(json.dumps({"checkpoint": target.name, "completed_steps": step}) + "\n")
        pointer.replace(output / "latest.json")
    if world > 1:
        dist.barrier()
    return target


def text_metrics(rows, vocabulary):
    """Natural-distribution sequence metrics; unknown predictions are errors."""

    def canonical(text):
        return " ".join((text or "").lower().split())

    labeled = [row for row in rows if row["label"] and row["label"].strip()]
    classes = sorted({canonical(label) for label in vocabulary if label and label.strip()})
    per_class = {}
    for label in classes:
        tp = sum(canonical(row["label"]) == label == canonical(row["prediction"]) for row in labeled)
        fp = sum(canonical(row["label"]) != label == canonical(row["prediction"]) for row in labeled)
        fn = sum(canonical(row["prediction"]) != label == canonical(row["label"]) for row in labeled)
        per_class[label] = {"f1": 2 * tp / max(1, 2 * tp + fp + fn), "recall": tp / max(1, tp + fn), "support": tp + fn}
    return {
        "exact_match": sum(row["label"] == row["prediction"] for row in labeled) / max(1, len(labeled)),
        "normalized_exact_match": sum(canonical(row["label"]) == canonical(row["prediction"]) for row in labeled)
        / max(1, len(labeled)),
        "macro_f1": float(np.mean([value["f1"] for value in per_class.values()])) if classes else 0.0,
        "unknown_rate": sum(canonical(row["prediction"]) not in classes for row in rows) / max(1, len(rows)),
        "invalid_generation_rate": sum(row["status"] != "ok" for row in rows) / max(1, len(rows)),
        "per_class": per_class,
    }


@torch.no_grad()
def evaluate(model, dataset, device, *, samples, batch_size, draws, vocabulary, action_enabled):
    """Fixed frame indices and fixed flow noise/time; all conditions are generated."""
    model = unwrap(model)
    rank, world = distributed_context()
    saved_rng, previous_training = random_state(), model.training
    model.eval()
    indices = np.unique(np.linspace(0, len(dataset) - 1, min(samples, len(dataset)), dtype=int))[rank::world]
    totals = torch.zeros(5, device=device, dtype=torch.float64)
    rows = []
    try:
        for begin in range(0, len(indices), batch_size):
            current_indices = indices[begin : begin + batch_size]
            batch = collate_subtask([dataset[int(index)] for index in current_indices]).to(device)
            context = model.prepare_context(batch.observation, batch.global_prompts)
            valid_count = batch.target_mask.any(dim=1).sum()
            totals[0] += model.subtask_loss(context, batch.target_ids, batch.target_mask) * valid_count
            totals[1] += valid_count
            predictions, statuses, _ = model.generate_subtask(context)
            for index, prediction, status, label, episode, frame in zip(
                current_indices,
                predictions,
                statuses,
                batch.labels,
                batch.episode_indices.tolist(),
                batch.frame_indices.tolist(),
                strict=True,
            ):
                rows.append(
                    {
                        "index": int(index),
                        "episode": episode,
                        "frame": frame,
                        "label": label,
                        "prediction": prediction,
                        "status": status,
                    }
                )
            if action_enabled:
                prefix = model.action_prefix(context, predictions)
                for draw in range(draws):
                    noises, times = [], []
                    for index in current_indices:
                        generator = torch.Generator().manual_seed(100000 + int(index) * draws + draw)
                        noises.append(torch.randn(batch.actions.shape[1:], generator=generator))
                        times.append(
                            np.random.default_rng(200000 + int(index) * draws + draw).beta(1.5, 1) * 0.999 + 0.001
                        )
                    errors = model.action_loss(
                        context,
                        prefix,
                        batch.actions,
                        noise=torch.stack(noises).to(device),
                        time=torch.tensor(times, device=device, dtype=torch.float32),
                        reduction="none",
                    )
                    totals[2] += errors[..., :14].mean(dim=(1, 2)).sum()
                    totals[3] += errors.mean(dim=(1, 2)).sum()
                    totals[4] += len(current_indices)
        if world > 1:
            dist.all_reduce(totals)
            all_rows = [None] * world
            dist.all_gather_object(all_rows, rows)
            rows = [row for rank_rows in all_rows for row in rank_rows]
        rows.sort(key=lambda row: row["index"])
        result = {
            "loss_subtask": float(totals[0] / totals[1].clamp_min(1)),
            "frames": len(rows),
            **text_metrics(rows, vocabulary),
        }
        if action_enabled:
            result.update(
                flow_generated_native14_normalized=float(totals[2] / totals[4]),
                flow_generated_all32=float(totals[3] / totals[4]),
                draws=draws,
            )
        return result, rows
    finally:
        model.train(previous_training)
        restore_random_state(saved_rng)
