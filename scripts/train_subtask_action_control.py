"""B1 (no subtask) and O (GT condition) with the M2+M3 frozen-action budget."""

import argparse
import contextlib
import dataclasses
import json
import os
from pathlib import Path
import random
import shutil
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
from train_subtask_hierarchy import build_run_config
from train_subtask_pytorch import StepBatchSampler
from train_subtask_pytorch import learning_rate

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.training import config
from openpi.training import data_loader
from openpi.training import hierarchy_training as training
from openpi.training.action_control import ActionControl
from openpi.training.action_control import ActionOptimizer
from openpi.training.runtime_provenance import archive_runtime
from openpi.training.stage1_data import sha256_file
from openpi.training.subtask_batch import SubtaskTrainingDataset
from openpi.training.subtask_batch import collate_subtask
from openpi.training.subtask_curriculum import selection_eligible


class MatchedSampler:
    """Restart the sampler at the exact M2->M3 boundary, as in the main method."""

    def __init__(self, length, args, start, rank, world):
        self.samplers = []
        for offset, steps in [
            (0, args.warmup_phase_steps),
            (args.warmup_phase_steps, args.steps - args.warmup_phase_steps),
        ]:
            if start < offset + steps:
                self.samplers.append(
                    StepBatchSampler(
                        length,
                        args.batch_size,
                        args.accumulation,
                        steps,
                        max(0, start - offset),
                        args.seed,
                        rank,
                        world,
                    )
                )

    def __iter__(self):
        for sampler in self.samplers:
            yield from sampler

    def __len__(self):
        return sum(len(sampler) for sampler in self.samplers)


@torch.no_grad()
def evaluate(model, dataset, device, args):
    model = training.unwrap(model)
    rng, prior = training.random_state(), model.training
    model.eval()
    rank, world = training.distributed_context()
    indices = np.unique(np.linspace(0, len(dataset) - 1, min(args.eval_samples, len(dataset)), dtype=int))[rank::world]
    total = torch.zeros(3, device=device, dtype=torch.float64)
    try:
        for begin in range(0, len(indices), args.batch_size):
            current = indices[begin : begin + args.batch_size]
            batch = collate_subtask([dataset[int(index)] for index in current]).to(device)
            context = model.prepare_action_context(batch.observation, batch.global_prompts)
            prefix = model.action_prefix(context, batch.labels if args.condition == "oracle" else [""] * len(current))
            for draw in range(args.eval_draws):
                noise = torch.stack(
                    [
                        torch.randn(
                            batch.actions.shape[1:],
                            generator=torch.Generator().manual_seed(100000 + int(index) * args.eval_draws + draw),
                        )
                        for index in current
                    ]
                ).to(device)
                times = [
                    np.random.default_rng(200000 + int(index) * args.eval_draws + draw).beta(1.5, 1) * 0.999 + 0.001
                    for index in current
                ]
                error = model.action_loss(
                    context,
                    prefix,
                    batch.actions,
                    noise=noise,
                    time=torch.tensor(times, device=device, dtype=torch.float32),
                    reduction="none",
                )
                total += torch.stack(
                    [
                        error[..., :14].mean((1, 2)).sum(),
                        error.mean((1, 2)).sum(),
                        torch.tensor(len(current), device=device),
                    ]
                )
        if world > 1:
            dist.all_reduce(total)
        return {
            "flow_native14_normalized": float(total[0] / total[2]),
            "flow_all32": float(total[1] / total[2]),
            "draws": int(total[2]),
        }
    finally:
        model.train(prior)
        training.restore_random_state(rng)


def train(args):
    if not 1 <= args.warmup_phase_steps <= args.steps // 5 or args.steps - args.warmup_phase_steps < 4:
        raise ValueError("Match a positive M2 phase of at most 20%, followed by at least four M3 steps")
    if (
        min(
            args.batch_size,
            args.accumulation,
            args.eval_samples,
            args.eval_draws,
            args.eval_every,
            args.checkpoint_every,
        )
        < 1
    ):
        raise ValueError("Counts must be positive")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', '0')}" if args.device == "cuda" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group(
            "nccl" if device.type == "cuda" else "gloo", **({"device_id": device} if device.type == "cuda" else {})
        )
    rank, world = training.distributed_context()
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    parent = json.loads((args.initialize_from / "metadata.json").read_text())
    if parent["stage"] != "m0" or (parent["config"].get("engineering_smoke") and not args.engineering_smoke):
        raise ValueError("Research controls must initialize from the same completed C0")
    if not args.engineering_smoke:
        rows = [json.loads(line) for line in (args.initialize_from.parent / "metrics.jsonl").read_text().splitlines()]
        if not any(
            row.get("event") == "complete" and row["completed_steps"] == parent["config"]["steps"] for row in rows
        ):
            raise ValueError("Parent C0 run is incomplete")
    cfg = config.get_config("pi05_piper_stage1")
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, dtype=args.precision))
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    assets = Path(dc.split_manifest).parent
    payload = [None]
    if rank == 0:
        # Reuse the main method's audited data/runtime/source identity checks.
        identity_args = argparse.Namespace(**vars(args))
        identity_args.stage = "m1"
        identity_args.overfit_samples = 0
        run_config = build_run_config(identity_args, cfg, dc, parent, world)
        run_config.update(
            stage=f"control_{'b1' if args.condition == 'none' else 'o'}",
            condition=args.condition,
            decoder_unused=True,
            action_schedule="same two stage LR resets and sampler restarts as M2/M3",
            selection="same final generated-only-budget eligible update range as G",
        )
        for source in [Path(__file__), Path("src/openpi/training/action_control.py")]:
            run_config["sources"][str(source)] = sha256_file(source)
        payload[0] = json.loads(json.dumps(run_config))
    if world > 1:
        dist.broadcast_object_list(payload, src=0)
    run_config = payload[0]
    start, best, resume = 0, None, None
    if args.resume:
        resume = args.output / json.loads((args.output / "latest.json").read_text())["checkpoint"]
        metadata = json.loads((resume / "metadata.json").read_text())
        if metadata["config"] != run_config:
            raise ValueError("Exact control resume requires unchanged configuration/source/runtime/data")
        start, best = metadata["completed_steps"], metadata["best"]
    elif rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
        archive_runtime(run_config["runtime"], args.output / "runtime_sources")
        for name, expected in run_config["sources"].items():
            source = Path(name)
            target = args.output / "sources" / source.resolve().relative_to(Path.cwd())
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if sha256_file(target) != expected:
                raise ValueError("Source changed during archiving")
    if world > 1:
        dist.barrier()
    raw = data_loader.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
    dataset = SubtaskTrainingDataset(raw, dc)
    val_dc = dataclasses.replace(dc, split="val")
    validation = SubtaskTrainingDataset(
        data_loader.create_torch_dataset(val_dc, cfg.model.action_horizon, cfg.model), val_dc
    )
    model = ActionControl(PI0Pytorch(cfg.model).to(device), condition=args.condition)
    safetensors.torch.load_model(
        model if resume else model.base, (resume or args.initialize_from) / "model.safetensors", strict=True
    )
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
        )
    optimizer = ActionOptimizer(training.unwrap(model), args.lr_action)
    if resume:
        saved = torch.load(training.checkpoint_state_path(resume, rank, world), map_location="cpu", weights_only=False)
        if saved["rank"] != rank or saved["world_size"] != world:
            raise ValueError("Control resume rank/world size changed")
        optimizer.load_state_dict(saved["branch_optimizers"])
        training.restore_random_state(saved["rng"])
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=MatchedSampler(len(dataset), args, start, rank, world),
        collate_fn=collate_subtask,
        num_workers=args.workers,
        multiprocessing_context="spawn" if args.workers else None,
        persistent_workers=args.workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
    )

    def log(row):
        if rank == 0:
            with (args.output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)

    log(
        {
            "event": "ready",
            "stage": run_config["stage"],
            "effective_batch": args.batch_size * args.accumulation * world,
            "action_parameters": sum(p.numel() for p in training.unwrap(model).action_parameters()),
            "start": start,
        }
    )
    if not args.resume:
        log({"event": "validation", "step": 0, **evaluate(model, validation, device, args)})
    iterator = iter(loader)
    model.train()
    completed = start
    for step in range(start, args.steps):
        begun = time.perf_counter()
        optimizer.zero_grad()
        phase_step = step if step < args.warmup_phase_steps else step - args.warmup_phase_steps
        phase_steps = (
            args.warmup_phase_steps if step < args.warmup_phase_steps else args.steps - args.warmup_phase_steps
        )
        lr = learning_rate(phase_step, args.warmup_action, phase_steps, args.lr_action)
        for group in optimizer.optimizer.param_groups:
            group["lr"] = lr
        loss_total = torch.zeros((), device=device)
        for micro in range(args.accumulation):
            batch = next(iterator).to(device)
            with model.no_sync() if world > 1 and micro < args.accumulation - 1 else contextlib.nullcontext():
                loss = model(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite control action loss")
                (loss / args.accumulation).backward()
            loss_total += loss.detach() / args.accumulation
        norm = optimizer.step()
        optimizer.zero_grad()
        if world > 1:
            dist.all_reduce(loss_total)
        completed = step + 1
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        log(
            {
                "event": "train",
                "stage": run_config["stage"],
                "step": completed,
                "loss_action": float(loss_total / world),
                "grad_norms": {"action": norm},
                "lr_action": lr,
                "seconds": time.perf_counter() - begun,
            }
        )
        improved = False
        if completed % args.eval_every == 0 or completed in {args.steps, args.stop_after}:
            metrics = evaluate(model, validation, device, args)
            log({"event": "validation", "step": completed, **metrics})
            eligible = completed > args.warmup_phase_steps and selection_eligible(
                "m3", completed - args.warmup_phase_steps, args.steps - args.warmup_phase_steps, completed
            )
            score = [metrics["flow_native14_normalized"]]
            improved = eligible and (best is None or score < best["selection_score"])
            if improved:
                best = {"step": completed, "selection_score": score}
        if improved or completed % args.checkpoint_every == 0 or completed in {args.steps, args.stop_after}:
            target = training.save_checkpoint(
                model, optimizer, args.output, completed, run_config, best, {"subtask": 0, "action": completed}, assets
            )
            log({"event": "checkpoint", "step": completed, "path": str(target)})
            if improved and rank == 0:
                (args.output / "best.json").write_text(json.dumps({"checkpoint": target.name, **best}) + "\n")
        if args.stop_after and completed >= args.stop_after:
            break
    log(
        {
            "event": "complete" if completed == args.steps else "stopped_at_checkpoint",
            "stage": run_config["stage"],
            "completed_steps": completed,
            "best": best,
        }
    )
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition", choices=["none", "oracle"], required=True)
    p.add_argument("--initialize-from", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--warmup-phase-steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accumulation", type=int, default=2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--precision", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--lr-action", type=float, default=1e-5)
    p.add_argument("--warmup-action", type=int, default=200)
    p.add_argument("--eval-samples", type=int, default=128)
    p.add_argument("--eval-draws", type=int, default=2)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--engineering-smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after", type=int, default=0)
    train(p.parse_args())
