"""Reproducible stage-one training. M0 uses the unchanged official pi05 objective.

Supports single-process or DDP with sharded AdamW states and accumulation. Dataset
order is indexed by optimizer step so worker prefetch cannot change resume order.
All logs and checkpoints stay local; no experiment service is contacted.
"""

import argparse
import contextlib
import dataclasses
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist

from openpi.models.model import Observation
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.training import config as config_lib
from openpi.training import data_loader
from openpi.training.stage1_data import manifest_digest
from openpi.training.stage1_data import sha256_file
from openpi.training.stage1_optimizer import MixedPrecisionZeroAdamW


class StepBatchSampler:
    """Uniform action-window sampling, reproducible at optimizer-step boundaries."""

    def __init__(self, length, batch_size, accumulation, steps, start, seed, rank=0, world_size=1):
        self.length, self.batch_size, self.accumulation = length, batch_size, accumulation
        self.steps, self.start, self.seed = steps, start, seed
        self.rank, self.world_size = rank, world_size

    def __iter__(self):
        for step in range(self.start, self.steps):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, step]))
            for _ in range(self.accumulation):
                indices = rng.integers(0, self.length, size=self.batch_size * self.world_size)
                yield indices.reshape(self.world_size, self.batch_size)[self.rank].tolist()

    def __len__(self):
        return (self.steps - self.start) * self.accumulation


def collate_observation(samples):
    batch = jax.tree.map(lambda *values: np.stack([np.asarray(value) for value in values]), *samples)
    return jax.tree.map(torch.as_tensor, batch)


def to_device(batch, device):
    batch = jax.tree.map(lambda value: value.to(device), batch)
    return Observation.from_dict(batch), batch["actions"].float()


def learning_rate(step, warmup, total, peak):
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    fraction = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * fraction)))


def random_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda_current": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_random_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda_current") is not None:
        torch.cuda.set_rng_state(state["cuda_current"])
    elif "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def distributed_context():
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def save_checkpoint(model, optimizer, output, step, run_config, best, assets):
    target = output / f"step_{step:06d}"
    temporary = output / f".step_{step:06d}.tmp"
    rank, world_size = distributed_context()
    if rank == 0:
        if target.exists() or temporary.exists():
            raise FileExistsError(f"Refusing to overwrite checkpoint {target}")
        temporary.mkdir()
    if world_size > 1:
        dist.barrier()
        torch.save(
            {"optimizer": optimizer.local_state_dict(), "rng": random_state()},
            temporary / f"training_rank_{rank:03d}.pt",
        )
    else:
        torch.save({"optimizer": optimizer.state_dict(), "rng": random_state()}, temporary / "training.pt")
    if rank == 0:
        safetensors.torch.save_model(unwrap_model(model), temporary / "model.safetensors")
        metadata = {"schema_version": 2, "completed_steps": step, "stage": "m0", "config": run_config, "best": best}
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        shutil.copytree(assets, temporary / "assets" / "eggplant_potato")
    # Every optimizer shard must finish writing before publication by rank zero.
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        temporary.rename(target)
        latest_tmp = output / ".latest.json.tmp"
        latest_tmp.write_text(json.dumps({"checkpoint": target.name, "completed_steps": step}) + "\n")
        latest_tmp.replace(output / "latest.json")
    if world_size > 1:
        dist.barrier()
    return target


@torch.no_grad()
def evaluate(model, dataset, device, samples, batch_size, draws=2):
    """Fixed natural-distribution validation frames and independently fixed flow draws."""
    model = unwrap_model(model)
    rank, world_size = distributed_context()
    previous_training = model.training
    saved_rng = random_state()
    model.eval()
    indices = np.unique(np.linspace(0, len(dataset) - 1, min(samples, len(dataset)), dtype=int))
    global_frames = len(indices)
    indices = indices[rank::world_size]
    total = torch.zeros(2, device=device)
    count = 0
    try:
        for begin in range(0, len(indices), batch_size):
            batch_indices = indices[begin : begin + batch_size]
            observation, actions = to_device(
                collate_observation([dataset[int(index)] for index in batch_indices]), device
            )
            for draw in range(draws):
                noises, times = [], []
                for index in batch_indices:
                    generator = torch.Generator().manual_seed(100000 + int(index) * draws + draw)
                    noises.append(torch.randn(actions.shape[1:], generator=generator))
                    times.append(np.random.default_rng(200000 + int(index) * draws + draw).beta(1.5, 1) * 0.999 + 0.001)
                noise = torch.stack(noises).to(device)
                flow_time = torch.tensor(times, device=device, dtype=torch.float32)
                losses = model(observation, actions, noise=noise, time=flow_time)
                total += torch.stack([losses[..., :14].mean(dim=(1, 2)).sum(), losses.mean(dim=(1, 2)).sum()])
                count += len(batch_indices)
        totals = torch.cat([total, torch.tensor([count], device=device)])
        if world_size > 1:
            dist.all_reduce(totals)
        result = (totals[:2] / totals[2]).cpu().tolist()
        return {"flow_mse_native14": result[0], "flow_mse_all32": result[1], "frames": global_frames, "draws": draws}
    finally:
        model.train(previous_training)
        restore_random_state(saved_rng)


def train(args):
    if args.stage != "m0":
        raise NotImplementedError("M1-M3 are implemented after the M0 real-model gate")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no device is available")
    if args.steps < 1 or args.batch_size < 1 or args.accumulation < 1 or args.eval_samples < 1:
        raise ValueError("Steps, batch size, accumulation and validation size must be positive")
    if args.eval_every < 1 or args.checkpoint_every < 1 or args.warmup < 0:
        raise ValueError("Invalid evaluation/checkpoint interval or warmup")
    torch.set_num_threads(args.cpu_threads)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if args.device == "cuda" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if args.gpu_memory_limit_gib:
            limit = args.gpu_memory_limit_gib * 2**30 / torch.cuda.get_device_properties(device).total_memory
            torch.cuda.set_per_process_memory_fraction(limit, device)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group(
            "nccl" if device.type == "cuda" else "gloo", **({"device_id": device} if device.type == "cuda" else {})
        )
    rank, world_size = distributed_context()
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    cfg = config_lib.get_config("pi05_piper_stage1")
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, dtype=args.precision))
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    assets = Path(dc.split_manifest).parent
    audit = json.loads((assets / "audit.json").read_text())
    manifest = json.loads(Path(dc.split_manifest).read_text())
    if (
        manifest["manifest_sha256"] != manifest_digest(manifest)
        or audit["manifest_sha256"] != manifest["manifest_sha256"]
    ):
        raise ValueError("Split/audit provenance mismatch")
    if (
        audit["norm_stats_sha256"] != sha256_file(assets / "norm_stats.json")
        or manifest["action_horizon"] != cfg.model.action_horizon
    ):
        raise ValueError("Normalization/horizon does not match the audited training split")
    base_weights = args.base_checkpoint / "model.safetensors"
    source_files = [
        Path(__file__),
        Path("src/openpi/models_pytorch/pi0_pytorch.py"),
        Path("src/openpi/models_pytorch/gemma_pytorch.py"),
        Path("src/openpi/policies/piper_policy.py"),
        Path("src/openpi/training/stage1_data.py"),
        Path("src/openpi/training/config.py"),
        Path("src/openpi/training/data_loader.py"),
        Path("src/openpi/training/local_lerobot_dataset.py"),
        Path("src/openpi/training/stage1_optimizer.py"),
    ]
    run_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in {"resume", "stop_after"}
    }
    run_config.update(
        {
            "world_size": world_size,
            "optimizer_sharding": "zero1_by_dtype_local_checkpoint" if world_size > 1 else "none",
            "split_sha256": manifest["manifest_sha256"],
            "norm_sha256": audit["norm_stats_sha256"],
            "base_weights_sha256": sha256_file(base_weights),
            "model": dataclasses.asdict(cfg.model),
            "sources": {str(path): sha256_file(path) for path in source_files},
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }
    )
    start = 0
    best = {"flow_mse_native14": float("inf"), "step": None}
    resume_path = None
    if args.resume:
        latest = json.loads((args.output / "latest.json").read_text())
        resume_path = args.output / latest["checkpoint"]
        metadata = json.loads((resume_path / "metadata.json").read_text())
        # JSON roundtrip normalizes tuples in the model configuration.
        if metadata["config"] != json.loads(json.dumps(run_config)):
            raise ValueError("Resume configuration, code, dataset, or checkpoint provenance changed")
        start, best = metadata["completed_steps"], metadata["best"]
    elif rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    if world_size > 1:
        dist.barrier()
    train_raw = data_loader.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
    train_dataset = data_loader.transform_dataset(train_raw, dc)
    val_dc = dataclasses.replace(dc, split="val")
    val_raw = data_loader.create_torch_dataset(val_dc, cfg.model.action_horizon, cfg.model)
    val_dataset = data_loader.transform_dataset(val_raw, val_dc)
    # Fail on dataset/collation problems before allocating the large model.
    collate_observation([train_dataset[0], train_dataset[len(train_dataset) - 1]])
    collate_observation([val_dataset[0], val_dataset[len(val_dataset) - 1]])
    model = PI0Pytorch(cfg.model).to(device)
    safetensors.torch.load_model(
        model, (resume_path / "model.safetensors") if resume_path else base_weights, strict=True
    )
    model.gradient_checkpointing_enable()
    optimizer_options = {"lr": args.lr, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.01}
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
        )
        optimizer = MixedPrecisionZeroAdamW(model.parameters(), **optimizer_options)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_options)
    sampler = StepBatchSampler(
        len(train_dataset), args.batch_size, args.accumulation, args.steps, start, args.seed, rank, world_size
    )
    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=sampler,
        collate_fn=collate_observation,
        num_workers=args.workers,
        multiprocessing_context="spawn" if args.workers else None,
        persistent_workers=args.workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    if resume_path:
        checkpoint_file = f"training_rank_{rank:03d}.pt" if world_size > 1 else "training.pt"
        checkpoint = torch.load(resume_path / checkpoint_file, map_location="cpu", weights_only=False)
        if world_size > 1:
            optimizer.load_local_state_dict(checkpoint["optimizer"])
        else:
            optimizer.load_state_dict(checkpoint["optimizer"])
        restore_random_state(checkpoint["rng"])
        del checkpoint
    print(
        json.dumps(
            {
                "event": "ready",
                "stage": args.stage,
                "start": start,
                "train_frames": len(train_dataset),
                "val_frames": len(val_dataset),
                "parameters": sum(p.numel() for p in model.parameters()),
                "effective_batch": args.batch_size * args.accumulation * world_size,
                "rank": rank,
                "world_size": world_size,
            }
        ),
        flush=True,
    )

    def log(payload):
        if rank != 0:
            return
        with (args.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(payload) + "\n")
        print(json.dumps(payload), flush=True)

    if not args.resume:
        initial = evaluate(model, val_dataset, device, args.eval_samples, args.batch_size, args.eval_draws)
        log({"event": "validation", "step": 0, **initial})
    iterator = iter(loader)
    model.train()
    completed = start
    for step in range(start, args.steps):
        begun = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, args.warmup, args.steps, args.lr)
        loss_sum = 0.0
        for micro_step in range(args.accumulation):
            observation, actions = to_device(next(iterator), device)
            sync = (
                model.no_sync() if world_size > 1 and micro_step < args.accumulation - 1 else contextlib.nullcontext()
            )
            with sync:
                losses = model(observation, actions)
                loss = losses.mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite action loss at step {step}")
                (loss / args.accumulation).backward()
            loss_sum += loss.item() / args.accumulation
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        completed = step + 1
        loss_value = torch.tensor(loss_sum, device=device)
        if world_size > 1:
            dist.all_reduce(loss_value)
            loss_value /= world_size
        if device.type == "cuda":
            torch.cuda.synchronize()
        log(
            {
                "event": "train",
                "step": completed,
                "loss_action": float(loss_value),
                "grad_norm": float(norm),
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": time.perf_counter() - begun,
                "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None,
            }
        )
        improved = False
        if completed % args.eval_every == 0 or completed in {args.steps, args.stop_after}:
            validation = evaluate(model, val_dataset, device, args.eval_samples, args.batch_size, args.eval_draws)
            log({"event": "validation", "step": completed, **validation})
            improved = validation["flow_mse_native14"] < best["flow_mse_native14"]
            if improved:
                best = {"flow_mse_native14": validation["flow_mse_native14"], "step": completed}
        if improved or completed % args.checkpoint_every == 0 or completed in {args.steps, args.stop_after}:
            target = save_checkpoint(model, optimizer, args.output, completed, run_config, best, assets)
            log({"event": "checkpoint", "step": completed, "path": str(target), "best": best})
            if improved and rank == 0:
                (args.output / "best.json").write_text(json.dumps({"checkpoint": target.name, **best}) + "\n")
        if args.stop_after and completed >= args.stop_after:
            break
    log(
        {
            "event": "complete" if completed == args.steps else "stopped_at_checkpoint",
            "stage": args.stage,
            "completed_steps": completed,
            "best": best,
        }
    )
    if world_size > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["m0"], default="m0")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--precision", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--gpu-memory-limit-gib", type=float, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, default=Path("checkpoints/pi05_base_pytorch"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--stop-after",
        type=int,
        default=0,
        help="Stop at this completed step without changing the LR schedule; resume with the same --steps.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-samples", type=int, default=128)
    parser.add_argument("--eval-draws", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
