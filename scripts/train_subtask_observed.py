"""M1 subtask warmup, M2 GT action warmup, M3 generated-condition training.

The M0 trainer remains immutable while its C0 experiment runs. Stage transitions
strictly load all model weights and carry existing branch optimizer states;
resume within a stage additionally requires unchanged code/configuration and RNG.
"""

import argparse
import contextlib
import dataclasses
import hashlib
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
from subtask_runtime import CachedDataset
from subtask_runtime import StepTimer
from subtask_runtime import collate_transfer
from subtask_runtime import reduce_timing_max
import torch
import torch.distributed as dist
from train_subtask_pytorch import StepBatchSampler
from train_subtask_pytorch import learning_rate
from visualize_subtask_step import render_first_step

from openpi.models.subtask_tokenizer import SubtaskTextCodec
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pi05_subtask_pytorch import Pi05SubtaskPytorch
from openpi.models_pytorch.subtask_decoder import SubtaskDecoderConfig
from openpi.training import config as config_lib
from openpi.training import data_loader
from openpi.training import hierarchy_training as training
from openpi.training.stage1_data import manifest_digest
from openpi.training.stage1_data import sha256_file
from openpi.training.subtask_batch import BalancedSubtaskSampler
from openpi.training.subtask_batch import SubtaskTrainingDataset
from openpi.training.subtask_batch import collate_subtask
from openpi.training.subtask_batch import predicted_condition_ratio


def overfit_indices(labels, tasks, count):
    buckets = {}
    for index, (task, label) in enumerate(zip(tasks, labels, strict=True)):
        if label and label.strip():
            buckets.setdefault((task, label), []).append(index)
    if count < len(buckets) or count % len(buckets):
        raise ValueError("Overfit sample count must be a positive multiple of task/subtask groups")
    selected = []
    for key in sorted(buckets):
        candidates = buckets[key]
        number = count // len(buckets)
        if number > len(candidates):
            raise ValueError("Not enough distinct frames in an overfit group")
        positions = np.linspace(0, len(candidates) - 1, number + 2, dtype=int)[1:-1]
        selected.extend(candidates[int(position)] for position in positions)
    return selected


def validate_transition(args, metadata):
    expected = {"m1": "m0", "m2": "m1", "m3": "m2"}[args.stage]
    if metadata["stage"] != expected:
        raise ValueError(f"Stage {args.stage} requires a {expected} checkpoint")
    if metadata["config"].get("engineering_smoke") and not args.engineering_smoke:
        raise ValueError("Engineering/overfit weights cannot initialize a research run")
    if args.overfit_samples and not args.engineering_smoke:
        raise ValueError("Overfit debug requires --engineering-smoke")
    if args.stage == "m2" and args.steps * 4 > args.m3_planned_steps:
        raise ValueError("M2 GT updates must be no more than 20% of M2+M3 updates")
    if args.stage == "m3":
        if args.steps < 4 or args.steps != metadata["config"]["m3_planned_steps"]:
            raise ValueError("M3 steps must match the M2 plan and cover four schedule phases")
        if metadata["counters"]["action"] * 4 > args.steps:
            raise ValueError("Inherited GT warmup exceeds 20% of conditional action updates")
    if not args.engineering_smoke:
        metrics = args.initialize_from.parent / "metrics.jsonl"
        complete = any(
            row.get("event") == "complete" and row.get("completed_steps") == metadata["config"]["steps"]
            for row in map(json.loads, metrics.read_text().splitlines())
        )
        if not complete:
            raise ValueError("Finish the parent run before initializing the next research stage")


def build_run_config(args, cfg, dc, parent, world):
    assets = Path(dc.split_manifest).parent
    manifest = json.loads((assets / "split.json").read_text())
    audit = json.loads((assets / "audit.json").read_text())
    norm_sha = sha256_file(assets / "norm_stats.json")
    if (
        manifest["manifest_sha256"] != manifest_digest(manifest)
        or audit["manifest_sha256"] != manifest["manifest_sha256"]
    ):
        raise ValueError("Split/audit provenance mismatch")
    if audit["norm_stats_sha256"] != norm_sha or manifest["action_horizon"] != cfg.model.action_horizon:
        raise ValueError("Normalization/horizon provenance mismatch")
    if parent["config"]["split_sha256"] != manifest["manifest_sha256"] or parent["config"]["norm_sha256"] != norm_sha:
        raise ValueError("Parent checkpoint uses different training assets")
    for name in ["split.json", "norm_stats.json"]:
        if sha256_file(args.initialize_from / "assets" / "eggplant_potato" / name) != sha256_file(assets / name):
            raise ValueError("Parent checkpoint assets differ from current assets")
    sources = [
        Path(__file__),
        Path("scripts/subtask_runtime.py"),
        Path("scripts/visualize_subtask_step.py"),
        Path("scripts/train_subtask_pytorch.py"),
        *[
            Path("src/openpi") / name
            for name in [
                "models_pytorch/pi05_subtask_pytorch.py",
                "models_pytorch/subtask_decoder.py",
                "models_pytorch/pi0_pytorch.py",
                "models_pytorch/gemma_pytorch.py",
                "models/subtask_tokenizer.py",
                "models/tokenizer.py",
                "transforms.py",
                "policies/piper_policy.py",
                "training/subtask_batch.py",
                "training/hierarchy_training.py",
                "training/stage1_optimizer.py",
                "training/stage1_data.py",
                "training/local_lerobot_dataset.py",
                "training/config.py",
                "training/data_loader.py",
            ]
        ],
    ]
    parent_hash = sha256_file(args.initialize_from / "model.safetensors")
    result = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in {"resume", "stop_after"}
    }
    result.update(
        world_size=world,
        model=dataclasses.asdict(cfg.model),
        decoder=dataclasses.asdict(SubtaskDecoderConfig()),
        use_quantile_norm=dc.use_quantile_norm,
        tokenizer_model_sha256=hashlib.sha256(SubtaskTextCodec().processor.serialized_model_proto()).hexdigest(),
        split_sha256=manifest["manifest_sha256"],
        norm_sha256=norm_sha,
        parent_weights_sha256=parent_hash,
        c0_weights_sha256=parent_hash if args.stage == "m1" else parent["config"]["c0_weights_sha256"],
        sources={str(path): sha256_file(path) for path in sources},
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        optimizer_transition="carry existing branches; initialize newly enabled branch; restart stage LR schedule",
        evaluation_source="training_overfit_subset" if args.overfit_samples else "validation_natural",
        prompt_contract="global task plus normalized native14 state; condition generated or explicit training curriculum",
    )
    return json.loads(json.dumps(result))


def train(args):
    if (
        min(
            args.steps,
            args.batch_size,
            args.accumulation,
            args.eval_samples,
            args.eval_every,
            args.eval_draws,
            args.checkpoint_every,
        )
        < 1
    ):
        raise ValueError("Training, batch and evaluation counts must be positive")
    if not 0 <= args.condition_dropout < 1 or min(args.warmup_subtask, args.warmup_action) < 0:
        raise ValueError("Invalid dropout or warmup")
    if args.overfit_samples and args.stage != "m1":
        raise ValueError("Small-sample overfit is a dedicated M1 gate")
    torch.set_num_threads(args.cpu_threads)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if args.device == "cuda" else "cpu")
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
    validate_transition(args, parent)
    cfg = config_lib.get_config("pi05_piper_stage1")
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, dtype=args.precision))
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    assets = Path(dc.split_manifest).parent
    config_payload = [build_run_config(args, cfg, dc, parent, world) if rank == 0 else None]
    if world > 1:
        dist.broadcast_object_list(config_payload, src=0)
    run_config = config_payload[0]
    start, best = 0, None
    counters = {"subtask": 0, "action": 0} if args.stage == "m1" else parent["counters"].copy()
    resume_path = None
    if args.resume:
        pointer = json.loads((args.output / "latest.json").read_text())
        resume_path = args.output / pointer["checkpoint"]
        saved = json.loads((resume_path / "metadata.json").read_text())
        if saved["schema_version"] != 3 or saved["config"] != run_config:
            raise ValueError("Resume configuration, source, data or checkpoint provenance changed")
        start, best, counters = saved["completed_steps"], saved["best"], saved["counters"]
    elif rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
        for name, expected_hash in run_config["sources"].items():
            source = Path(name)
            relative = source.resolve().relative_to(Path.cwd().resolve())
            archived = args.output / "sources" / relative
            archived.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, archived)
            if sha256_file(archived) != expected_hash:
                raise ValueError("Source changed between fingerprinting and archiving")

    if world > 1:
        dist.barrier()
    raw = data_loader.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
    dataset = SubtaskTrainingDataset(raw, dc)
    labels, tasks = dataset.labels_without_video(), list(raw.hf_dataset["task"])
    vocabulary = sorted({label for label in labels if label and label.strip()})
    if args.overfit_samples:
        selected = overfit_indices(labels, tasks, args.overfit_samples)
        dataset = torch.utils.data.Subset(dataset, selected)
        labels, tasks = [labels[index] for index in selected], [tasks[index] for index in selected]
        evaluation_dataset = dataset
        if rank == 0 and not args.resume:
            (args.output / "overfit_indices.json").write_text(json.dumps(selected) + "\n")
    else:
        val_dc = dataclasses.replace(dc, split="val")
        evaluation_dataset = SubtaskTrainingDataset(
            data_loader.create_torch_dataset(val_dc, cfg.model.action_horizon, cfg.model), val_dc
        )
    collate_subtask([dataset[0], dataset[len(dataset) - 1]])
    collate_subtask([evaluation_dataset[0]])
    base = PI0Pytorch(cfg.model).to(device)
    model = Pi05SubtaskPytorch(base)
    if resume_path or args.stage != "m1":
        safetensors.torch.load_model(model, (resume_path or args.initialize_from) / "model.safetensors", strict=True)
    else:
        safetensors.torch.load_model(model.base, args.initialize_from / "model.safetensors", strict=True)
    model.set_stage(args.stage)
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
        )
    optimizers = training.BranchOptimizers(training.unwrap(model), args.stage, args.lr_subtask, args.lr_action)
    state_path = resume_path or (args.initialize_from if args.stage != "m1" else None)
    if state_path:
        state = torch.load(
            training.checkpoint_state_path(state_path, rank, world), map_location="cpu", weights_only=False
        )
        if state["rank"] != rank or state["world_size"] != world:
            raise ValueError("Optimizer transition/resume requires the same rank and world size")
        optimizers.load_state_dict(state["branch_optimizers"], transition=resume_path is None)
        if resume_path:
            training.restore_random_state(state["rng"])
        del state
    sampler_options = {
        "batch_size": args.batch_size,
        "accumulation": args.accumulation,
        "steps": args.steps,
        "start": start,
        "seed": args.seed,
        "rank": rank,
        "world_size": world,
    }

    sampler = (
        BalancedSubtaskSampler(labels, tasks=tasks, **sampler_options)
        if args.stage == "m1"
        else StepBatchSampler(len(dataset), **sampler_options)
    )
    loader = torch.utils.data.DataLoader(
        CachedDataset(dataset, args.cache_samples or args.overfit_samples),
        batch_sampler=sampler,
        collate_fn=collate_transfer,
        num_workers=args.workers,
        multiprocessing_context="spawn" if args.workers else None,
        persistent_workers=args.workers > 0,
        pin_memory=device.type == "cuda",
        **({"prefetch_factor": args.prefetch_factor} if args.workers else {}),
        generator=torch.Generator().manual_seed(args.seed),
    )

    def log(payload):
        if rank == 0:
            with (args.output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(payload) + "\n")
            print(json.dumps(payload), flush=True)

    def validate(step):
        evaluation_begun = time.perf_counter()
        metrics, predictions = training.evaluate(
            model,
            evaluation_dataset,
            device,
            samples=args.eval_samples,
            batch_size=args.batch_size,
            draws=args.eval_draws,
            vocabulary=vocabulary,
            action_enabled=args.stage != "m1",
        )
        log({"event": "validation", "step": step, **metrics})
        log({"event": "performance", "step": step, "validation_seconds": time.perf_counter() - evaluation_begun})
        if rank == 0:
            (args.output / f"predictions_{step:06d}.json").write_text(json.dumps(predictions, indent=2) + "\n")
        return metrics

    log(
        {
            "event": "ready",
            "stage": args.stage,
            "start": start,
            "effective_batch": args.batch_size * args.accumulation * world,
            "train_frames": len(dataset),
            "evaluation_frames": len(evaluation_dataset),
            "branch_parameters": {
                key: sum(parameter.numel() for parameter in parameters)
                for key, parameters in optimizers.parameters.items()
            },
        }
    )
    if not args.resume:
        validate(0)
    iterator = iter(loader)
    model.train()
    completed = start
    for step in range(start, args.steps):
        begun = time.perf_counter()
        timer = StepTimer(device, step - start < args.profile_steps)
        first_batch = None
        data_wait_seconds = 0.0
        optimizers.zero_grad()
        for branch, optimizer in optimizers.optimizers.items():
            peak, warmup = (
                (args.lr_subtask, args.warmup_subtask) if branch == "subtask" else (args.lr_action, args.warmup_action)
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step, warmup, args.steps, peak)
        ratio = predicted_condition_ratio(step, args.steps) if args.stage == "m3" else 0.0
        totals = torch.zeros(6, device=device)
        for micro in range(args.accumulation):
            phase_begun = timer.now()
            cpu_batch = next(iterator)
            data_wait_seconds += time.perf_counter() - phase_begun
            timer.add("data_wait", phase_begun)
            if step == 0 and micro == 0 and rank == 0:
                first_batch = cpu_batch
            phase_begun = timer.now()
            batch = cpu_batch.to(device)
            timer.add("host_to_device", phase_begun)
            choices = np.random.random(len(batch.labels)) < ratio
            dropped = np.random.random(len(batch.labels)) < (args.condition_dropout if args.stage == "m3" else 0)
            sync = model.no_sync() if world > 1 and micro < args.accumulation - 1 else contextlib.nullcontext()
            with sync:
                phase_begun = timer.now()
                result = model(batch, use_prediction=choices, drop_condition=dropped)
                loss = result["loss_subtask"] + result.get("loss_action", 0)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at {args.stage} step {step}")
                timer.add("forward", phase_begun)
                phase_begun = timer.now()
                (loss / args.accumulation).backward()
                timer.add("backward_including_ddp", phase_begun)
            totals[0] += result["loss_subtask"].detach() / args.accumulation
            totals[1] += result.get("loss_action", torch.zeros((), device=device)).detach() / args.accumulation
            totals[2:] += torch.tensor(
                [
                    result.get("generated_count", 0),
                    result.get("invalid_generation_count", 0),
                    result.get("empty_condition_count", 0),
                    len(batch.labels),
                ],
                device=device,
            )
        phase_begun = timer.now()
        norms = optimizers.step()
        timer.add("optimizer_including_zero", phase_begun)
        optimizers.zero_grad()
        counters["subtask"] += 1
        counters["action"] += int(args.stage != "m1")
        if world > 1:
            dist.all_reduce(totals)
        totals[:2] /= world
        completed = step + 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        log(
            {
                "event": "train",
                "stage": args.stage,
                "step": completed,
                "loss_subtask": float(totals[0]),
                "loss_action": float(totals[1]) if args.stage != "m1" else None,
                "grad_norms": norms,
                "learning_rates": {
                    name: optimizer.param_groups[0]["lr"] for name, optimizer in optimizers.optimizers.items()
                },
                "predicted_ratio_schedule": ratio,
                "generated_count": int(totals[2]),
                "invalid_generation_count": int(totals[3]),
                "empty_condition_count": int(totals[4]),
                "examples": int(totals[5]),
                "seconds": time.perf_counter() - begun,
                "data_wait_seconds_rank0": data_wait_seconds,
                "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None,
            }
        )
        if timer.enabled:
            log({"event": "performance", "step": completed, **reduce_timing_max(timer.values, device)})
        if completed == 1:
            visual_begun = time.perf_counter()
            if rank == 0:
                try:
                    render_first_step(
                        training.unwrap(model),
                        first_batch.to(device),
                        dc,
                        args.output / "first_step",
                        origin="live first update; rank-0 first microbatch",
                        limit=args.visualization_samples,
                    )
                except Exception as error:
                    # Preserve training and expose the failure instead of silently losing diagnostics.
                    log({"event": "visualization_error", "step": 1, "error": f"{type(error).__name__}: {error}"})
            if world > 1:
                dist.barrier()
            log(
                {
                    "event": "performance",
                    "step": completed,
                    "first_visualization_seconds": time.perf_counter() - visual_begun,
                }
            )
        improved = False
        if completed % args.eval_every == 0 or completed in {args.steps, args.stop_after}:
            metrics = validate(completed)
            score = (
                [-metrics["macro_f1"], metrics["loss_subtask"]]
                if args.stage == "m1"
                else [metrics["flow_generated_native14_normalized"]]
            )
            improved = best is None or score < best["selection_score"]
            if improved:
                best = {"selection_score": score, "step": completed}
        if improved or completed % args.checkpoint_every == 0 or completed in {args.steps, args.stop_after}:
            save_begun = time.perf_counter()
            target = training.save_checkpoint(
                model, optimizers, args.output, completed, run_config, best, counters, assets
            )
            log({"event": "checkpoint", "step": completed, "path": str(target), "best": best})
            log({"event": "performance", "step": completed, "checkpoint_seconds": time.perf_counter() - save_begun})
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
            "counters": counters,
        }
    )
    if world > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["m1", "m2", "m3"], required=True)
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--m3-planned-steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--cache-samples", type=int, default=0)
    parser.add_argument("--profile-steps", type=int, default=10)
    parser.add_argument("--visualization-samples", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--precision", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--lr-subtask", type=float, default=1e-4)
    parser.add_argument("--lr-action", type=float, default=1e-5)
    parser.add_argument("--warmup-subtask", type=int, default=200)
    parser.add_argument("--warmup-action", type=int, default=200)
    parser.add_argument("--condition-dropout", type=float, default=0.1)
    parser.add_argument("--eval-samples", type=int, default=128)
    parser.add_argument("--eval-draws", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overfit-samples", type=int, default=0)
    parser.add_argument("--engineering-smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, default=0)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
