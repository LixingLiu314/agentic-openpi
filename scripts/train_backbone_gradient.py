"""Seed42 action->B ablations with the user-specified 5000/256/cosine recipe."""

import argparse
import contextlib
import dataclasses
import datetime
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

from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.backbone_gradient import BackboneGradientModel, cosine_lr
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.subtask_decoder import SubtaskDecoderConfig
from openpi.training import config as config_lib, data_loader, hierarchy_training as training
from openpi.training.research_checkpoint import require_research_checkpoint
from openpi.training.runtime_provenance import archive_runtime
from openpi.training.stage1_data import sha256_file
from openpi.training.stage1_optimizer import MixedPrecisionZeroAdamW
from openpi.training.subtask_batch import SubtaskTrainingDataset, collate_subtask


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class Optimizers:
    def __init__(self, model, world):
        self.parameters = {"subtask": model.subtask_parameters(),
                           "action_backbone": model.action_parameters() + model.backbone_parameters()}
        ids = [id(p) for params in self.parameters.values() for p in params]
        if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in model.parameters() if p.requires_grad}:
            raise ValueError("Optimizer ownership mismatch")
        self.world = world
        factory = MixedPrecisionZeroAdamW if world > 1 else torch.optim.AdamW
        self.optimizers = {name: factory(params, lr=2.5e-5, betas=(0.9, 0.95), eps=1e-8,
                                         weight_decay=1e-10) for name, params in self.parameters.items()}

    def zero_grad(self):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=True)

    def step(self, lr):
        norms = {name: float(torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True))
                 for name, params in self.parameters.items()}
        for optimizer in self.optimizers.values():
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
        return norms

    def state_dict(self):
        return {name: optimizer.local_state_dict() if self.world > 1 else optimizer.state_dict()
                for name, optimizer in self.optimizers.items()}

    def load_state_dict(self, values):
        if set(values) != set(self.optimizers):
            raise ValueError("Changed optimizer branches")
        for name, value in values.items():
            if self.world > 1:
                self.optimizers[name].load_local_state_dict(value)
            else:
                self.optimizers[name].load_state_dict(value)


def save_checkpoint(model, optimizers, args, config, step, best, counters, assets):
    rank, world = training.distributed_context()
    model = training.unwrap(model)
    target = args.output / f"step_{step:06d}"
    temporary = args.output / f".step_{step:06d}.tmp"
    if rank == 0:
        if target.exists() or temporary.exists():
            raise FileExistsError(target)
        temporary.mkdir()
    if world > 1:
        dist.barrier()
    torch.save({"branch_optimizers": optimizers.state_dict(), "rng": training.random_state(),
                "rank": rank, "world_size": world}, training.checkpoint_state_path(temporary, rank, world))
    if rank == 0:
        safetensors.torch.save_file(model.deployment_state(), temporary / "model.safetensors")
        if args.mode == "limited":
            safetensors.torch.save_model(model, temporary / "training_model.safetensors")
        weights_hash = sha256_file(temporary / "model.safetensors")
        metadata = {"schema_version": 4, "stage": "backbone_grad", "variant": "action_backbone_v1",
                    "completed_steps": step, "config": config, "best": best, "counters": counters,
                    "weights_sha256": weights_hash,
                    "training_weights_sha256": sha256_file(temporary / "training_model.safetensors")
                    if args.mode == "limited" else weights_hash,
                    "export": "LoRA merged into ordinary M3 tensor layout; complete B/S/A weights"}
        write_json(temporary / "metadata.json", metadata)
        shutil.copytree(assets, temporary / "assets/eggplant_potato")
        shutil.copytree(args.output / "sources", temporary / "sources")
        shutil.copytree(args.output / "runtime_sources", temporary / "runtime_sources")
    if world > 1:
        dist.barrier()
    if rank == 0:
        temporary.rename(target)
        write_json(args.output / "latest.json", {"checkpoint": target.name, "completed_steps": step})
    if world > 1:
        dist.barrier()
    return target


def train(args):
    torch.set_num_threads(args.cpu_threads)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if args.device == "cuda" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if not 0 < args.memory_fraction <= 1:
            raise ValueError("Invalid CUDA allocator fraction")
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo",
                                timeout=datetime.timedelta(minutes=45))
    rank, world = training.distributed_context()
    if args.batch_size * args.accumulation * world != args.global_batch:
        raise ValueError("microbatch * accumulation * world must equal global_batch")
    if not args.engineering_smoke and (args.steps, args.global_batch, args.seed, args.checkpoint_every,
                                      args.warmup, args.peak_lr, args.decay_lr) != (5000, 256, 42, 500, 500, 2.5e-5, 2.5e-6):
        raise ValueError("Formal configuration differs from user recipe")
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    parent = json.loads((args.initialize_from / "metadata.json").read_text())
    require_research_checkpoint(args.initialize_from, parent)
    if parent["stage"] != "m3" or parent["completed_steps"] != 3500:
        raise ValueError("Both experiments require original research M3 step3500")
    cfg = config_lib.get_config("pi05_piper_stage1")
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(Pi0Config(**parent["config"]["model"]),
                                                            dtype=args.precision))
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    assets = Path(dc.split_manifest).parent
    args.stage, args.overfit_samples = "backbone_grad", 0
    payload = [build_run_config(args, cfg, dc, parent, world) if rank == 0 else None]
    if rank == 0:
        payload[0].update(
            variant="action_backbone_v1", global_batch_size=args.global_batch,
            optimizer_transition="fresh S and A+B optimizers from identical M3 weights",
            gradient_contract="CE->S only; action->A+B only; discrete generated text bridge",
            action_conditions="100 percent generated; independent 0.1 empty-condition dropout; no GT curriculum",
            lr_schedule={"warmup_steps":args.warmup, "peak_lr":args.peak_lr, "decay_steps":args.steps,
                         "decay_lr":args.decay_lr, "applies_to":"all S/A/B trainable parameters"},
            optimizer={"name":"AdamW", "betas":[0.9,0.95], "eps":1e-8, "weight_decay":1e-10,
                       "clip_gradient_norm":1.0, "clipping_groups":["S", "A+B"]},
            training_source="original M3 continuation; not initialization from Aloha obstacle or R1",
        )
        for name in ["scripts/train_backbone_gradient.py", "scripts/run_backbone_gradient_pair.py",
                     "scripts/check_backbone_gradient_checkpoint.py", "scripts/visualize_subtask_step.py",
                     "src/openpi/models_pytorch/backbone_gradient.py",
                     "src/openpi/policies/backbone_gradient_policy.py"]:
            payload[0]["sources"][name] = sha256_file(Path(name))
        payload[0]["parent_identity"] = {"checkpoint":str(args.initialize_from.resolve()),
                                         "completed_steps":parent["completed_steps"]}
    if world > 1:
        dist.broadcast_object_list(payload, src=0)
    config = payload[0]
    start, best, resume_path = 0, None, None
    counters = dict(parent["counters"], backbone=0)
    if args.resume:
        resume_path = args.output / json.loads((args.output / "latest.json").read_text())["checkpoint"]
        saved = json.loads((resume_path / "metadata.json").read_text())
        if saved["config"] != config:
            raise ValueError("Exact resume rejected changed config/source/runtime/data")
        start, best, counters = saved["completed_steps"], saved["best"], saved["counters"]
    elif rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "run_config.json", config)
        archive_runtime(config["runtime"], args.output / "runtime_sources")
        for name, digest in config["sources"].items():
            relative = Path(name).resolve().relative_to(Path.cwd().resolve())
            destination = args.output / "sources" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(name, destination)
            if sha256_file(destination) != digest:
                raise ValueError("Source changed during archival")
    if world > 1:
        dist.barrier()

    raw = data_loader.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
    dataset = SubtaskTrainingDataset(raw, dc)
    val_dc = dataclasses.replace(dc, split="val")
    validation = SubtaskTrainingDataset(data_loader.create_torch_dataset(val_dc, cfg.model.action_horizon, cfg.model), val_dc)
    vocabulary = sorted({value for value in dataset.labels_without_video() if value and value.strip()})
    model = BackboneGradientModel(PI0Pytorch(cfg.model).to(device), SubtaskDecoderConfig(**parent["config"]["decoder"]))
    safetensors.torch.load_model(model, args.initialize_from / "model.safetensors", strict=True)
    model.enable_backbone(args.mode, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, last_layers=args.last_layers)
    if resume_path:
        filename = "training_model.safetensors" if args.mode == "limited" else "model.safetensors"
        if sha256_file(resume_path / filename) != saved["training_weights_sha256"]:
            raise ValueError("Resume weights fingerprint mismatch")
        safetensors.torch.load_model(model, resume_path / filename, strict=True)
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None,
                                                         find_unused_parameters=True, gradient_as_bucket_view=True,
                                                         broadcast_buffers=False)
    plain = training.unwrap(model)
    optimizers = Optimizers(plain, world)
    if resume_path:
        saved_state = torch.load(training.checkpoint_state_path(resume_path, rank, world), map_location="cpu", weights_only=False)
        if (saved_state["rank"], saved_state["world_size"]) != (rank, world):
            raise ValueError("Resume rank/world mismatch")
        optimizers.load_state_dict(saved_state["branch_optimizers"])
        training.restore_random_state(saved_state["rng"])
        del saved_state
    sampler = StepBatchSampler(len(dataset), args.batch_size, args.accumulation, args.steps,
                               start=start, seed=args.seed, rank=rank, world_size=world)
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_subtask,
                                         num_workers=args.workers, multiprocessing_context="spawn" if args.workers else None,
                                         persistent_workers=args.workers > 0, generator=torch.Generator().manual_seed(args.seed))
    wb = None
    if rank == 0 and args.wandb:
        try:
            import wandb
            wb = wandb.init(project="agentic-openpi-pi05-subtask", name=args.output.name,
                            id=args.output.name, resume="allow", config=config, dir=str(args.output),
                            settings=wandb.Settings(x_disable_stats=True))
        except Exception as error:
            print(json.dumps({"event":"wandb_unavailable", "error":str(error)}), flush=True)

    def log(value):
        if rank == 0:
            with (args.output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(value) + "\n")
            print(json.dumps(value), flush=True)
            if wb is not None:
                wb.log(value)

    def validate(step):
        result, rows = training.evaluate(model, validation, device, samples=args.eval_samples,
                                          batch_size=args.batch_size, draws=args.eval_draws,
                                          vocabulary=vocabulary, action_enabled=True)
        log({"event":"validation", "step":step, **result})
        if rank == 0:
            write_json(args.output / f"validation_{step:06d}.json", {"metrics":result, "predictions":rows})
        return result

    log({"event":"ready", "mode":args.mode, "start":start, "global_batch":args.global_batch,
         "train_frames":len(dataset), "validation_frames":len(validation),
         "trainable_parameters":{"S":sum(p.numel() for p in plain.subtask_parameters()),
                                 "A":sum(p.numel() for p in plain.action_parameters()),
                                 "B":sum(p.numel() for p in plain.backbone_parameters())}})
    if not args.resume:
        # LoRA construction consumes RNG; reset it so the two arms start with
        # the same flow noise/time and dropout streams, not merely the same seed label.
        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        validate(0)
    iterator = iter(loader)
    model.train()
    for step in range(start, args.steps):
        begun = time.perf_counter()
        optimizers.zero_grad()
        totals = torch.zeros(5, device=device, dtype=torch.float64)
        for micro in range(args.accumulation):
            batch = next(iterator).to(device)
            if step == 0 and micro == 0 and rank == 0:
                first_batch = batch
            dropped = np.random.random(len(batch.labels)) < args.condition_dropout
            sync = model.no_sync() if world > 1 and micro + 1 < args.accumulation else contextlib.nullcontext()
            with sync:
                result = model(batch, drop_condition=dropped)
                loss = result["loss_subtask"] + result["loss_action"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at step {step}")
                (loss / args.accumulation).backward()
            totals[0] += result["loss_subtask"].detach() / args.accumulation
            totals[1] += result["loss_action"].detach() / args.accumulation
            totals[2:] += torch.tensor([result["generated_count"], result["invalid_generation_count"],
                                       result["empty_condition_count"]], device=device)
        if step == start:
            b_grads = [p.grad for p in plain.backbone_parameters() if p.grad is not None]
            if not b_grads or not any(bool(torch.count_nonzero(value)) for value in b_grads):
                raise AssertionError("No nonzero action gradient reached B")
            if args.mode == "full":
                vision = plain.base.paligemma_with_expert.paligemma.vision_tower
                if not any(p.grad is not None and bool(torch.count_nonzero(p.grad)) for p in vision.parameters()):
                    raise AssertionError("Full B did not receive visual gradients")
            log({"event":"backbone_gradient_verified", "step":step + 1, "mode":args.mode,
                 "B_tensors_with_gradient":len(b_grads)})
        lr = cosine_lr(step, warmup=args.warmup, decay_steps=args.steps, peak=args.peak_lr, end=args.decay_lr)
        norms = optimizers.step(lr)
        optimizers.zero_grad()
        for key in ("subtask", "action", "backbone"):
            counters[key] += 1
        if world > 1:
            dist.all_reduce(totals)
        totals[:2] /= world
        completed = step + 1
        log({"event":"train", "step":completed, "loss_subtask":float(totals[0]), "loss_action":float(totals[1]),
             "generated_count":int(totals[2]), "invalid_generation_count":int(totals[3]),
             "empty_condition_count":int(totals[4]), "lr":lr, "grad_norms":norms,
             "seconds":time.perf_counter() - begun})
        if completed == 1 and rank == 0 and not args.engineering_smoke:
            from visualize_subtask_step import render_first_step
            render_first_step(plain, first_batch, dc, args.output / "first_update", step=1,
                              origin="first actual minibatch; policy after first optimizer update", limit=1)
            del first_batch
        if completed == 50 and rank == 0:
            write_json(args.output / "milestone_50.json", {"completed_steps":50, "time":time.time()})
        if completed % args.checkpoint_every == 0 or completed in {args.steps, args.stop_after}:
            metrics = validate(completed)
            score = [metrics["flow_generated_native14_normalized"]]
            improved = best is None or score < best["selection_score"]
            if improved:
                best = {"step":completed, "selection_score":score}
            target = save_checkpoint(model, optimizers, args, config, completed, best, counters, assets)
            if rank == 0 and improved:
                write_json(args.output / "best.json", {"checkpoint":target.name, **best})
            log({"event":"checkpoint", "step":completed, "path":str(target), "best":best})
        if args.stop_after and completed >= args.stop_after:
            break
    log({"event":"complete" if completed == args.steps else "stopped_at_checkpoint",
         "completed_steps":completed, "best":best, "counters":counters})
    if wb is not None:
        wb.finish()
    if world > 1:
        dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["limited", "full"], required=True)
    p.add_argument("--initialize-from", type=Path, default=Path("checkpoints/pi05_piper_stage1/m3_pilot_seed42/step_003500"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--global-batch", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accumulation", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--peak-lr", type=float, default=2.5e-5)
    p.add_argument("--decay-lr", type=float, default=2.5e-6)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--eval-samples", type=int, default=128)
    p.add_argument("--eval-draws", type=int, default=2)
    p.add_argument("--condition-dropout", type=float, default=0.1)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--last-layers", type=int, default=2)
    p.add_argument("--precision", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--memory-fraction", type=float, default=0.55,
                   help="Bound this rank's CUDA allocator while sharing GPUs with the retained RobotWin task")
    p.add_argument("--engineering-smoke", action="store_true")
    p.add_argument("--stop-after", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--wandb", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
