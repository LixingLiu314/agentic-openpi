"""Official fresh limited B + recurrent S, with actor text only in reach labels."""

import argparse
import contextlib
import dataclasses
import datetime
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
from train_subtask_pytorch import StepBatchSampler

from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.backbone_gradient import cosine_lr
from openpi.models_pytorch.official_backbone_gradient import initialize_official, tensor_digest
from openpi.models_pytorch.decision_recurrent import DecisionRecurrentModel, VARIANT, SCHEMA_VERSION, DISPLAY_SET, EXPERIMENTS
from openpi.models_pytorch.semantic_recurrent import stable_training_frames
from openpi.training.decision_data import ASSETS, load_assets, DecisionStreamSampler, GroundingSamples, row_weights
from openpi.training.reach_arm_data import data_config, ANNOTATIONS_SHA256
from openpi.training.recurrent_sequence import SequenceDataset, EpisodeStreamSampler, episode_rows, collate_sequence
from openpi.training.reach_arm_evaluation import evaluate_recurrent
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.subtask_decoder import SubtaskDecoderConfig
from openpi.training import config as config_lib, data_loader, hierarchy_training as training
from openpi.training.official_gradient_provenance import build_run_config
from openpi.training.runtime_provenance import archive_runtime
from openpi.training.stage1_data import sha256_file
from openpi.training.stage1_optimizer import MixedPrecisionZeroAdamW
from openpi.training.subtask_batch import SubtaskTrainingDataset, collate_subtask
from openpi.training.decoded_video_cache import create_cached_dataset, collate_pinned_subtask, worker_init
from openpi.training.wandb_standard import configure_run, log_event


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class Optimizers:
    def __init__(self, model, world):
        self.observed = {"action": model.action_parameters(), "backbone": model.backbone_parameters()}
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
        observed = {name: float(torch.linalg.vector_norm(torch.stack([
            torch.linalg.vector_norm(p.grad.detach().float()) for p in params if p.grad is not None])))
            for name, params in self.observed.items()}
        norms = {name: float(torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True))
                 for name, params in self.parameters.items()}
        norms.update(observed)
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


def save_checkpoint(model, optimizers, args, config, step, best, counters, assets, carry):
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
                "rank": rank, "world_size": world, "recurrent_state": None if carry is None else carry.detach().cpu()}, training.checkpoint_state_path(temporary, rank, world))
    if rank == 0:
        safetensors.torch.save_file(model.deployment_state(), temporary / "model.safetensors")
        if args.mode == "limited":
            safetensors.torch.save_model(model, temporary / "training_model.safetensors")
        weights_hash = sha256_file(temporary / "model.safetensors")
        metadata = {"schema_version": SCHEMA_VERSION, "stage": "recurrent_subtask", "variant": VARIANT,
                    "completed_steps": step, "config": config, "best": best, "counters": counters,
                    "weights_sha256": weights_hash,
                    "training_weights_sha256": sha256_file(temporary / "training_model.safetensors")
                    if args.mode == "limited" else weights_hash,
                    "export": "Complete B/S/A weights; optional LoRA merged; official-initialized experiment"}
        write_json(temporary / "metadata.json", metadata)
        shutil.copytree(assets, temporary / "assets/eggplant_potato")
        shutil.copytree(ASSETS, temporary / "assets/decision")
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
    if not args.engineering_smoke and (args.steps != 5000 or
            (args.global_batch, args.seed, args.checkpoint_every, args.warmup, args.peak_lr, args.decay_lr)
            != (256, 42, 500, 500, 2.5e-5, 2.5e-6)):
        raise ValueError("Formal configuration differs from user recipe")
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    cfg = config_lib.get_config("pi05_piper_stage1")
    cfg = dataclasses.replace(cfg, model=Pi0Config(pi05=True, dtype=args.precision, pytorch_compile_mode=None))
    dc = data_config(cfg.model)
    assets = Path(dc.split_manifest).parent
    args.stage, args.overfit_samples = "recurrent_subtask", 0
    if (args.mode, args.arm, args.accumulation, args.unroll, args.lora_rank, args.lora_alpha, args.last_layers,
            args.condition_dropout) != ("limited", "recurrent", 1, 4, 16, 32, 2, .1):
        raise ValueError("Candidate fixes limited B, recurrent S and unchanged generated-condition recipe")
    payload = [build_run_config(args, cfg, dc, world) if rank == 0 else None]
    if world > 1:
        dist.broadcast_object_list(payload, src=0)
    config = payload[0]
    config.update(variant=VARIANT, stage="recurrent_subtask", experiment=args.experiment,
                  display_set=DISPLAY_SET, experiment_family="recurrent-decision-grounding-pair",
                  memory_tokens=4, memory_input="observations only; no teacher-forced labels or S-to-A hidden features",
                  sampling="task/layout/actor-balanced causal episode streams from frame0; include object-reach starts; truncated gradients per update",
                  checkpoint_selection="final step5000 is primary; best validation score is optional comparator",
                  dataset="eggplant_potato_reach_arm_v1", dataset_root=dc.local_root, dataset_repo_id=dc.repo_id,
                  dataset_split_manifest=dc.split_manifest, label_version="reach_arm_v1",
                  annotations_sha256=ANNOTATIONS_SHA256,
                  budget_reason="User authorized two new experiments, each 5000 updates",
                  semantic_weight=4.0, action_rank_coefficient=.1,
                  action_rank_margin=.01, action_rank_max_pairs_per_rank=4,
                  action_rank_condition="exact-correct generated and non-dropped only; train-vocabulary negatives",
                  loss_metrics="standard CE is unweighted, action flow is main all32; auxiliary losses have separate names",
                  gradient_observability="A and B norms measured separately before unchanged joint A+B norm1 clipping",
                  evaluation_version="reach_arm_causal_v1: all episode starts, first second and reach starts included in flow frames")
    decisions,points,decision_ready=load_assets()
    train_prompts=sorted({r["prompt"] for r in decisions if r["split"]=="train"})
    config.update(decision_assets_sha256=sha256_file(ASSETS/"READY.json"),
                  decision_asset_files=decision_ready["files"], global_prompts=train_prompts,
                  prefix_weights={"first25":2.0,"remaining25":1.0,"normalization":"mean weight"},
                  decision_row_weight=2.0, rank_per_type=2,
                  grounding_coefficient=.05 if args.experiment=="decision_grounded" else 0.,
                  grounding_aux_per_rank=2 if args.experiment=="decision_grounded" else 0,
                  grounding_spec="S-only task-conditioned head, cam_high16x16, sigma1patch; predicted S context token; no GT point or hidden feature to A")
    raw = create_cached_dataset(dc, cfg.model.action_horizon, args.decoded_cache)
    dataset = SubtaskTrainingDataset(raw, dc)
    vocabulary = sorted({value for value in dataset.labels_without_video() if value and value.strip()})
    config["semantic_vocabulary"] = vocabulary
    stable_frames = stable_training_frames(raw.hf_dataset,horizon=25,past=7)
    config["action_rank_stability"] = "same training label across past7/current/future24 frames, clipped at episode ends"
    config["action_rank_stable_train_frames"] = len(stable_frames)
    config["engineering_condition_fixture"] = args.engineering_smoke
    new_sources = ["scripts/train_semantic_recurrent.py", "src/openpi/models_pytorch/semantic_recurrent.py",
                   "src/openpi/policies/semantic_recurrent_policy.py", "scripts/check_semantic_recurrent.py",
                   "scripts/run_semantic_pair.py", "scripts/evaluate_semantic_recurrent.py","scripts/train_reach_arm_candidate.py", "src/openpi/models_pytorch/recurrent_subtask.py",
                   "src/openpi/training/recurrent_sequence.py", "src/openpi/training/recurrent_evaluation.py",
                   "src/openpi/policies/recurrent_subtask_policy.py", "src/openpi/policies/reach_arm_subtask_policy.py",
                   "src/openpi/training/reach_arm_data.py", "src/openpi/training/reach_arm_evaluation.py",
                   "scripts/check_reach_arm_candidate.py", "scripts/run_reach_arm_candidate.py",
                   "scripts/evaluate_reach_arm_candidate.py", "scripts/build_reach_arm_cache.py",
                   "scripts/verify_reach_arm_cache.py"]
    new_sources += ["scripts/train_decision_recurrent.py","src/openpi/models_pytorch/decision_recurrent.py",
                    "src/openpi/training/decision_data.py","src/openpi/training/decision_assets.py","src/openpi/policies/decision_recurrent_policy.py",
                    "scripts/check_decision_recurrent.py","scripts/evaluate_decision_recurrent.py",
                    "scripts/run_decision_pair.py","scripts/prepare_decision_assets.py","scripts/decision_data_test.py","scripts/verify_decision_startup.py"]
    config["sources"].update({name: sha256_file(Path(name)) for name in new_sources})
    start, best, resume_path = 0, None, None
    counters = {"subtask":0, "action":0, "backbone":0}
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

    val_dc = dataclasses.replace(dc, split="val")
    validation = SubtaskTrainingDataset(create_cached_dataset(val_dc, cfg.model.action_horizon, args.decoded_cache), val_dc)
    vocabulary = sorted({value for value in dataset.labels_without_video() if value and value.strip()})
    model = initialize_official(cfg.model, SubtaskDecoderConfig(), args.initialize_from / "model.safetensors",
                                seed=args.seed, device=device)
    reference_s_hash = tensor_digest(model.decoder.state_dict())
    reference_keys = set(model.decoder.state_dict())
    initial = model
    model = DecisionRecurrentModel(initial.base, SubtaskDecoderConfig(), grounded=args.experiment=="decision_grounded", recurrent=True,
                                  seed=args.seed, unroll=args.unroll)
    del initial
    model.configure_decisions(vocabulary, train_prompts, args.experiment)
    initial_s_hash = tensor_digest({k: v for k, v in model.decoder.state_dict().items() if k in reference_keys})
    if initial_s_hash != reference_s_hash:
        raise AssertionError("Shared S initialization differs from the official experiment decoder")
    initial_complete_s_hash = tensor_digest(model.decoder.state_dict())
    initial_common_s_hash = tensor_digest({k:v for k,v in model.decoder.state_dict().items() if not k.startswith("ground_")})
    if world > 1:
        hashes = [None] * world
        dist.all_gather_object(hashes, initial_s_hash)
        if len(set(hashes)) != 1:
            raise AssertionError("Ranks have different initial S weights")
    if rank == 0 and not args.resume:
        write_json(args.output / "initialization.json", {
            "initialization":"official_pi05_base", "official_weights_sha256":config["official_weights_sha256"],
            "decoder_sha256":initial_s_hash, "complete_decoder_sha256":initial_complete_s_hash, "common_recurrent_decoder_sha256":initial_common_s_hash, "arm":args.arm, "decoder_seed":args.seed, "inherited_training_updates":0,
            "initial_counters":counters, "all_ranks_identical_S":True,
        })
    if rank == 0 and not args.resume:
        write_json(args.output / "semantic_rules.json", model.semantic_audit)
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
    carry = None
    if resume_path:
        saved_state = torch.load(training.checkpoint_state_path(resume_path, rank, world), map_location="cpu", weights_only=False)
        if (saved_state["rank"], saved_state["world_size"]) != (rank, world):
            raise ValueError("Resume rank/world mismatch")
        optimizers.load_state_dict(saved_state["branch_optimizers"])
        training.restore_random_state(saved_state["rng"])
        carry = saved_state["recurrent_state"]
        if carry is not None:
            carry = carry.to(device)
        restored = {"rank": rank, "step": start, "carry_shape": None if carry is None else list(carry.shape),
                    "carry_sha256": None if carry is None else tensor_digest({"carry":carry}),
                    "optimizer_branches": sorted(saved_state["branch_optimizers"])}
        assert carry is not None and list(carry.shape) == [args.batch_size//args.unroll,4,512]
        assert restored["carry_sha256"] == tensor_digest({"carry":saved_state["recurrent_state"]})
        if world > 1:
            gathered = [None] * world
            dist.all_gather_object(gathered, restored)
        else:
            gathered = [restored]
        if rank == 0:
            write_json(args.output / f"resume_state_{start:06d}.json", {"passed":True,"ranks":gathered})
        del saved_state
    grounding = GroundingSamples(dataset,points) if plain.grounded else None
    sampler = DecisionStreamSampler(episode_rows(raw.hf_dataset), decisions, engineering=args.engineering_smoke, batch_size=args.batch_size,
                                    unroll=args.unroll, steps=args.steps, start=start, seed=args.seed, rank=rank)
    loader = torch.utils.data.DataLoader(SequenceDataset(dataset), batch_sampler=sampler, collate_fn=collate_sequence,
                                         num_workers=args.workers, multiprocessing_context="spawn" if args.workers else None,
                                         persistent_workers=args.workers > 0, pin_memory=device.type == "cuda",
                                         worker_init_fn=worker_init, prefetch_factor=2 if args.workers else None,
                                         generator=torch.Generator().manual_seed(args.seed))
    wb = None
    if rank == 0 and args.wandb:
        try:
            import wandb
            wb = wandb.init(entity="xiahy23-tsinghua-university", project="agentic-openpi-pi05-subtask",
                             name=f"{args.experiment} | s{args.seed}",
                             id=args.output.name, resume="allow", config=config, dir=str(args.output),
                             group=config["display_set"], job_type="engineering" if args.engineering_smoke else "train",
                             tags=["display-v1", "engineering" if args.engineering_smoke else "research", "reach-arm", "limited", "recurrent", "seed42"],
                             settings=wandb.Settings(x_disable_stats=True, disable_git=True, save_code=False))
            configure_run(wb)
            wb.define_metric("decision/*", step_metric="trainer/step", step_sync=False)
            wb.define_metric("semantic/*", step_metric="trainer/step", step_sync=False)
            wb.define_metric("sequence/*", step_metric="trainer/step", step_sync=False)
            wb.summary.update({"source_status":"running", "last_optimizer_step":start})
        except Exception as error:
            raise RuntimeError("Required W&B initialization failed") from error

    def log(value):
        if rank == 0:
            with (args.output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(value) + "\n")
            print(json.dumps(value), flush=True)
            if wb is not None:
                if value.get("event") != "train" or value["step"] == 1 or value["step"] % 10 == 0:
                    log_event(wb, value, config)
                if value.get("event") == "train" and (value["step"] == 1 or value["step"] % 10 == 0):
                    wb.log({"trainer/step":value["step"], "perf/data_wait_seconds_max":value["data_wait_seconds_max"],
                            "optim/grad_norm_a":value["grad_norms"]["action"],
                            "optim/grad_norm_b":value["grad_norms"]["backbone"],
                            "semantic/weighted_ce":value["semantic_ce"],
                            "semantic/action_rank_hinge":value["action_rank_hinge"],
                            "semantic/action_rank_pairs":value["rank_pairs"],
                            **{"decision/"+k:value[k] for k in ["rank_arm_pairs","rank_object_pairs","rank_arm_hinge","rank_object_hinge",
                               "grounding_ce","grounding_distance_px","first25_native_mse","action_weighted_flow"]}})
                    wb.summary["last_optimizer_step"] = value["step"]
                if value.get("event") in ("checkpoint", "complete"):
                    wb.summary.update({"source_status":value["event"], "best_checkpoint":value.get("best"),
                                       "completed_optimizer_steps":value.get("completed_steps", value.get("step"))})

    def validate(step):
        result, rows = evaluate_recurrent(model, validation, device, samples=args.eval_samples,
                                          draws=args.eval_draws, vocabulary=vocabulary,
                                          engineering=args.engineering_smoke)
        log({"event":"validation", "step":step, **result})
        if wb is not None:
            wb.log({"trainer/step":step, **{"sequence/"+k:v for k,v in result.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)}})
        if rank == 0:
            write_json(args.output / f"validation_{step:06d}.json", {"metrics":result, "predictions":rows})
        return result

    log({"event":"ready", "mode":args.mode, "arm":args.arm, "start":start, "global_batch":args.global_batch,
         "train_frames":len(dataset), "validation_frames":len(validation),
         "trainable_parameters":{"S":sum(p.numel() for p in plain.subtask_parameters()),
                                 "A":sum(p.numel() for p in plain.action_parameters()),
                                 "B":sum(p.numel() for p in plain.backbone_parameters())}})
    if not args.resume:
        # LoRA construction consumes RNG; reset it so all three arms start with
        # the same flow noise/time and dropout streams, not merely the same seed label.
        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        validate(0)
    startup_process = None
    iterator = iter(loader)
    model.train()
    for step in range(start, args.steps):
        begun = time.perf_counter()
        optimizers.zero_grad()
        totals = torch.zeros(16, device=device, dtype=torch.float64)
        data_wait = 0.0
        for micro in range(args.accumulation):
            wait_started = time.perf_counter()
            host_batch = next(iterator)
            data_wait += time.perf_counter() - wait_started
            if step == start and micro == 0 and device.type == "cuda" and not host_batch.actions.is_pinned():
                raise AssertionError("Pinned batch transport was not activated")
            batch = host_batch.to(device, non_blocking=True)
            if step == 0 and micro == 0 and rank == 0:
                first_batch = batch
            dropped = np.random.random(len(batch.labels)) < args.condition_dropout
            rank_stable = [(int(ep), int(frame)) in stable_frames for ep, frame in zip(host_batch.episode_indices, host_batch.frame_indices)]
            if config["engineering_condition_fixture"]:
                rank_stable = [True] * len(batch.labels)
            weights=row_weights(host_batch.episode_indices,host_batch.frame_indices,decisions)
            ground_batch,ground_points=(None,None) if grounding is None else grounding.batch(grounding.select(step,rank),device)
            if config["engineering_condition_fixture"]:dropped=[False]*len(batch.labels)
            sync = model.no_sync() if world > 1 and micro + 1 < args.accumulation else contextlib.nullcontext()
            with sync:
                fixture_context = contextlib.nullcontext()
                if config["engineering_condition_fixture"]:
                    # Engineering-only capacity fixture: force the extra action branch
                    # at micro32 before S has learned words. Never enabled in formal runs.
                    from unittest.mock import patch
                    from openpi.models_pytorch.subtask_decoder import SubtaskGeneration
                    fixture = SubtaskGeneration(batch.target_ids, batch.target_mask,
                        torch.ones(len(batch.labels), dtype=torch.bool, device=device), torch.zeros(len(batch.labels), device=device))
                    fixture_context = patch.object(plain.decoder, "generate_projected", return_value=fixture)
                with fixture_context:
                    result = model(batch, carry=carry, drop_condition=dropped, rank_offset=step, rank_stable_mask=rank_stable, decision_weights=weights,
                                   ground_batch=ground_batch,ground_points=ground_points)
                carry = result["carry"]
                loss = result["loss_subtask"] + result["loss_action"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at step {step}")
                (loss / args.accumulation).backward()
            totals[0] += result["loss_subtask_unweighted"].detach() / args.accumulation
            totals[1] += result["loss_action_main"].detach() / args.accumulation
            totals[5] += result["loss_semantic"].detach()
            totals[6] += result["loss_action_rank"].detach()
            totals[7] += result["rank_pairs"]
            for index,key in enumerate(["rank_arm_pairs","rank_object_pairs","rank_arm_hinge","rank_object_hinge",
                    "grounding_ce","grounding_distance_px","first25_native_mse","loss_action_weighted"],8):
                totals[index] += result[key]
            totals[2:5] += torch.tensor([result["generated_count"], result["invalid_generation_count"],
                                       result["empty_condition_count"]], device=device)
        if step == start:
            b_grads = [p.grad for p in plain.backbone_parameters() if p.grad is not None]
            if args.mode == "frozen":
                if b_grads or any(p.requires_grad or p.grad is not None for p in plain.base.paligemma_with_expert.paligemma.parameters()):
                    raise AssertionError("Frozen B acquired trainable parameters or gradients")
            elif not b_grads or not any(bool(torch.count_nonzero(value)) for value in b_grads):
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
        for key in ("subtask", "action") + (() if args.mode == "frozen" else ("backbone",)):
            counters[key] += 1
        if world > 1:
            dist.all_reduce(totals)
        wait_tensor = torch.tensor(data_wait, device=device, dtype=torch.float64)
        if world > 1:
            dist.all_reduce(wait_tensor, op=dist.ReduceOp.MAX)
        totals[:2] /= world
        totals[5:7] /= world
        totals[10:] /= world
        completed = step + 1
        log({"event":"train", "step":completed, "loss_subtask":float(totals[0]), "loss_action":float(totals[1]),
             "generated_count":int(totals[2]), "invalid_generation_count":int(totals[3]),
             "empty_condition_count":int(totals[4]), "semantic_ce":float(totals[5]),
             "action_rank_hinge":float(totals[6]), "rank_pairs":int(totals[7]),
             "rank_arm_pairs":int(totals[8]),"rank_object_pairs":int(totals[9]),
             **{k:float(totals[i]) for i,k in enumerate(["rank_arm_hinge","rank_object_hinge","grounding_ce",
                "grounding_distance_px","first25_native_mse","action_weighted_flow"],10)}, "lr":lr, "grad_norms":norms,
             "data_wait_seconds_max":float(wait_tensor),
             "seconds":time.perf_counter() - begun})
        if completed == 1 and rank == 0 and not args.engineering_smoke:
            from visualize_subtask_step import render_first_step
            render_first_step(plain, first_batch, dc, args.output / "first_update", step=1,
                               origin="first actual sequence minibatch; reset-memory policy after first optimizer update", limit=1)
            if wb is not None:
                for key, suffix in (("inputs", "inputs.png"), ("actions", "actions.png")):
                    matches = list((args.output / "first_update").glob("*" + suffix))
                    if matches:
                        wb.log({"trainer/step":1, "media/first_update_" + key:wandb.Image(str(matches[0]))})
            del first_batch
        if completed == 10 and rank == 0 and not args.engineering_smoke:
            with (args.output/"startup_verification.log").open("x") as stream:
                startup_process=subprocess.Popen([os.path.abspath(".stage1_staging/wandb_display_env/bin/python"),
                    "scripts/verify_decision_startup.py","--output",str(args.output)],stdout=stream,stderr=subprocess.STDOUT)
        if completed == 50 and rank == 0:
            write_json(args.output / "milestone_50.json", {"completed_steps":50, "time":time.time()})
        if completed % args.checkpoint_every == 0 or completed in {args.steps, args.stop_after}:
            if rank == 0 and not args.engineering_smoke and completed>=500:
                if startup_process is not None and startup_process.poll() not in (None,0):
                    raise RuntimeError("Startup W&B verification failed; see startup_failure.json")
                if not (args.output/"startup_verified.json").exists():
                    raise RuntimeError("Startup verification did not finish before first save")
            metrics = validate(completed)
            score = [metrics["selection_score"]]
            improved = best is None or score < best["selection_score"]
            if improved:
                best = {"step":completed, "selection_score":score}
            target = save_checkpoint(model, optimizers, args, config, completed, best, counters, assets, carry)
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
    p.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    p.add_argument("--mode", choices=["limited"], default="limited")
    p.add_argument("--arm", choices=["recurrent"], default="recurrent")
    p.add_argument("--unroll", type=int, default=4)
    p.add_argument("--initialize-from", type=Path, default=Path("checkpoints/pi05_base_pytorch"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--global-batch", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--accumulation", type=int, default=1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--decoded-cache", type=Path, required=True)
    p.add_argument("--eval-batch-size", type=int, default=2,
                   help="Preserve original fixed validation batching independently of training packing")
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
    p.add_argument("--memory-fraction", type=float, default=0.9,
                   help="User-authorized larger-batch capacity; unrelated tasks remain protected")
    p.add_argument("--engineering-smoke", action="store_true")
    p.add_argument("--stop-after", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--wandb", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
