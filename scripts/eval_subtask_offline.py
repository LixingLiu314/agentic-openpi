"""Dense validation semantics, paired action interventions and complete GPU policy timing.

Labels enter scoring and the explicitly named offline oracle only. Normal and
no-image generations start at BOS without targets. The test split is deliberately
not exposed during protocol development.
"""

import argparse
from collections import Counter
import dataclasses
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
import torch.distributed as dist

from openpi.models.pi0_config import Pi0Config
from openpi.policies.piper_policy import JOINT_MASK
from openpi.policies.subtask_policy import create_subtask_policy
from openpi.shared import normalize
from openpi.training import config
from openpi.training import data_loader
from openpi.training.evaluation_protocol import validate_evaluation_request
from openpi.training.hierarchy_training import distributed_context
from openpi.training.stage1_data import sha256_file
from openpi.training.subtask_batch import SubtaskTrainingDataset
from openpi.training.subtask_batch import collate_subtask
from openpi.training.subtask_evaluation import aggregate_actions
from openpi.training.subtask_evaluation import native_action_metrics
from openpi.training.subtask_evaluation import semantic_metrics
from openpi.training.subtask_evaluation import shuffled_conditions
from openpi.training.subtask_evaluation import transition_metrics


def gather_rows(rows, world):
    if world == 1:
        return rows
    partitions = [None] * world
    dist.all_gather_object(partitions, rows)
    return [row for partition in partitions for row in partition]


def fixed_indices(length, count):
    return np.arange(length) if count == 0 else np.unique(np.linspace(0, length - 1, min(count, length), dtype=int))


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n")


@torch.no_grad()
def evaluate(args):
    if (
        min(args.batch_size, args.draws, args.num_steps) < 1
        or min(args.semantic_samples, args.action_samples, args.latency_samples) < 0
    ):
        raise ValueError("Invalid evaluation counts")
    torch.set_num_threads(args.cpu_threads)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if args.device == "cuda" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group(
            "nccl" if device.type == "cuda" else "gloo", **({"device_id": device} if device.type == "cuda" else {})
        )
    rank, world = distributed_context()
    metadata = json.loads((args.checkpoint / "metadata.json").read_text())
    test_protocol = validate_evaluation_request(args, metadata)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()
    policy = create_subtask_policy(args.checkpoint, device=device, num_steps=args.num_steps, allow_engineering=True)
    model = policy.model.eval()
    cfg = config.get_config("pi05_piper_stage1")
    model_config = Pi0Config(**metadata["config"]["model"])
    stats = normalize.load(args.checkpoint / "assets/eggplant_potato")
    dc = dataclasses.replace(cfg.data.create(cfg.assets_dirs, model_config), split=args.split, norm_stats=stats)
    raw = data_loader.create_torch_dataset(dc, model_config.action_horizon, model_config)
    dataset = SubtaskTrainingDataset(raw, dc)
    labels = dataset.labels_without_video()
    episode_lengths = Counter(int(x) for x in raw.hf_dataset["episode_index"])
    # Vocabulary derives from the immutable training split, not validation predictions.
    train_dc = dataclasses.replace(dc, split="train")
    train_raw = data_loader.create_torch_dataset(train_dc, model_config.action_horizon, model_config)
    vocabulary = sorted(set(train_raw.hf_dataset["subtask"]))
    task_vocabulary = {}
    for task, label in zip(train_raw.hf_dataset["task"], train_raw.hf_dataset["subtask"], strict=True):
        task_vocabulary.setdefault(task, set()).add(label)
    del train_raw
    selected = fixed_indices(len(dataset), args.semantic_samples)
    if args.action_samples:
        selected = np.union1d(selected, fixed_indices(len(dataset), args.action_samples))
    indices = selected[rank::world]
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, indices.tolist()),
        batch_size=args.batch_size,
        collate_fn=collate_subtask,
        num_workers=args.workers,
        multiprocessing_context="spawn" if args.workers else None,
    )
    rows, no_image_rows, seen = [], [], 0
    for cpu_batch in loader:
        batch = cpu_batch.to(device)
        context = model.prepare_context(batch.observation, batch.global_prompts)
        texts, statuses, generation = model.generate_subtask(context)
        current_indices = indices[seen : seen + len(texts)]
        current = [
            {
                "index": int(index),
                "episode": int(episode),
                "frame": int(frame),
                "episode_length": episode_lengths[int(episode)],
                "task": task,
                "label": label,
                "prediction": text,
                "status": status,
                "sequence_score": float(score),
            }
            for index, episode, frame, task, label, text, status, score in zip(
                current_indices,
                batch.episode_indices.tolist(),
                batch.frame_indices.tolist(),
                batch.global_prompts,
                batch.labels,
                texts,
                statuses,
                generation.mean_log_probability.tolist(),
                strict=True,
            )
        ]
        rows.extend(current)
        if args.no_image:
            blank = dataclasses.replace(
                batch.observation,
                images={name: torch.full_like(value, -1) for name, value in batch.observation.images.items()},
            )
            blank_context = model.prepare_context(blank, batch.global_prompts)
            blank_texts, blank_status, blank_generation = model.generate_subtask(blank_context)
            no_image_rows.extend(
                dict(row, prediction=text, status=status, sequence_score=float(score))
                for row, text, status, score in zip(
                    current, blank_texts, blank_status, blank_generation.mean_log_probability.tolist(), strict=True
                )
            )
        seen += len(texts)
        if rank == 0 and seen % (args.batch_size * 25) == 0:
            print(
                json.dumps({"event": "semantic_progress", "rank0_frames": seen, "rank0_total": len(indices)}),
                flush=True,
            )
    write_json(args.output / f"semantics_rank_{rank:03d}.json", rows)
    rows = gather_rows(rows, world)
    if args.no_image:
        write_json(args.output / f"no_image_rank_{rank:03d}.json", no_image_rows)
        no_image_rows = gather_rows(no_image_rows, world)
    row_by_index = {row["index"]: row for row in rows}
    action_indices = fixed_indices(len(dataset), args.action_samples) if args.action_samples else []
    action_semantics = [row_by_index[int(index)] for index in action_indices]
    shuffled = shuffled_conditions(action_semantics)
    action_rows = []
    native_windows, target_windows = [], []
    for raw_index in action_indices[rank::world]:
        index = int(raw_index)
        sample = raw[index]
        batch = collate_subtask([dataset[index]]).to(device)
        context = model.prepare_context(batch.observation, batch.global_prompts)
        generated = row_by_index[index]["prediction"]
        target = np.asarray(sample["action"], dtype=np.float64)
        state = np.asarray(sample["observation.state"], dtype=np.float64)
        episode, frame = int(sample["episode_index"]), int(sample["frame_index"])
        valid = min(model_config.action_horizon, episode_lengths[episode] - frame)
        crossing = any(label != sample["subtask"] for label in labels[index : index + valid])
        conditions = {"generated": generated, "oracle": sample["subtask"], "drop": "", "shuffle": shuffled[index]}
        for condition_name, condition in conditions.items():
            prefix = model.action_prefix(context, [condition])
            for draw in range(args.draws):
                generator = torch.Generator().manual_seed(100000 + index * args.draws + draw)
                noise = torch.randn(batch.actions.shape, generator=generator).to(device)
                flow_time = np.random.default_rng(200000 + index * args.draws + draw).beta(1.5, 1) * 0.999 + 0.001
                flow_error = model.action_loss(
                    context,
                    prefix,
                    batch.actions,
                    noise=noise,
                    time=torch.tensor([flow_time], device=device, dtype=torch.float32),
                    reduction="none",
                )
                prediction = (
                    model.sample_actions_from_prefix(context, prefix, noise=noise, num_steps=args.num_steps)[0, :, :14]
                    .float()
                    .cpu()
                    .numpy()
                )
                native = (prediction + 1) * 0.5 * (stats["actions"].q99 - stats["actions"].q01 + 1e-6) + stats[
                    "actions"
                ].q01
                native[:, np.asarray(JOINT_MASK)] += state[np.asarray(JOINT_MASK)]
                native_windows.append(native)
                target_windows.append(target)
                action_rows.append(
                    {
                        "index": index,
                        "episode": episode,
                        "frame": frame,
                        "task": sample["task"],
                        "draw": draw,
                        "condition_mode": condition_name,
                        "condition": condition,
                        "generated": generated,
                        "condition_changed": condition != generated,
                        "crosses_subtask_boundary": crossing,
                        "valid_horizon": valid,
                        "flow_native14_normalized": float(flow_error[..., :14].mean()),
                        "flow_all32": float(flow_error.mean()),
                        **native_action_metrics(native, target, valid_horizon=valid),
                    }
                )
    if action_rows:
        np.savez_compressed(
            args.output / f"native_rank_{rank:03d}.npz",
            predicted=np.stack(native_windows),
            target=np.stack(target_windows),
            index=np.asarray([row["index"] for row in action_rows]),
            draw=np.asarray([row["draw"] for row in action_rows]),
            condition=np.asarray([row["condition_mode"] for row in action_rows]),
            valid_horizon=np.asarray([row["valid_horizon"] for row in action_rows]),
        )
    write_json(args.output / f"actions_rank_{rank:03d}.json", action_rows)
    action_rows = gather_rows(action_rows, world)
    timing_rows = []
    latency_indices = fixed_indices(len(dataset), args.latency_samples)[rank::world] if args.latency_samples else []
    for iteration, index in enumerate(latency_indices):
        sample = raw[int(index)]
        observation = {
            "images": {
                camera: np.asarray(sample[f"observation.images.{camera}"])
                for camera in ["cam_high", "cam_left_wrist", "cam_right_wrist"]
            },
            "state": np.asarray(sample["observation.state"]),
            "prompt": sample["task"],
        }
        noise = torch.randn(
            (model_config.action_horizon, model_config.action_dim),
            generator=torch.Generator().manual_seed(800000 + int(index)),
        ).numpy()
        if iteration == 0:
            for _ in range(3):
                policy.infer(observation, noise=noise)
        output = policy.infer(observation, noise=noise)
        timing_rows.append(
            {
                "index": int(index),
                "rank": rank,
                "subtask": output["subtask"],
                "status": output["subtask_status"],
                **output["policy_timing"],
            }
        )
    write_json(args.output / f"timing_rank_{rank:03d}.json", timing_rows)
    timing_rows = gather_rows(timing_rows, world)
    if rank == 0:
        report = {
            "checkpoint": str(args.checkpoint),
            "evaluation_config": {
                key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
            },
            "native_action_archives": "native_rank_*.npz preserve paired full windows for selecting a common execution length; absent when actions disabled",
            "checkpoint_stage": metadata["stage"],
            "engineering_checkpoint": metadata["config"]["engineering_smoke"],
            "weights_sha256": sha256_file(args.checkpoint / "model.safetensors"),
            "split": args.split,
            "test_protocol": test_protocol,
            "split_sha256": metadata["config"]["split_sha256"],
            "norm_sha256": metadata["config"]["norm_sha256"],
            "world_size": world,
            "flow_steps": args.num_steps,
            "noise_draws": args.draws,
            "sources": {
                str(path): sha256_file(path)
                for path in [
                    Path(__file__),
                    Path("src/openpi/training/subtask_evaluation.py"),
                    Path("src/openpi/training/evaluation_protocol.py"),
                    Path("src/openpi/training/research_checkpoint.py"),
                    Path("src/openpi/policies/subtask_policy.py"),
                ]
            },
            "semantics": semantic_metrics(rows, vocabulary, task_vocabulary=task_vocabulary),
            "transition_metrics": transition_metrics(rows, tolerance=10)
            if len(rows) == len(dataset)
            else {"unavailable": "requires dense complete episodes"},
            "action_conditions": {},
            "timing": {
                "frames": len(timing_rows),
                "warmup_per_rank": 3,
                "includes": "CPU transforms/tokenizer, vision, two prefixes, BOS generation, ten/default flow steps, native output; excludes sensor transport and video loading",
                "p50_p95_ms": {
                    key: np.percentile([row[key] for row in timing_rows], [50, 95]).tolist()
                    for key in timing_rows[0]
                    if key.endswith("_ms")
                }
                if timing_rows
                else {},
            },
            "action_units": "native data numbers; controller units unverified; no clipping",
            "shuffle_protocol": "within-task permutation of generated texts; best changed fraction among 32 seeded permutations; preserves prediction histogram",
        }
        for mode in sorted({row["condition_mode"] for row in action_rows}):
            group = [row for row in action_rows if row["condition_mode"] == mode]
            report["action_conditions"][mode] = {
                "overall": aggregate_actions(group),
                "condition_changed_fraction": float(np.mean([row["condition_changed"] for row in group])),
                "by_task": {
                    task: aggregate_actions([row for row in group if row["task"] == task])
                    for task in sorted({row["task"] for row in group})
                },
                "by_boundary": {
                    str(crossing): aggregate_actions(
                        [row for row in group if row["crosses_subtask_boundary"] == crossing]
                    )
                    for crossing in [False, True]
                },
            }
        if args.no_image:
            report["no_image"] = {
                "definition": "all three normalized image tensors set to -1 (black); state/task retained; no retraining",
                "semantics": semantic_metrics(no_image_rows, vocabulary, task_vocabulary=task_vocabulary),
            }
        write_json(args.output / "report.json", report)
        print(
            json.dumps(
                {
                    "event": "complete",
                    "report": str(args.output / "report.json"),
                    "semantic_frames": len(rows),
                    "action_draws": len(action_rows),
                    "timing_frames": len(timing_rows),
                }
            ),
            flush=True,
        )
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--test-protocol", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--semantic-samples", type=int, default=0, help="0 means every validation frame")
    parser.add_argument("--action-samples", type=int, default=128, help="0 disables action evaluation")
    parser.add_argument("--latency-samples", type=int, default=64, help="0 disables policy timing")
    parser.add_argument("--draws", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--no-image", action="store_true")
    evaluate(parser.parse_args())
