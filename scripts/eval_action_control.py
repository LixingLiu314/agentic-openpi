"""Common native-action evaluator for C0, B1 and the trained oracle control."""

import argparse
import dataclasses
import json
import os
from pathlib import Path
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist

from openpi import transforms
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.policies.piper_policy import JOINT_MASK
from openpi.policies.piper_policy import PiperInputs
from openpi.policies.piper_policy import PiperOutputs
from openpi.policies.policy import Policy
from openpi.shared import normalize
from openpi.training import config
from openpi.training import data_loader
from openpi.training.action_control import ActionControl
from openpi.training.evaluation_protocol import validate_evaluation_request
from openpi.training.hierarchy_training import distributed_context
from openpi.training.stage1_data import manifest_digest
from openpi.training.stage1_data import sha256_file
from openpi.training.subtask_batch import SubtaskTrainingDataset
from openpi.training.subtask_batch import collate_subtask
from openpi.training.subtask_evaluation import aggregate_actions
from openpi.training.subtask_evaluation import native_action_metrics


def gather(rows, world):
    if world == 1:
        return rows
    partitions = [None] * world
    dist.all_gather_object(partitions, rows)
    return [row for partition in partitions for row in partition]


@torch.no_grad()
def evaluate(args):
    if min(args.samples, args.draws, args.num_steps) < 1 or args.latency_samples < 0:
        raise ValueError("Evaluation counts must be positive")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', '0')}" if args.device == "cuda" else "cpu")
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
    if metadata["stage"] not in {"m0", "control_b1", "control_o"}:
        raise ValueError("Expected C0/B1/O checkpoint")
    condition = "oracle" if metadata["stage"] == "control_o" else "none"
    cfg = config.get_config("pi05_piper_stage1")
    model_config = Pi0Config(**metadata["config"]["model"])
    assets = args.checkpoint / "assets/eggplant_potato"
    if sha256_file(assets / "norm_stats.json") != metadata["config"]["norm_sha256"]:
        raise ValueError("Normalization identity changed")
    if manifest_digest(json.loads((assets / "split.json").read_text())) != metadata["config"]["split_sha256"]:
        raise ValueError("Split identity changed")
    stats = normalize.load(assets)
    dc = dataclasses.replace(cfg.data.create(cfg.assets_dirs, model_config), split=args.split, norm_stats=stats)
    raw = data_loader.create_torch_dataset(dc, model_config.action_horizon, model_config)
    wrapper = SubtaskTrainingDataset(raw, dc)
    labels = wrapper.labels_without_video()
    model = ActionControl(PI0Pytorch(model_config).to(device), condition=condition).eval()
    safetensors.torch.load_model(
        model.base if metadata["stage"] == "m0" else model, args.checkpoint / "model.safetensors", strict=True
    )
    indices = np.unique(np.linspace(0, len(raw) - 1, min(args.samples, len(raw)), dtype=int))[rank::world]
    rows = []
    native_windows, target_windows = [], []
    for raw_index in indices:
        index = int(raw_index)
        sample = raw[index]
        batch = collate_subtask([wrapper[index]]).to(device)
        context = model.prepare_action_context(batch.observation, batch.global_prompts)
        prefix = model.action_prefix(context, batch.labels if condition == "oracle" else [""])
        target = np.asarray(sample["action"], dtype=np.float64)
        state = np.asarray(sample["observation.state"], dtype=np.float64)
        episode, frame = int(sample["episode_index"]), int(sample["frame_index"])
        position = raw.episode_positions[episode]
        length = int(raw.episode_data_index["to"][position] - raw.episode_data_index["from"][position])
        valid = min(model_config.action_horizon, length - frame)
        crossing = any(label != sample["subtask"] for label in labels[index : index + valid])
        for draw in range(args.draws):
            generator = torch.Generator().manual_seed(100000 + index * args.draws + draw)
            noise = torch.randn(batch.actions.shape, generator=generator).to(device)
            flow_time = np.random.default_rng(200000 + index * args.draws + draw).beta(1.5, 1) * 0.999 + 0.001
            error = model.action_loss(
                context,
                prefix,
                batch.actions,
                noise=noise,
                time=torch.tensor([flow_time], device=device, dtype=torch.float32),
                reduction="none",
            )
            predicted = (
                model.sample_actions_from_prefix(context, prefix, noise=noise, num_steps=args.num_steps)[0, :, :14]
                .float()
                .cpu()
                .numpy()
            )
            native = (predicted + 1) * 0.5 * (stats["actions"].q99 - stats["actions"].q01 + 1e-6) + stats["actions"].q01
            native[:, np.asarray(JOINT_MASK)] += state[np.asarray(JOINT_MASK)]
            native_windows.append(native)
            target_windows.append(target)
            rows.append(
                {
                    "index": index,
                    "episode": episode,
                    "frame": frame,
                    "task": sample["task"],
                    "draw": draw,
                    "condition_mode": condition,
                    "crosses_subtask_boundary": crossing,
                    "valid_horizon": valid,
                    "flow_native14_normalized": float(error[..., :14].mean()),
                    "flow_all32": float(error.mean()),
                    **native_action_metrics(native, target, valid_horizon=valid),
                }
            )
    if rows:
        np.savez_compressed(
            args.output / f"native_rank_{rank:03d}.npz",
            predicted=np.stack(native_windows),
            target=np.stack(target_windows),
            index=np.asarray([row["index"] for row in rows]),
            draw=np.asarray([row["draw"] for row in rows]),
            condition=np.asarray([row["condition_mode"] for row in rows]),
            valid_horizon=np.asarray([row["valid_horizon"] for row in rows]),
        )
    (args.output / f"actions_rank_{rank:03d}.json").write_text(json.dumps(rows, indent=2) + "\n")
    rows = gather(rows, world)
    timing = []
    if args.latency_samples and condition == "none":
        model_transforms = config.ModelTransformFactory()(model_config)
        policy = Policy(
            model.base,
            transforms=[PiperInputs(), transforms.Normalize(stats, use_quantiles=True), *model_transforms.inputs],
            output_transforms=[
                *model_transforms.outputs,
                transforms.Unnormalize(stats, use_quantiles=True),
                transforms.AbsoluteActions(JOINT_MASK),
                PiperOutputs(),
            ],
            sample_kwargs={"num_steps": args.num_steps},
            pytorch_device=str(device),
            is_pytorch=True,
        )
        timing_indices = np.unique(np.linspace(0, len(raw) - 1, min(args.latency_samples, len(raw)), dtype=int))[
            rank::world
        ]
        for iteration, index in enumerate(timing_indices):
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
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            begun = time.perf_counter()
            output = policy.infer(observation, noise=noise)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = (time.perf_counter() - begun) * 1000
            if output["actions"].shape != (model_config.action_horizon, 14) or not np.isfinite(output["actions"]).all():
                raise ValueError("Invalid native policy output")
            timing.append({"index": int(index), "rank": rank, "complete_policy_ms": elapsed})
    (args.output / f"timing_rank_{rank:03d}.json").write_text(json.dumps(timing, indent=2) + "\n")
    timing = gather(timing, world)
    if rank == 0:
        report = {
            "checkpoint": str(args.checkpoint),
            "evaluation_config": {
                key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
            },
            "native_action_archives": "native_rank_*.npz preserve paired full windows for a common execution length",
            "stage": metadata["stage"],
            "condition": condition,
            "weights_sha256": sha256_file(args.checkpoint / "model.safetensors"),
            "split": args.split,
            "test_protocol": test_protocol,
            "split_sha256": metadata["config"]["split_sha256"],
            "norm_sha256": metadata["config"]["norm_sha256"],
            "samples": args.samples,
            "noise_draws": args.draws,
            "flow_steps": args.num_steps,
            "world_size": world,
            "overall": aggregate_actions(rows),
            "by_task": {
                task: aggregate_actions([row for row in rows if row["task"] == task])
                for task in sorted({row["task"] for row in rows})
            },
            "by_boundary": {
                str(crossing): aggregate_actions([row for row in rows if row["crosses_subtask_boundary"] == crossing])
                for crossing in [False, True]
            },
            "complete_policy_ms_p50_p95": np.percentile(
                [row["complete_policy_ms"] for row in timing], [50, 95]
            ).tolist()
            if timing
            else None,
            "timing_frames": len(timing),
            "timing_scope": "external synchronized wall clock around complete official Policy.infer; includes transforms/tokenizer, vision/prefix, all flow steps and native outputs; excludes video loading/sensor transport; three warmups per rank; unavailable for GT oracle",
            "units": "native source numbers, no clipping; controller units unverified",
            "sources": {
                str(path): sha256_file(path)
                for path in [
                    Path(__file__),
                    Path("src/openpi/training/action_control.py"),
                    Path("src/openpi/training/subtask_evaluation.py"),
                    Path("src/openpi/training/evaluation_protocol.py"),
                    Path("src/openpi/training/research_checkpoint.py"),
                    Path("src/openpi/policies/policy.py"),
                ]
            },
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps({"event": "complete", "report": str(args.output / "report.json"), "overall": report["overall"]}),
            flush=True,
        )
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--test-protocol", type=Path)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--draws", type=int, default=2)
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--latency-samples", type=int, default=128)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--cpu-threads", type=int, default=4)
    evaluate(p.parse_args())
