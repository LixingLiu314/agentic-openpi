"""Fixed-draw C0 validation in normalized flow and native Piper action units."""

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
from train_subtask_pytorch import collate_observation
from train_subtask_pytorch import to_device

from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.policies.piper_policy import JOINT_MASK
from openpi.shared import normalize
from openpi.training import config
from openpi.training import data_loader
from openpi.training.hierarchy_training import distributed_context
from openpi.training.stage1_data import sha256_file
from openpi.training.subtask_batch import SubtaskTrainingDataset


@torch.no_grad()
def evaluate(args):
    if min(args.samples, args.draws, args.num_steps) < 1:
        raise ValueError("Evaluation counts must be positive")
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
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()
    metadata = json.loads((args.checkpoint / "metadata.json").read_text())
    if metadata["stage"] != "m0":
        raise ValueError("This evaluator expects the official-action M0 checkpoint")
    model_config = Pi0Config(**metadata["config"]["model"])
    cfg = config.get_config("pi05_piper_stage1")
    dc = dataclasses.replace(cfg.data.create(cfg.assets_dirs, model_config), split="val")
    asset_dir = args.checkpoint / "assets/eggplant_potato"
    if sha256_file(asset_dir / "norm_stats.json") != metadata["config"]["norm_sha256"]:
        raise ValueError("Checkpoint normalization fingerprint mismatch")
    stats = normalize.load(asset_dir)
    dc = dataclasses.replace(dc, norm_stats=stats)
    raw = data_loader.create_torch_dataset(dc, model_config.action_horizon, model_config)
    wrapper = SubtaskTrainingDataset(raw, dc)
    labels = list(raw.hf_dataset["subtask"])
    model = PI0Pytorch(model_config).to(device).eval()
    safetensors.torch.load_model(model, args.checkpoint / "model.safetensors", strict=True)
    indices = np.unique(np.linspace(0, len(raw) - 1, min(args.samples, len(raw)), dtype=int))[rank::world]
    rows = []
    for index in indices:
        sample = raw[int(index)]
        target = np.asarray(sample["action"], dtype=np.float64)
        original_state = np.asarray(sample["observation.state"], dtype=np.float64)
        observation, actions = to_device(collate_observation([wrapper.transform(sample)]), device)
        episode, frame = int(sample["episode_index"]), int(sample["frame_index"])
        position = raw.episode_positions[episode]
        length = int(raw.episode_data_index["to"][position] - raw.episode_data_index["from"][position])
        valid_horizon = min(model_config.action_horizon, length - frame)
        crossing = any(label != sample["subtask"] for label in labels[int(index) : int(index) + valid_horizon])
        for draw in range(args.draws):
            generator = torch.Generator().manual_seed(100000 + int(index) * args.draws + draw)
            noise = torch.randn(actions.shape, generator=generator).to(device)
            flow_time = np.random.default_rng(200000 + int(index) * args.draws + draw).beta(1.5, 1) * 0.999 + 0.001
            flow_error = model(
                observation, actions, noise=noise, time=torch.tensor([flow_time], device=device, dtype=torch.float32)
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            begun = time.perf_counter()
            predicted = model.sample_actions(device, observation, noise=noise, num_steps=args.num_steps)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            sampling_ms = (time.perf_counter() - begun) * 1000
            predicted = predicted[0, :, :14].float().cpu().numpy()
            native = (predicted + 1) * 0.5 * (stats["actions"].q99 - stats["actions"].q01 + 1e-6) + stats["actions"].q01
            native[:, np.asarray(JOINT_MASK)] += original_state[np.asarray(JOINT_MASK)]
            if not np.isfinite(native).all():
                raise FloatingPointError("Nonfinite native actions")
            errors = (native - target) ** 2
            row = {
                "index": int(index),
                "episode": episode,
                "frame": frame,
                "task": sample["task"],
                "draw": draw,
                "crosses_subtask_boundary": crossing,
                "valid_horizon": valid_horizon,
                "flow_native14_normalized": float(flow_error[..., :14].mean()),
                "flow_all32": float(flow_error.mean()),
                "native_mse_per_dim_h50": errors.mean(axis=0).tolist(),
                "native_mse_per_dim_first10": errors[: min(10, valid_horizon)].mean(axis=0).tolist(),
                "native_mse_per_dim_valid_horizon": errors[:valid_horizon].mean(axis=0).tolist(),
                "gripper_accuracy_h50": float(((native[:, [6, 13]] >= 0.045) == (target[:, [6, 13]] >= 0.045)).mean()),
                "gripper_out_of_demonstrated_range": float(
                    ((native[:, [6, 13]] < 0) | (native[:, [6, 13]] > 0.09)).mean()
                ),
                "sampling_ms": sampling_ms,
            }
            rows.append(row)
    (args.output / f"rank_{rank:03d}.json").write_text(json.dumps(rows, indent=2) + "\n")
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, rows)
        rows = [row for partition in gathered for row in partition]
    if rank == 0:

        def aggregate(group):
            if not group:
                return {"draws": 0}
            result = {"draws": len(group), "frames": len({row["index"] for row in group})}
            for metric in [
                "flow_native14_normalized",
                "flow_all32",
                "gripper_accuracy_h50",
                "gripper_out_of_demonstrated_range",
            ]:
                result[metric] = float(np.mean([row[metric] for row in group]))
            for scope in ["h50", "first10", "valid_horizon"]:
                mse = np.mean([row[f"native_mse_per_dim_{scope}"] for row in group], axis=0)
                result[f"native_rmse_per_dim_{scope}"] = np.sqrt(mse).tolist()
                result[f"native_joint_rmse_{scope}"] = float(np.sqrt(mse[np.asarray(JOINT_MASK)].mean()))
                result[f"native_gripper_rmse_{scope}"] = float(np.sqrt(mse[[6, 13]].mean()))
            return result

        report = {
            "checkpoint": str(args.checkpoint),
            "weights_sha256": sha256_file(args.checkpoint / "model.safetensors"),
            "split": "val",
            "split_sha256": metadata["config"]["split_sha256"],
            "norm_sha256": metadata["config"]["norm_sha256"],
            "samples": args.samples,
            "noise_draws": args.draws,
            "flow_steps": args.num_steps,
            "world_size": world,
            "units": "native source numbers; controller physical units still require deployment verification",
            "overall": aggregate(rows),
            "by_task": {
                task: aggregate([row for row in rows if row["task"] == task])
                for task in sorted({row["task"] for row in rows})
            },
            "by_boundary": {
                str(crossing): aggregate([row for row in rows if row["crosses_subtask_boundary"] == crossing])
                for crossing in [False, True]
            },
            "model_sampling_ms_p50_p95": np.percentile([row["sampling_ms"] for row in rows], [50, 95]).tolist(),
            "timing_scope": "model.sample_actions only; excludes input transforms and does not establish complete policy latency",
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--draws", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    evaluate(parser.parse_args())
