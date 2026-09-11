"""Isolated batch-one policy timing and CUDA memory after checkpoint loading."""

import argparse
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
from openpi_client.action_chunk_broker import ActionChunkBroker
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
from openpi.policies.subtask_policy import create_subtask_policy
from openpi.shared import normalize
from openpi.training import config
from openpi.training import data_loader
from openpi.training.action_control import ActionControl
from openpi.training.research_checkpoint import require_research_checkpoint
from openpi.training.stage1_data import manifest_digest
from openpi.training.stage1_data import sha256_file


def driver_processes():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


def verify_chunk_broker(output):
    class ReplayPolicy:
        calls = 0

        def infer(self, _observation):
            self.calls += 1
            return output

        def reset(self):
            self.calls = 0

    replay = ReplayPolicy()
    broker = ActionChunkBroker(replay, action_horizon=50)
    for step in range(50):
        sliced = broker.infer({})
        np.testing.assert_array_equal(sliced["actions"], output["actions"][step])
        for key, value in output.items():
            if key != "actions":
                assert sliced[key] == value, key
    assert replay.calls == 1
    np.testing.assert_array_equal(broker.infer({})["actions"], output["actions"][0])
    assert replay.calls == 2
    broker.reset()
    np.testing.assert_array_equal(broker.infer({})["actions"], output["actions"][0])
    assert replay.calls == 1


def load_policy(checkpoint, device, num_steps):
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    require_research_checkpoint(checkpoint, metadata)
    model_config = Pi0Config(**metadata["config"]["model"])
    stats = normalize.load(checkpoint / "assets/eggplant_potato")
    if sha256_file(checkpoint / "assets/eggplant_potato/norm_stats.json") != metadata["config"]["norm_sha256"]:
        raise ValueError("Checkpoint normalization changed")
    if metadata["stage"] == "m3":
        policy = create_subtask_policy(checkpoint, device=str(device), num_steps=num_steps)
    elif metadata["stage"] in {"m0", "control_b1"}:
        base = PI0Pytorch(model_config).to(device).eval()
        if metadata["stage"] == "m0":
            safetensors.torch.load_model(base, checkpoint / "model.safetensors", strict=True)
        else:
            control = ActionControl(base, condition="none")
            safetensors.torch.load_model(control, checkpoint / "model.safetensors", strict=True)
            # Only the base policy remains deployed; the unused S is not retained.
            del control
        base.eval()
        model_transforms = config.ModelTransformFactory()(model_config)
        policy = Policy(
            base,
            transforms=[PiperInputs(), transforms.Normalize(stats, use_quantiles=True), *model_transforms.inputs],
            output_transforms=[
                *model_transforms.outputs,
                transforms.Unnormalize(stats, use_quantiles=True),
                transforms.AbsoluteActions(JOINT_MASK),
                PiperOutputs(),
            ],
            sample_kwargs={"num_steps": num_steps},
            pytorch_device=str(device),
            is_pytorch=True,
        )
    else:
        raise ValueError("Oracle is not a deployable policy benchmark")
    dc = dataclasses.replace(
        config.get_config("pi05_piper_stage1").data.create(
            config.get_config("pi05_piper_stage1").assets_dirs, model_config
        ),
        split="val",
        norm_stats=stats,
    )
    if manifest_digest(json.loads(Path(dc.split_manifest).read_text())) != metadata["config"]["split_sha256"]:
        raise ValueError("Profile dataset split differs from checkpoint")
    return policy, model_config, dc


@torch.no_grad()
def main(args):
    if min(args.samples, args.warmup, args.num_steps) < 1:
        raise ValueError("Counts must be positive")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', '0')}")
    torch.cuda.set_device(device)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group("nccl", device_id=device)
    rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        before_processes = driver_processes()
    if world > 1:
        dist.barrier()
    policy, model_config, dc = load_policy(args.checkpoint, device, args.num_steps)
    raw = data_loader.create_torch_dataset(dc, model_config.action_horizon, model_config)
    indices = np.unique(np.linspace(0, len(raw) - 1, min(args.samples, len(raw)), dtype=int))[rank::world]
    torch.cuda.empty_cache()
    rows = []
    for iteration, raw_index in enumerate(indices):
        index = int(raw_index)
        sample = raw[index]
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
            generator=torch.Generator().manual_seed(800000 + index),
        ).numpy()
        if iteration == 0:
            for _ in range(args.warmup):
                policy.infer(observation, noise=noise)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        before = torch.cuda.memory_allocated(device)
        begun = time.perf_counter()
        output = policy.infer(observation, noise=noise)
        torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - begun) * 1000
        if output["actions"].shape != (50, 14) or not np.isfinite(output["actions"]).all():
            raise ValueError("Invalid native deployment output")
        if iteration == 0:
            verify_chunk_broker(output)
        rows.append(
            {
                "rank": rank,
                "index": index,
                "complete_policy_ms": elapsed,
                "allocated_before_bytes": before,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "policy_timing": output.get("policy_timing"),
            }
        )
    (args.output / f"rank_{rank:03d}.json").write_text(json.dumps(rows, indent=2) + "\n")
    partitions = [None] * world
    if world > 1:
        dist.all_gather_object(partitions, rows)
    else:
        partitions[0] = rows
    if rank == 0:
        rows = [row for partition in partitions for row in partition]
        report = {
            "checkpoint": str(args.checkpoint),
            "weights_sha256": sha256_file(args.checkpoint / "model.safetensors"),
            "samples": len(rows),
            "warmup_per_rank": args.warmup,
            "chunk_broker_contract": "Actual first inference output replayed through all 50 action slices, then replan/reset; native actions and semantic/timing metadata preserved on every rank",
            "world_size": world,
            "flow_steps": args.num_steps,
            "policy_ms_p50_p95": np.percentile([row["complete_policy_ms"] for row in rows], [50, 95]).tolist(),
            "max_peak_allocated_gib": max(row["peak_allocated_bytes"] for row in rows) / 1024**3,
            "max_peak_reserved_gib": max(row["peak_reserved_bytes"] for row in rows) / 1024**3,
            "max_call_incremental_allocated_gib": max(
                row["peak_allocated_bytes"] - row["allocated_before_bytes"] for row in rows
            )
            / 1024**3,
            "scope": "Single policy per GPU, batch one, isolated from evaluation flow-loss contexts; complete synchronized Policy.infer including transforms/tokenizer and native outputs, excluding video decode/transport. Memory is PyTorch allocated/reserved (including model) after warmup, not total driver memory; external GPU processes are listed separately.",
            "driver_processes_before_loading": before_processes,
            "driver_processes_after_inference": driver_processes(),
            "sources": {
                str(path): sha256_file(path)
                for path in [
                    Path(__file__),
                    Path("src/openpi/policies/subtask_policy.py"),
                    Path("src/openpi/policies/policy.py"),
                    Path("src/openpi/training/action_control.py"),
                    Path("src/openpi/training/research_checkpoint.py"),
                ]
            },
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"event": "complete", **report}), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--cpu-threads", type=int, default=4)
    main(parser.parse_args())
