"""Bounded M1 eight-GPU compute benchmark: equal global batch, real observations.

This writes measurements only, never training checkpoints. Decoded observations
are resident before timing, so results measure model/transfer/optimizer throughput
and are not a claim about full-dataset end-to-end throughput.
"""

import argparse
import contextlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import safetensors.torch
from subtask_runtime import StepTimer
from subtask_runtime import collate_transfer
from subtask_runtime import reduce_timing_max
from sync_subtask_wandb import gpu_metrics
import torch
import torch.distributed as dist

from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pi05_subtask_pytorch import Pi05SubtaskPytorch
from openpi.training import config as config_lib
from openpi.training import data_loader
from openpi.training.hierarchy_training import BranchOptimizers
from openpi.training.research_checkpoint import require_research_checkpoint
from openpi.training.stage1_data import manifest_digest
from openpi.training.stage1_data import sha256_file
from openpi.training.subtask_batch import SubtaskTrainingDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=30)
    args = parser.parse_args()
    if args.updates < 10:
        raise ValueError("Use at least ten measured updates")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 8:
        raise ValueError("This fixed global-batch benchmark requires eight ranks")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(42)
    metadata = json.loads((args.checkpoint / "metadata.json").read_text())
    if metadata["stage"] != "m0":
        raise ValueError("Initialize this bounded M1 benchmark from the common C0")
    require_research_checkpoint(args.checkpoint, metadata)
    mc = Pi0Config(**metadata["config"]["model"])
    cfg = config_lib.get_config("pi05_piper_stage1")
    dc = cfg.data.create(cfg.assets_dirs, mc)
    if dc.split != "train":
        raise ValueError("Throughput observations must come from training only")
    if manifest_digest(json.loads(Path(dc.split_manifest).read_text())) != metadata["config"]["split_sha256"]:
        raise ValueError("Benchmark split identity differs from C0")
    norm_path = Path("assets/pi05_piper_stage1/eggplant_potato/norm_stats.json")
    if sha256_file(norm_path) != metadata["config"]["norm_sha256"]:
        raise ValueError("Benchmark normalization differs from C0")
    raw = data_loader.create_torch_dataset(dc, mc.action_horizon, mc)
    dataset = SubtaskTrainingDataset(raw, dc)
    indices = json.loads(
        Path("checkpoints/pi05_piper_stage1/m1_overfit32_ddp8_seed42/overfit_indices.json").read_text()
    )
    if len(indices) != 32 or len(set(indices)) != 32 or not all(0 <= i < len(dataset) for i in indices):
        raise ValueError("Expected 32 distinct valid training observations")
    # Every rank reads just its union of four-example and two-example partitions.
    needed = sorted(
        set(
            [indices[rank * 2 + k] for k in range(2)]
            + [indices[world * 2 + rank * 2 + k] for k in range(2)]
            + indices[rank * 4 : rank * 4 + 4]
        )
    )
    samples = {index: dataset[index] for index in needed}
    base = PI0Pytorch(mc).to(device)
    model = Pi05SubtaskPytorch(base)
    safetensors.torch.load_model(base, args.checkpoint / "model.safetensors", strict=True)
    model.set_stage("m1")
    original_decoder = {key: value.detach().cpu().clone() for key, value in model.decoder.state_dict().items()}
    ddp = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device.index],
        find_unused_parameters=True,
        gradient_as_bucket_view=True,
        broadcast_buffers=False,
    )
    results = []
    driver_before = (
        subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        if rank == 0
        else None
    )
    for microbatch, accumulation in [(2, 2), (4, 1)]:
        model.decoder.load_state_dict(original_decoder)
        torch.manual_seed(42)
        optimizers = BranchOptimizers(model, "m1", 1e-4, 1e-5)
        batches = []
        for micro in range(accumulation):
            chosen = indices[
                micro * world * microbatch + rank * microbatch : micro * world * microbatch + (rank + 1) * microbatch
            ]
            batches.append(collate_transfer([samples[i] for i in chosen]).pin_memory())
        durations, components, gpu_rows = [], [], []
        stop = threading.Event()

        def monitor(stop=stop, gpu_rows=gpu_rows):
            while not stop.is_set():
                with contextlib.suppress(Exception):
                    gpu_rows.append(gpu_metrics())
                stop.wait(0.5)

        thread = None
        torch.cuda.reset_peak_memory_stats()
        for step in range(args.updates + 5):
            if step == 5 and rank == 0:
                thread = threading.Thread(target=monitor, daemon=True)
                thread.start()
            dist.barrier()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            timer = StepTimer(device, enabled=step == 4)
            optimizers.zero_grad()
            micro_losses = []
            for micro, cpu_batch in enumerate(batches):
                phase = timer.now()
                batch = cpu_batch.to(device)
                timer.add("transfer", phase)
                with ddp.no_sync() if micro < accumulation - 1 else contextlib.nullcontext():
                    phase = timer.now()
                    loss = ddp(batch)["loss_subtask"]
                    micro_losses.append(loss.detach())
                    timer.add("forward", phase)
                    phase = timer.now()
                    (loss / accumulation).backward()
                    timer.add("backward", phase)
            phase = timer.now()
            optimizers.step()
            timer.add("optimizer", phase)
            torch.cuda.synchronize()
            duration = torch.tensor(time.perf_counter() - begin, device=device)
            dist.all_reduce(duration, op=dist.ReduceOp.MAX)
            if not torch.stack(micro_losses).isfinite().all():
                raise ValueError("Nonfinite benchmark loss")
            if step >= 5:
                durations.append(float(duration))
            if timer.enabled:
                components = reduce_timing_max(timer.values, device)
        if thread:
            stop.set()
            thread.join(timeout=6)
        peak = torch.tensor(torch.cuda.max_memory_allocated() / 2**30, device=device)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        if rank == 0:
            utilization = [
                value for row in gpu_rows for key, value in row.items() if key.endswith("utilization_percent")
            ]
            result = {
                "microbatch": microbatch,
                "accumulation": accumulation,
                "global_batch": world * microbatch * accumulation,
                "updates": len(durations),
                "mean_update_seconds": statistics.mean(durations),
                "median_update_seconds": statistics.median(durations),
                "p95_update_seconds": float(np.percentile(durations, 95)),
                "all_update_seconds_max_rank": durations,
                "examples_per_second": world * microbatch * accumulation / statistics.mean(durations),
                "max_rank_peak_allocated_gib": float(peak),
                "gpu_utilization_mean_percent": statistics.mean(utilization) if utilization else None,
                "gpu_utilization_samples": len(utilization),
                "profile_one_warmup_update": components,
            }
            results.append(result)
            print(json.dumps(result), flush=True)
        del optimizers, batches
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(
                json.dumps(
                    {
                        "scope": "M1, decoded real frames resident, same 32 examples/update, 5 warmup updates; not full-run timing",
                        "checkpoint": str(args.checkpoint),
                        "world_size": world,
                        "results": results,
                        "throughput_ratio": results[1]["examples_per_second"] / results[0]["examples_per_second"],
                        "weights_sha256": sha256_file(args.checkpoint / "model.safetensors"),
                        "metadata_sha256": sha256_file(args.checkpoint / "metadata.json"),
                        "split_sha256": metadata["config"]["split_sha256"],
                        "norm_sha256": metadata["config"]["norm_sha256"],
                        "training_indices": indices,
                        "variant_order": ["microbatch2_accum2", "microbatch4_accum1"],
                        "comparison_limits": "Single ordered run with equal examples, initial decoder state and fresh optimizers; stochastic per-example operations and optimizer trajectories need not be bitwise equivalent across batch partitions. No training-quality or whole-dataset throughput claim.",
                        "driver_processes_before_timing": driver_before,
                        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda},
                        "sources": {
                            str(path): sha256_file(path)
                            for path in [
                                Path(__file__),
                                Path("scripts/subtask_runtime.py"),
                                Path("src/openpi/training/research_checkpoint.py"),
                                Path("src/openpi/models_pytorch/pi05_subtask_pytorch.py"),
                                Path("src/openpi/models_pytorch/subtask_decoder.py"),
                                Path("src/openpi/training/subtask_batch.py"),
                            ]
                        },
                    },
                    indent=2,
                )
                + "\n"
            )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
