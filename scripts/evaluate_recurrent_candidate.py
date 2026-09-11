"""Final validation-only memory ablation and text-to-action condition controls."""
import argparse
import dataclasses
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import torch.distributed as dist
from openpi.models.pi0_config import Pi0Config
from openpi.policies.recurrent_subtask_policy import create_recurrent_subtask_policy
from openpi.training import config as config_lib
from openpi.training.decoded_video_cache import create_cached_dataset
from openpi.training.recurrent_evaluation import evaluate_recurrent
from openpi.training.subtask_batch import SubtaskTrainingDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda:%d" % rank)
    torch.cuda.set_device(device)
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group("nccl")
    policy = create_recurrent_subtask_policy(args.checkpoint, device=device)
    cfg = config_lib.get_config("pi05_piper_stage1")
    dc = dataclasses.replace(cfg.data.create(cfg.assets_dirs, cfg.model), split="val")
    dataset = SubtaskTrainingDataset(create_cached_dataset(dc, 50, args.cache), dc)
    labels = sorted(set(dataset.labels_without_video()))
    result = {}
    for reset in ([False, True] if policy.model.decoder.recurrent else [False]):
        metrics, rows = evaluate_recurrent(policy.model, dataset, device, vocabulary=labels,
                                           reset_each=reset, condition_controls=not reset)
        result["reset_each" if reset else "normal"] = dict(metrics=metrics, predictions=rows)
    if rank == 0:
        args.output.write_text(json.dumps(result, indent=2))
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
