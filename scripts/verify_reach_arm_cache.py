"""Exact old/new data-path comparison, including every episode's tail window."""
import argparse
import dataclasses
import json
from pathlib import Path
import time

import numpy as np
import torch
from openpi.training import config as config_lib, data_loader
from openpi.training.decoded_video_cache import create_cached_dataset, collate_pinned_subtask
from openpi.training.subtask_batch import SubtaskTrainingDataset
from openpi.training.reach_arm_data import data_config


def equal(a, b, path="sample"):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            equal(a[key], b[key], path + "/" + key)
    elif isinstance(a, (np.ndarray, torch.Tensor)):
        x, y = np.asarray(a), np.asarray(b)
        assert x.dtype == y.dtype and np.array_equal(x, y), (path, x.shape, y.shape)
    else:
        assert a == b, (path, a, b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = config_lib.get_config("pi05_piper_stage1")
    dc = data_config(cfg.model)
    report = {"passed":False, "splits":{}, "scope":"exact transformed samples; all episode endpoints; random and label-boundary frames; CPU throughput"}
    begun = time.perf_counter()
    for split in ("train", "val"):
        split_dc = dataclasses.replace(dc, split=split)
        raw = data_loader.create_torch_dataset(split_dc, cfg.model.action_horizon, cfg.model)
        cached = create_cached_dataset(split_dc, cfg.model.action_horizon, args.cache)
        old, new = SubtaskTrainingDataset(raw, split_dc), SubtaskTrainingDataset(cached, split_dc)
        assert len(old) == len(new)
        starts = [int(x) for x in raw.episode_data_index["from"]]
        ends = [int(x)-1 for x in raw.episode_data_index["to"]]
        labels = old.labels_without_video()
        assert labels == new.labels_without_video()
        boundaries = [i for i in range(1,len(labels)) if labels[i] != labels[i-1] and i not in starts]
        rng = np.random.default_rng(42)
        selected = set(starts + ends + rng.choice(len(old), 32, replace=False).tolist())
        for i in rng.choice(boundaries, min(32,len(boundaries)), replace=False):
            selected.update([int(i)-1, int(i)])
        for n,index in enumerate(sorted(selected)):
            equal(old[index], new[index])
            if (n+1) % 100 == 0:
                print(json.dumps({"split":split,"verified":n+1,"total":len(selected)}),flush=True)
        # Same 64 real samples in each path; excludes initialization; no GPU speed claim.
        indices = rng.choice(len(old),64,replace=False).tolist()
        timings = {}
        for name, dataset in (("original_pyav",old),("cached_rgb",new)):
            for i in indices[:4]:
                dataset[i]
            start = time.perf_counter()
            for i in indices:
                dataset[i]
            timings[name] = time.perf_counter()-start
        batch = collate_pinned_subtask([new[starts[0]],new[ends[0]]])
        moved = batch.to("cpu")
        assert torch.equal(batch.actions,moved.actions)
        report["splits"][split] = {"frames":len(old),"episodes":len(starts),"exact_samples":len(selected),
                                      "all_episode_starts_and_tails":True,"timed_samples":len(indices),
                                      "read_seconds":timings,"cpu_read_speedup":timings["original_pyav"]/timings["cached_rgb"]}
    report.update(passed=True, seconds=time.perf_counter()-begun)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    main()
