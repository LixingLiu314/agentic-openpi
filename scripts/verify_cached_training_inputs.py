"""Compare every input in the exact eight-update distributed engineering schedule."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import time


def verify_rank(rank):
    import numpy as np
    import torch
    import jax
    from train_subtask_pytorch import StepBatchSampler
    from verify_piper_rgb_cache import equal
    from openpi.training import config as cfg_lib, data_loader
    from openpi.training.decoded_video_cache import create_cached_dataset, collate_pinned_subtask
    from openpi.training.subtask_batch import SubtaskTrainingDataset, collate_subtask
    torch.set_num_threads(1)
    cfg = cfg_lib.get_config("pi05_piper_stage1")
    dc = cfg.data.create(cfg.assets_dirs,cfg.model)
    old = SubtaskTrainingDataset(data_loader.create_torch_dataset(dc,50,cfg.model),dc)
    new = SubtaskTrainingDataset(create_cached_dataset(dc,50,".stage1_staging/piper_rgb224_cache_v2"),dc)
    count = 0
    for indices in StepBatchSampler(len(old),32,1,8,0,42,rank,8):
        old_samples,new_samples = [],[]
        for index in indices:
            a,b = old[index],new[index]
            equal(a,b)
            old_samples.append(a);new_samples.append(b)
            count += 1
        a,b = collate_subtask(old_samples),collate_pinned_subtask(new_samples)
        for x,y in zip(jax.tree.leaves(a.observation),jax.tree.leaves(b.observation),strict=True):
            assert torch.equal(x,y)
        for key in ("actions","target_ids","target_mask","episode_indices","frame_indices"):
            assert torch.equal(getattr(a,key),getattr(b,key))
        assert a.labels == b.labels and a.global_prompts == b.global_prompts
    return {"rank":rank,"exact_samples":count,"collated_observations_exact":True}


if __name__ == "__main__":
    begun=time.perf_counter()
    with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context("spawn")) as pool:
        ranks=list(pool.map(verify_rank,range(8)))
    report={"passed":True,"ranks":ranks,"total_samples":sum(r["exact_samples"] for r in ranks),
            "seconds":time.perf_counter()-begun,"scope":"every exact original/cached input and collated tensor in the full 8-update engineering schedule"}
    Path("logs/pi05_piper_backbone_grad/batch_retry_20260908/all_engineering_inputs_exact.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)
