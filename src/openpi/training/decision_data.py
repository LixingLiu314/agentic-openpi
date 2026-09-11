"""Training-only decision sampling and sparse visually reviewed target points."""
import dataclasses
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from openpi.training.recurrent_sequence import EpisodeStreamSampler, episode_rows
from openpi.training.decoded_video_cache import collate_pinned_subtask
from openpi.training.stage1_data import sha256_file

from openpi.training.decision_assets import ASSETS, load_assets


class DecisionStreamSampler(EpisodeStreamSampler):
    """Balance episode starts by existing task/layout/lid arm, retaining causality.

    Every normal episode starts at frame zero. Sampling may land on a known
    training reach boundary; boundaries never enter the model observation.
    Engineering fixture uses real decision frames to exercise both extra paths.
    """
    def __init__(self,rows,decisions,*,engineering=False,**kwargs):
        super().__init__(rows,**kwargs)
        self.decisions={int(r["episode"]):r for r in decisions}
        self.engineering=engineering
        if not set(rows)<=set(self.decisions):raise ValueError("Missing episode decisions")

    def __iter__(self):
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.rank,20260910]))
        groups=defaultdict(list)
        for ep in sorted(self.rows):
            r=self.decisions[ep];groups[(r["prompt"],r["layout"],r["lid_arm"])].append(ep)
        keys=sorted(groups);group_queue=[];episode_queues={k:[] for k in keys}
        def begin():
            if not group_queue:group_queue.extend(rng.permutation(len(keys)).tolist())
            key=keys[group_queue.pop()]
            if not episode_queues[key]:episode_queues[key].extend(rng.permutation(groups[key]).tolist())
            return episode_queues[key].pop(),0
        streams=[None]*(self.batch_size//self.unroll)
        for step in range(self.steps):
            batch=[]
            for t in range(self.unroll):
                for stream,state in enumerate(streams):
                    reset=state is None
                    ep,pos=begin() if reset else state
                    indices,times=self.rows[ep]
                    if self.engineering:
                        # Fixed real start/object observations per stream; capacity only.
                        pos=0 if stream%2==0 else self.decisions[ep]["object_start"]
                    batch.append((int(indices[pos]),reset))
                    next_pos=int(np.searchsorted(times,times[pos]+rng.uniform(self.min_interval,self.max_interval)))
                    boundary=self.decisions[ep]["object_start"]
                    if pos<boundary<next_pos:next_pos=boundary
                    streams[stream]=(ep,pos) if self.engineering else (None if next_pos>=len(indices) else (ep,next_pos))
            if step>=self.start:yield batch


def row_weights(episodes,frames,decisions):
    by_ep={int(r["episode"]):r for r in decisions}
    result=[]
    for ep,frame in zip(episodes,frames,strict=True):
        r=by_ep[int(ep)];f=int(frame)
        decision=f<45 or r["object_start"]<=f<min(r["object_end"],r["object_start"]+30)
        result.append(2.0 if decision else 1.0)
    return result


class GroundingSamples:
    """Cache only reviewed frames from the requested split, never action targets.

    Both visible objects can supply point supervision with their matching global
    task. Alternate tasks are used solely by the localization loss, not A loss.
    """
    def __init__(self,dataset,points,*,split="train"):
        rows=episode_rows(dataset.raw_dataset.hf_dataset)
        self.samples=[];self.groups=defaultdict(list)
        for row in points["rows"]:
            if row["split"]!=split:continue
            if not row["reviewed"] or row["episode"] not in rows:raise ValueError("Unreviewed or wrong-split grounding row")
            index=int(rows[row["episode"]][0][row["frame"]])
            sample=dataset[index]
            for entity,point in sorted(row["points"].items()):
                if not point["visible"]:continue
                prompt=row["prompt"].replace(row["target"],entity)
                xy=point["xy"]
                if len(xy)!=2 or not np.isfinite(xy).all() or not (0<=np.array(xy)).all() or not (np.array(xy)<=1).all():
                    raise ValueError("Invalid reviewed point")
                i=len(self.samples)
                self.samples.append(dict(sample=sample,prompt=prompt,point=xy,bbox=point["bbox"],entity=entity,
                                         episode=row["episode"],frame=row["frame"]))
                self.groups[entity].append(i)
        if not self.samples:raise ValueError("No reviewed samples")

    def select(self,step,rank,count=2):
        rng=np.random.default_rng(np.random.SeedSequence([42,step,rank,4202]))
        keys=sorted(self.groups)
        return [int(rng.choice(self.groups[keys[(step+i)%len(keys)]])) for i in range(count)]

    def batch(self,indices,device):
        selected=[self.samples[i] for i in indices]
        host=collate_pinned_subtask([r["sample"] for r in selected])
        # prepare_context rebuilds all prompt/state tokens from global_prompts.
        # Pre-existing tokenized_prompt is not read in that path.
        host=dataclasses.replace(host,global_prompts=tuple(r["prompt"] for r in selected))
        if str(device).startswith("cuda"):host=host.pin_memory()
        return host.to(device,non_blocking=True),torch.tensor([r["point"] for r in selected],device=device,dtype=torch.float32)
