"""Final causal validation and offline initial-role interventions; no robot I/O."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS","cpu")
os.environ.setdefault("HF_HUB_OFFLINE","1")

import jax
import numpy as np
import torch
import torch.distributed as dist

from openpi.models.model import Observation
from openpi.models.pi0_config import Pi0Config
from openpi.policies.reach_arm_subtask_policy import create_reach_arm_policy
from openpi.training.reach_arm_data import data_config
from openpi.training.decoded_video_cache import create_cached_dataset
from openpi.training.reach_arm_evaluation import evaluate_recurrent, phase_actor
from openpi.training.subtask_batch import SubtaskTrainingDataset


def dominant_arm(actions,state):
    d=[float(np.linalg.norm(actions[:,a:a+6]-state[a:a+6],axis=1).max()) for a in (0,7)]
    i=int(np.argmax(d))
    return (("left","right")[i] if d[i]>.1 and d[i]>2*max(d[1-i],1e-6) else "ambiguous"),d


@torch.no_grad()
def text_intervention(policy, observation, text, noise):
    inputs=policy.input_transform(observation)
    tensors=jax.tree.map(lambda v:torch.as_tensor(np.asarray(v),device=policy.device)[None],inputs)
    context=policy.model.prepare_context(Observation.from_dict(tensors),[observation["prompt"]])
    prefix=policy.model.action_prefix(context,[text])
    action=policy.model.sample_actions_from_prefix(context,prefix,
             noise=torch.as_tensor(noise,device=policy.device)[None],num_steps=policy.num_steps)
    return policy.output_transform(dict(state=tensors["state"][0].cpu().numpy(),actions=action[0].float().cpu().numpy()))["actions"]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--cache",type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(4)
    local_rank=int(os.environ.get("LOCAL_RANK",0)); world=int(os.environ.get("WORLD_SIZE",1))
    device=torch.device(f"cuda:{local_rank}");torch.cuda.set_device(device)
    if world>1:dist.init_process_group("nccl")
    rank=dist.get_rank() if world>1 else 0
    policy=create_reach_arm_policy(args.checkpoint,device=device)
    dc=data_config(Pi0Config(pi05=True),split="val")
    raw=create_cached_dataset(dc,50,args.cache)
    dataset=SubtaskTrainingDataset(raw,dc)
    vocabulary=sorted(set(dataset.labels_without_video()))
    report={}
    for reset in (False,True):
        metrics,rows=evaluate_recurrent(policy.model,dataset,device,vocabulary=vocabulary,
                                         reset_each=reset,condition_controls=not reset)
        report["reset_each" if reset else "normal"]=dict(metrics=metrics,predictions=rows)
    manifest=json.loads(Path(dc.split_manifest).read_text())
    records={r["episode_index"]:r for r in manifest["episodes"]}
    rows=[]
    for ep in sorted(raw.episodes)[rank::world]:
        index=int(raw.episode_data_index["from"][raw.episode_positions[ep]])
        sample=raw[index]; record=records[ep]
        obs=dict(prompt=sample["task"],state=np.asarray(sample["observation.state"]).copy(),
                 images={n:np.asarray(sample[f"observation.images.{n}"]).copy()
                         for n in ("cam_high","cam_left_wrist","cam_right_wrist")})
        original_actor=phase_actor(sample["subtask"])[1]
        assert original_actor in {"left","right"}
        for draw in (0,1):
            noise=np.random.default_rng(4250+draw).standard_normal((50,32)).astype(np.float32)
            for goal in ("original","switched"):
                goal_obs=dict(obs)
                expected=original_actor
                if goal=="switched":
                    goal_obs["prompt"]=("Put the sweet potato into the box" if "eggplant" in obs["prompt"]
                                         else "Put the eggplant into the box")
                    if record["layout_visual"].startswith("opposite_sides"):
                        expected="left" if original_actor=="right" else "right"
                output=policy.new_session().infer(goal_obs,noise=noise)
                action_actor,displacement=dominant_arm(output["actions"],obs["state"])
                rows.append(dict(episode=ep,layout=record["layout_visual"],draw=draw,kind="goal_"+goal,
                                 prompt=goal_obs["prompt"],expected_actor=expected,subtask=output["subtask"],
                                 subtask_actor=phase_actor(output["subtask"])[1],
                                 action_actor=action_actor,joint_displacement_LR=displacement))
            phase=phase_actor(sample["subtask"])[0]
            reverse="left" if original_actor=="right" else "right"
            for condition,text in [("correct_actor",sample["subtask"]),("actor_removed",phase),
                                    ("actor_reversed",phase+f" with the {reverse} arm")]:
                action=text_intervention(policy,obs,text,noise)
                actor,displacement=dominant_arm(action,obs["state"])
                rows.append(dict(episode=ep,layout=record["layout_visual"],draw=draw,kind=condition,
                                 expected_actor=original_actor,subtask=text,action_actor=actor,
                                 joint_displacement_LR=displacement))
    if world>1:
        gathered=[None]*world;dist.all_gather_object(gathered,rows)
        rows=[r for part in gathered for r in part]
    if rank==0:
        summary={}
        for kind in sorted({r["kind"] for r in rows}):
            subset=[r for r in rows if r["kind"]==kind]
            summary[kind]=dict(samples=len(subset),correct_action_actor=sum(r["action_actor"]==r["expected_actor"] for r in subset),
                               ambiguous_actions=sum(r["action_actor"]=="ambiguous" for r in subset))
            if kind.startswith("goal_"):
                summary[kind]["correct_subtask_actor"]=sum(r["subtask_actor"]==r["expected_actor"] for r in subset)
        report["initial_role_controls"]=dict(summary=summary,rows=rows,
            scope="Validation initial observations, two common noise draws, fresh memory for each goal. Joint-dominance proxy and offline text interventions, not physical success. Switched-goal actor reference comes from previously audited layouts.")
        args.output.write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(dict(passed=True,output=str(args.output),normal=report["normal"]["metrics"],role_controls=summary)),flush=True)
    if world>1:dist.destroy_process_group()


if __name__=="__main__":main()
