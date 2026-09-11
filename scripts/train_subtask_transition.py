"""Seed42 R1 S-only refinement with immutable M3 B/A and autonomous evaluation."""
import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

os.environ.setdefault("JAX_PLATFORMS","cpu")
os.environ.setdefault("HF_HUB_OFFLINE","1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
os.environ.setdefault("NCCL_ALGO","Ring")
os.environ.setdefault("NCCL_PROTO","Simple")
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
from train_subtask_pytorch import learning_rate
from visualize_subtask_step import render_first_step
from openpi import transforms
from openpi.policies.piper_policy import JOINT_MASK
from openpi.policies.subtask_policy import create_subtask_policy
from openpi.training import config as config_lib, data_loader
from openpi.training.hierarchy_training import random_state, restore_random_state
from openpi.training.runtime_provenance import capture_runtime, archive_runtime
from openpi.training.stage1_data import sha256_file
from openpi.training.subtask_batch import SubtaskTrainingDataset, collate_subtask
from openpi.training.subtask_transition import BoundarySampler, read_jsonl, text_loss_per_example, evaluation_report, weighted_numerator


def write_json(path,value):
    path=Path(path)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    tmp.replace(path)


def tensor_digest(module):
    digest=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((tuple(value.shape),value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class TextObjective(torch.nn.Module):
    def __init__(self,decoder):
        super().__init__()
        self.decoder=decoder
    def forward(self,ids,mask,memory,memory_mask,embedding):
        return text_loss_per_example(self.decoder,ids,mask,memory,memory_mask,embedding)


def build_dataset(dc,model_config,rows):
    raw=data_loader.create_torch_dataset(dc,model_config.action_horizon,model_config)
    dataset=SubtaskTrainingDataset(raw,dc)
    episodes=list(raw.hf_dataset["episode_index"])
    frames=list(raw.hf_dataset["frame_index"])
    labels=dataset.labels_without_video()
    if len(rows)!=len(dataset) or any((int(e),int(f),l)!=(r["episode"],r["frame"],r["label"])
            for e,f,l,r in zip(episodes,frames,labels,rows,strict=True)):
        raise ValueError("D0 scalar table and runtime dataset ordering/labels differ")
    return dataset


@torch.no_grad()
def validate(model,dataset,rows,boundaries,device,rank,world,batch_size,limit=0):
    state=random_state()
    previous=model.training
    model.eval()
    result=[]
    indices=list(range(len(rows)))[rank::world]
    if limit:
        indices=indices[:limit]
    try:
        for offset in range(0,len(indices),batch_size):
            ids=indices[offset:offset+batch_size]
            batch=collate_subtask([dataset[i] for i in ids]).to(device)
            context=model.prepare_context(batch.observation,batch.global_prompts)
            texts,statuses,_=model.generate_subtask(context)
            result.extend(dict(rows[i],prediction=t,status=s) for i,t,s in zip(ids,texts,statuses,strict=True))
        if world>1:
            combined=[None]*world
            dist.all_gather_object(combined,result)
            result=[r for part in combined for r in part]
        result.sort(key=lambda r:r["index"])
        return evaluation_report(result,boundaries),result
    finally:
        model.train(previous)
        restore_random_state(state)


@torch.no_grad()
def action_diagnostic(model,dataset,rows,events,dc,device,rank,world,limit=0):
    """Paired native first15 generated/old/new/drop on fixed boundary examples."""
    state=random_state()
    previous=model.training
    model.eval()
    inverse=transforms.Unnormalize(dc.norm_stats,use_quantiles=dc.use_quantile_norm)
    absolute=transforms.AbsoluteActions(JOINT_MASK)
    by_key={(r["episode"],r["frame"]):r["index"] for r in rows}
    selected=[]
    seen=set()
    for event in events:
        key=(event["task"],event["transition"])
        if key in seen:
            continue
        seen.add(key)
        for delta in [-3,0,3,15]:
            frame=event["frame"]+delta
            if (event["episode"],frame) in by_key:
                selected.append((by_key[(event["episode"],frame)],event,"boundary"))
    for index in np.unique(np.linspace(0,len(rows)-1,min(128,len(rows)),dtype=int)):
        selected.append((int(index),{"event_id":"natural_validation","old":rows[index]["label"],"new":rows[index]["label"]},"natural"))
    if limit:
        selected=selected[:limit]
    output=[]
    try:
        for index,event,kind in selected[rank::world]:
            batch=collate_subtask([dataset[index]]).to(device)
            context=model.prepare_context(batch.observation,batch.global_prompts)
            text,status,_=model.generate_subtask(context)
            noise=torch.randn(batch.actions.shape,generator=torch.Generator().manual_seed(42+index)).to(device)
            obs_state=batch.observation.state[0].float().cpu().numpy()
            target=absolute(inverse({"state":obs_state.copy(),"actions":batch.actions[0].cpu().numpy().copy()}))["actions"][:,:14]
            valid=min(15,rows[index]["episode_length"]-rows[index]["frame"])
            entry={"index":index,"episode":rows[index]["episode"],"frame":rows[index]["frame"],"event_id":event["event_id"],
                   "generated_text":text[0],"status":status[0],"valid_first_steps":valid,"kind":kind,"conditions":{}}
            for name,condition in [("generated",text[0]),("old",event["old"]),("new",event["new"]),("drop","")]:
                if kind=="natural" and name!="generated":continue
                prefix=model.action_prefix(context,[condition])
                actions=model.sample_actions_from_prefix(context,prefix,noise=noise.clone(),num_steps=10)[0].float().cpu().numpy()
                native=absolute(inverse({"state":obs_state.copy(),"actions":actions.copy()}))["actions"][:valid,:14]
                truth=target[:valid]
                commanded=native.copy()
                commanded[:,[6,13]]=np.clip(commanded[:,[6,13]],0,.09)
                entry["conditions"][name]={"joint_mse":float(np.mean((native[:,JOINT_MASK[:14]]-truth[:,JOINT_MASK[:14]])**2)),
                    "gripper_mse":float(np.mean((native[:,[6,13]]-truth[:,[6,13]])**2)),
                    "native_first15":native.tolist(),"command_first15":commanded.tolist()}
            output.append(entry)
        if world>1:
            combined=[None]*world
            dist.all_gather_object(combined,output)
            output=[r for part in combined for r in part]
        output.sort(key=lambda r:r["index"])
        means={name:{metric:float(np.mean([r["conditions"][name][metric] for r in output if name in r["conditions"]])) for metric in ["joint_mse","gripper_mse"]}
               for name in ["generated","old","new","drop"]} if output else {}
        grouped={kind:{metric:float(np.mean([r["conditions"]["generated"][metric] for r in output if r["kind"]==kind]))
                       for metric in ["joint_mse","gripper_mse"]} for kind in sorted({r["kind"] for r in output})}
        return {"scope":"Fixed validation boundary examples; no physical completion labels or success claims",
                "examples":len(output),"means":means,"generated_by_group":grouped,"rows":output}
    finally:
        model.train(previous)
        restore_random_state(state)


def selection(report,baseline):
    guards=report["stable_exact_match"]>=baseline["stable_exact_match"]-.01
    score=[]
    for name,current in [("dense",report["dense"]), *[("low_"+k,v) for k,v in report["low_rate"].items()]]:
        base=baseline["dense"] if name=="dense" else baseline["low_rate"][name[4:]]
        guards &= current["early_over_100ms"]<=base["early_over_100ms"]+.02*base["events"]
        guards &= current["failed_or_unconfirmed"]<=base["failed_or_unconfirmed"]
        if name!="dense":
            score.append(current["failed_or_unconfirmed"])
    p95s=[v["late_p50_p95_seconds"][1] if v["late_p50_p95_seconds"] else 999 for v in report["low_rate"].values()]
    score.extend([float(np.mean(p95s)),report["dense"]["late_p50_p95_seconds"][1] if report["dense"]["late_p50_p95_seconds"] else 999,
                  report["dense"]["transient_correct_runs"],-report["stable_exact_match"]])
    return bool(guards),score


def train(args):
    if args.seed!=42 or args.steps<1:
        raise ValueError("R1 is fixed to seed42 and positive steps")
    if args.val_limit and not args.engineering_smoke:
        raise ValueError("Research validation must cover all fixed val frames")
    torch.set_num_threads(args.cpu_threads)
    torch.use_deterministic_algorithms(True)
    rank=int(os.environ.get("RANK",0));world=int(os.environ.get("WORLD_SIZE",1));local=int(os.environ.get("LOCAL_RANK",0))
    device=torch.device(f"cuda:{local}" if args.device=="cuda" else "cpu")
    if device.type=="cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32=False
    if world>1:
        dist.init_process_group("nccl" if device.type=="cuda" else "gloo",**({"device_id":device} if device.type=="cuda" else {}))
    def barrier():
        if world>1:dist.barrier()
    random.seed(42+rank);np.random.seed(42+rank);torch.manual_seed(42+rank)
    protocol=json.loads((args.assets/"protocol.json").read_text())
    for name,digest in protocol["files"].items():
        if sha256_file(args.assets/name)!=digest:raise ValueError("D0 asset changed: "+name)
    train_rows=read_jsonl(args.assets/"frames_train.jsonl")
    val_rows=read_jsonl(args.assets/"frames_val.jsonl")
    events=[e for e in read_jsonl(args.assets/"boundary_index.jsonl") if e["split"]=="val"]
    baseline=json.loads((args.assets/"baseline_report.json").read_text())
    cfg=config_lib.get_config("pi05_piper_stage1")
    parent=json.loads((args.initialize_from/"metadata.json").read_text())
    for key in ["split_sha256","norm_sha256"]:
        if parent["config"][key]!=protocol[key]:raise ValueError("Parent/data mismatch")
    payload=[None]
    if rank==0:
        actual=sha256_file(args.initialize_from/"model.safetensors")
        if actual!=protocol["parent_weights_sha256"]:raise ValueError("Wrong M3 baseline")
        config={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ["resume","stop_after"]}
        source_paths={Path(name) for name in parent["config"]["sources"]}
        source_paths.update([Path(__file__),Path("scripts/audit_subtask_transitions.py"),Path("scripts/visualize_subtask_step.py"),
                             Path("scripts/serve_transition_policy.py"),Path("src/openpi/training/subtask_transition.py"),
                             Path("src/openpi/policies/subtask_transition_policy.py"),Path("scripts/sync_transition_wandb.py"),Path("scripts/sync_subtask_wandb.py")])
        config.update(world_size=world,stage="r1",variant="r1_boundary_ce_v1",parent_weights_sha256=actual,
                      split_sha256=protocol["split_sha256"],norm_sha256=protocol["norm_sha256"],
                      protocol_sha256=sha256_file(args.assets/"protocol.json"),runtime=capture_runtime(),
                      sources={str(p.resolve()):sha256_file(p) for p in sorted(source_paths)},
                      label_shift_frames=0,gradient_owners="S only; B/A frozen",optimizer="fresh replicated AdamW",
                      deterministic_training="math SDPA; fixed parameter-order allreduce; Ring/Simple; CUBLAS :4096:8")
        payload[0]=json.loads(json.dumps(config))
    if world>1:dist.broadcast_object_list(payload,src=0)
    config=payload[0]
    resume=None;start=0;best=None
    if args.resume:
        resume=args.output/json.loads((args.output/"latest.json").read_text())["checkpoint"]
        saved=json.loads((resume/"metadata.json").read_text())
        if saved["config"]!=config:raise ValueError("Exact resume provenance changed")
        start=saved["completed_steps"];best=saved["best"]
    elif rank==0:
        args.output.mkdir(parents=True,exist_ok=False)
        write_json(args.output/"run_config.json",config)
        shutil.copytree(args.assets,args.output/"transition_assets")
        archive_runtime(config["runtime"],args.output/"runtime_sources")
        for name,digest in config["sources"].items():
            p=Path(name);dest=args.output/"sources"/p.resolve().relative_to(Path.cwd())
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
            if sha256_file(dest)!=digest:raise ValueError("Source changed while archiving")
    barrier()
    def log(row):
        if rank==0:
            with (args.output/"metrics.jsonl").open("a") as f:f.write(json.dumps(row,allow_nan=False)+"\n")
            print(json.dumps(row,allow_nan=False),flush=True)
    policy=create_subtask_policy(args.initialize_from,device=str(device),num_steps=10)
    model=policy.model
    model.set_stage("m1")
    dc=cfg.data.create(cfg.assets_dirs,model.base.config)
    dataset=build_dataset(dc,model.base.config,train_rows)
    val_dc=dataclasses.replace(dc,split="val")
    val_dataset=build_dataset(val_dc,model.base.config,val_rows)
    if resume:safetensors.torch.load_model(model.decoder,resume/"decoder.safetensors",strict=True)
    objective=TextObjective(model.decoder)
    optimizer=torch.optim.AdamW(model.subtask_parameters(),lr=args.lr,betas=(.9,.95),eps=1e-8,weight_decay=.01,foreach=False)
    if resume:
        state=torch.load(resume/f"training_rank_{rank:03d}.pt",map_location="cpu",weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        restore_random_state(state["rng"])
    initial=[tensor_digest(model.base) if rank==0 else None]
    if world>1:dist.broadcast_object_list(initial,src=0)
    initial_hash=initial[0]
    if resume and initial_hash!=saved["initial_base_tensor_hash"]:raise ValueError("Resume base changed")
    sampler=BoundarySampler(train_rows,batch_size=args.batch_size,accumulation=args.accumulation,steps=args.steps,start=start,seed=42,rank=rank,world_size=world)
    loader=torch.utils.data.DataLoader(dataset,batch_sampler=sampler,collate_fn=collate_subtask,num_workers=args.workers,
                                     **({"multiprocessing_context":"spawn","persistent_workers":True} if args.workers else {}),
                                     generator=torch.Generator().manual_seed(42+rank))
    iterator=iter(loader)
    def save(step):
        temporary=args.output/f".step_{step:06d}.tmp";target=args.output/f"step_{step:06d}"
        if rank==0:
            if target.exists():raise FileExistsError(target)
            temporary.mkdir(exist_ok=False)
        barrier()
        torch.save({"optimizer":optimizer.state_dict(),"rng":random_state(),"rank":rank,"world_size":world},temporary/f"training_rank_{rank:03d}.pt")
        if rank==0:
            base_hash=tensor_digest(model.base)
            if base_hash!=initial_hash:raise RuntimeError("Frozen B/A tensors changed")
            safetensors.torch.save_model(model.decoder,temporary/"decoder.safetensors")
            meta={"schema_version":1,"variant":"r1_boundary_ce_v1","completed_steps":step,"config":config,"best":best,
                  "initial_base_tensor_hash":initial_hash,"base_tensor_hash":base_hash,"frozen_base_equal":True,
                  "decoder_sha256":sha256_file(temporary/"decoder.safetensors")}
            write_json(temporary/"metadata.json",meta)
        barrier()
        if rank==0:
            temporary.rename(target)
            write_json(args.output/"latest.json",{"checkpoint":target.name,"completed_steps":step})
        barrier()
        return target
    log({"event":"ready","stage":"r1","start":start,"effective_batch":args.batch_size*args.accumulation*world,
         "train_frames":len(dataset),"validation_frames":len(val_dataset),"seed":42,"physical_readiness_reviewed":False})
    if rank==0:
        import psutil
        write_json(args.output/"trainer_process.json",{"pid":os.getpid(),"created":psutil.Process().create_time()})
        if args.wandb:
            with (args.output/"wandb_bridge.log").open("a") as stream:
                subprocess.Popen([sys.executable,"scripts/sync_transition_wandb.py","--output",str(args.output.resolve())],
                                 stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    if not resume:
        diagnostics=action_diagnostic(model,val_dataset,val_rows,events,val_dc,device,rank,world,limit=2 if args.engineering_smoke else 0)
        if rank==0:write_json(args.output/"r0_action_diagnostic.json",diagnostics)
        log({"event":"baseline_action_diagnostic","examples":diagnostics["examples"],"means":diagnostics["means"]})
    for step in range(start,args.steps):
        begun=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        lr=learning_rate(step,args.warmup,args.steps,args.lr)
        for group in optimizer.param_groups:group["lr"]=lr
        # All weights are one in R1a. Normalize across every rank and microbatch.
        batches=[next(iterator).to(device) for _ in range(args.accumulation)]
        denominator=sum(b.target_mask.any(1).sum() for b in batches).to(dtype=torch.float32)
        if world>1:dist.all_reduce(denominator)
        if denominator.item()<=0:raise ValueError("Global update has no valid supervision")
        numerator_total=torch.zeros((),device=device)
        for micro,batch in enumerate(batches):
            context=model.prepare_context(batch.observation,batch.global_prompts)
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                losses,valid=objective(batch.target_ids,batch.target_mask,context.memory,context.memory_mask,model.embedding_weight)
                numerator,_=weighted_numerator(losses,valid,torch.ones_like(losses))
                loss=numerator*world/denominator
                if not torch.isfinite(loss):raise FloatingPointError("Nonfinite text loss")
                loss.backward()
            numerator_total+=numerator.detach()
        # Stable parameter order avoids DDP's first-iteration bucket rebuild
        # changing floating-point reduction order after a process restart.
        if world>1:
            for parameter in model.subtask_parameters():
                if parameter.grad is None:raise RuntimeError("Missing S gradient")
                dist.all_reduce(parameter.grad)
                parameter.grad.div_(world)
        norm=float(torch.nn.utils.clip_grad_norm_(model.subtask_parameters(),1,error_if_nonfinite=True))
        if step==start:
            if any(p.grad is not None for p in model.base.parameters()):raise RuntimeError("B/A received gradients")
            if not norm>0:raise RuntimeError("S received no gradient")
            log({"event":"gradient_gate","step":step+1,"frozen_base_gradients_absent":True,"subtask_grad_norm":norm})
        optimizer.step()
        if world>1:dist.all_reduce(numerator_total)
        completed=step+1
        row={"event":"train","step":completed,"loss_subtask":float(numerator_total/denominator),"grad_norm":norm,
             "lr":lr,"examples":int(denominator),"seconds":time.perf_counter()-begun}
        log(row)
        if completed==1:
            if rank==0:
                render_first_step(model,batches[0],dc,args.output/"first_step",step=1,origin="R1 live first update; frozen M3 B/A",limit=1)
                indices=sampler.global_indices(step).flatten().tolist()
                write_json(args.output/"first_step"/"sampling.json",[train_rows[i] for i in indices])
            barrier()
        if completed%args.eval_every==0 or completed==args.steps:
            metrics,predictions=validate(model,val_dataset,val_rows,events,device,rank,world,args.eval_batch_size,args.val_limit)
            eligible,score=selection(metrics,baseline)
            if args.engineering_smoke:eligible=False
            if eligible and (best is None or score<best["score"]):best={"step":completed,"score":score}
            if rank==0:
                write_json(args.output/f"validation_{completed:06d}.json",metrics)
                write_json(args.output/f"predictions_{completed:06d}.json",predictions)
            log({"event":"validation","step":completed,"exact_match":metrics["exact_match"],
                 "stable_exact_match":metrics["stable_exact_match"],"boundary_exact_match":metrics["boundary_exact_match"],
                 "proxy_eligible":eligible,"score":score})
        if completed==50 or completed%args.eval_every==0 or completed in [args.steps,args.stop_after]:
            checkpoint=save(completed)
            if rank==0 and best and best["step"]==completed:write_json(args.output/"best.json",{"checkpoint":checkpoint.name,**best})
        if completed==50 and rank==0:
            write_json(args.output/"milestone_50.json",dict(row,checkpoint="step_000050",instructions="No further assistant polling; autonomous training/evaluation continues"))
        if args.stop_after and completed==args.stop_after:
            log({"event":"stopped_at_checkpoint","completed_steps":completed})
            if world>1:dist.destroy_process_group()
            return
    # Autonomous post-training check, including selected candidate's native actions.
    selected=best["step"] if best else args.steps
    target=args.output/f"step_{selected:06d}"
    safetensors.torch.load_model(model.decoder,target/"decoder.safetensors",strict=True)
    diagnostic=action_diagnostic(model,val_dataset,val_rows,events,val_dc,device,rank,world,limit=2 if args.engineering_smoke else 0)
    if rank==0:
        write_json(args.output/"selected_action_diagnostic.json",diagnostic)
        base_action=json.loads((args.output/"r0_action_diagnostic.json").read_text())
        action_pass=all(diagnostic["generated_by_group"][group][key]<=base_action["generated_by_group"][group][key]*1.05+1e-12
                        for group in diagnostic["generated_by_group"] for key in ["joint_mse","gripper_mse"])
        metrics=json.loads((args.output/f"validation_{selected:06d}.json").read_text())
        base_p95=[v["late_p50_p95_seconds"][1] for v in baseline["low_rate"].values() if v["late_p50_p95_seconds"]]
        new_p95=[v["late_p50_p95_seconds"][1] for v in metrics["low_rate"].values() if v["late_p50_p95_seconds"]]
        improvement=bool(len(new_p95)==len(base_p95) and new_p95 and np.mean(new_p95)<=.7*np.mean(base_p95))
        qualifying=bool(best and action_pass and improvement)
        metadata=json.loads((target/"metadata.json").read_text())
        result={"checkpoint":target.name,"decoder_sha256":metadata["decoder_sha256"],"proxy_gates_passed":qualifying,
                "action_gate_passed":action_pass,"label_proxy_p95_reduction_30percent":improvement,"physical_readiness_reviewed":False,"status":"awaiting_human_review_and_robot_trial" if qualifying else "no_qualified_candidate",
                "seed":42,"completed_steps":args.steps,"best":best}
        write_json(args.output/"candidate.json",result)
        (args.output/"RESULT.md").write_text("# R1 seed42 result\n\n"+json.dumps(result,indent=2)+"\n\nPhysical readiness labels and robot task success were not evaluated.\n")
    barrier()
    log({"event":"complete","completed_steps":args.steps,"best":best})
    if world>1:dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--initialize-from",type=Path,default=Path("checkpoints/pi05_piper_stage1/m3_pilot_seed42/step_003500"))
    p.add_argument("--assets",type=Path,default=Path("assets/pi05_piper_transition/eggplant_potato/d0_v1"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--steps",type=int,default=1000)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--batch-size",type=int,default=2)
    p.add_argument("--accumulation",type=int,default=2)
    p.add_argument("--lr",type=float,default=1e-5)
    p.add_argument("--warmup",type=int,default=50)
    p.add_argument("--eval-every",type=int,default=250)
    p.add_argument("--eval-batch-size",type=int,default=4)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--cpu-threads",type=int,default=4)
    p.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    p.add_argument("--engineering-smoke",action="store_true")
    p.add_argument("--val-limit",type=int,default=0)
    p.add_argument("--stop-after",type=int,default=0)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--wandb",action="store_true")
    args=p.parse_args()
    train(args)

if __name__=="__main__":
    main()
