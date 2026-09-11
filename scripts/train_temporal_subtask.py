"""R2 cached-frozen-B training, with separate fit/calibration/original validation.

No image or action model runs during updates. Raw frozen B features are identical
to deployment features; S memory projection, temporal layers, D, and text decoder
remain differentiable. Original B/A and their artifacts remain unchanged.
"""
import argparse
from collections import defaultdict
import contextlib
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('NCCL_ALGO','Ring')
os.environ.setdefault('NCCL_PROTO','Simple')
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist

from verify_temporal_subtask_cpu import load_parts
from openpi.models.subtask_tokenizer import SubtaskTextCodec
from openpi.models_pytorch.temporal_subtask import TemporalSubtask,TemporalConfig
from openpi.training.hierarchy_training import random_state,restore_random_state
from openpi.training.stage1_data import sha256_file
from openpi.training.subtask_transition import BoundarySampler,read_jsonl
from openpi.training.transition_feature_cache import EpisodeFeatures
from openpi.training.transition_sequence import assemble_inputs,batch_inputs,completion_target
from openpi.training.transition_rollout import rollout,previous_rollout_text
from openpi.training.transition_metrics_v2 import causal_sample,report,semantic_gates


def write_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def cosine_lr(step,warmup=500,steps=5000,peak=2.5e-5,end=2.5e-6):
    initial=peak/(warmup+1)
    if step<warmup:return initial+(peak-initial)*step/warmup
    progress=min(1.,max(0.,(step-warmup)/(steps-warmup)))
    return end+(peak-end)*.5*(1+math.cos(math.pi*progress))


def gather(rows,world):
    if world==1:return rows
    pieces=[None]*world;dist.all_gather_object(pieces,rows)
    return sorted([r for p in pieces for r in p],key=lambda r:(r['episode'],r['frame']))


def quality_score(reports,baseline):
    """Calibration selection; final validation is not used for checkpoint choice."""
    dense=reports['dense'];base=baseline['dense'];f=dense['frames'];bf=base['frames']
    safeguards=(f['stable_em']>=bf['stable_em']-.01 and f['far_stage_frames']<=bf['far_stage_frames'])
    miss=sum(r['events']['model_missed_or_unstable'] for r in reports.values())
    late=float(np.mean([r['events']['capped_late_cost_mean'] for k,r in reports.items() if k.startswith('low_')]))
    return [int(not safeguards),miss,-f['boundary_em'],late,f['backward_changes']+f['forward_jumps']]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--assets',type=Path,default=Path('assets/pi05_piper_transition/eggplant_potato/r2_v1'))
    p.add_argument('--cache',type=Path,default=Path('assets/pi05_piper_transition/eggplant_potato/r2_frozen_m3_cache_v1'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--steps',type=int,default=5000);p.add_argument('--global-batch',type=int,default=256)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--warmup',type=int,default=500)
    p.add_argument('--save-every',type=int,default=500);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int)
    p.add_argument('--engineering-smoke',action='store_true')
    p.add_argument('--wandb',action='store_true')
    args=p.parse_args()
    if args.seed!=42:raise ValueError('Only seed42 is authorized')
    if not args.engineering_smoke and (args.steps,args.global_batch,args.warmup,args.save_every)!=(5000,256,500,500):
        raise ValueError('Formal run must use the user fixed common recipe')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.manual_seed(42);np.random.seed(42);random.seed(42)
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    device=torch.device(f'cuda:{local}' if args.device=='cuda' else 'cpu')
    if device.type=='cuda':torch.cuda.set_device(device);torch.backends.cuda.matmul.allow_tf32=False
    if world>1:dist.init_process_group('nccl' if device.type=='cuda' else 'gloo',**({'device_id':device} if device.type=='cuda' else {}))
    barrier=lambda:dist.barrier() if world>1 else None
    if args.global_batch%(args.batch_size*world):raise ValueError('Global batch must exactly match packing')
    accumulation=args.global_batch//(args.batch_size*world)
    protocol=json.loads((args.assets/'protocol.json').read_text())
    parent=Path(protocol['parent_checkpoint'])
    # The protocol hash alone cannot detect a changed sidecar whose recorded hash stayed fixed.
    asset_check=[all(sha256_file(args.assets/name)==digest for name,digest in protocol['files'].items()) if rank==0 else None]
    if world>1:dist.broadcast_object_list(asset_check,src=0)
    if not asset_check[0]:raise ValueError('Training/calibration/validation sidecar bytes changed')
    if not args.engineering_smoke and not (args.cache/'cache_complete.json').exists():raise ValueError('Formal observation cache incomplete')
    identity=json.loads((args.cache/'cache_identity.json').read_text())
    if identity['parent_weights_sha256']!=protocol['parent_weights_sha256']:raise ValueError('Cache/parent mismatch')
    if not args.engineering_smoke and identity['engineering_frames']:raise ValueError('Engineering cache cannot train a formal model')
    if identity['protocol_sha256']!=sha256_file(args.assets/'protocol.json'):raise ValueError('Cache protocol mismatch')
    checked=[sha256_file(parent/'model.safetensors')==protocol['parent_weights_sha256'] if rank==0 else None]
    if world>1:dist.broadcast_object_list(checked,src=0)
    if not checked[0]:raise ValueError('Parent model bytes changed')
    # Every rank reads the frozen vocabulary and original decoder only, not B/A weights.
    initial,embedding=load_parts(parent)
    embedding=embedding.to(device,dtype=torch.bfloat16 if device.type=='cuda' else torch.float32).detach()
    model=TemporalSubtask().to(device);model.decoder.load_state_dict(initial,strict=True);del initial
    codec=SubtaskTextCodec()
    # Fingerprint the sources actually imported from the immutable launch snapshot.
    source_objects=[TemporalSubtask.__init__,assemble_inputs,rollout,report,load_parts,
                    EpisodeFeatures.__init__,BoundarySampler.__init__,random_state]
    source_paths={Path(__file__).resolve(),*[Path(f.__code__.co_filename).resolve() for f in source_objects]}
    sources={str(q):sha256_file(q) for q in sorted(source_paths)}
    visualizer=Path(__file__).with_name('render_temporal_first_step.py')
    sources[str(visualizer)]=sha256_file(visualizer)
    config={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ['resume','stop_after']}
    config.update(world_size=world,accumulation=accumulation,variant='r2_temporal_completion_v1',
                  parent_weights_sha256=protocol['parent_weights_sha256'],protocol_sha256=sha256_file(args.assets/'protocol.json'),
                  cache_identity=identity,sources=sources,source_manifest_sha256=os.environ.get('OPENPI_R2_SOURCE_MANIFEST_SHA256'),
                  runtime={'python':sys.version,'torch':str(torch.__version__),'numpy':np.__version__,
                           'safetensors':safetensors.__version__,'torch_cuda':torch.version.cuda},
                  temporal_config=dataclasses.asdict(model.config),
                  decoder_config=dataclasses.asdict(model.decoder.config),gradient_contract='CE and D -> S/T/D only; B/A absent from optimizer',
                  history_corruption={'probability':.1,'source':'frozen model outputs on other FIT observations; synthetic wrong-active history, never validation labels'},
                  schedule={'warmup':args.warmup,'steps':args.steps,'peak':2.5e-5,'end':2.5e-6,'weight_decay':1e-10})
    own_predictions={};best=None;start=0;resume=None
    optimizer=torch.optim.AdamW(model.parameters(),lr=cosine_lr(0,args.warmup,args.steps),betas=(.9,.95),eps=1e-8,weight_decay=1e-10,foreach=False)
    if args.resume:
        resume=args.output/json.loads((args.output/'latest.json').read_text())['checkpoint']
        meta=json.loads((resume/'metadata.json').read_text())
        if meta['config']!=config:raise ValueError('Resume configuration/source/data mismatch')
        safetensors.torch.load_model(model,resume/'temporal.safetensors',strict=True)
        state=torch.load(resume/f'training_rank_{rank:03d}.pt',map_location='cpu',weights_only=False)
        optimizer.load_state_dict(state['optimizer']);restore_random_state(state['rng'])
        start=meta['completed_steps'];best=meta['best']
        own_predictions={int(k):v for k,v in json.loads((resume/'own_history.json').read_text()).items()}
    elif rank==0:
        args.output.mkdir(parents=True,exist_ok=False);write_json(args.output/'run_config.json',config)
        shutil.copy2(args.assets/'protocol.json',args.output/'protocol.json')
        for name,digest in sources.items():
            source=Path(name);dest=args.output/'sources'/source.resolve().relative_to(Path.cwd())
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
            if sha256_file(dest)!=digest:raise ValueError('Source changed during capture')
    barrier()
    wb=None
    if rank==0 and args.wandb:
        import wandb
        wb=wandb.init(project='agentic-openpi-pi05-subtask',name=args.output.name,dir=str(args.output),config=config,
                      settings=wandb.Settings(x_disable_stats=True))
    def log(value):
        if rank==0:
            with (args.output/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(value,allow_nan=False)+'\n')
            print(json.dumps(value,allow_nan=False),flush=True)
            if wb and value.get('event')=='train':wb.log({k:v for k,v in value.items() if k!='event'},step=value['step'])
    all_train_rows=read_jsonl(args.assets/'frames_train.jsonl')
    fit=set(protocol['fit_episodes'])
    train_rows=[r for r in all_train_rows if r['episode'] in fit]
    if args.engineering_smoke:
        available=[EpisodeFeatures(q.parent) for q in sorted((args.cache/'train').glob('episode_*/complete.json'))]
        train_rows=[r for ep in available for r in ep.rows]
    row_by_index={r['index']:r for r in train_rows}
    episodes={}
    def episode(split,ep):
        key=(split,ep)
        if key not in episodes:episodes[key]=EpisodeFeatures(args.cache/split/f'episode_{ep:06d}')
        return episodes[key]
    if not args.engineering_smoke:
        sampler=BoundarySampler(train_rows,batch_size=args.batch_size,accumulation=accumulation,steps=args.steps,seed=42,world_size=world)
    events=read_jsonl(args.assets/'boundary_index.jsonl')
    def evaluate(split,ids,threshold,*,all_rates=True):
        selected={ep:episode(split,ep) for i,ep in enumerate(sorted(ids)) if i%world==rank}
        event_subset=[e for e in events if e['split']==split and e['episode'] in ids]
        modes=[('low_0.0',.764,0.)]
        if all_rates:modes=[('dense',None,0.),*modes,('low_0.255',.764,.255),('low_0.509',.764,.509)]
        reports={};preds={}
        for name,period,phase in modes:
            outputs=gather(rollout(model,embedding,selected,codec,device,period=period,phase=phase,stay_threshold=threshold),world)
            preds[name]=outputs;reports[name]=report(outputs,event_subset)
        return reports,preds
    def reference(split,ids):
        outputs=[r for ep in sorted(ids) for r in episode(split,ep).rows]
        subset=[e for e in events if e['split']==split and e['episode'] in ids]
        return {'dense':report(outputs,subset),**{f'low_{phase}':report(causal_sample(outputs,.764,phase),subset) for phase in [0.,.255,.509]}}
    calibration=set(protocol['calibration_episodes'])
    calibration_base=reference('train',calibration) if not args.engineering_smoke else None
    corruption_pool=defaultdict(set)
    if not args.engineering_smoke:
        for ep in sorted(fit):
            for row in episode('train',ep).rows:
                if row.get('status')=='ok' and row['prediction']:corruption_pool[row['task']].add(row['prediction'])
    corruption_pool={task:sorted(values) for task,values in corruption_pool.items()}
    def save(step):
        temporary=args.output/f'.step_{step:06d}.tmp';target=args.output/f'step_{step:06d}'
        if rank==0:temporary.mkdir(exist_ok=False)
        barrier()
        torch.save({'optimizer':optimizer.state_dict(),'rng':random_state()},temporary/f'training_rank_{rank:03d}.pt')
        if rank==0:
            safetensors.torch.save_model(model,temporary/'temporal.safetensors')
            write_json(temporary/'own_history.json',own_predictions)
            write_json(temporary/'metadata.json',{'schema_version':2,'variant':'r2_temporal_completion_v1','completed_steps':step,
                       'config':config,'best':best,'temporal_sha256':sha256_file(temporary/'temporal.safetensors'),
                       'engineering_smoke':args.engineering_smoke,'physical_readiness_reviewed':False})
        barrier()
        if rank==0:temporary.rename(target);write_json(args.output/'latest.json',{'checkpoint':target.name})
        barrier()
    for step in range(start,args.steps):
        rng=np.random.default_rng(np.random.SeedSequence([42,step,1701]))
        indices=(sampler.global_indices(step).flatten().tolist() if not args.engineering_smoke else rng.choice(list(row_by_index),args.global_batch).tolist())
        all_examples=[]
        for offset,index in enumerate(indices):
            row=row_by_index[index];ep=episode('train',row['episode']);frame=row['frame']
            history=ep.past(frame,float(rng.uniform(.55,1.0)),3)
            previous=history[-1] if rng.random()>.2 else frame-1 if frame else None
            active=ep.rows[previous]['prediction'] if previous is not None else ''
            own=previous_rollout_text(own_predictions,row['episode'],frame)
            if own is not None and rng.random()<.8:active=own
            if row['task'] in corruption_pool and rng.random()<.1:active=str(rng.choice(corruption_pool[row['task']]))
            if rng.random()<.15:active=''
            done=completion_target(ep.rows,frame,active)
            all_examples.append((row,history,active,done))
        global_done=sum(item[3]!=-100 for item in all_examples)
        local_examples=[x for i,x in enumerate(all_examples) if (i//args.batch_size)%world==rank]
        assert len(local_examples)*world==args.global_batch
        lr=cosine_lr(step,args.warmup,args.steps)
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True);text_total=0.;done_total=0.
        for offset in range(0,len(local_examples),args.batch_size):
            micro=local_examples[offset:offset+args.batch_size]
            inputs=batch_inputs([assemble_inputs(episode('train',r['episode']),r['frame'],hist,active,codec) for r,hist,active,_ in micro],device)
            ids,mask=codec.targets([r['label'] for r,_,_,_ in micro]);done=torch.tensor([item[3] for item in micro],device=device)
            amp=torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else contextlib.nullcontext()
            with amp:
                result=model(inputs,torch.tensor(ids,device=device),torch.tensor(mask,device=device),done,embedding)
                loss=result['text_loss']*len(micro)/args.global_batch+.3*result['completion_loss']*result['completion_valid']/max(1,global_done)
            loss.backward();text_total+=float(result['text_loss'].detach())*len(micro);done_total+=float(result['completion_loss'].detach())*int(result['completion_valid'])
        for parameter in model.parameters():
            if parameter.grad is None:raise RuntimeError('Trainable parameter unexpectedly absent from loss graph')
            if world>1:dist.all_reduce(parameter.grad,op=dist.ReduceOp.SUM)
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True));optimizer.step()
        values=torch.tensor([text_total,done_total],device=device,dtype=torch.float64)
        if world>1:dist.all_reduce(values)
        completed=step+1
        if completed==1 and rank==0:
            row,history,active,done=all_examples[0]
            first_inputs=batch_inputs([assemble_inputs(episode('train',row['episode']),row['frame'],history,active,codec)],device)
            amp=torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else contextlib.nullcontext()
            with amp:generation,probabilities=model.generate(first_inputs,embedding)
            texts,statuses=codec.decode(generation)
            first=args.output/'first_step';first.mkdir(exist_ok=False)
            write_json(first/'sample.json',{'row':row,'history':history,'active_text':active,'completion_target':done,
                                           'prediction':texts[0],'status':statuses[0],'completion_probabilities':probabilities[0].float().tolist()})
            subprocess.run([sys.executable,str(visualizer),'--directory',str(first)],check=True)
        if completed==1:barrier()
        if completed<=50 or completed%50==0:
            log({'event':'train','step':completed,'text_loss':float(values[0])/args.global_batch,'completion_loss':float(values[1])/max(1,global_done),
                 'lr':lr,'grad_norm':norm,'residual_scale':float(model.residual_scale.detach())})
        if completed%args.save_every==0 or completed==args.steps:
            if not args.engineering_smoke:
                saved_rng=random_state()
                options=[]
                for threshold in [1.01,.8,.95]:
                    quick,_=evaluate('train',calibration,threshold,all_rates=False)
                    q=quick['low_0.0'];f=q['frames'];e=q['events']
                    options.append(([e['model_missed_or_unstable'],f['forward_jumps']+f['backward_changes'],-f['boundary_em'],e['capped_late_cost_mean']],threshold))
                threshold=min(options)[1]
                metrics,_=evaluate('train',calibration,threshold)
                score=quality_score(metrics,calibration_base)
                if best is None or score<best['score']:best={'step':completed,'score':score,'stay_threshold':threshold}
                if rank==0:write_json(args.output/f'calibration_{completed:06d}.json',{'reports':metrics,'threshold_trials':options,'selected_threshold':threshold,'score':score})
                log({'event':'calibration','step':completed,'boundary_em':metrics['dense']['frames']['boundary_em'],'score':score,'best':best})
                # Refresh model-owned training histories, without using calibration/val targets as inputs.
                selected={ep:episode('train',ep) for i,ep in enumerate(sorted(fit)) if i%world==rank}
                predictions=gather(rollout(model,embedding,selected,codec,device,period=.764,stay_threshold=threshold),world)
                own_predictions=defaultdict(list)
                for r in predictions:own_predictions[r['episode']].append({'frame':r['frame'],'prediction':r['prediction']})
                restore_random_state(saved_rng)
            save(completed)
        elif completed==50 or completed==args.stop_after:save(completed)
        if completed==50 and rank==0:write_json(args.output/'milestone_50.json',{'completed_steps':50,'instruction':'No assistant step polling after this point; finite evaluation continues.'})
        if completed==args.stop_after:
            log({'event':'stopped_at_checkpoint','step':completed})
            if wb:wb.finish()
            if world>1:dist.destroy_process_group()
            return
    if not args.engineering_smoke:
        safetensors.torch.load_model(model,args.output/f"step_{best['step']:06d}"/'temporal.safetensors',strict=True)
        val_ids=set(r['episode'] for r in read_jsonl(args.assets/'frames_val.jsonl'))
        validation,predictions=evaluate('val',val_ids,best['stay_threshold'])
        baseline=reference('val',val_ids);gates=semantic_gates(validation,baseline)
        if rank==0:
            write_json(args.output/'selected_validation.json',validation);write_json(args.output/'baseline_validation.json',baseline)
            for cadence,preds in predictions.items():write_json(args.output/f'predictions_{cadence}.json',preds)
            write_json(args.output/'semantic_gates.json',gates)
            write_json(args.output/'best.json',{'checkpoint':f"step_{best['step']:06d}",**best})
            write_json(args.output/'training_complete.json',{'completed_steps':args.steps,'seed':42,'best':best,'semantic_passed':gates['passed'],
                                                           'action_gate_pending':True,'native_policy_gate_pending':True,'goal_achieved':False})
    log({'event':'complete','steps':args.steps,'best':best})
    if wb:wb.finish()
    if world>1:dist.destroy_process_group()


if __name__=='__main__':
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):main()
