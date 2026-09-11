"""Paired native action check: every validation boundary, causal low-rate prior.

Only offline oracle diagnostics receive old/new labels. Deployment generation
receives a current observation, actual past observations, and past model text.
"""
import argparse
import contextlib
import dataclasses
import json
import os
from pathlib import Path
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('HF_HUB_OFFLINE','1')
import numpy as np
import torch
import torch.distributed as dist

from train_subtask_transition import build_dataset
from openpi import transforms
from openpi.policies.piper_policy import JOINT_MASK
from openpi.policies.temporal_subtask_policy import create_temporal_policy
from openpi.training import config as config_lib
from openpi.training.subtask_batch import collate_subtask
from openpi.training.subtask_transition import read_jsonl
from openpi.training.transition_feature_cache import EpisodeFeatures,compress_memory
from openpi.training.transition_sequence import assemble_inputs,batch_inputs,choose_text,select_observed_history


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=Path('assets/pi05_piper_transition/eggplant_potato/r2_frozen_m3_cache_v1'))
    p.add_argument('--device',default='cuda');p.add_argument('--engineering-smoke',action='store_true');args=p.parse_args()
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    device=torch.device(f'cuda:{local}' if args.device=='cuda' else 'cpu');torch.set_num_threads(4)
    if device.type=='cuda':torch.cuda.set_device(device)
    if world>1:dist.init_process_group('nccl' if device.type=='cuda' else 'gloo',**({'device_id':device} if device.type=='cuda' else {}))
    policy=create_temporal_policy(args.checkpoint,device=str(device),require_candidate=False,allow_engineering=args.engineering_smoke)
    metadata=json.loads((args.checkpoint/'metadata.json').read_text())
    assets=Path(metadata['config']['assets'])
    if not args.engineering_smoke and json.loads((args.checkpoint.parent/'best.json').read_text())['checkpoint']!=args.checkpoint.name:
        raise ValueError('Action replay must use the same calibration-selected checkpoint as its causal prediction history')
    rows=read_jsonl(assets/'frames_val.jsonl');events=[e for e in read_jsonl(assets/'boundary_index.jsonl') if e['split']=='val']
    low=([] if args.engineering_smoke else json.loads((args.checkpoint.parent/'predictions_low_0.0.json').read_text()))
    by_ep={ep:[r for r in low if r['episode']==ep] for ep in {r['episode'] for r in rows}}
    by_key={(r['episode'],r['frame']):r for r in rows}
    anchors=[]
    for event in events:
        for delta in [-3,0,3,15]:
            row=by_key.get((event['episode'],event['frame']+delta))
            if row:anchors.append((row,event,'boundary'))
    # Six or seven natural examples per episode, with task-stratified remainder.
    rng=np.random.default_rng(42);extra=set()
    for task in sorted({r['task'] for r in rows}):
        eps=sorted({r['episode'] for r in rows if r['task']==task});extra.update(rng.choice(eps,4,replace=False).tolist())
    for ep in sorted(by_ep):
        group=[r for r in rows if r['episode']==ep];count=6+int(ep in extra)
        for index in np.linspace(0,len(group)-1,count+2,dtype=int)[1:-1]:
            r=group[int(index)];anchors.append((r,{'event_id':'natural','old':r['label'],'new':r['label']},'natural'))
    if args.engineering_smoke:
        first=EpisodeFeatures(sorted((args.cache/'val').glob('episode_*/complete.json'))[0].parent).rows[0]
        anchors=[(first,{'event_id':'engineering','old':first['label'],'new':first['label']},'natural')]
    cfg=config_lib.get_config('pi05_piper_stage1');dc=dataclasses.replace(cfg.data.create(cfg.assets_dirs,policy.base.model.base.config),split='val')
    dataset=build_dataset(dc,policy.base.model.base.config,rows)
    inverse=transforms.Unnormalize(dc.norm_stats,use_quantiles=dc.use_quantile_norm);absolute=transforms.AbsoluteActions(JOINT_MASK)
    caches={};results=[]
    for row,event,kind in anchors[rank::world]:
        ep,frame=row['episode'],row['frame']
        if ep not in caches:caches[ep]=EpisodeFeatures(args.cache/'val'/f'episode_{ep:06d}')
        previous=[r for r in by_ep[ep] if r['frame']<frame]
        positions=select_observed_history([r['timestamp'] for r in previous],row['timestamp'])
        history=[previous[i]['frame'] if i is not None else None for i in positions]
        active=previous[-1]['prediction'] if previous else ''
        inputs=batch_inputs([assemble_inputs(caches[ep],frame,history,active,policy.base.model.codec)],device)
        batch=collate_subtask([dataset[row['index']]]).to(device)
        context=policy.base.model.prepare_context(batch.observation,batch.global_prompts)
        # Current features use the actual native inference path, rather than assuming cache/batch parity.
        summary,mask=compress_memory(context.memory,context.memory_mask)
        inputs['memory']=context.memory;inputs['memory_mask']=context.memory_mask
        inputs['summaries'][:,-1]=summary;inputs['summary_masks'][:,-1]=mask;inputs['states'][:,-1]=context.state
        amp=torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else contextlib.nullcontext()
        with amp:generation,prob=policy.temporal.generate(inputs,policy.base.model.embedding_weight)
        text,status=policy.base.model.codec.decode(generation)
        selected,_=choose_text(text[0],status[0],active,prob[0],policy.stay_threshold)
        r0,_,_=policy.base.model.generate_subtask(context)
        noise=torch.randn(batch.actions.shape,generator=torch.Generator().manual_seed(42+row['index'])).to(device)
        state=batch.observation.state[0].float().cpu().numpy()
        valid=min(15,row['episode_length']-frame)
        target=absolute(inverse({'state':state.copy(),'actions':batch.actions[0].cpu().numpy().copy()}))['actions'][:valid,:14]
        entry={'episode':ep,'frame':frame,'kind':kind,'event_id':event['event_id'],'r0_text':r0[0],'r2_text':selected,
               'previous_active':active,'history_frames':history,'valid_steps':valid,'conditions':{}}
        conditions=[('r0_generated',r0[0]),('r2_generated',selected)]
        if kind=='boundary':conditions.extend([('old',event['old']),('new',event['new']),('drop','')])
        for name,condition in conditions:
            prefix=policy.base.model.action_prefix(context,[condition])
            output=policy.base.model.sample_actions_from_prefix(context,prefix,noise=noise.clone(),num_steps=10)[0].float().cpu().numpy()
            native=absolute(inverse({'state':state.copy(),'actions':output.copy()}))['actions'][:valid,:14]
            if not np.isfinite(native).all():raise ValueError('Nonfinite action prediction')
            entry['conditions'][name]={'joint_mse':float(np.mean((native[:,JOINT_MASK[:14]]-target[:,JOINT_MASK[:14]])**2)),
                                       'gripper_mse':float(np.mean((native[:,[6,13]]-target[:,[6,13]])**2)),
                                       'command_gripper_mse':float(np.mean((np.clip(native[:,[6,13]],0,.09)-np.clip(target[:,[6,13]],0,.09))**2)),
                                       'native_first15':native.tolist()}
        results.append(entry)
    if world>1:
        pieces=[None]*world;dist.all_gather_object(pieces,results);results=[r for piece in pieces for r in piece]
    if rank==0:
        results.sort(key=lambda r:(r['episode'],r['frame'],r['event_id']))
        observed_events={r['event_id'] for r in results if r['kind']=='boundary'}
        if not args.engineering_smoke and observed_events!={e['event_id'] for e in events}:
            raise ValueError('Action replay failed to cover every validation boundary')
        means={};checks={}
        for group in ['boundary','natural']:
            subset=[r for r in results if r['kind']==group]
            if not subset:
                if not args.engineering_smoke:raise ValueError('Missing action comparison group')
                continue
            means[group]={condition:{key:float(np.mean([r['conditions'][condition][key] for r in subset]))
                                    for key in ['joint_mse','gripper_mse','command_gripper_mse']}
                          for condition in ['r0_generated','r2_generated']}
            for key in ['joint_mse','gripper_mse','command_gripper_mse']:
                checks[group+'_'+key]=means[group]['r2_generated'][key]<=means[group]['r0_generated'][key]*1.05+1e-12
        report={'passed':all(checks.values()),'engineering_smoke':args.engineering_smoke,'checks':checks,'examples':len(results),'boundary_events':len(events),
                'boundary_events_actually_evaluated':len(observed_events),
                'natural_examples':sum(r['kind']=='natural' for r in results),'means':means,'rows':results,
                'scope':'paired native first15 actions with causal low-rate model history; no physical success claim'}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
