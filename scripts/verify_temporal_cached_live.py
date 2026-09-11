"""Compare cached and native R2 inputs/predictions over four actual observations."""
import argparse
import contextlib
import dataclasses
import json
import os
from pathlib import Path
from types import SimpleNamespace
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('HF_HUB_OFFLINE','1')
import jax
import numpy as np
import torch
from openpi.models.model import Observation
from openpi.policies.temporal_subtask_policy import create_temporal_policy
from openpi.training import config as config_lib,data_loader
from openpi.training.subtask_transition import read_jsonl
from openpi.training.transition_feature_cache import compress_memory,encode_tensor,EpisodeFeatures
from openpi.training.transition_sequence import assemble_inputs,batch_inputs,choose_text,select_observed_history


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--cache',type=Path)
    args=p.parse_args();torch.set_num_threads(4);device=torch.device(args.device)
    policy=create_temporal_policy(args.checkpoint,device=args.device,allow_engineering=True,require_candidate=False)
    cfg=config_lib.get_config('pi05_piper_stage1');dc=dataclasses.replace(cfg.data.create(cfg.assets_dirs,policy.base.model.base.config),split='val')
    dataset=data_loader.create_torch_dataset(dc,50,policy.base.model.base.config)
    rows=read_jsonl(Path('assets/pi05_piper_transition/eggplant_potato/r2_v1/frames_val.jsonl'))
    indices=[0,23,46,69];observations=[];features=[];times=[]
    with torch.no_grad():
        for index in indices:
            frame=dataset[index];times.append(rows[index]['timestamp'])
            obs={'state':np.asarray(frame['observation.state']).copy(),'prompt':frame['task'],
                 'images':{name:np.asarray(frame['observation.images.'+name]).copy() for name in ['cam_high','cam_left_wrist','cam_right_wrist']}}
            observations.append(obs)
            transformed=policy.base.input_transform(obs)
            tensors=jax.tree.map(lambda x:torch.as_tensor(np.asarray(x),device=device)[None],transformed)
            context=policy.base.model.prepare_context(Observation.from_dict(tensors),[obs['prompt']])
            summary,mask=compress_memory(context.memory,context.memory_mask)
            features.append({'memory':context.memory[0].cpu().clone(),'memory_mask':context.memory_mask[0].cpu().clone(),
                             'summary':summary[0].cpu().clone(),'summary_mask':mask[0].cpu().clone(),'state':context.state[0].cpu().clone()})
    ep=SimpleNamespace(times=np.array(times),dtype=str(features[0]['summary'].dtype).split('.')[-1],summary=np.stack([encode_tensor(f['summary']) for f in features]),
                       summary_mask=np.stack([f['summary_mask'].numpy() for f in features]),state=np.stack([f['state'].numpy() for f in features]),
                       current=lambda index:features[index])
    original_generate=policy.temporal.generate;captured={}
    def capture(inputs,embedding):
        captured.clear();captured.update({k:v.detach().clone() for k,v in inputs.items()})
        return original_generate(inputs,embedding)
    policy.temporal.generate=capture
    active='';results=[]
    for i,obs in enumerate(observations):
        session={'run_id':'parity','sequence':i+1,'observation_time':times[i],'mode':'offline'}
        live=policy.infer(dict(obs,session=session),noise=np.random.default_rng(42+i).standard_normal((50,32),dtype=np.float32))
        history=select_observed_history(times[:i],times[i])
        cached=batch_inputs([assemble_inputs(ep,i,history,active,policy.base.model.codec)],device)
        for name,value in cached.items():assert torch.equal(value,captured[name]),f'Native/cache input differs: frame{i} {name}'
        amp=torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else contextlib.nullcontext()
        with amp:generation,completion=original_generate(cached,policy.base.model.embedding_weight)
        texts,statuses=policy.base.model.codec.decode(generation)
        chosen,reason=choose_text(texts[0],statuses[0],active,completion[0],policy.stay_threshold)
        assert live['subtask']==chosen and live['raw_subtask_candidate']==texts[0]
        assert np.array_equal(np.array(live['completion_proxy_probabilities']),completion[0].float().cpu().numpy())
        results.append({'original_frame':indices[i],'history_valid_count':int(cached['frame_valid'].sum()),'subtask':chosen,'all_inputs_bit_exact':True})
        active=chosen
    assert results[-1]['history_valid_count']==4
    batch_cache=[]
    if args.cache:
        actual=EpisodeFeatures(args.cache/'val'/f"episode_{rows[0]['episode']:06d}")
        cached_active='';live_active=''
        for i,index in enumerate(indices):
            hist=select_observed_history(times[:i],times[i])
            stored=batch_inputs([assemble_inputs(actual,index,[indices[h] if h is not None else None for h in hist],cached_active,policy.base.model.codec)],device)
            single=batch_inputs([assemble_inputs(ep,i,hist,live_active,policy.base.model.codec)],device)
            differences={}
            for name in ['memory','summaries','states']:
                a,b=stored[name].float(),single[name].float()
                relative=float((a-b).norm()/b.norm().clamp_min(1e-8));differences[name]=relative
                if relative>.03:raise AssertionError(f'Formal-cache/live feature divergence: {index} {name} {relative}')
            for name in ['memory_mask','summary_masks','frame_valid','ages']:
                if not torch.equal(stored[name],single[name]):raise AssertionError(f'Formal-cache/live layout mismatch: {name}')
            with torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else contextlib.nullcontext():
                sg,sp=original_generate(stored,policy.base.model.embedding_weight)
                lg,lp=original_generate(single,policy.base.model.embedding_weight)
            st,ss=policy.base.model.codec.decode(sg);lt,ls=policy.base.model.codec.decode(lg)
            cached_active,_=choose_text(st[0],ss[0],cached_active,sp[0],policy.stay_threshold)
            live_active,_=choose_text(lt[0],ls[0],live_active,lp[0],policy.stay_threshold)
            if st!=lt or ss!=ls or cached_active!=live_active:raise AssertionError('Formal-cache/live generated text differs')
            difference=float((sp.float()-lp.float()).abs().max())
            if difference>.02:raise AssertionError('Formal-cache/live completion probabilities differ')
            batch_cache.append({'frame':index,'relative_l2':differences,'text_equal':True,'completion_max_abs_diff':difference})
    result={'passed':True,'device':args.device,'scope':'real recorded four-observation live/cache equivalence; no robot I/O',
            'frames':results,'formal_batch_cache_comparison':batch_cache}
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
