"""Native 50x14 R2 policy check on a recorded validation observation; no robot I/O."""
import argparse
import dataclasses
import json
import os
from pathlib import Path
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('HF_HUB_OFFLINE','1')
import numpy as np
import torch
from openpi.policies.temporal_subtask_policy import create_temporal_policy
from openpi.training import config as config_lib,data_loader


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--allow-engineering',action='store_true')
    p.add_argument('--require-candidate',action='store_true');args=p.parse_args()
    torch.set_num_threads(4)
    policy=create_temporal_policy(args.checkpoint,device=args.device,allow_engineering=args.allow_engineering,require_candidate=args.require_candidate)
    cfg=config_lib.get_config('pi05_piper_stage1')
    dc=dataclasses.replace(cfg.data.create(cfg.assets_dirs,policy.base.model.base.config),split='val')
    dataset=data_loader.create_torch_dataset(dc,50,policy.base.model.base.config);frame=dataset[0]
    obs={'state':np.asarray(frame['observation.state']).copy(),'prompt':frame['task'],
         'images':{name:np.asarray(frame['observation.images.'+name]).copy() for name in ['cam_high','cam_left_wrist','cam_right_wrist']},
         'session':{'run_id':'gate','sequence':1,'observation_time':0.,'mode':'offline'}}
    original={'state':obs['state'].copy(),'images':{k:v.copy() for k,v in obs['images'].items()}}
    noise=np.random.default_rng(42).standard_normal((50,32),dtype=np.float32)
    a=policy.infer(obs,noise=noise)
    isolated=policy.new_session()
    assert not isolated.history and not isolated.active
    b=isolated.infer(obs,noise=noise)
    assert a['subtask']==b['subtask'] and np.array_equal(a['actions'],b['actions'])
    assert a['actions'].shape==(50,14) and np.isfinite(a['actions']).all()
    assert np.array_equal(original['state'],obs['state'])
    assert all(np.array_equal(original['images'][k],v) for k,v in obs['images'].items())
    try:policy.infer(obs,noise=noise)
    except ValueError:pass
    else:raise AssertionError('Duplicate request accepted')
    try:policy.infer(dict(obs,subtask='grasp'),noise=noise)
    except ValueError:pass
    else:raise AssertionError('External subtask accepted')
    policy.reset()
    c=policy.infer(obs,noise=noise)
    assert np.array_equal(c['actions'],a['actions'])
    result={'passed':True,'native_shape':[50,14],'fixed_noise_reset_equal':True,'connection_state_isolated':True,
            'input_unchanged':True,'duplicate_rejected':True,'external_subtask_rejected':True,'subtask':a['subtask'],
            'raw_candidate':a['raw_subtask_candidate'],'completion_probabilities':a['completion_proxy_probabilities'],
            'scope':'Recorded-observation policy engineering; no physical actions or capability claim'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
