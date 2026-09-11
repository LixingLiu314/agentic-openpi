"""Compare M3/R2 on identical reconstructed failed-trial observations, without I/O."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('HF_HUB_OFFLINE','1')
import numpy as np
import torch
from openpi.policies.temporal_subtask_policy import create_temporal_policy


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--replay',type=Path,default=Path('assets/pi05_piper_transition/eggplant_potato/r2_failure_replay_v1'))
    p.add_argument('--device',default='cuda:0');p.add_argument('--engineering-smoke',action='store_true');args=p.parse_args();torch.set_num_threads(4)
    protocol=json.loads((args.replay/'protocol.json').read_text());constraints=protocol['constraints']
    policy=create_temporal_policy(args.checkpoint,device=args.device,require_candidate=False,allow_engineering=args.engineering_smoke)
    results=[]
    records=[r for r in protocol['records'] if not args.engineering_smoke or r['query'] in [1,94,95]]
    for sequence,r in enumerate(records,start=1):
        path=args.replay/r['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Replay input changed')
        with np.load(path) as saved:
            obs={'state':saved['state'].copy(),'prompt':r['prompt'],'images':{name:saved[name].copy() for name in ['cam_high','cam_left_wrist','cam_right_wrist']}}
        noise=np.random.default_rng(42+r['query']).standard_normal((50,32),dtype=np.float32)
        base=policy.base.infer(obs,noise=noise)
        output=policy.infer(dict(obs,session={'run_id':protocol['run'],'sequence':sequence,'observation_time':r['timestamp'],'mode':'offline'}),noise=noise)
        results.append({'query':r['query'],'timestamp':r['timestamp'],'r0_text':base['subtask'],'r0_status':base['subtask_status'],
                        'r2_text':output['subtask'],'r2_status':output['subtask_status'],'r2_candidate':output['raw_subtask_candidate'],
                        'completion_probabilities':output['completion_proxy_probabilities']})
    def summarize(prefix):
        forbidden=[r['query'] for r in results if r['query'] in constraints['forbidden_final_put_queries'] and 'put the lid' in r[prefix+'_text'].lower()]
        closed=[r['query'] for r in results if r['query'] in constraints['closed_gripper_queries'] and r[prefix+'_text'] in constraints['closed_gripper_allowed_text']]
        invalid=[r['query'] for r in results if r[prefix+'_status']!='ok']
        return {'forbidden_final_put_queries':forbidden,'closed_gripper_recognized_queries':closed,'invalid_queries':invalid}
    baseline,current=summarize('r0'),summarize('r2')
    checks={'no_final_put_before_object_placement':len(current['forbidden_final_put_queries'])<=constraints['max_forbidden_final_put'],
            'recognizes_closed_gripper_phase':len(current['closed_gripper_recognized_queries'])>=constraints['required_closed_gripper_allowed_count'],
            'valid_generation_guard':len(current['invalid_queries'])<=constraints['max_invalid_generations']}
    result={'passed':all(checks.values()),'checks':checks,'r0':baseline,'r2':current,'rows':results,'approximate_inputs':True,
            'engineering_smoke':args.engineering_smoke,'engineering_execution_passed':True,'observations':len(results),
            'protocol_sha256':hashlib.sha256((args.replay/'protocol.json').read_bytes()).hexdigest(),'physical_success_claimed':False}
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
