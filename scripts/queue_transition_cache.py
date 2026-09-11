"""Identity-checked queue behind existing work; one immutable observation-cache job."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import gpu_reservation


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.root.mkdir(parents=True,exist_ok=True)
    directory=Path('logs/pi05_subtask_stage1/gpu_reservation')
    dependency=Path('logs/pi05_piper_backbone_grad/pair_seed42.process.json')
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=8',
             str(args.snapshot/'scripts/cache_transition_observations.py'),'--output',str(args.output),'--batch-size','8']
    reason=None
    try:
        while True:
            dependent=False
            if dependency.exists():
                state=json.loads(dependency.read_text())
                dependent=gpu_reservation.alive(state['pid'],state['created'])
            active=gpu_reservation.active_job(directory)
            current='waiting_for_backbone_pair' if dependent else 'waiting_for_managed_job' if active else 'ready'
            if current!=reason:
                gpu_reservation.write_json(args.root/'queue_state.json',{'status':current,'dependency':str(dependency),
                                                                       'verified_live_dependency':dependent,'command':command,'time':time.time()})
                print(json.dumps({'event':current}),flush=True);reason=current
            if dependent or active:
                time.sleep(30);continue
            try:
                code=gpu_reservation.run(directory,args.root/'cache_managed.log',command)
            except BlockingIOError:
                time.sleep(5);continue
            except RuntimeError as error:
                if str(error)=='Another managed job is still active':
                    time.sleep(5);continue
                raise
            if code:raise RuntimeError(f'Cache extraction exited with {code}; preserve partial evidence and diagnose before resume')
            protocol=json.loads(Path('assets/pi05_piper_transition/eggplant_potato/r2_v1/protocol.json').read_text())
            expected={'train':106079,'val':13515};actual={};episode_count=0
            for split,count in expected.items():
                records=[json.loads(q.read_text()) for q in (args.output/split).glob('episode_*/complete.json')]
                actual[split]=sum(r['frames'] for r in records);episode_count+=len(records)
                if actual[split]!=count or any(r['cache_identity']['engineering_frames'] for r in records):
                    raise RuntimeError('Incomplete or engineering cache used as formal cache')
            gpu_reservation.write_json(args.output/'cache_complete.json',{'complete':True,'frames':actual,'episodes':episode_count,
                                                                         'seed':42,'parent_weights_sha256':protocol['parent_weights_sha256'],'time':time.time()})
            gpu_reservation.write_json(args.root/'cache_exit.json',{'exit_code':0,'complete':True,'time':time.time()})
            return
    except Exception as error:
        gpu_reservation.write_json(args.root/'cache_failure.json',{'error':str(error),'time':time.time()})
        raise


if __name__=='__main__':main()
