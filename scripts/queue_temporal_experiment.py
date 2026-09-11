"""Wait for the existing cache and all live leases, then reserve one finite job."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import gpu_reservation


def dependency_state(root,cache):
    identity=root/'cache_queue.process.json'
    if not identity.exists():raise FileNotFoundError('Existing cache queue identity required')
    state=json.loads(identity.read_text())
    if gpu_reservation.alive(state['pid'],state['created']):return 'waiting_for_cache_job'
    if (root/'cache_failure.json').exists():raise RuntimeError('Cache job failed; diagnose before any training')
    if not (cache/'cache_complete.json').exists():raise RuntimeError('Cache job exited without complete cache')
    record=json.loads((cache/'cache_complete.json').read_text())
    if record.get('frames')!={'train':106079,'val':13515} or record.get('episodes')!=178:
        raise RuntimeError('Unexpected formal cache coverage')
    pair=Path('logs/pi05_piper_backbone_grad/pair_seed42.process.json')
    if pair.exists():
        state=json.loads(pair.read_text())
        if gpu_reservation.alive(state['pid'],state['created']):return 'waiting_for_backbone_pair'
    if gpu_reservation.active_job(Path('logs/pi05_subtask_stage1/gpu_reservation')):return 'waiting_for_managed_job'
    return 'ready'


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    command=[sys.executable,str(Path(__file__).with_name('run_temporal_experiment.py')),
             '--root',str(args.root),'--cache',str(args.cache),'--output',str(args.output)]
    previous=None
    try:
        while True:
            phase=dependency_state(args.root,args.cache)
            if phase!=previous:
                gpu_reservation.write_json(args.root/'training_queue_state.json',{'status':phase,'command':command,'time':time.time()})
                print(json.dumps({'event':phase}),flush=True);previous=phase
            if phase!='ready':time.sleep(30);continue
            try:
                code=gpu_reservation.run(Path('logs/pi05_subtask_stage1/gpu_reservation'),args.root/'experiment_managed.log',command)
            except BlockingIOError:
                time.sleep(5);continue
            except RuntimeError as error:
                if str(error)=='Another managed job is still active':time.sleep(5);continue
                raise
            gpu_reservation.write_json(args.root/'training_queue_exit.json',{'exit_code':code,'time':time.time()})
            if code:raise RuntimeError('Finite experiment failed; inspect preserved experiment_failure.json')
            return
    except Exception as error:
        gpu_reservation.write_json(args.root/'training_queue_failure.json',{'error':str(error),'time':time.time()})
        raise


if __name__=='__main__':main()
