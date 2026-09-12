"""Dispatch exactly one committed, CPU-preflighted finite N1 waiting process."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
from run_native_n1 import verify,matching_process,write,predecessor_status

GIT='/media/raid/workspace/surongpeng/anaconda3/bin/git'

def queue_environment(source=None):
    environment=dict(os.environ if source is None else source)
    # CPU preflight may be invoked with an empty visibility mask. The waiter
    # itself does no GPU work, but its eventual eight-rank children must not
    # inherit that CPU-only override. Preserve deliberate nonempty masks.
    if environment.get('CUDA_VISIBLE_DEVICES')=='':
        environment.pop('CUDA_VISIBLE_DEVICES')
    return environment

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    if (a.root/'launcher.process.json').exists() or (a.root/'source_manifest.json').exists():
        raise FileExistsError('This N1 root has already been dispatched; no duplicate launch')
    cpu=json.loads((a.root/'cpu_preflight.json').read_text());assert cpu['passed'] and not cpu['gpu_checks_performed']
    view=json.loads((a.root/'wandb_workspace.json').read_text());assert view['verified_remote_roundtrip']
    changes=subprocess.check_output([GIT,'status','--porcelain'],text=True).strip()
    if changes:raise ValueError('Commit scoped implementation before dispatch: '+changes)
    commit=subprocess.check_output([GIT,'rev-parse','HEAD'],text=True).strip()
    predecessor=Path('logs/pi05_parallel_action_stop_20260911/attempt_04')
    previous=predecessor_status(predecessor)
    if Path('checkpoints/pi05_piper_native/n1_action_stop_seed42_v1').exists():
        raise FileExistsError('Formal output already exists; no implicit overwrite or resume')
    command=[sys.executable,'scripts/run_native_n1.py','--root',str(a.root)]
    with (a.root/'launcher.log').open('x') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=queue_environment())
    # A bounded dispatch handshake only, not training progress polling.
    for _ in range(20):
        if child.poll() is not None:raise RuntimeError('Queue exited at dispatch: '+(a.root/'launcher.log').read_text())
        if (a.root/'phase.json').exists():break
        time.sleep(.25)
    else:raise RuntimeError('Dispatch unconfirmed; inspect existing PID, never launch a duplicate')
    identity=json.loads((a.root/'launcher.process.json').read_text())
    proc=matching_process(identity)
    visibility=proc.environ().get('CUDA_VISIBLE_DEVICES')
    if visibility=='':raise ValueError('CPU-only visibility leaked into the future GPU queue')
    phase=json.loads((a.root/'phase.json').read_text())
    if phase['phase'] not in {'waiting_predecessor','acquiring_reservation',
                             'engineering_save_resume','engineering_native_gradient_gate'}:
        raise ValueError('Unexpected dispatch phase: '+str(phase))
    verify(a.root);verify(predecessor)
    receipt=dict(queued=phase['phase']=='waiting_predecessor',dispatched=True,phase=phase['phase'],
        formal_training_started=False,engineering_started=phase['phase'].startswith('engineering_'),
        gpu_work_started=phase['phase'].startswith('engineering_'),
        root=str(a.root),output='checkpoints/pi05_piper_native/n1_action_stop_seed42_v1',
        identity=identity,predecessor=previous,dispatch_git_commit=commit,
        cwd=str(Path.cwd()),cpu_preflight_passed=True,wandb_view=view['url'],
        cuda_visibility_override=visibility,cpu_only_mask_removed=True,time=time.time())
    write(a.root/'dispatch_verified.json',receipt);print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
