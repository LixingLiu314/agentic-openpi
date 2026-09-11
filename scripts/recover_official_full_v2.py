"""Finite recovery of the same full-B arm after its logging checker failed."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import psutil

from gpu_reservation import run_concurrent
from run_official_gradient_trio import checked, training_command, verify_sources, write_json


def pipeline(args):
    verify_sources(args.root)
    output=Path('checkpoints/pi05_piper_official_grad/full_seed42_v2')
    assert not output.exists(),'Do not overwrite a recovery run'
    command=training_command('full',output,args.cache)
    command[command.index('scripts/train_official_backbone_gradient.py')]='scripts/train_official_backbone_gradient_observed_v2.py'
    command+=['--wandb']
    write_json(args.root/'phase.json',{'phase':'formal_training','mode':'full','output':str(output),'time':time.time(),'recovery':True})
    with (args.root/'full_formal.log').open('x') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        write_json(args.root/'full_formal.process.json',{'pid':process.pid,'created':psutil.Process(process.pid).create_time(),'command':command,'started':time.time()})
        try:
            checked([sys.executable,'scripts/verify_official_full_wandb_startup_v2.py','--output',str(output),
                     '--identity',str(args.root/'full_formal.process.json'),'--report',str(args.root/'full_wandb_startup.json')],args.root/'full_wandb_startup.log')
            code=process.wait()
            if code:raise RuntimeError(f'Full recovery training exited {code}')
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    verify_sources(args.root)
    latest=json.loads((output/'latest.json').read_text())
    assert latest['completed_steps']==5000
    selected=output/json.loads((output/'best.json').read_text())['checkpoint']
    write_json(args.root/'phase.json',{'phase':'final_native_gate','mode':'full','time':time.time()})
    checked([sys.executable,'scripts/check_official_gradient_checkpoint.py','--checkpoint',str(selected),'--output',str(output/'policy_load_gate.json')],args.root/'full_final_native.log')
    write_json(output/'candidate.json',{'checkpoint':selected.name,'mode':'full','initialization':'official_pi05_base','policy_load_gate_passed':True,'status':'loadable_experimental_candidate','robot_test_performed':False})
    write_json(args.root/'full_complete.json',{'completed':True,'output':str(output),'candidate':str(selected),'steps':5000,'global_batch':256,'time':time.time()})
    write_json(args.root/'phase.json',{'phase':'complete','time':time.time()})
    write_json(args.root/'complete.json',{'completed':True,'modes':['frozen','limited','full'],'time':time.time()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--managed',action='store_true')
    args=parser.parse_args()
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        if args.managed:pipeline(args)
        else:
            with (args.root/'sequence.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                verify_sources(args.root)
                checked([sys.executable,'scripts/verify_official_full_wandb_startup_v2.py','--preflight','--report',str(args.root/'wandb_preflight.json')],args.root/'wandb_preflight.log')
                command=[sys.executable,str(Path(__file__).absolute()),'--root',str(args.root),'--cache',str(args.cache),'--managed']
                code=run_concurrent(Path('logs/pi05_subtask_stage1/gpu_reservation'),args.root/'managed.log',command)
                write_json(args.root/'supervisor_exit.json',{'code':code,'time':time.time()})
                if code:raise RuntimeError(f'Full recovery pipeline exited {code}')
    except BaseException as error:
        write_json(args.root/'failure.json',{'error':repr(error),'traceback':traceback.format_exc(),'time':time.time()})
        write_json(args.root/'phase.json',{'phase':'failed','time':time.time(),'error':repr(error)})
        raise


if __name__=='__main__':main()
