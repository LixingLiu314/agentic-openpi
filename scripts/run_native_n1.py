"""One finite N1 queue: predecessor exit -> eight-rank gates -> fresh formal.

Waits on the predecessor launcher (including its final gate), never on progress
files or GPU telemetry. No other experiment is restarted or signalled.
"""
import argparse
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import select
import signal
import statistics
import subprocess
import sys
import time
import traceback
import psutil
from gpu_reservation import run_concurrent

def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def verify(root):
    for path,digest in json.loads((root/'source_manifest.json').read_text()).items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise ValueError('Dispatched source changed: '+path)

def open_pidfd(pid):
    if hasattr(os,'pidfd_open'):return os.pidfd_open(pid)
    libc=ctypes.CDLL(None,use_errno=True)
    function=getattr(libc,'pidfd_open',None)
    if function is not None:
        function.argtypes=[ctypes.c_int,ctypes.c_uint];function.restype=ctypes.c_int
        result=function(pid,0)
    else:
        if platform.system()!='Linux' or platform.machine()!='x86_64':
            raise RuntimeError('Unsupported pidfd ABI; no polling fallback')
        libc.syscall.restype=ctypes.c_long
        result=libc.syscall(ctypes.c_long(434),ctypes.c_int(pid),ctypes.c_uint(0))
    if result<0:
        error=ctypes.get_errno();raise OSError(error,os.strerror(error))
    return result

def matching_process(record):
    p=psutil.Process(record['pid'])
    if abs(p.create_time()-record['created'])>.02:
        raise ValueError('PID was reused; refusing this predecessor identity')
    actual=p.cmdline()
    # Compare recorded argv, not an invented absolute-path script spelling.
    if actual!=record['command']:
        raise ValueError('Predecessor command identity changed')
    return p

def wait_predecessor(root,predecessor):
    record=json.loads((predecessor/'formal_launcher.process.json').read_text())
    expected='scripts/run_parallel_action_stop.py'
    if not any(Path(x).resolve()==Path(expected).resolve() for x in record['command'][1:]):
        raise ValueError('Not the requested predecessor launcher')
    write(root/'predecessor.json',dict(root=str(predecessor),**record))
    verify(predecessor)
    descriptor=None
    try:
        matching_process(record)
        descriptor=open_pidfd(record['pid'])
        matching_process(record)  # close check/open PID reuse race
    except (psutil.NoSuchProcess,ProcessLookupError):
        pass
    if descriptor is not None:
        try:
            write(root/'phase.json',dict(phase='waiting_predecessor',time=time.time(),
                predecessor=record,mechanism='pidfd launcher-exit event; includes predecessor final gates'))
            poll=select.poll();poll.register(descriptor,select.POLLIN);poll.poll()
        finally:os.close(descriptor)
    complete=json.loads((predecessor/'complete.json').read_text())
    if not complete.get('completed') or not complete.get('native_gate_passed') or complete.get('completed_steps')!=5000:
        raise ValueError('Predecessor did not successfully complete all final gates')
    write(root/'predecessor_completed.json',dict(passed=True,predecessor=complete,time=time.time()))

def checked(cmd,log):
    with log.open('x') as out:
        child=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        write(log.with_suffix('.process.json'),dict(pid=child.pid,
            created=psutil.Process(child.pid).create_time(),command=cmd,time=time.time()))
        try:code=child.wait()
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    if code:raise RuntimeError('Child exit '+str(code)+': '+str(log))

def train_cmd(output,cache,steps,batch,accum):
    return [sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8',
        'scripts/train_native_n1.py','--output',str(output),'--decoded-cache',str(cache),
        '--steps',str(steps),'--batch-size',str(batch),'--accumulation',str(accum),'--workers','4']

def pipeline(a):
    root=a.root
    def phase(name,**kw):
        verify(root);write(root/'phase.json',dict(phase=name,time=time.time(),**kw))
    for batch,accum in [(32,1),(16,2),(8,4)]:
        output=root/('engineering_b'+str(batch));prefix='b'+str(batch)
        cmd=train_cmd(output,a.cache,8,batch,accum)+['--engineering-smoke','--warmup','2',
            '--checkpoint-every','4','--eval-samples','8','--eval-draws','1']
        phase('engineering_save_resume',batch=batch,accumulation=accum)
        try:
            checked(cmd+['--stop-after','4'],root/(prefix+'_first.log'))
            checked(cmd+['--resume'],root/(prefix+'_resume.log'))
        except RuntimeError:
            logs='\n'.join(p.read_text(errors='replace') for p in root.glob(prefix+'_*.log'))
            if 'cuda out of memory' not in logs.lower():raise
            write(root/(prefix+'_capacity_failure.json'),dict(passed=False,reason='measured CUDA out of memory',time=time.time()))
            continue
        ckpt=output/'step_000008'
        resume=json.loads((output/'resume_state_000004.json').read_text())
        assert resume['passed'] and len(resume['ranks'])==8
        assert {r['rank'] for r in resume['ranks']}==set(range(8))
        assert all(r['carry_shape'] is None and r['optimizer_branches']==['action','backbone'] for r in resume['ranks'])
        metadata=json.loads((ckpt/'metadata.json').read_text())
        assert metadata['counters']==dict(action=8,backbone=8)
        for rank in range(8):
            assert (ckpt/('training_rank_%03d.pt'%rank)).exists(),list(ckpt.glob('*.pt'))
        phase('engineering_native_gradient_gate',batch=batch)
        checked([sys.executable,'scripts/check_native_n1.py','--checkpoint',str(ckpt),
            '--allow-engineering','--output',str(root/'engineering_native.json')],root/'engineering_native.log')
        native=json.loads((root/'engineering_native.json').read_text());assert native['passed']
        rows=[json.loads(s) for s in (output/'metrics.jsonl').read_text().splitlines()]
        seconds=[r['seconds'] for r in rows if r.get('event')=='train' and r['step'] in [2,3,4,6,7,8]]
        gate=dict(passed=True,batch=batch,accumulation=accum,mean_update_seconds=statistics.mean(seconds),
            checkpoint=str(ckpt),time=time.time())
        write(root/'engineering_passed.json',gate)
        break
    else:raise RuntimeError('All measured native N1 batch capacities failed')
    phase('formal_training',output=str(a.output),steps=5000)
    checked(train_cmd(a.output,a.cache,5000,gate['batch'],gate['accumulation'])+['--wandb'],root/'formal.log')
    final=a.output/'step_005000';phase('final_native_gate',checkpoint=str(final))
    checked([sys.executable,'scripts/check_native_n1.py','--checkpoint',str(final),
        '--output',str(a.output/'policy_load_gate.json')],root/'final_native.log')
    candidate=dict(checkpoint=str(final),completed_steps=5000,native_gate_passed=True,primary=True,
        status='loadable_experimental_candidate',robot_test_performed=False,time=time.time())
    write(a.output/'candidate.json',candidate);write(root/'candidate_ready.json',candidate)
    write(root/'complete.json',dict(completed=True,**candidate));phase('complete',checkpoint=str(final))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--predecessor',type=Path,default=Path('logs/pi05_parallel_action_stop_20260911/attempt_04'))
    p.add_argument('--cache',type=Path,default=Path('.stage1_staging/piper_rgb224_reach_arm_v1'))
    p.add_argument('--output',type=Path,default=Path('checkpoints/pi05_piper_native/n1_action_stop_seed42_v1'))
    p.add_argument('--managed',action='store_true');a=p.parse_args()
    a.root.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    if a.managed:
        pipeline(a);return
    with (a.root/'sequence.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=a.root/'source_manifest.json'
        if manifest.exists():raise FileExistsError('N1 queue root already dispatched; no implicit restart')
        files=list(Path('src/openpi').rglob('*.py'))+list(Path('scripts').glob('*native*.py'))
        files += [Path('scripts/gpu_reservation.py'),Path('scripts/train_subtask_pytorch.py'),
            Path('packages/openpi-client/src/openpi_client/image_tools.py')]
        write(manifest,{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in files})
        proc=psutil.Process()
        write(a.root/'launcher.process.json',dict(pid=proc.pid,created=proc.create_time(),command=proc.cmdline(),time=time.time()))
        try:
            wait_predecessor(a.root,a.predecessor)
            verify(a.root)
            code=run_concurrent(Path('logs/pi05_subtask_stage1/gpu_reservation'),a.root/'managed.log',
                [sys.executable,__file__,'--root',str(a.root),'--cache',str(a.cache),'--output',str(a.output),'--managed'])
            write(a.root/'exit.json',dict(code=code,time=time.time()))
            if code:raise RuntimeError('Managed N1 pipeline exit '+str(code))
        except BaseException as error:
            write(a.root/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),time=time.time()))
            raise
if __name__=='__main__':main()
