"""Finite identity-recorded Action-stop gates and training, under reservation lease."""
import argparse,fcntl,hashlib,json,os,signal,statistics,subprocess,sys,time,traceback
from pathlib import Path
import psutil
from gpu_reservation import run_concurrent

def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def checked(cmd,log):
    with log.open('x') as out:
        p=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        write(log.with_suffix('.process.json'),dict(pid=p.pid,created=psutil.Process(p.pid).create_time(),command=cmd,time=time.time()))
        try:code=p.wait()
        finally:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    if code:raise RuntimeError('Child exit '+str(code)+': '+str(log))

def verify(root):
    for path,digest in json.loads((root/'source_manifest.json').read_text()).items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path

def train_cmd(output,cache,steps,batch,accum):
    return [sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8',
        'scripts/train_parallel_action_stop.py','--output',str(output),'--decoded-cache',str(cache),
        '--steps',str(steps),'--batch-size',str(batch),'--accumulation',str(accum),'--workers','4']

def pipeline(a):
    root=a.root
    def phase(name,**kw):
        verify(root);write(root/'phase.json',dict(phase=name,time=time.time(),**kw))
    if a.stage=='gates':
        for batch,accum in [(32,1),(16,2),(8,4)]:
            output=root/('engineering_b'+str(batch));prefix='b'+str(batch)
            cmd=train_cmd(output,a.cache,8,batch,accum)+['--engineering-smoke','--warmup','2','--checkpoint-every','4','--eval-samples','8','--eval-draws','1']
            phase('engineering_save_resume',batch=batch,accumulation=accum)
            try:
                checked(cmd+['--stop-after','4'],root/(prefix+'_first.log'))
                checked(cmd+['--resume'],root/(prefix+'_resume.log'))
            except RuntimeError:
                logs='\n'.join(p.read_text(errors='replace') for p in root.glob(prefix+'_*.log'))
                if 'out of memory' not in logs.lower():raise
                write(root/(prefix+'_capacity_failure.json'),dict(capacity_passed=False,reason='Measured CUDA out of memory',time=time.time()))
                continue
            ckpt=output/'step_000008'
            resume=json.loads((output/'resume_state_000004.json').read_text())
            assert resume['passed'] and len(resume['ranks'])==8
            metadata=json.loads((ckpt/'metadata.json').read_text())
            assert metadata['counters']==dict(subtask=8,action=8,backbone=8)
            phase('engineering_native_gradient_gate',batch=batch)
            checked([sys.executable,'scripts/check_parallel_action_stop.py','--checkpoint',str(ckpt),
                     '--allow-engineering','--output',str(root/'engineering_native.json')],root/'engineering_native.log')
            rows=[json.loads(s) for s in (output/'metrics.jsonl').read_text().splitlines()]
            secs=[r['seconds'] for r in rows if r.get('event')=='train' and r['step'] in [2,3,4,6,7,8]]
            write(root/'engineering_passed.json',dict(passed=True,batch=batch,accumulation=accum,
                mean_update_seconds=statistics.mean(secs),checkpoint=str(ckpt),time=time.time()))
            phase('gates_complete');return
        raise RuntimeError('All measured batch capacities failed')
    gate=json.loads((root/'engineering_passed.json').read_text());assert gate['passed']
    output=a.output
    phase('formal_training',output=str(output),steps=5000)
    checked(train_cmd(output,a.cache,5000,gate['batch'],gate['accumulation'])+['--wandb'],root/'formal.log')
    final=output/'step_005000';phase('final_native_gate',checkpoint=str(final))
    checked([sys.executable,'scripts/check_parallel_action_stop.py','--checkpoint',str(final),
             '--output',str(output/'policy_load_gate.json')],root/'final_native.log')
    candidate=dict(checkpoint=str(final),completed_steps=5000,native_gate_passed=True,primary=True,
                   status='loadable_experimental_candidate',robot_test_performed=False,time=time.time())
    write(output/'candidate.json',candidate);write(root/'candidate_ready.json',candidate)
    write(root/'complete.json',dict(completed=True,**candidate));phase('complete',checkpoint=str(final))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=Path('.stage1_staging/piper_rgb224_reach_arm_v1'))
    p.add_argument('--output',type=Path,default=Path('checkpoints/pi05_piper_parallel/action_stop_seed42_v1'))
    p.add_argument('--reuse-gates',type=Path)
    p.add_argument('--stage',choices=['gates','formal'],required=True);p.add_argument('--managed',action='store_true')
    a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    if a.managed:
        try:pipeline(a)
        except BaseException as e:
            write(a.root/(a.stage+'_failure.json'),dict(error=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
        return
    with (a.root/'sequence.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.stage=='gates':
            manifest=a.root/'source_manifest.json'
            if manifest.exists():raise FileExistsError('Immutable dispatched source manifest already exists')
            files=list(Path('src/openpi').rglob('*.py'))+list(Path('scripts').glob('*parallel*.py'))
            files += [Path('scripts/gpu_reservation.py'),Path('scripts/train_subtask_pytorch.py'),Path('scripts/visualize_subtask_step.py')]
            write(manifest,{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
        elif a.reuse_gates:
            # Only startup-reader / orchestration fixes may reuse integration
            # evidence. Training, model, data, optimizer and native tests must
            # remain byte-identical to the already passed eight-rank gates.
            if (a.root/'source_manifest.json').exists():raise FileExistsError('Recovery root already dispatched')
            prior=a.reuse_gates
            old=json.loads((prior/'source_manifest.json').read_text())
            changed=[p for p,h in old.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
            allowed={'scripts/verify_parallel_startup.py','scripts/run_parallel_action_stop.py'}
            if not set(changed)<=allowed:raise ValueError('Computational source changed: '+str(changed))
            gate=json.loads((prior/'engineering_passed.json').read_text())
            native=json.loads((prior/'engineering_native.json').read_text())
            assert json.loads((prior/'gates_exit.json').read_text())['code']==0 and gate['passed'] and native['passed']
            assert native['action_gradients']['B']==0 and native['action_gradients']['S']==0
            assert native['ce_gradients']['B']>0 and native['ce_gradients']['A']==0
            write(a.root/'engineering_passed.json',{**gate,'reused_from':str(prior)})
            write(a.root/'reused_gate_evidence.json',dict(prior_root=str(prior),changed_noncomputational_sources=changed,
                native_report_sha256=hashlib.sha256((prior/'engineering_native.json').read_bytes()).hexdigest(),time=time.time()))
            write(a.root/'source_manifest.json',{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in old})
        verify(a.root)
        code=run_concurrent(Path('logs/pi05_subtask_stage1/gpu_reservation'),a.root/(a.stage+'_managed.log'),
            [sys.executable,__file__,'--root',str(a.root),'--cache',str(a.cache),'--output',str(a.output),'--stage',a.stage,'--managed'])
        write(a.root/(a.stage+'_exit.json'),dict(code=code,time=time.time()))
        raise SystemExit(code)
if __name__=='__main__':main()
