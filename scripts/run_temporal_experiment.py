"""Finite reserved R2 experiment: engineering gates, formal training, final checks.

This process waits for its children, never polls optimizer steps or GPU telemetry.
Failures retain every artifact and do not select or deploy a different model.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil


def write(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate_record(checkpoint,semantic,action,native,stress):
    meta=json.loads((checkpoint/'metadata.json').read_text())
    if meta['engineering_smoke']:raise ValueError('Cannot qualify an engineering checkpoint')
    flags={'semantic_gates_passed':semantic['passed'],
           'action_gate_passed':action['passed'] and not action.get('engineering_smoke',True),
           'policy_load_gate_passed':native['passed'],
           'stress_gate_passed':stress['passed'] and not stress.get('engineering_smoke',True) and stress.get('observations')==99}
    passed=all(flags.values())
    return {'status':'qualified_offline_candidate' if passed else 'failed_candidate',
            'checkpoint':checkpoint.name,'temporal_sha256':meta['temporal_sha256'],
            'parent_weights_sha256':meta['config']['parent_weights_sha256'],
            'protocol_sha256':meta['config']['protocol_sha256'],**flags,
            'physical_readiness_reviewed':False,'physical_success_claimed':False,
            'goal_achieved':False,'review_required':'Inspect paired final evidence; a finite job never completes the exploration goal itself.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);args=p.parse_args()
    scripts=Path(__file__).resolve().parent;args.root.mkdir(parents=True,exist_ok=True)
    def run(name,command):
        log=args.root/(name+'.log')
        if log.exists():raise FileExistsError('Refusing to overwrite or rerun an existing phase: '+str(log))
        write(args.root/'experiment_state.json',{'phase':name,'status':'running','command':command,'time':time.time()})
        print(json.dumps({'phase':name,'event':'started'}),flush=True)
        with log.open('x') as stream:
            child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
            write(args.root/(name+'.process.json'),{'pid':child.pid,'created':psutil.Process(child.pid).create_time(),'command':command})
            code=child.wait()
        write(args.root/(name+'.exit.json'),{'exit_code':code,'time':time.time()})
        if code:raise RuntimeError(f'{name} exited {code}; inspect {log}')
        print(json.dumps({'phase':name,'event':'finished'}),flush=True)
    def distributed(script,*extra):
        return [sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=8',str(scripts/script),*map(str,extra)]
    def single(script,*extra):return [sys.executable,str(scripts/script),*map(str,extra)]
    try:
        manifest=scripts.parent/'source_manifest.json'
        if digest(manifest)!=os.environ['OPENPI_R2_SOURCE_MANIFEST_SHA256']:raise ValueError('Snapshot manifest changed')
        for relative,expected in json.loads(manifest.read_text()).items():
            if digest(scripts.parent/relative)!=expected:raise ValueError('Snapshot source changed: '+relative)
        full=args.root/'engineering_gpu_uninterrupted';resumed=args.root/'engineering_gpu_resumed'
        common=['--cache',str(args.cache),'--engineering-smoke','--steps','2','--global-batch','32','--batch-size','2',
                '--warmup','1','--save-every','2','--seed','42']
        run('engineering_gpu_full',distributed('train_temporal_subtask.py',*common,'--output',full))
        run('engineering_gpu_part1',distributed('train_temporal_subtask.py',*common,'--output',resumed,'--stop-after','1'))
        run('engineering_gpu_resume',distributed('train_temporal_subtask.py',*common,'--output',resumed,'--resume'))
        run('engineering_gpu_compare',single('compare_temporal_resume.py','--first',full/'step_000002','--second',resumed/'step_000002',
                                             '--world','8','--output',args.root/'engineering_gpu_compare.json'))
        run('engineering_gpu_native',single('check_temporal_checkpoint.py','--checkpoint',resumed/'step_000002','--device','cuda:0',
                                            '--allow-engineering','--output',args.root/'engineering_gpu_native.json'))
        run('engineering_gpu_history',single('verify_temporal_cached_live.py','--checkpoint',resumed/'step_000002','--device','cuda:0',
                                             '--cache',args.cache,'--output',args.root/'engineering_gpu_history.json'))
        for name in ['engineering_gpu_compare','engineering_gpu_native','engineering_gpu_history']:
            if not json.loads((args.root/(name+'.json')).read_text())['passed']:raise RuntimeError(name+' failed')
        write(args.root/'engineering_passed.json',{'passed':True,'seed':42,'world_size':8,'time':time.time()})
        # Formal weights are freshly initialized from M3; engineering weights are never inherited.
        run('formal_training',distributed('train_temporal_subtask.py','--cache',args.cache,'--output',args.output,
                                          '--seed','42','--steps','5000','--global-batch','256','--batch-size','8',
                                          '--warmup','500','--save-every','500','--wandb'))
        complete=json.loads((args.output/'training_complete.json').read_text())
        if complete['completed_steps']!=5000:raise RuntimeError('Formal training incomplete')
        best=json.loads((args.output/'best.json').read_text());checkpoint=args.output/best['checkpoint']
        run('selected_native',single('check_temporal_checkpoint.py','--checkpoint',checkpoint,'--device','cuda:0','--output',args.output/'policy_load_gate.json'))
        run('selected_actions',distributed('evaluate_temporal_actions.py','--checkpoint',checkpoint,'--cache',args.cache,'--output',args.output/'action_gate.json'))
        run('selected_failure_replay',single('evaluate_transition_failure_replay.py','--checkpoint',checkpoint,'--output',args.output/'stress_gate.json'))
        paths={k:args.output/v for k,v in {'semantic':'semantic_gates.json','action':'action_gate.json',
                                         'native':'policy_load_gate.json','stress':'stress_gate.json'}.items()}
        reports={k:json.loads(v.read_text()) for k,v in paths.items()}
        candidate=candidate_record(checkpoint,**reports)
        candidate['evidence_sha256']={k:digest(v) for k,v in paths.items()}
        write(args.output/'candidate.json',candidate)
        if candidate['status']=='qualified_offline_candidate':
            run('qualified_native',single('check_temporal_checkpoint.py','--checkpoint',checkpoint,'--device','cuda:0',
                                         '--require-candidate','--output',args.output/'qualified_load_gate.json'))
        result={'complete':True,'candidate':candidate,'checkpoint':str(checkpoint),'seed':42,
                'goal_achieved':False,'time':time.time()}
        write(args.root/'experiment_result.json',result)
        write(args.root/'experiment_state.json',{'status':'complete','phase':'final_evidence_ready','time':time.time()})
        print(json.dumps(result),flush=True)
    except Exception as error:
        candidate_path=args.output/'candidate.json'
        if candidate_path.exists():
            failed=json.loads(candidate_path.read_text());failed.update(status='failed_candidate',policy_load_gate_passed=False,
                                                                        qualification_error=str(error))
            write(candidate_path,failed)
        write(args.root/'experiment_failure.json',{'error':str(error),'time':time.time(),'goal_achieved':False})
        raise


if __name__=='__main__':main()
