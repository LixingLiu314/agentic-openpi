"""Seal current R2 sources and enqueue one finite training/evaluation sequence."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import psutil
import gpu_reservation


def main():
    root=Path.cwd();logs=root/'logs/pi05_piper_transition/r2_exploration_seed42'
    identity=logs/'training_queue.process.json'
    with (logs/'training_launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if identity.exists():
            state=json.loads(identity.read_text())
            if gpu_reservation.alive(state['pid'],state['created']):print(json.dumps({'already_queued':True,**state}));return
            raise FileExistsError('Prior launch is terminal; inspect evidence before a versioned recovery')
        for name in ['native_cpu_gate.json','native_cached_history_gate_v2.json','actions_cpu_gate_v1.json','prelaunch_cpu_checks.json']:
            if not json.loads((logs/name).read_text())['passed']:raise RuntimeError('Missing prelaunch CPU gate: '+name)
        smoke=json.loads((logs/'failure_replay_cpu_gate.json').read_text())
        if not smoke.get('engineering_execution_passed') or smoke['observations']!=3:
            raise RuntimeError('Recorded failure replay execution check missing')
        if not (logs/'trainer_cpu_gate_v3/step_000002/metadata.json').exists():raise RuntimeError('Updated CPU trainer resume missing')
        if not (logs/'trainer_cpu_gate_v3/first_step/input_output.png').exists():raise RuntimeError('First-update visualization missing')
        replay=root/'assets/pi05_piper_transition/eggplant_potato/r2_failure_replay_v1/protocol.json'
        r=json.loads(replay.read_text())
        if r['observations']!=99 or r['missing_queries']:raise RuntimeError('Failure diagnostic inputs incomplete')
        snapshot=root/'.stage1_staging/r2_training_source_v1'
        snapshot.mkdir(parents=True,exist_ok=False);files={}
        for directory in ['src','scripts']:
            for source in sorted((root/directory).rglob('*.py')):
                if '__pycache__' in source.parts:continue
                relative=source.relative_to(root);target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,target);files[str(relative)]=hashlib.sha256(target.read_bytes()).hexdigest()
        manifest=snapshot/'source_manifest.json';manifest.write_text(json.dumps(files,sort_keys=True,indent=2)+'\n')
        digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
        cache=root/'assets/pi05_piper_transition/eggplant_potato/r2_frozen_m3_cache_v1'
        output=root/'checkpoints/pi05_piper_transition/r2_temporal_seed42_v1'
        if output.exists():raise FileExistsError(output)
        command=[sys.executable,str(snapshot/'scripts/queue_temporal_experiment.py'),'--root',str(logs),'--cache',str(cache),'--output',str(output)]
        env=dict(os.environ,PYTHONPATH=str(snapshot/'src'),JAX_PLATFORMS='cpu',HF_HUB_OFFLINE='1',
                 OPENPI_R2_SOURCE_MANIFEST_SHA256=digest,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',WANDB_DISABLE_STATS='true')
        with (logs/'training_queue.log').open('x') as stream:
            child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env=env)
        state={'pid':child.pid,'created':psutil.Process(child.pid).create_time(),'command':command,
               'source_snapshot':str(snapshot),'source_manifest_sha256':digest,'seed':42,'output':str(output),
               'replay_protocol_sha256':hashlib.sha256(replay.read_bytes()).hexdigest(),
               'scope':'Queued engineering gates -> formal 5000/global256 -> final semantic/native/action/stress evidence; no robot I/O'}
        gpu_reservation.write_json(identity,state);print(json.dumps(state,indent=2))


if __name__=='__main__':main()
