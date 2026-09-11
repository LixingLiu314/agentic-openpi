"""Freeze a source snapshot and enqueue one cache job without touching other jobs."""
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
    root=Path.cwd()
    logs=root/'logs/pi05_piper_transition/r2_exploration_seed42';logs.mkdir(parents=True,exist_ok=True)
    identity=logs/'cache_queue.process.json'
    with (logs/'cache_launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if identity.exists():
            state=json.loads(identity.read_text())
            if gpu_reservation.alive(state['pid'],state['created']):
                print(json.dumps({'already_queued':True,**state}));return
            raise FileExistsError('Previous cache launch is terminal; inspect its results before a reviewed recovery')
        gate=json.loads((logs/'cache_cpu_gate/rank_000_complete.json').read_text())
        if gate['frames']!=4 or gate['cache_identity']['engineering_frames']!=2:raise RuntimeError('Missing CPU observation-cache gate')
        for p in (logs/'cache_cpu_gate').glob('*/episode_*/complete.json'):
            if json.loads(p.read_text())['frames']!=2:raise RuntimeError('Wrong engineering cache')
        snapshot=root/'.stage1_staging/r2_cache_source_v1'
        if snapshot.exists():raise FileExistsError(snapshot)
        snapshot.mkdir(parents=True)
        files={}
        for directory in ['src','scripts']:
            for source in sorted((root/directory).rglob('*.py')):
                if '__pycache__' in source.parts:continue
                relative=source.relative_to(root);target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,target);files[str(relative)]=hashlib.sha256(target.read_bytes()).hexdigest()
        manifest=snapshot/'source_manifest.json';manifest.write_text(json.dumps(files,sort_keys=True,indent=2)+'\n')
        digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
        output=root/'assets/pi05_piper_transition/eggplant_potato/r2_frozen_m3_cache_v1'
        command=[sys.executable,str(snapshot/'scripts/queue_transition_cache.py'),'--root',str(logs),
                 '--snapshot',str(snapshot),'--output',str(output)]
        environment=dict(os.environ,PYTHONPATH=str(snapshot/'src'),JAX_PLATFORMS='cpu',HF_HUB_OFFLINE='1',
                         OPENPI_R2_SOURCE_MANIFEST_SHA256=digest,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        with (logs/'cache_queue.log').open('x') as stream:
            process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env=environment)
        state={'pid':process.pid,'created':psutil.Process(process.pid).create_time(),'command':command,
               'source_snapshot':str(snapshot),'source_manifest_sha256':digest,'output':str(output),'seed':42,
               'scope':'frozen M3 observation feature extraction; not optimizer training; queued behind existing jobs'}
        gpu_reservation.write_json(identity,state);print(json.dumps(state,indent=2))


if __name__=='__main__':main()
