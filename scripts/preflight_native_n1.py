"""CPU-only pre-dispatch contract checks, including real pidfd child-exit test."""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time
from unittest.mock import patch
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['JAX_PLATFORMS']='cpu'
os.environ['HF_HUB_OFFLINE']='1'
import psutil
import torch
from openpi.models.pi0_config import Pi0Config
from openpi.training import config as config_lib
from openpi.models.subtask_tokenizer import SubtaskTextCodec
from openpi.training.reach_arm_data import data_config
from openpi.training.decoded_video_cache import create_cached_dataset
from openpi.training.subtask_batch import SubtaskTrainingDataset
from openpi.training.native_subtask_provenance import build_run_config
from openpi.training.native_subtask_wandb import event_payload
from run_native_n1 import verify,open_pidfd,matching_process,train_cmd
from train_native_n1 import parse_args

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    a.root.mkdir(parents=True,exist_ok=True)
    tests=subprocess.run([sys.executable,'scripts/test_native_n1.py'],capture_output=True,text=True,check=True)
    (a.root/'cpu_structure.log').write_text(tests.stdout+tests.stderr)
    structural=json.loads(tests.stdout.strip().splitlines()[-1]);assert structural['passed']
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(0.5)'])
    process=psutil.Process(child.pid)
    record=dict(pid=child.pid,created=process.create_time(),command=process.cmdline())
    matching_process(record)
    try:matching_process({**record,'created':record['created']-1})
    except ValueError:pass
    else:raise AssertionError('Reused PID identity was accepted')
    descriptor=open_pidfd(child.pid)
    poll=select.poll();poll.register(descriptor,select.POLLIN)
    assert poll.poll(5000),'Real pidfd child exit was not delivered'
    os.close(descriptor);assert child.wait()==0
    verify(Path('logs/pi05_parallel_action_stop_20260911/attempt_04'))
    predecessor=json.loads(Path('logs/pi05_parallel_action_stop_20260911/attempt_04/formal_launcher.process.json').read_text())
    matching_process(predecessor)
    with patch.object(sys,'argv',['train_native_n1.py','--output','checkpoints/pi05_piper_native/n1_action_stop_seed42_v1',
        '--decoded-cache','.stage1_staging/piper_rgb224_reach_arm_v1']):args=parse_args()
    assert (args.arm,args.unroll,args.mode,args.global_batch,args.seed)==('stateless',1,'action_stop',256,42)
    args.stage='native_subtask';args.overfit_samples=0
    cfg=config_lib.get_config('pi05_piper_stage1')
    cfg=dataclasses.replace(cfg,model=Pi0Config(pi05=True,dtype='bfloat16',pytorch_compile_mode=None))
    dc=data_config(cfg.model)
    prov=build_run_config(args,cfg,dc,8)
    assert 'decoder' not in prov and prov['inherited_training_updates']==0
    raw=create_cached_dataset(dc,50,args.decoded_cache)
    dataset=SubtaskTrainingDataset(raw,dc)
    vocabulary=sorted(set(dataset.labels_without_video()))
    codec=SubtaskTextCodec();ids,mask=codec.targets(vocabulary)
    assert len(vocabulary)==15 and max(mask.sum(1))<=16
    assert any('with the left arm' in v for v in vocabulary) and any('with the right arm' in v for v in vocabulary)
    assert dc.local_root=='Datasets/eggplant_potato_reach_arm_v1'
    manifest=json.loads(Path(dc.split_manifest).read_text())
    # Read labels/cache metadata only. No full model/GPU work is allowed here.
    train_payload=event_payload(dict(event='train',step=10,loss_subtask=3.,loss_action=.4,
        grad_norms=dict(backbone=2.,action=.2,backbone_clip_scale=.5,action_clip_scale=1.)),prov)
    assert train_payload['train/subtask_ce']==3. and train_payload['optim/grad_norm_b']==2.
    assert not any(k.endswith('_s') for k in train_payload)
    assert all('val/' not in k for k in train_payload)
    cmd=train_cmd(Path('example'),Path('cache'),5000,32,1)
    assert '--nproc-per-node=8' in cmd and 'scripts/train_native_n1.py' in cmd
    result=dict(passed=True,time=time.time(),cpu_structure=structural,pidfd_exit_event=True,
        predecessor_source_manifest_unchanged=True,predecessor_identity_verified=True,
        official_weights_sha256=prov['official_weights_sha256'],
        data_root=dc.local_root,repo_id=dc.repo_id,split_sha256=prov['split_sha256'],norm_sha256=prov['norm_sha256'],
        train_frames=len(dataset),labels=vocabulary,max_target_tokens_including_eos=int(mask.sum(1).max()),
        optimizer_owners=['B','A'],native_text_parameters_only=True,wandb_mapping=True,
        gpu_checks_performed=False)
    (a.root/'cpu_preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
