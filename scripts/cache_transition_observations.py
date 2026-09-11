"""Cache each original train/val observation once using the unchanged frozen M3 B.

Runs only inside the managed resource queue. Does not inspect GPU telemetry.
"""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('HF_HUB_OFFLINE','1')
import numpy as np
import torch

from train_subtask_transition import build_dataset,write_json
from openpi.policies.subtask_policy import create_subtask_policy
from openpi.training import config as config_lib
from openpi.training.stage1_data import sha256_file
from openpi.training.subtask_batch import collate_subtask
from openpi.training.subtask_transition import read_jsonl
from openpi.training.transition_feature_cache import compress_memory,storage_dtype,encode_tensor,decode_array


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--assets',type=Path,default=Path('assets/pi05_piper_transition/eggplant_potato/r2_v1'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--engineering-frames',type=int,default=0)
    args=parser.parse_args()
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    torch.set_num_threads(4)
    torch.manual_seed(42)
    device=torch.device(f'cuda:{local}' if args.device=='cuda' else 'cpu')
    if device.type=='cuda':torch.cuda.set_device(device)
    protocol=json.loads((args.assets/'protocol.json').read_text())
    protocol_sha=sha256_file(args.assets/'protocol.json')
    parent=Path(protocol['parent_checkpoint'])
    if sha256_file(parent/'model.safetensors')!=protocol['parent_weights_sha256']:raise ValueError('Parent changed')
    for name,digest in protocol['files'].items():
        if sha256_file(args.assets/name)!=digest:raise ValueError('Protocol asset changed: '+name)
    args.output.mkdir(parents=True,exist_ok=True)
    config={'schema_version':1,'parent_weights_sha256':protocol['parent_weights_sha256'],'protocol_sha256':protocol_sha,
            'feature_scope':'current observation and global task only; raw frozen B prefix; no GT text',
            'seed':42,'engineering_frames':args.engineering_frames,'source_sha256':sha256_file(Path(__file__)),
            'cache_module_sha256':sha256_file(Path(compress_memory.__code__.co_filename)),
            'source_manifest_sha256':os.environ.get('OPENPI_R2_SOURCE_MANIFEST_SHA256')}
    identity=args.output/'cache_identity.json'
    if identity.exists() and json.loads(identity.read_text())!=config:raise ValueError('Cache identity differs')
    if rank==0 and not identity.exists():write_json(identity,config)
    policy=create_subtask_policy(parent,device=str(device),num_steps=10)
    model=policy.model.eval()
    # CPU engineering checks use FP32; they never populate the formal CUDA cache.
    if device.type=='cpu':model.float()
    cfg=config_lib.get_config('pi05_piper_stage1')
    dc=cfg.data.create(cfg.assets_dirs,model.base.config)
    total=0
    for split in ['train','val']:
        rows=read_jsonl(args.assets/f'frames_{split}.jsonl')
        dataset=build_dataset(dataclasses.replace(dc,split=split),model.base.config,rows)
        episodes=sorted(set(r['episode'] for r in rows))
        if args.engineering_frames:episodes=episodes[:1]
        for ep in episodes[rank::world]:
            all_rows=[r for r in rows if r['episode']==ep]
            selected=all_rows[:args.engineering_frames] if args.engineering_frames else all_rows
            directory=args.output/split/f'episode_{ep:06d}'
            if (directory/'complete.json').exists():
                prior=json.loads((directory/'complete.json').read_text())
                if prior['cache_identity']!=config or prior['frames']!=len(selected):raise ValueError('Existing episode cache differs')
                total+=len(selected);continue
            directory.mkdir(parents=True,exist_ok=True)
            begun=time.time();arrays=None;predictions=[];max_error=0.
            for offset in range(0,len(selected),args.batch_size):
                chunk=selected[offset:offset+args.batch_size]
                batch=collate_subtask([dataset[r['index']] for r in chunk]).to(device)
                context=model.prepare_context(batch.observation,batch.global_prompts)
                summary,summary_mask=compress_memory(context.memory,context.memory_mask)
                texts,statuses,_=model.generate_subtask(context)
                if arrays is None:
                    n=len(selected);dtype=context.memory.dtype
                    arrays={name:np.lib.format.open_memmap(directory/(name+'.npy'),mode='w+',dtype=kind,shape=(n,*shape))
                            for name,kind,shape in [('memory',storage_dtype(dtype),context.memory.shape[1:]),
                                                    ('mask',np.bool_,context.memory_mask.shape[1:]),
                                                    ('summary',storage_dtype(dtype),summary.shape[1:]),
                                                    ('summary_mask',np.bool_,summary_mask.shape[1:]),
                                                    ('state',np.float32,context.state.shape[1:])]}
                end=offset+len(chunk)
                for name,tensor in [('memory',context.memory),('mask',context.memory_mask),('summary',summary),
                                    ('summary_mask',summary_mask),('state',context.state.float())]:
                    arrays[name][offset:end]=encode_tensor(tensor)
                predictions.extend(dict(r,prediction=t,status=s) for r,t,s in zip(chunk,texts,statuses,strict=True))
            for array in arrays.values():array.flush()
            write_json(directory/'rows.json',predictions)
            metadata={'schema_version':1,'episode':ep,'split':split,'frames':len(selected),
                      'dtype':str(dtype).split('.')[-1],'cache_identity':config,'elapsed_seconds':time.time()-begun,
                      'files':{name+'.npy':{'shape':list(array.shape),'dtype':str(array.dtype),'bytes':(directory/(name+'.npy')).stat().st_size}
                               for name,array in arrays.items()},'rows_sha256':sha256_file(directory/'rows.json')}
            # Round-trip stored representation must be bit-exact, including BF16 bits.
            assert torch.equal(decode_array(arrays['memory'][-1],metadata['dtype']),context.memory[-1].cpu())
            assert torch.equal(decode_array(arrays['summary'][-1],metadata['dtype']),summary[-1].cpu())
            assert torch.equal(torch.tensor(arrays['mask'][-1]),context.memory_mask[-1].cpu())
            write_json(directory/'complete.json',metadata)
            total+=len(selected)
            print(json.dumps({'event':'episode_cached','rank':rank,'split':split,'episode':ep,'frames':len(selected),
                              'roundtrip_exact':True,'seconds':metadata['elapsed_seconds']}),flush=True)
    write_json(args.output/f'rank_{rank:03d}_complete.json',{'rank':rank,'world_size':world,'frames':total,'cache_identity':config})


if __name__=='__main__':main()
