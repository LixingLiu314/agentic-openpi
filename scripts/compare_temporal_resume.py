"""Bit-exact parameters, every rank's optimizer/RNG, and model-owned histories."""
import argparse
import json
from pathlib import Path
import numpy as np
import safetensors.torch
import torch


def equal(a,b,path='root'):
    if type(a)!=type(b):raise AssertionError(f'Type mismatch: {path}')
    if isinstance(a,torch.Tensor):
        if a.dtype!=b.dtype or a.shape!=b.shape or not torch.equal(a,b):raise AssertionError(f'Tensor mismatch: {path}')
    elif isinstance(a,np.ndarray):
        if a.dtype!=b.dtype or not np.array_equal(a,b):raise AssertionError(f'Array mismatch: {path}')
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise AssertionError(f'Keys mismatch: {path}')
        for k in a:equal(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b):raise AssertionError(f'Length mismatch: {path}')
        for i,(x,y) in enumerate(zip(a,b,strict=True)):equal(x,y,path+f'[{i}]')
    elif a!=b:raise AssertionError(f'Value mismatch: {path}')


def compare(first,second,world):
    a=safetensors.torch.load_file(first/'temporal.safetensors')
    b=safetensors.torch.load_file(second/'temporal.safetensors');equal(a,b,'parameters')
    for rank in range(world):
        name=f'training_rank_{rank:03d}.pt'
        equal(torch.load(first/name,weights_only=False,map_location='cpu'),
              torch.load(second/name,weights_only=False,map_location='cpu'),f'rank{rank}')
    equal(json.loads((first/'own_history.json').read_text()),json.loads((second/'own_history.json').read_text()),'own_history')
    return {'passed':True,'world_size':world,'tensor_count':len(a),'parameters_bit_exact':True,
            'every_rank_optimizer_rng_bit_exact':True,'own_history_equal':True,'engineering_only':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--first',type=Path,required=True);p.add_argument('--second',type=Path,required=True)
    p.add_argument('--world',type=int,default=8);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=compare(a.first,a.second,a.world);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
