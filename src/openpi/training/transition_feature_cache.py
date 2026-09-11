"""Exact frozen observation-feature storage; supervision stays in sidecars."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def compress_memory(memory, mask, views=3, grid=4):
    """Pool contextual image patches, retaining view/position order and one text mean."""
    b,length,width=memory.shape
    patches=256
    if length<=views*patches:
        raise ValueError('Expected three 16x16 image streams followed by prompt tokens')
    visual=memory[:,:views*patches].float().reshape(b*views,16,16,width).permute(0,3,1,2)
    pooled=F.adaptive_avg_pool2d(visual,(grid,grid)).permute(0,2,3,1).reshape(b,views*grid*grid,width)
    view_mask=mask[:,:views*patches].reshape(b,views,patches).all(-1)
    pooled_mask=view_mask[:,:,None].expand(b,views,grid*grid).reshape(b,-1)
    text_mask=mask[:,views*patches:]
    text=(memory[:,views*patches:].float()*text_mask[:,:,None]).sum(1)/text_mask.sum(1,keepdim=True).clamp_min(1)
    return torch.cat([pooled,text[:,None]],1).to(memory.dtype),torch.cat([pooled_mask,text_mask.any(1,keepdim=True)],1)


def storage_dtype(dtype):
    return {torch.bfloat16:np.uint16,torch.float32:np.float32,torch.float16:np.float16}[dtype]


def encode_tensor(tensor):
    tensor=tensor.detach().cpu().contiguous()
    return tensor.view(torch.uint16).numpy() if tensor.dtype==torch.bfloat16 else tensor.numpy()


def decode_array(array, dtype_name):
    result=torch.from_numpy(np.array(array,copy=True))
    return result.view(torch.bfloat16) if dtype_name=='bfloat16' else result


class EpisodeFeatures:
    def __init__(self,path):
        self.path=Path(path)
        self.metadata=json.loads((self.path/'complete.json').read_text())
        if self.metadata['schema_version']!=1:raise ValueError('Unknown observation cache schema')
        self.memory=np.load(self.path/'memory.npy',mmap_mode='r')
        self.mask=np.load(self.path/'mask.npy',mmap_mode='r')
        self.summary=np.load(self.path/'summary.npy',mmap_mode='r')
        self.summary_mask=np.load(self.path/'summary_mask.npy',mmap_mode='r')
        self.state=np.load(self.path/'state.npy',mmap_mode='r')
        self.rows=json.loads((self.path/'rows.json').read_text())
        self.times=np.array([r['timestamp'] for r in self.rows])
        self.dtype=self.metadata['dtype']

    def current(self,index):
        return {'memory':decode_array(self.memory[index],self.dtype),'memory_mask':torch.tensor(self.mask[index]),
                'summary':decode_array(self.summary[index],self.dtype),'summary_mask':torch.tensor(self.summary_mask[index]),
                'state':torch.tensor(self.state[index])}

    def past(self,index,period=.764,count=3):
        current=self.times[index]
        requested=current-period*np.arange(count,0,-1)
        ids=np.searchsorted(self.times,requested,side='right')-1
        return [int(i) if i>=0 and i<index else None for i in ids]
