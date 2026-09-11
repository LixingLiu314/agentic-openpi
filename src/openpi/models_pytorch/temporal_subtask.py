"""R2: causal observed history and model-owned active text, still autoregressive.

Only raw frozen-backbone features/embedding weights are detached. The temporal
encoder and completion head are inside the trainable S branch. No dataset stage
lookup, episode clock, future frame, or externally selected subtask is accepted.
"""
import dataclasses

import torch
from torch import nn
import torch.nn.functional as F

from openpi.models_pytorch.subtask_decoder import SubtaskDecoder,SubtaskDecoderConfig,SubtaskGeneration


@dataclasses.dataclass(frozen=True)
class TemporalConfig:
    history_count:int=4
    summary_tokens:int=49
    state_dim:int=14
    temporal_layers:int=2
    max_history_seconds:float=4.0


class TemporalSubtask(nn.Module):
    def __init__(self,decoder_config=None,config=None):
        super().__init__()
        self.decoder=SubtaskDecoder(decoder_config)
        self.config=config or TemporalConfig()
        c=self.decoder.config;w=c.width
        self.spatial_position=nn.Parameter(torch.randn(self.config.summary_tokens+1,w)*.02)
        self.current_marker=nn.Parameter(torch.randn(1,1,w)*.02)
        self.active_marker=nn.Parameter(torch.randn(1,1,w)*.02)
        self.summary_cls=nn.Parameter(torch.randn(1,1,w)*.02)
        self.state_projection=nn.Sequential(nn.Linear(self.config.state_dim,w),nn.GELU(),nn.Linear(w,w))
        self.time_projection=nn.Sequential(nn.Linear(4,w),nn.GELU(),nn.Linear(w,w))
        self.temporal=nn.ModuleList([nn.TransformerEncoderLayer(w,c.heads,c.mlp_dim,dropout=0.,activation='gelu',
                                                             batch_first=True,norm_first=True) for _ in range(self.config.temporal_layers)])
        self.temporal_norm=nn.LayerNorm(w)
        self.memory_norm=nn.LayerNorm(w)
        self.temporal_attention=nn.MultiheadAttention(w,c.heads,dropout=0.,batch_first=True)
        self.residual_scale=nn.Parameter(torch.zeros(()))
        self.completion=nn.Sequential(nn.LayerNorm(w),nn.Linear(w,w//2),nn.GELU(),nn.Linear(w//2,3))

    def compose(self,memory,memory_mask,summaries,summary_masks,states,ages,frame_valid,active_ids,active_mask,embedding):
        """Ages are elapsed seconds before CURRENT observation, never absolute time.

        Summaries contain oldest..current observations, including current as last.
        All tensors here are deployment-available; target tokens enter loss only.
        """
        b,k,n,d=summaries.shape
        if k!=self.config.history_count or n!=self.config.summary_tokens:
            raise ValueError('Wrong observed-history layout')
        if memory.shape[0]!=b or memory_mask.shape!=memory.shape[:2]:raise ValueError('Current memory mismatch')
        if ages.shape!=(b,k) or frame_valid.shape!=(b,k):raise ValueError('History time/mask mismatch')
        if not frame_valid[:,-1].all() or not torch.all(ages[:,-1]==0):raise ValueError('Current frame must be valid at age zero')
        if not torch.isfinite(ages).all() or (ages[frame_valid]<0).any():raise ValueError('Future or nonfinite history')
        if ((ages>self.config.max_history_seconds)&frame_valid).any():raise ValueError('Expired observation history')
        if ((ages[:,:-1]<=0)&frame_valid[:,:-1]).any():raise ValueError('Past observations must precede current')
        dtype=self.decoder.memory_projection.weight.dtype
        projected=self.decoder.memory_projection(memory.detach().to(dtype))
        features=self.decoder.memory_projection(summaries.detach().to(dtype))
        state_tokens=self.state_projection(states[...,:self.config.state_dim].detach().to(dtype))[:,:,None]
        features=torch.cat([features,state_tokens],dim=2)+self.spatial_position[None,None]
        age=ages.to(dtype)
        encoding=torch.stack([age/3.,age.sin(),age.cos(),torch.log1p(age)],-1)
        features=features+self.time_projection(encoding)[:,:,None]
        features[:,-1]=features[:,-1]+self.current_marker
        observed_mask=torch.cat([summary_masks,torch.ones((b,k,1),dtype=torch.bool,device=memory.device)],-1)&frame_valid[:,:,None]
        active=self.decoder.input_projection(F.embedding(active_ids,embedding.detach()).to(dtype))+self.active_marker
        features=torch.cat([self.summary_cls.expand(b,-1,-1),features.flatten(1,2),active],1)
        mask=torch.cat([torch.ones((b,1),dtype=torch.bool,device=memory.device),observed_mask.flatten(1,2),active_mask],1)
        for layer in self.temporal:features=layer(features,src_key_padding_mask=~mask)
        features=self.temporal_norm(features)
        completion=self.completion(features[:,0])
        residual,_=self.temporal_attention(self.memory_norm(projected),features,features,key_padding_mask=~mask,need_weights=False)
        # Zero initialization exactly preserves the original observation-to-text map.
        adapted=projected+self.residual_scale.tanh()*residual
        return adapted,completion

    def logits(self,input_ids,projected_memory,memory_mask,embedding):
        c=self.decoder.config
        if input_ids.ndim!=2 or not 1<=input_ids.shape[1]<=c.max_tokens:raise ValueError('Wrong autoregressive input')
        dtype=self.decoder.input_projection.weight.dtype
        hidden=self.decoder.input_projection(F.embedding(input_ids,embedding.detach()).to(dtype))
        hidden=hidden+self.decoder.position_embedding[None,:input_ids.shape[1]]
        causal=torch.ones((input_ids.shape[1],input_ids.shape[1]),device=input_ids.device,dtype=torch.bool).triu(1)
        for layer in self.decoder.layers:
            hidden=layer(hidden,projected_memory,tgt_mask=causal,tgt_key_padding_mask=input_ids.eq(c.pad_id),memory_key_padding_mask=~memory_mask)
        projected=self.decoder.output_projection(self.decoder.output_norm(hidden))
        return F.linear(projected.to(embedding.dtype),embedding.detach())

    def forward(self,inputs,target_ids,target_mask,completion_targets,embedding,completion_weight=.3):
        projected,done=self.compose(**inputs,embedding=embedding)
        c=self.decoder.config
        shifted=torch.full_like(target_ids,c.pad_id);shifted[:,0]=c.bos_id
        shifted[:,1:]=torch.where(target_mask[:,:-1],target_ids[:,:-1],c.pad_id)
        logits=self.logits(shifted,projected,inputs['memory_mask'],embedding)
        targets=torch.where(target_mask,target_ids,c.pad_id)
        ce=F.cross_entropy(logits.float().flatten(0,1),targets.flatten(),reduction='none').view_as(target_ids)
        per_example=(ce*target_mask).sum(1)/target_mask.sum(1).clamp_min(1)
        done_loss=F.cross_entropy(done.float(),completion_targets,ignore_index=-100,reduction='none')
        valid=completion_targets.ne(-100)
        done_loss=done_loss.sum()/valid.sum().clamp_min(1)
        return {'loss':per_example.mean()+completion_weight*done_loss,'text_loss':per_example.mean(),
                'completion_loss':done_loss,'completion_valid':valid.sum(),'completion_logits':done}

    @torch.no_grad()
    @torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH])
    def generate(self,inputs,embedding):
        previous=self.training;self.eval()
        try:
            memory,done=self.compose(**inputs,embedding=embedding)
            c=self.decoder.config;b=memory.shape[0];device=memory.device
            ids=torch.full((b,1),c.bos_id,dtype=torch.long,device=device)
            output=torch.full((b,c.max_tokens),c.pad_id,dtype=torch.long,device=device)
            mask=torch.zeros_like(output,dtype=torch.bool);ended=torch.zeros(b,dtype=torch.bool,device=device)
            scores=torch.zeros(b,dtype=torch.float32,device=device)
            for position in range(c.max_tokens):
                logits=self.logits(ids,memory,inputs['memory_mask'],embedding)[:,-1].float()
                logits[:,[c.bos_id,c.pad_id]]=-torch.inf
                next_ids=logits.argmax(-1);active=~ended
                output[:,position]=torch.where(active,next_ids,c.pad_id);mask[:,position]=active
                scores+=torch.where(active,logits.log_softmax(-1).gather(1,next_ids[:,None]).squeeze(1),0)
                ended|=active&next_ids.eq(c.eos_id)
                if ended.all():break
                ids=torch.cat([ids,output[:,position:position+1]],1)
            return SubtaskGeneration(output,mask,ended,scores/mask.sum(1).clamp_min(1)),done.softmax(-1)
        finally:self.train(previous)
