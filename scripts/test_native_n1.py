"""CPU gates with real tiny PaliGemma/Gemma layers (no mocked attention math)."""
import dataclasses
import copy
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('HF_HUB_OFFLINE','1')
import numpy as np
import safetensors.torch
import torch
from torch import nn
from transformers import GemmaConfig, GemmaForCausalLM, PaliGemmaConfig, PaliGemmaForConditionalGeneration, SiglipVisionConfig
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
from openpi.models_pytorch.native_subtask import NativeSubtaskModel
from openpi.models.subtask_tokenizer import SubtaskTextCodec
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

class Codec:
    pad_id=0; eos_id=1; bos_id=2
    processor=SimpleNamespace(encode=lambda s:[3,4], decode=lambda t:' '.join(map(str,t)))
    def __init__(self,*args):pass
    def prompts(self,prompts,states):
        ids=np.tile([2,5,6,0,0],(len(prompts),1))
        return ids,ids!=0
    decode=SubtaskTextCodec.decode

class TinyWrapper(PaliGemmaWithExpertModel):
    def __init__(self):
        nn.Module.__init__(self)
        text=GemmaConfig(hidden_size=32,intermediate_size=64,num_attention_heads=2,
            num_key_value_heads=1,head_dim=16,num_hidden_layers=2,vocab_size=64,
            hidden_activation='gelu_pytorch_tanh',use_adarms=False)
        vision=SiglipVisionConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=1,
            num_attention_heads=2,image_size=4,patch_size=2,projection_dim=32)
        self.paligemma=PaliGemmaForConditionalGeneration(PaliGemmaConfig(text_config=text.to_dict(),
            vision_config=vision.to_dict(),projection_dim=32,image_token_index=63))
        self.gemma_expert=GemmaForCausalLM(text)
        self.gemma_expert.model.embed_tokens=None
        self.paligemma.language_model.config._attn_implementation='eager'

class TinyBase(nn.Module):
    pi05=True
    config=SimpleNamespace(max_token_len=5,action_horizon=2,action_dim=4)
    def __init__(self):
        super().__init__();self.paligemma_with_expert=TinyWrapper()
        self.action_in_proj=nn.Linear(4,32);self.action_out_proj=nn.Linear(32,4)
        self.time_mlp_in=nn.Linear(1,32);self.time_mlp_out=nn.Linear(32,32)
    def gradient_checkpointing_disable(self):pass
    def _preprocess_observation(self,o,train=False):
        return [o.image],[torch.ones(o.state.shape[0],dtype=torch.bool)],None,None,o.state
    def _prepare_attention_masks_4d(self,m):
        return torch.where(m[:,None],0.,-2.3819763e38)
    def embed_suffix(self,state,actions,time):
        h=self.action_in_proj(actions)+self.time_mlp_out(torch.tanh(self.time_mlp_in(time[:,None])))[:,None]
        valid=torch.ones(h.shape[:2],dtype=torch.bool); ar=torch.zeros_like(valid);ar[:,0]=True
        return h,valid,ar,None

def nonzero(parameters):
    return sum(float(p.grad.detach().float().square().sum()) for p in parameters if p.grad is not None)

def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    from train_native_n1 import verify_resume_config, Optimizers
    original=dict(training_git_commit='1'*40,sources={'model.py':'same'},world_size=8)
    assert verify_resume_config({**original,'training_git_commit':'2'*40},original)==original
    for changed in [{'sources':{'model.py':'changed'}},{'world_size':4}]:
        try:verify_resume_config({**original,**changed},original)
        except ValueError:pass
        else:raise AssertionError('Changed resume contract accepted')
    with patch('openpi.models_pytorch.native_subtask.SubtaskTextCodec',Codec):
        model=NativeSubtaskModel(TinyBase(),max_tokens=4)
        clone=NativeSubtaskModel(TinyBase(),max_tokens=4)
    assert not hasattr(model,'decoder')
    assert not any(isinstance(m,nn.TransformerDecoderLayer) for m in model.modules())
    assert set(model.state_dict())=={'base.'+k for k in model.base.state_dict()}
    optim=Optimizers(model,1)
    assert set(optim.parameters)=={'action','backbone'}
    observation=SimpleNamespace(image=torch.randn(2,3,4,4),state=torch.randn(2,4))
    batch=SimpleNamespace(observation=observation,global_prompts=['task']*2,actions=torch.randn(2,2,4),
        target_ids=torch.tensor([[8,9,1,0],[10,11,12,1]]),
        target_mask=torch.tensor([[True,True,True,False],[True,True,True,True]]))
    noise=torch.randn_like(batch.actions);time=torch.tensor([.3,.4])
    def joint(ids=None):
        return model.joint_outputs(observation,batch.global_prompts,batch.actions,
            batch.target_ids if ids is None else ids,batch.target_mask,noise=noise,time=time)
    ids,valid=model.teacher_inputs(batch.target_ids,batch.target_mask)
    assert ids.tolist()==[[3,4,8,9,1],[3,4,10,11,12]]
    model.eval()
    with torch.no_grad():
        logits,act=joint()
        changed=batch.target_ids.clone();changed[:,:]=15
        other,alt=joint(changed)
        torch.testing.assert_close(act,alt,rtol=0,atol=0)
        torch.testing.assert_close(logits[:,0],other[:,0],rtol=0,atol=0)
        changed=batch.target_ids.clone();changed[:,1]=15
        later,_=joint(changed)
        torch.testing.assert_close(logits[:,:2],later[:,:2],rtol=0,atol=0)
        context=model.prepare_context(observation,batch.global_prompts)
        cached,_=model.logits_cached(context,ids,valid)
        torch.testing.assert_close(logits,cached[:,len(model.cue_ids)-1:],rtol=2e-5,atol=1e-6)
        # Independent reference: run original HF Gemma once over prefix+text
        # with prefix-block/text-causal mask, not our text helper.
        w=model.base.paligemma_with_expert
        h=torch.cat([context.prefix,model.embed_text(ids)],1)
        mask=torch.cat([context.mask,valid],1)
        ar=torch.cat([torch.zeros_like(context.mask),torch.ones_like(valid)],1)
        positions=torch.cat([context.mask.cumsum(1)-1,
            context.mask.sum(1)[:,None]+torch.arange(ids.shape[1])[None]],1)
        out=w.paligemma.language_model(inputs_embeds=h,
            attention_mask=model.base._prepare_attention_masks_4d(make_att_2d_masks(mask,ar)),
            position_ids=positions,use_cache=False).last_hidden_state
        ref=w.paligemma.lm_head(out[:,context.prefix.shape[1]:])
        torch.testing.assert_close(cached[valid],ref[valid],rtol=2e-5,atol=1e-6)
        # Token-at-a-time cache parity uses a fully valid stream.
        all_valid=torch.ones_like(ids,dtype=torch.bool)
        full,_=model.logits_cached(context,ids,all_valid)
        past=None; pieces=[]
        for i in range(ids.shape[1]):
            step,past=model.logits_cached(context,ids[:,i:i+1],past=past);pieces.append(step)
        torch.testing.assert_close(full,torch.cat(pieces,1),rtol=2e-5,atol=1e-6)
        before=[(k.clone(),v.clone()) for k,v in context.pairs]
        model.generate_subtask(context)
        for (k,v),(kk,vv) in zip(before,context.pairs):
            torch.testing.assert_close(k,kk,rtol=0,atol=0);torch.testing.assert_close(v,vv,rtol=0,atol=0)
        pref=model.action_prefix(context,['wrong']*2)
        assert pref.cache is context.cache and pref.mask is context.mask
        # EOS contributes; PAD target value does not.
        original_ce=model.ce(logits,batch.target_ids,batch.target_mask)
        padded=batch.target_ids.clone();padded[~batch.target_mask]=60
        torch.testing.assert_close(original_ce,model.ce(logits,padded,batch.target_mask),rtol=0,atol=0)
    model.train();model.zero_grad(set_to_none=True)
    model(batch,noise=noise,time=time)['loss_subtask'].backward()
    b=model.base.paligemma_with_expert.paligemma
    ce=dict(B=nonzero(model.backbone_parameters()),A=nonzero(model.action_parameters()),
        vision=nonzero(b.vision_tower.parameters()),projector=nonzero(b.multi_modal_projector.parameters()),
        head=nonzero([b.lm_head.weight]),language=nonzero(b.language_model.layers.parameters()))
    assert ce['A']==0 and all(v>0 for k,v in ce.items() if k!='A'),ce
    model.zero_grad(set_to_none=True)
    model(batch,noise=noise,time=time)['loss_action'].backward()
    flow=dict(B=nonzero(model.backbone_parameters()),A=nonzero(model.action_parameters()))
    assert flow['B']==0 and flow['A']>0,flow
    model.zero_grad(set_to_none=True)
    (sum(model(batch,noise=noise,time=time)[k] for k in ['loss_subtask','loss_action'])).backward()
    optim.step(1e-4);optim.zero_grad()
    with tempfile.TemporaryDirectory(prefix='n1_cpu_') as folder:
        path=Path(folder)/'model.safetensors'
        safetensors.torch.save_file(model.deployment_state(),path)
        safetensors.torch.load_model(clone,path,strict=True);clone.verify_tied_head()
        clone_optim=Optimizers(clone,1);clone_optim.load_state_dict(copy.deepcopy(optim.state_dict()))
        for current in [model,clone]:
            current.train();losses=current(batch,noise=noise,time=time)
            (losses['loss_subtask']+losses['loss_action']).backward()
        optim.step(1e-4);clone_optim.step(1e-4)
        for k,v in model.state_dict().items():torch.testing.assert_close(v,clone.state_dict()[k],rtol=0,atol=0)
    print(json.dumps(dict(passed=True,real_tiny_hf_layers=2,native_head_tied=True,
        no_new_text_parameters=True,causal_shift=True,action_gt_invariant=True,
        native_joint_and_incremental_cache_parity=True,prefix_cache_immutable=True,
        ce_gradients=ce,flow_gradients=flow,optimizer_resume=True)))
if __name__=='__main__':main()
