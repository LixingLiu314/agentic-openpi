"""CPU structural regression using actual parallel_layer and Gemma attention math."""
import types
import torch
from torch import nn
from openpi.models_pytorch.parallel_subtask import parallel_layer, ParallelDecoder
from openpi.models_pytorch.subtask_decoder import SubtaskDecoderConfig

class Norm(nn.Module):
    def __init__(self):super().__init__();self.norm=nn.LayerNorm(16)
    def forward(self,x,cond=None):return self.norm(x),None
class MLP(nn.Module):
    def __init__(self):super().__init__();self.up_proj=nn.Linear(16,32);self.down=nn.Linear(32,16)
    def forward(self,x):return self.down(torch.nn.functional.gelu(self.up_proj(x)))
class Layer(nn.Module):
    def __init__(self):
        super().__init__();self.input_layernorm=Norm();self.post_attention_layernorm=Norm();self.mlp=MLP()
        self.self_attn=nn.Module();self.self_attn.head_dim=8;self.self_attn.scaling=8**-.5
        self.self_attn.num_key_value_groups=1
        for k in ['q_proj','k_proj','v_proj','o_proj']:setattr(self.self_attn,k,nn.Linear(16,16))
def main():
    torch.manual_seed(42)
    b=nn.Module();b.layers=nn.ModuleList([Layer(),Layer()])
    a=nn.Module();a.layers=nn.ModuleList([Layer(),Layer()])
    b.rotary_emb=lambda x,pos:(torch.ones(x.shape[0],x.shape[1],8),torch.zeros(x.shape[0],x.shape[1],8))
    w=types.SimpleNamespace(paligemma=types.SimpleNamespace(language_model=b,model=types.SimpleNamespace(language_model=b)),gemma_expert=types.SimpleNamespace(model=a))
    pos=torch.arange(5)[None].expand(2,-1);mask=torch.zeros(2,1,5,5);mask[:,:,:3,3:]=-1e9
    def forward():
        p=torch.randn(2,3,16,requires_grad=True);s=torch.randn(2,2,16,requires_grad=True)
        for i in range(2):p,s=parallel_layer(w,i,p,s,mask,pos,None)
        return p,s
    p,s=forward();s.square().mean().backward()
    assert all(v.grad is None or torch.count_nonzero(v.grad)==0 for v in b.parameters())
    assert any(v.grad is not None and torch.count_nonzero(v.grad)>0 for v in a.parameters())
    b.zero_grad(set_to_none=True);a.zero_grad(set_to_none=True)
    p,s=forward();p.square().mean().backward()
    assert all(v.grad is None or torch.count_nonzero(v.grad)==0 for v in a.parameters())
    assert any(v.grad is not None and torch.count_nonzero(v.grad)>0 for v in b.parameters())
    b.zero_grad(set_to_none=True);a.zero_grad(set_to_none=True)
    p,s=forward();params=list(b.parameters())+list(a.parameters())
    lp,la=p.square().mean(),s.square().mean()
    gp=torch.autograd.grad(lp,params,allow_unused=True,retain_graph=True)
    ga=torch.autograd.grad(la,params,allow_unused=True,retain_graph=True)
    combined=torch.autograd.grad(lp+la,params,allow_unused=True)
    for param,left,right,total in zip(params,gp,ga,combined):
        left=torch.zeros_like(param) if left is None else left
        right=torch.zeros_like(param) if right is None else right
        total=torch.zeros_like(param) if total is None else total
        torch.testing.assert_close(total,left+right,rtol=1e-5,atol=1e-6)
    d=ParallelDecoder(SubtaskDecoderConfig(memory_dim=16,embedding_dim=16,width=16,heads=2,layers=1,mlp_dim=32),recurrent=True)
    h=torch.randn(4,3,16,requires_grad=True);m=torch.ones(4,3,dtype=torch.bool)
    projected,_,carry=d.compose_sequence(h,m,resets=torch.tensor([True,False,False,False]),unroll=4)
    projected[-1].square().mean().backward()
    assert h.grad[0].abs().sum()>0 and h.grad[-1].abs().sum()>0
    assert d.memory_update.weight_hh.grad.abs().sum()>0
    print('PASS: 2-layer action->B blocked; CE->A blocked; merged loss gradients equal separate routes; recurrent CE reaches current and past B features')
if __name__=='__main__':main()
