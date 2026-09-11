import pytest
import torch
from openpi.models_pytorch.subtask_decoder import SubtaskDecoderConfig
from openpi.models_pytorch.temporal_subtask import TemporalSubtask,TemporalConfig


def setup():
    torch.manual_seed(42)
    cfg=SubtaskDecoderConfig(memory_dim=32,embedding_dim=32,width=16,heads=2,layers=2,mlp_dim=32,max_tokens=6)
    model=TemporalSubtask(cfg,TemporalConfig(summary_tokens=3,temporal_layers=1))
    embedding=torch.randn(64,32,requires_grad=True)
    inputs={'memory':torch.randn(2,9,32,requires_grad=True),'memory_mask':torch.ones(2,9,dtype=torch.bool),
            'summaries':torch.randn(2,4,3,32,requires_grad=True),'summary_masks':torch.ones(2,4,3,dtype=torch.bool),
            'states':torch.randn(2,4,14),'ages':torch.tensor([[2.1,1.4,.7,0.]]).expand(2,-1),
            'frame_valid':torch.ones(2,4,dtype=torch.bool),'active_ids':torch.tensor([[3,4,1],[5,1,0]]),
            'active_mask':torch.tensor([[True,True,True],[True,True,False]])}
    return model,embedding,inputs


def test_zero_residual_preserves_base_logits():
    model,embedding,x=setup();ids=torch.tensor([[2,3,4],[2,5,6]])
    memory,_=model.compose(**x,embedding=embedding)
    original=model.decoder(ids,x['memory'],x['memory_mask'],embedding)
    actual=model.logits(ids,memory,x['memory_mask'],embedding)
    assert torch.equal(original,actual)


def test_gradients_reach_temporal_done_decoder_but_never_frozen_inputs():
    model,embedding,x=setup()
    result=model(x,torch.tensor([[3,4,1],[5,6,1]]),torch.ones(2,3,dtype=torch.bool),torch.tensor([0,1]),embedding)
    result['loss'].backward()
    assert model.decoder.memory_projection.weight.grad.abs().sum()>0
    assert model.temporal[0].self_attn.in_proj_weight.grad.abs().sum()>0
    assert model.completion[-1].weight.grad.abs().sum()>0
    assert model.residual_scale.grad.abs()>0
    assert x['memory'].grad is None and x['summaries'].grad is None and embedding.grad is None


def test_future_or_expired_observation_is_rejected():
    model,embedding,x=setup();x['ages']=x['ages'].clone();x['ages'][0,0]=-.1
    with pytest.raises(ValueError,match='Future'):model.compose(**x,embedding=embedding)
    x['ages'][0,0]=5.
    with pytest.raises(ValueError,match='Expired'):model.compose(**x,embedding=embedding)


def test_text_logits_are_causal():
    model,embedding,x=setup();memory,_=model.compose(**x,embedding=embedding)
    a=torch.tensor([[2,3,4],[2,5,6]]);b=a.clone();b[:,-1]=9
    assert torch.equal(model.logits(a,memory,x['memory_mask'],embedding)[:,:-1],model.logits(b,memory,x['memory_mask'],embedding)[:,:-1])


def test_masked_history_content_does_not_change_prediction():
    model,embedding,x=setup();model.residual_scale.data.fill_(.5)
    x['frame_valid'][:,0]=False
    a,d=model.compose(**x,embedding=embedding)
    x['summaries']=x['summaries'].detach().clone();x['summaries'][:,0].add_(100)
    b,e=model.compose(**x,embedding=embedding)
    assert torch.equal(a,b) and torch.equal(d,e)
