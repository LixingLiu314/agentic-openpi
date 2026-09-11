import numpy as np
import pytest
import torch
from openpi.models_pytorch.subtask_decoder import SubtaskDecoder,SubtaskDecoderConfig
from openpi.training.subtask_transition import BoundarySampler,event_metrics,sampled_rows,text_loss_per_example,weighted_numerator


def test_loss_preserves_existing_length_normalization_and_gradient_ownership():
    torch.manual_seed(42)
    config=SubtaskDecoderConfig(memory_dim=8,embedding_dim=8,width=8,heads=2,layers=1,mlp_dim=16,max_tokens=4)
    decoder=SubtaskDecoder(config)
    memory=torch.randn(3,2,8,requires_grad=True)
    embed=torch.randn(12,8,requires_grad=True)
    ids=torch.tensor([[4,1,0],[5,6,1],[0,0,0]])
    mask=ids!=0
    memory_mask=torch.ones(3,2,dtype=torch.bool)
    existing=decoder.compute_loss(ids,mask,memory,memory_mask,embed)
    values,valid=text_loss_per_example(decoder,ids,mask,memory,memory_mask,embed)
    assert torch.allclose(existing,values.sum()/valid.sum())
    numerator,denominator=weighted_numerator(values,valid,torch.tensor([1.,3.,9.]))
    (numerator/denominator).backward()
    assert denominator.item()==4
    assert memory.grad is None and embed.grad is None
    assert decoder.memory_projection.weight.grad.abs().sum()>0


def test_global_weighted_gradient_with_uneven_ranks_and_accumulation():
    x=torch.tensor([1.,2.,4.,8.]);weights=torch.tensor([1.,0.,2.,7.])
    full=torch.tensor(.4,requires_grad=True)
    (((full*x)**2*weights).sum()/weights.sum()).backward()
    grads=[]
    for rank in range(2):
        value=torch.tensor(.4,requires_grad=True)
        for micro in range(2):
            i=rank+2*micro
            ((value*x[i])**2*weights[i]*2/weights.sum()).backward()
        grads.append(value.grad)
    assert torch.allclose(sum(grads)/2,full.grad)
    with pytest.raises(ValueError):weighted_numerator(x,torch.ones(4,dtype=torch.bool),-weights)


def rows_for_sampling():
    return [{"index":i,"episode":i//8,"task":"a" if i<16 else "b","transition":"old -> new",
             "boundary_event":None if i%2==0 else str(i//8),"boundary_side":"pre" if i%4<2 else "post"} for i in range(32)]


def test_sampler_resume_rank_partition_and_exact_half():
    rows=rows_for_sampling()
    options=dict(batch_size=2,accumulation=2,steps=5,world_size=2,seed=42)
    a=BoundarySampler(rows,rank=0,**options);b=BoundarySampler(rows,rank=1,**options)
    for step in range(5):
        draw=a.global_indices(step)
        assert sum(rows[i]["boundary_event"] is not None for i in draw.flat)==4
    assert list(BoundarySampler(rows,start=3,rank=0,**options))==list(a)[6:]
    assert list(a)!=list(b)


def event():
    return {"event_id":"v:1:30","episode":1,"task":"a","transition":"old -> new","new":"new",
            "t_label":1.,"cell_start":0.,"cell_end":3.,"new_end_time":3.}


def prediction(t,label):
    return {"episode":1,"timestamp":t,"prediction":label}


def test_transient_change_does_not_improve_stable_latency():
    rows=[prediction(0,"old"),prediction(1,"new"),prediction(1.1,"old"),prediction(1.5,"new"),prediction(1.9,"new")]
    r=event_metrics(rows,[event()])
    assert r["stable"]==1 and r["late_p50_p95_seconds"]==[.5,.5]
    assert r["events_detail"][0]["confirm_observation_time"]==1.9
    assert r["transient_correct_runs"]==1


def test_held_display_and_missing_tail_are_not_stable_observations():
    r=event_metrics([prediction(0,"old"),prediction(1,"new")],[event()])
    assert r["stable"]==0 and r["failed_or_unconfirmed"]==1
    assert r["late_p50_p95_seconds"] is None


def test_arrival_from_wrong_stage_still_detected_and_recorded():
    r=event_metrics([prediction(0,"wrong"),prediction(1,"new"),prediction(1.5,"new")],[event()])
    assert r["stable"]==1 and r["events_detail"][0]["previous_prediction"]=="wrong"


def test_sampling_uses_floor_and_retains_real_times():
    rows=[prediction(i/30,"old") for i in range(60)]
    selected=sampled_rows(rows,.764)
    assert [r["timestamp"] for r in selected]==[0,22/30,45/30]
    assert all(r["timestamp"]<=i*.764 for i,r in enumerate(selected))


def test_short_stage_is_separate_from_missing():
    e=event();e["new_end_time"]=1.1
    r=event_metrics([prediction(0,"old"),prediction(1,"new"),prediction(1.1,"next")],[e])
    assert r["short_stage"]==1 and r["failed_or_unconfirmed"]==0
