"""Real native N1 loading, causal isolation and separately measured gradient gate."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('HF_HUB_OFFLINE','1')
import numpy as np
import torch
from openpi.policies.native_subtask_policy import create_native_policy
from openpi.training.reach_arm_data import data_config
from openpi.training import config,data_loader
from openpi.training.subtask_batch import SubtaskTrainingDataset,collate_subtask

def grad_norm(params):
    values=[p.grad.detach().float().norm() for p in params if p.grad is not None]
    return float(torch.stack(values).norm()) if values else 0.

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--allow-engineering',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42)
    if str(a.device).startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(.9,torch.device(a.device))
    policy=create_native_policy(a.checkpoint,device=a.device,allow_engineering=a.allow_engineering)
    model=policy.model;model.verify_tied_head()
    assert not hasattr(model,'decoder') and not any(isinstance(m,torch.nn.TransformerDecoderLayer) for m in model.modules())
    dc=data_config(config.get_config('pi05_piper_stage1').model,split='val')
    raw=data_loader.create_torch_dataset(dc,50,model.base.config)
    sample=raw[0]
    obs=dict(prompt=sample['task'],state=np.asarray(sample['observation.state']).copy(),
        images={k:np.asarray(sample['observation.images.'+k]).copy() for k in ['cam_high','cam_left_wrist','cam_right_wrist']})
    original_state=obs['state'].copy();original_images={k:v.copy() for k,v in obs['images'].items()}
    noise=np.random.default_rng(4250).standard_normal((50,32)).astype('float32')
    left,right=policy.new_session(),policy.new_session()
    first=left.infer(obs,noise=noise);second=right.infer(obs,noise=noise);again=left.infer(obs,noise=noise)
    np.testing.assert_array_equal(first['actions'],second['actions'])
    np.testing.assert_array_equal(first['actions'],again['actions'])
    assert first['actions'].shape==(50,14) and np.isfinite(first['actions']).all()
    left.reset();np.testing.assert_array_equal(first['actions'],left.infer(obs,noise=noise)['actions'])
    np.testing.assert_array_equal(original_state,obs['state'])
    for k in original_images:np.testing.assert_array_equal(original_images[k],obs['images'][k])
    try:policy.infer({**obs,'subtask':'injected'})
    except ValueError:pass
    else:raise AssertionError('External subtask accepted')
    result=dict(passed=True,native_shape=[50,14],fixed_noise_repeat=True,stateless=True,
        inputs_unchanged=True,external_subtask_rejected=True,native_head_tied=True,
        independent_text_network=False,metadata=policy.metadata)
    batch=collate_subtask([SubtaskTrainingDataset(raw,dc)[0]]).to(a.device)
    fixed=torch.randn_like(batch.actions);times=torch.full((1,),.4,device=a.device)
    model.eval()
    with torch.no_grad():
        context=model.prepare_context(batch.observation,batch.global_prompts)
        before_pairs=[(k.clone(),v.clone()) for k,v in context.pairs]
        clean=model.sample_actions_from_prefix(context,model.action_prefix(context),noise=fixed)
        model.generate_subtask(context)
        after=model.sample_actions_from_prefix(context,model.action_prefix(context,['wrong']),noise=fixed)
        torch.testing.assert_close(clean,after,rtol=0,atol=0)
        for (k,v),(kk,vv) in zip(before_pairs,context.pairs):
            torch.testing.assert_close(k,kk,rtol=0,atol=0);torch.testing.assert_close(v,vv,rtol=0,atol=0)
        result.update(action_unchanged_with_text_generation=True,prefix_cache_immutable=True)
    if a.allow_engineering:
        with torch.no_grad():
            def joint(ids):
                return model.joint_outputs(batch.observation,batch.global_prompts,batch.actions,
                    ids,batch.target_mask,noise=fixed,time=times)
            logits,velocity=joint(batch.target_ids)
            corrupted=batch.target_ids.clone();corrupted[:]=42
            other,altered=joint(corrupted)
            torch.testing.assert_close(velocity,altered,rtol=0,atol=0)
            torch.testing.assert_close(logits[:,0],other[:,0],rtol=0,atol=0)
            corrupted=batch.target_ids.clone();corrupted[:,1]=42
            later,_=joint(corrupted)
            torch.testing.assert_close(logits[:,:2],later[:,:2],rtol=0,atol=0)
            cached=model.teacher_logits_cached(context,batch.target_ids,batch.target_mask)
            torch.testing.assert_close(logits[batch.target_mask],cached[batch.target_mask],rtol=0,atol=0)
            noisy=times[:,None,None]*fixed+(1-times[:,None,None])*batch.actions
            native=model.base.denoise_step(context.state,context.mask,context.cache,noisy,times)
            torch.testing.assert_close(velocity,native,rtol=0,atol=0)
            result.update(native_action_max_abs=float((velocity-native).abs().max()),
                native_teacher_max_abs=float((logits[batch.target_mask]-cached[batch.target_mask]).abs().max()),
                action_gt_invariant=True,causal_future_invariant=True)
        del logits,velocity,other,altered,later,cached,native,context,before_pairs,clean,after
        model.train();model.zero_grad(set_to_none=True)
        model(batch,noise=fixed,time=times)['loss_subtask'].backward()
        b=model.base.paligemma_with_expert.paligemma
        ce=dict(B=grad_norm(model.backbone_parameters()),A=grad_norm(model.action_parameters()),
            vision=grad_norm(b.vision_tower.parameters()),projector=grad_norm(b.multi_modal_projector.parameters()),
            language=grad_norm(b.language_model.layers.parameters()),head=grad_norm([b.lm_head.weight]))
        assert ce['A']==0 and all(v>0 for k,v in ce.items() if k!='A'),ce
        model.zero_grad(set_to_none=True)
        model(batch,noise=fixed,time=times)['loss_action'].backward()
        act=dict(B=grad_norm(model.backbone_parameters()),A=grad_norm(model.action_parameters()))
        assert act['B']==0 and act['A']>0,act
        model.zero_grad(set_to_none=True)
        result.update(ce_gradients=ce,action_gradients=act)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
