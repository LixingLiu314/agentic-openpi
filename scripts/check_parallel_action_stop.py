"""Real native, source-routing, forward-parity and isolated-loss gradient gate."""
import argparse,json,os
from pathlib import Path
os.environ.setdefault("JAX_PLATFORMS","cpu")
os.environ.setdefault("HF_HUB_OFFLINE","1")
import numpy as np
import torch
from openpi.policies.parallel_subtask_policy import create_parallel_policy
from openpi.training.reach_arm_data import data_config
from openpi.training import config,data_loader
from openpi.training.subtask_batch import SubtaskTrainingDataset
from openpi.training.recurrent_sequence import collate_sequence

def grad_norm(params):
    vals=[p.grad.detach().float().norm() for p in params if p.grad is not None]
    return float(torch.stack(vals).norm()) if vals else 0.

def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cuda:0")
    p.add_argument("--allow-engineering",action="store_true");a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42)
    policy=create_parallel_policy(a.checkpoint,device=a.device,allow_engineering=a.allow_engineering)
    dc=data_config(config.get_config("pi05_piper_stage1").model,split="val")
    raw=data_loader.create_torch_dataset(dc,50,policy.model.base.config)
    r=raw[0];obs={"prompt":r["task"],"state":np.asarray(r["observation.state"]).copy(),
        "images":{k:np.asarray(r["observation.images."+k]).copy() for k in ["cam_high","cam_left_wrist","cam_right_wrist"]}}
    original_state=obs["state"].copy();original_images={k:v.copy() for k,v in obs["images"].items()}
    noise=np.random.default_rng(4250).standard_normal((50,32)).astype("float32")
    left,right=policy.new_session(),policy.new_session()
    first=left.infer(obs,noise=noise);second=right.infer(obs,noise=noise)
    assert policy._memory is None and left._memory is not right._memory
    torch.testing.assert_close(left._memory,right._memory,rtol=0,atol=0)
    again=left.infer(obs,noise=noise)
    np.testing.assert_array_equal(first["actions"],second["actions"])
    np.testing.assert_array_equal(first["actions"],again["actions"])
    assert first["actions"].shape==(50,14) and np.isfinite(first["actions"]).all()
    left.reset();np.testing.assert_array_equal(first["actions"],left.infer(obs,noise=noise)["actions"])
    np.testing.assert_array_equal(original_state,obs["state"])
    for k in original_images:np.testing.assert_array_equal(original_images[k],obs["images"][k])
    try:policy.infer({**obs,"subtask":"injected"})
    except ValueError:pass
    else:raise AssertionError("External subtask accepted")
    result={"passed":True,"native_shape":[50,14],"fixed_noise_repeat":True,"S_memory_independent_action":True,
            "inputs_unchanged":True,"external_subtask_rejected":True,"metadata":policy.metadata}
    if a.allow_engineering:
        model=policy.model
        batch=collate_sequence([(SubtaskTrainingDataset(raw,dc)[i],i==0) for i in [0,21,42,63]]).to(a.device)
        fixed=torch.randn_like(batch.actions);times=torch.full((4,),.4,device=a.device)
        model.eval()
        with torch.no_grad():
            h,m,v=model.joint_outputs(batch.observation,batch.global_prompts,batch.actions,noise=fixed,time=times)
            rh,rm,rv=model.joint_outputs(batch.observation,batch.global_prompts,batch.actions,noise=fixed,time=times,detached=False)
            torch.testing.assert_close(v,rv,rtol=.01,atol=.002)
            torch.testing.assert_close(h,rh,rtol=.01,atol=.03)
            context=model.prepare_context(batch.observation,batch.global_prompts)
            pref=model.action_prefix(context,["wrong"]*4)
            alt=model.action_prefix(context,[""]*4)
            torch.testing.assert_close(pref.mask,alt.mask,rtol=0,atol=0)
            noisy=times[:,None,None]*fixed+(1-times[:,None,None])*batch.actions
            cached=model.base.denoise_step(context.state,pref.mask,pref.cache,noisy,times)
            torch.testing.assert_close(v,cached,rtol=.015,atol=.003)
            action_max_diff=float((v-cached).abs().max())
        del h,m,v,rh,rm,rv,cached,context,pref,alt
        model.train();model.zero_grad(set_to_none=True)
        model(batch,noise=fixed,time=times)["loss_subtask"].backward()
        b=model.base.paligemma_with_expert.paligemma
        ce={"S":grad_norm(model.subtask_parameters()),"B":grad_norm(model.backbone_parameters()),
            "A":grad_norm(model.action_parameters()),"vision":grad_norm(list(b.vision_tower.parameters())),
            "language":grad_norm(list(b.language_model.parameters())),
            "memory":grad_norm(list(model.decoder.memory_update.parameters()))}
        assert ce["S"]>0 and ce["B"]>0 and ce["vision"]>0 and ce["language"]>0 and ce["memory"]>0 and ce["A"]==0,ce
        model.zero_grad(set_to_none=True)
        model(batch,noise=fixed,time=times)["loss_action"].backward()
        act={"S":grad_norm(model.subtask_parameters()),"B":grad_norm(model.backbone_parameters()),"A":grad_norm(model.action_parameters())}
        assert act["S"]==0 and act["B"]==0 and act["A"]>0,act
        model.zero_grad(set_to_none=True)
        result.update(ce_gradients=ce,action_gradients=act,joint_native_max_abs_diff=action_max_diff)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)
if __name__=="__main__":main()
