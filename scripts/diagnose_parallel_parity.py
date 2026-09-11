"""Bounded same-weights BF16/FP32 split/joint/cache parity diagnostic; no updates."""
import argparse,json,os
from pathlib import Path
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('HF_HUB_OFFLINE','1')
import torch
from openpi.policies.parallel_subtask_policy import create_parallel_policy
from openpi.training.reach_arm_data import data_config
from openpi.training import config,data_loader
from openpi.training.subtask_batch import SubtaskTrainingDataset
from openpi.training.recurrent_sequence import collate_sequence

def difference(x,y):
    delta=(x.float()-y.float()).abs()
    return dict(max_abs=float(delta.max()),rmse=float(delta.square().mean().sqrt()),
                reference_rms=float(y.float().square().mean().sqrt()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.cuda.set_per_process_memory_fraction(.55,0)
    model=create_parallel_policy(a.checkpoint,device='cuda:0',allow_engineering=True).model.eval()
    dc=data_config(config.get_config('pi05_piper_stage1').model,split='val')
    raw=data_loader.create_torch_dataset(dc,50,model.base.config)
    batch=collate_sequence([(SubtaskTrainingDataset(raw,dc)[i],i==0) for i in [0,21,42,63]]).to('cuda:0')
    fixed=torch.randn_like(batch.actions);times=torch.full((4,),.4,device='cuda:0')
    result={}
    with torch.no_grad():
        for precision in ['mixed_bfloat16','float32']:
            if precision=='float32':
                torch.backends.cuda.matmul.allow_tf32=False
                torch.backends.cudnn.allow_tf32=False
                torch.set_float32_matmul_precision('highest')
                model.float();torch.cuda.empty_cache()
            h,m,v=model.joint_outputs(batch.observation,batch.global_prompts,batch.actions,noise=fixed,time=times)
            rh,rm,rv=model.joint_outputs(batch.observation,batch.global_prompts,batch.actions,noise=fixed,time=times,detached=False)
            context=model.prepare_context(batch.observation,batch.global_prompts)
            pref=model.action_prefix(context,None)
            noisy=times[:,None,None]*fixed+(1-times[:,None,None])*batch.actions
            cached=model.base.denoise_step(context.state,pref.mask,pref.cache,noisy,times)
            result[precision]=dict(split_joint_action=difference(v,rv),split_joint_prefix=difference(h,rh),
                                   split_joint_valid_prefix=difference(h[m.bool()],rh[m.bool()]),
                                   split_native_valid_prefix=difference(h[m.bool()],context.memory[m.bool()]),
                                   split_cache_action=difference(v,cached),original_joint_cache_action=difference(rv,cached))
            print(json.dumps({precision:result[precision]}),flush=True)
            del h,m,v,rh,rm,rv,context,pref,cached
    a.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
