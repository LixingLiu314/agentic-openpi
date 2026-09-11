"""Real R1 decoder-delta loading, fixed-noise policy and optional exact comparison."""
import argparse
import dataclasses
import json
from pathlib import Path
import numpy as np
import safetensors.torch
import torch
from openpi.policies.subtask_transition_policy import create_transition_policy
from openpi.training import config as config_lib,data_loader


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--compare",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    if args.compare:
        a=safetensors.torch.load_file(args.checkpoint/"decoder.safetensors")
        b=safetensors.torch.load_file(args.compare/"decoder.safetensors")
        assert a.keys()==b.keys()
        assert all(torch.equal(a[k],b[k]) for k in a),"Resume decoder differs from uninterrupted updates"
    policy=create_transition_policy(args.checkpoint,device="cuda:0",allow_engineering=True,require_candidate=False)
    cfg=config_lib.get_config("pi05_piper_stage1")
    dc=dataclasses.replace(cfg.data.create(cfg.assets_dirs,policy.model.base.config),split="val")
    raw=data_loader.create_torch_dataset(dc,50,policy.model.base.config)
    frame=raw[0]
    cameras=["cam_high","cam_left_wrist","cam_right_wrist"]
    obs={"state":np.asarray(frame["observation.state"]),"prompt":frame["task"],
         "images":{name:np.asarray(frame["observation.images."+name]) for name in cameras}}
    original={"state":obs["state"].copy(),"images":{k:v.copy() for k,v in obs["images"].items()}}
    noise=np.random.default_rng(42).standard_normal((50,32),dtype=np.float32)
    a=policy.infer(obs,noise=noise);b=policy.infer(obs,noise=noise)
    assert a["actions"].shape==(50,14) and np.isfinite(a["actions"]).all()
    assert np.array_equal(a["actions"],b["actions"])
    assert np.array_equal(obs["state"],original["state"])
    assert all(np.array_equal(v,original["images"][k]) for k,v in obs["images"].items())
    try:policy.infer(dict(obs,subtask="grasp"),noise=noise)
    except ValueError:pass
    else:raise AssertionError("External subtask input was accepted")
    result={"exact_resume_equal":bool(args.compare),"native_shape":[50,14],"fixed_noise_equal":True,
            "input_unchanged":True,"external_subtask_rejected":True,"subtask":a["subtask"],"status":a["subtask_status"]}
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)

if __name__=="__main__":
    main()
