"""Actual checkpoint RTC inference without robot publishers or a GPU."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

# Allow this gate to run from an isolated deployment staging directory.
import openpi.models_pytorch
import openpi.policies
ROOT=Path(__file__).resolve().parents[1]
openpi.models_pytorch.__path__ = [str(ROOT/"src/openpi/models_pytorch"), *openpi.models_pytorch.__path__]
openpi.policies.__path__ = [str(ROOT/"src/openpi/policies"), *openpi.policies.__path__]
from openpi.policies.rtc_policy import RTCPolicy, PROTOCOL, native_overlap_to_model
from serve_subtask_policy import load_policy


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42)
    policy=load_policy(args.checkpoint,device="cpu")
    policy.model.float()
    noise=np.random.default_rng(42).standard_normal((50,32),dtype=np.float32)
    observation={"state":np.zeros(14,np.float32),"prompt":"Put the eggplant into the box",
                 "images":{k:np.zeros((480,640,3),np.uint8) for k in ("cam_high","cam_left_wrist","cam_right_wrist")}}
    wrapped=RTCPolicy(policy);one=wrapped.new_session();two=wrapped.new_session()
    first=dict(protocol=PROTOCOL,query_id=1,previous_query_id=None,consumed_steps=0,delay_steps=0)
    versions=[p._version for p in policy.model.parameters()]
    normal=policy.new_session().infer(observation,noise=noise)
    initial=one.infer(dict(observation,rtc=first),noise=noise)
    np.testing.assert_array_equal(normal["actions"],initial["actions"])
    again=two.infer(dict(observation,rtc=first),noise=noise)
    np.testing.assert_array_equal(initial["actions"],again["actions"])
    state=observation["state"].copy();state[[0,7]]=[.01,-.01]
    current=dict(observation,state=state)
    normalized=policy.input_transform(current)["state"]
    previous=initial["actions"][25:].copy();previous[:,[6,13]]=np.clip(previous[:,[6,13]],0,.09)
    converted=native_overlap_to_model(policy,normalized,previous)
    padded=np.zeros((50,32),np.float32);padded[:25,:14]=converted
    restored=policy.output_transform({"state":normalized.copy(),"actions":padded})["actions"][:25]
    np.testing.assert_allclose(restored,previous,atol=2e-6,rtol=2e-6)
    second=dict(protocol=PROTOCOL,query_id=2,previous_query_id=1,consumed_steps=25,delay_steps=8)
    started=time.monotonic();guided=one.infer(dict(current,rtc=second),noise=noise);seconds=time.monotonic()-started
    assert guided["actions"].shape==(50,14) and np.isfinite(guided["actions"]).all()
    assert guided["rtc"]["guidance"]=="soft_mask_endpoint_vjp"
    assert two._query_id==1 and one._query_id==2
    np.testing.assert_array_equal(two._previous,np.clip(again["actions"],
        np.array([-np.inf]*6+[0]+[-np.inf]*6+[0]),np.array([np.inf]*6+[.09]+[np.inf]*6+[.09])).astype(np.float32))
    assert versions==[p._version for p in policy.model.parameters()]
    assert all(p.grad is None for p in policy.model.parameters())
    for bad in [dict(second,query_id=4),dict(second,query_id=3,previous_query_id=1),
                dict(second,query_id=3,previous_query_id=2,consumed_steps=50)]:
        try:one.infer(dict(current,rtc=bad),noise=noise)
        except ValueError:pass
        else:raise AssertionError("Invalid RTC reference accepted")
    assert one._query_id==2
    one.reset();assert one._previous is None and one._query_id==0
    assert np.array_equal(observation["state"],np.zeros(14))
    out=dict(passed=True,checkpoint=str(args.checkpoint),actual_cpu_rtc=True,
             guided_seconds=seconds,unguided_exact_original=True,native_roundtrip=True,
             session_isolation=True,weights_unchanged=True,parameter_grads_absent=True,
             stale_requests_rejected=True,action_shape=[50,14],robot_motion=False,
             scope="Synthetic observations; not physical success or GPU latency")
    args.output.write_text(json.dumps(out,indent=2));print(json.dumps(out))


if __name__=="__main__":main()
