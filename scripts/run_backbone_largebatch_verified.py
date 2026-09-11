"""Release the batch32 restart after exact input and repeated-control diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
import time

import gpu_reservation
from run_backbone_largebatch_retry import checked, train_command, write_json


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def pipeline(root, cache):
    for name in ("data_equivalence.json", "all_engineering_inputs_exact.json", "original_sources_preserved.json"):
        if not json.loads((root/name).read_text())["passed"]:
            raise ValueError(f"Required evidence failed: {name}")
    if json.loads((root/"raw_repeat_exit.json").read_text())["exit_code"] != 0:
        raise ValueError("Original-loader repeated control failed")
    variants = {name:root/folder for name,folder in (
        ("original","full_b32_a1_gate"),("original_repeat","full_b32_raw_repeat_gate"),("cached_resumed","full_b32_cache_gate"))}
    evidence = {}
    for name,folder in variants.items():
        events=rows(folder/"metrics.jsonl")
        train=[r for r in events if r.get("event")=="train"]
        if [r["step"] for r in train] != list(range(1,9)):
            raise ValueError(f"Incomplete control: {name}")
        if not any(r.get("event")=="complete" and r.get("completed_steps")==8 for r in events):
            raise ValueError(f"Unfinished control: {name}")
        metadata=json.loads((folder/"step_000008/metadata.json").read_text())
        evidence[name]={"weight_sha256":metadata["weights_sha256"],"first_loss_subtask":train[0]["loss_subtask"],
                        "first_loss_action":train[0]["loss_action"],"first_s_grad_norm":train[0]["grad_norms"]["subtask"],
                        "last_loss_subtask":train[-1]["loss_subtask"],"last_loss_action":train[-1]["loss_action"]}
    baseline=evidence["original"]
    for current in evidence.values():
        for key in ("first_loss_subtask","first_loss_action"):
            if current[key] != baseline[key]:
                raise ValueError("Same-initialization loss differs despite identical scheduled input tensors")
        if abs(current["first_s_grad_norm"]-baseline["first_s_grad_norm"]) > 1e-6:
            raise ValueError("First-gradient discrepancy requires further diagnosis")
    if evidence["original_repeat"]["weight_sha256"] == baseline["weight_sha256"]:
        raise ValueError("Repeated original run is bit-exact; investigate cached trajectory before proceeding")
    native=root/"cached_native_policy_gate.json"
    checked([sys.executable,"scripts/check_backbone_gradient_checkpoint.py","--checkpoint",str(variants["cached_resumed"]/"step_000008"),
             "--output",str(native),"--allow-engineering"],root/"verified_cached_policy.log")
    write_json(root/"engineering_passed.json",{
        "passed":True,"world_size":8,"microbatch":32,"accumulation":1,"global_batch":256,
        "exact_individual_samples":547,"all_scheduled_training_inputs_exact":2048,
        "same_config_optimizer_rng_resume_completed":True,"native_cuda_policy_passed":True,
        "cross_independent_run_weights_bit_exact":False,"original_loader_repeat_also_not_bit_exact":True,
        "diagnosis":"Original and cached initial losses are exact; tiny first-gradient differences also occur in repeated unchanged original GPU training. Input optimization is byte-exact; independent GPU trajectory bitwise reproducibility is not established and is not claimed.",
        "controls":evidence})
    output=Path("checkpoints/pi05_piper_backbone_grad/full_seed42_b32_cache_v1")
    write_json(root/"formal_dispatch.json",{"time":time.time(),"output":str(output),"mode":"full",
              "initialization":"original M3 step3500; fresh optimizers; old completed limited model retained"})
    checked(train_command(output,cache)+["--wandb"],root/"full_formal.log")
    selected=output/json.loads((output/"best.json").read_text())["checkpoint"]
    checked([sys.executable,"scripts/check_backbone_gradient_checkpoint.py","--checkpoint",str(selected),
             "--output",str(output/"policy_load_gate.json")],root/"full_policy.log")
    write_json(output/"candidate.json",{"checkpoint":selected.name,"mode":"full","policy_load_gate_passed":True,
               "status":"loadable_experimental_candidate","robot_test_performed":False})
    write_json(root/"complete.json",{"completed":True,"output":str(output),"candidate":str(selected),
               "steps":5000,"global_batch":256,"time":time.time()})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--managed",action="store_true")
    args=parser.parse_args()
    if args.managed:
        signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        try:
            pipeline(args.root,args.cache)
        except BaseException as error:
            write_json(args.root/"verified_retry_failure.json",{"error":str(error),"type":type(error).__name__,"time":time.time()})
            raise
    else:
        command=[sys.executable,__file__,"--root",str(args.root),"--cache",str(args.cache),"--managed"]
        code=gpu_reservation.run_concurrent(Path("logs/pi05_subtask_stage1/gpu_reservation"),args.root/"verified_retry_managed.log",command)
        write_json(args.root/"verified_retry_supervisor_exit.json",{"code":code,"time":time.time()})
        raise SystemExit(code)


if __name__=="__main__":
    main()
