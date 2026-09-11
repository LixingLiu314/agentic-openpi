"""Finite formal run plus loading gate; no GPU telemetry and no periodic job watcher."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

def write_json(path,value):
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(value,indent=2)+"\n")
    tmp.replace(path)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    command=[sys.executable,"-m","torch.distributed.run","--standalone","--nnodes=1","--nproc-per-node=8",
             "scripts/train_subtask_transition.py","--output",str(args.output),"--steps","1000","--seed","42",
             "--batch-size","2","--accumulation","2","--lr","1e-5","--warmup","50",
             "--eval-every","250","--eval-batch-size","4","--workers","2","--wandb"]
    print(json.dumps({"event":"formal_launch","command":command}),flush=True)
    code=subprocess.run(command).returncode
    if code:
        if args.output.exists():write_json(args.output/"pipeline_result.json",{"status":"training_failed","exit_code":code})
        raise SystemExit(code)
    candidate=json.loads((args.output/"candidate.json").read_text())
    target=args.output/candidate["checkpoint"]
    code=subprocess.run([sys.executable,"scripts/check_transition_checkpoint.py","--checkpoint",str(target),
                         "--output",str(args.output/"policy_load_gate.json")]).returncode
    candidate["policy_load_gate_passed"]=code==0
    if code:
        candidate["proxy_gates_passed"]=False
        candidate["status"]="policy_loading_gate_failed"
    write_json(args.output/"candidate.json",candidate)
    write_json(args.output/"pipeline_result.json",{"status":"complete" if not code else "loading_gate_failed",
               "completed_steps":1000,"seed":42,"candidate":candidate,"finished":time.time()})
    (args.output/"RESULT.md").write_text("# R1 seed42 result\n\n"+json.dumps(candidate,indent=2)+
       "\n\nReports: validation_*.json, selected_action_diagnostic.json, policy_load_gate.json.\n"
       "\nPhysical readiness remains unreviewed; no robot actions were run.\n")
    raise SystemExit(code)

if __name__=="__main__":
    main()
