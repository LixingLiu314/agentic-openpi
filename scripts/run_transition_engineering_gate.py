"""Bounded eight-rank uninterrupted versus resume gate, executed by GPU supervisor."""
import json
from pathlib import Path
import subprocess
import sys

def main():
    root=Path(sys.argv[1])
    root.mkdir(parents=True,exist_ok=False)
    common=[sys.executable,"-m","torch.distributed.run","--standalone","--nnodes=1","--nproc-per-node=8",
            "scripts/train_subtask_transition.py","--steps","2","--eval-every","2","--engineering-smoke",
            "--val-limit","2","--workers","0","--warmup","2"]
    jobs=[("uninterrupted",common+["--output",str(root/"full")]),
          ("before_resume",common+["--output",str(root/"resume"),"--stop-after","1"]),
          ("after_resume",common+["--output",str(root/"resume"),"--resume"]),
          ("policy",[sys.executable,"scripts/check_transition_checkpoint.py","--checkpoint",str(root/"resume/step_000002"),
                     "--compare",str(root/"full/step_000002"),"--output",str(root/"policy_gate.json")])]
    for name,cmd in jobs:
        print(json.dumps({"gate":name,"command":cmd}),flush=True)
        with (root/(name+".log")).open("x") as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            (root/"failure.json").write_text(json.dumps({"gate":name,"exit_code":result.returncode})+"\n")
            raise SystemExit(result.returncode)
    (root/"passed.json").write_text(json.dumps({"passed":True,"world_size":8,"seed":42})+"\n")
    print("ENGINEERING_GATE_PASSED",flush=True)

if __name__=="__main__":
    main()
