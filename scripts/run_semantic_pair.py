"""Two finite fresh-initialized semantic experiments, each gated before training."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback

from gpu_reservation import run_concurrent
from run_reach_arm_candidate import checked, write_json

EXPERIMENTS=("semantic_s","semantic_s_actionrank")


def train_command(args, experiment, output, steps):
    return [sys.executable,"-m","torch.distributed.run","--standalone","--nproc-per-node=8",
            "scripts/train_semantic_recurrent.py","--experiment",experiment,"--output",str(output),
            "--decoded-cache",str(args.cache),"--steps",str(steps),"--batch-size","32","--unroll","4",
            "--accumulation","1","--workers","4"]


def verify_sources(root):
    for name,digest in json.loads((root/"source_manifest.json").read_text()).items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError("Dispatched source changed: "+name)


def pipeline(args):
    root=args.root
    def phase(name,**fields):
        verify_sources(root)
        write_json(root/"phase.json",dict(phase=name,time=time.time(),**fields))
    assert json.loads(Path("logs/pi05_reach_arm_20260910/cache_verified.json").read_text())["passed"]
    for experiment in EXPERIMENTS:
        phase("engineering_save_resume",experiment=experiment)
        output=root/(experiment+"_engineering")
        cmd=train_command(args,experiment,output,8)+["--engineering-smoke","--warmup","2",
                "--checkpoint-every","4","--eval-samples","8","--eval-draws","1"]
        checked(cmd+["--stop-after","4"],root/(experiment+"_gate_first.log"))
        checked(cmd+["--resume"],root/(experiment+"_gate_resume.log"))
        restored=json.loads((output/"resume_state_000004.json").read_text())
        assert restored["passed"] and len(restored["ranks"])==8
        phase("engineering_native_gradient_gate",experiment=experiment)
        checked([sys.executable,"scripts/check_semantic_recurrent.py","--checkpoint",str(output/"step_000008"),
                "--allow-engineering","--output",str(root/(experiment+"_engineering_native.json"))],
                root/(experiment+"_engineering_native.log"))
        rows=[json.loads(s) for s in (output/"metrics.jsonl").read_text().splitlines()]
        updates=[r for r in rows if r.get("event")=="train"]
        if experiment=="semantic_s_actionrank":
            assert all(r["rank_pairs"]==32 for r in updates)
        seconds=[r["seconds"] for r in updates if r["step"] in (2,3,4,6,7,8)]
        write_json(root/(experiment+"_engineering_passed.json"),dict(passed=True,mean_update_seconds=statistics.mean(seconds),
                   resume=restored,maximum_global_rank_pairs=max(r["rank_pairs"] for r in updates),time=time.time()))
    inits=[json.loads((root/(s+"_engineering")/"initialization.json").read_text()) for s in EXPERIMENTS]
    assert len({r["complete_decoder_sha256"] for r in inits})==1
    write_json(root/"engineering_passed.json",dict(passed=True,shared_fresh_S=True,time=time.time()))
    for experiment in EXPERIMENTS:
        output=Path("checkpoints/pi05_piper_semantic")/(experiment+"_seed42_v1")
        phase("formal_training",experiment=experiment,output=str(output),steps=5000)
        checked(train_command(args,experiment,output,5000)+["--wandb"],root/(experiment+"_formal.log"))
        final=output/"step_005000"
        metadata=json.loads((final/"metadata.json").read_text())
        assert metadata["completed_steps"]==5000 and metadata["counters"]==dict(subtask=5000,action=5000,backbone=5000)
        assert metadata["config"]["engineering_condition_fixture"] is False
        initial=json.loads((output/"initialization.json").read_text())
        assert initial["complete_decoder_sha256"]==inits[0]["complete_decoder_sha256"]
        phase("final_native_gate",experiment=experiment)
        checked([sys.executable,"scripts/check_semantic_recurrent.py","--checkpoint",str(final),
                 "--output",str(output/"policy_load_gate.json")],root/(experiment+"_final_native.log"))
        candidate=dict(checkpoint=str(final),completed_steps=5000,native_gate_passed=True,primary=True,
                       experiment=experiment,robot_test_performed=False,time=time.time())
        write_json(root/(experiment+"_candidate_ready.json"),candidate)
        write_json(output/"candidate.json",candidate)
        phase("final_diagnostics",experiment=experiment)
        checked([sys.executable,"-m","torch.distributed.run","--standalone","--nproc-per-node=8",
                "scripts/evaluate_semantic_recurrent.py","--checkpoint",str(final),"--cache",str(args.cache),
                "--output",str(root/(experiment+"_final_diagnostics.json"))],root/(experiment+"_final_diagnostics.log"))
        write_json(root/(experiment+"_complete.json"),dict(completed=True,checkpoint=str(final),time=time.time()))
    phase("complete")
    write_json(root/"complete.json",dict(completed=True,experiments=EXPERIMENTS,time=time.time()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("logs/pi05_semantic_pair_20260910"))
    p.add_argument("--cache",type=Path,default=Path(".stage1_staging/piper_rgb224_reach_arm_v1"))
    p.add_argument("--managed",action="store_true")
    args=p.parse_args();args.root.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    if args.managed:
        try:pipeline(args)
        except BaseException as error:
            write_json(args.root/"failure.json",dict(error=repr(error),traceback=traceback.format_exc(),time=time.time()))
            raise
        return
    with (args.root/"sequence.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=args.root/"source_manifest.json"
        if path.exists():raise FileExistsError("Refusing duplicate dispatch")
        files=sorted(Path("src/openpi").rglob("*.py"))+sorted(Path("scripts").glob("*semantic*.py"))
        files += [Path("scripts/run_reach_arm_candidate.py"),Path("scripts/gpu_reservation.py"),
                  Path("scripts/train_subtask_pytorch.py"),Path("scripts/visualize_subtask_step.py"),
                  Path("packages/openpi-client/src/openpi_client/image_tools.py")]
        write_json(path,{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
        code=run_concurrent(Path("logs/pi05_subtask_stage1/gpu_reservation"),args.root/"managed.log",
              [sys.executable,__file__,"--root",str(args.root),"--cache",str(args.cache),"--managed"])
        write_json(args.root/"exit.json",dict(code=code,time=time.time()))
        raise SystemExit(code)


if __name__=="__main__":main()
