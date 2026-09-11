"""Finite single-candidate pipeline with registered GPU lease and final-step delivery."""
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

import psutil
from gpu_reservation import run_concurrent


def write_json(path,value):
    temporary=path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value,indent=2)+"\n");temporary.replace(path)


def checked(command,log):
    with log.open("x") as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        write_json(log.with_suffix(".process.json"),dict(pid=process.pid,created=psutil.Process(process.pid).create_time(),
                                                        command=command,started=time.time()))
        try:
            code=process.wait()
        finally:
            # Own child process group only; never enumerate or signal unrelated jobs.
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
        if code:raise RuntimeError(f"Child exited {code}: {log}")


def command(output,cache,steps):
    return [sys.executable,"-m","torch.distributed.run","--standalone","--nnodes=1","--nproc-per-node=8",
            "scripts/train_reach_arm_candidate.py","--output",str(output),"--decoded-cache",str(cache),
            "--steps",str(steps),"--batch-size","32","--unroll","4","--accumulation","1", "--workers","4"]


def verify_sources(root):
    for name,digest in json.loads((root/"source_manifest.json").read_text()).items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f"Source changed after dispatch: {name}")


def pipeline(args):
    root=args.root
    def phase(name,**fields):
        verify_sources(root)
        write_json(root/"phase.json",dict(phase=name,time=time.time(),**fields))
    if args.stage=="gates":
        assert json.loads((root/"cache_verified.json").read_text())["passed"]
        phase("engineering_save_resume")
        output=root/"engineering"
        cmd=command(output,args.cache,8)+["--engineering-smoke","--warmup","2","--checkpoint-every","4",
                                        "--eval-samples","8","--eval-draws","1"]
        checked(cmd+["--stop-after","4"],root/"gate_first.log")
        checked(cmd+["--resume"],root/"gate_resume.log")
        checkpoint=output/"step_000008"
        metadata=json.loads((checkpoint/"metadata.json").read_text())
        assert metadata["counters"]==dict(subtask=8,action=8,backbone=8)
        assert len(list(checkpoint.glob("training_rank_*.pt")))==8
        restored=json.loads((output/"resume_state_000004.json").read_text())
        assert restored["passed"] and len(restored["ranks"])==8
        phase("engineering_native_gradient_export_gate")
        checked([sys.executable,"scripts/check_reach_arm_candidate.py","--checkpoint",str(checkpoint),
                 "--allow-engineering","--output",str(root/"engineering_native.json")],root/"engineering_native.log")
        rows=[json.loads(s) for s in (output/"metrics.jsonl").read_text().splitlines()]
        seconds=[r["seconds"] for r in rows if r.get("event")=="train" and r["step"] in (2,3,4,6,7,8)]
        result=dict(passed=True,mean_update_seconds=statistics.mean(seconds),updates=8,global_batch=256,
                    resume=restored,native=json.loads((root/"engineering_native.json").read_text()),time=time.time())
        write_json(root/"engineering_passed.json",result)
        phase("gates_complete")
        return
    gate=json.loads((root/"engineering_passed.json").read_text())
    assert gate["passed"]
    output=Path("checkpoints/pi05_piper_reach_arm/limited_recurrent_seed42_v1")
    phase("formal_training",output=str(output),steps=5000)
    checked(command(output,args.cache,5000)+["--wandb"],root/"formal.log")
    final=output/"step_005000"
    metadata=json.loads((final/"metadata.json").read_text())
    assert metadata["completed_steps"]==5000 and metadata["counters"]==dict(subtask=5000,action=5000,backbone=5000)
    initial=json.loads((output/"initialization.json").read_text())
    reference=json.loads((root/"engineering/initialization.json").read_text())
    assert initial["complete_decoder_sha256"]==reference["complete_decoder_sha256"]
    phase("final_native_gate",checkpoint=str(final))
    checked([sys.executable,"scripts/check_reach_arm_candidate.py","--checkpoint",str(final),
             "--output",str(output/"policy_load_gate.json")],root/"final_native.log")
    candidate=dict(checkpoint=str(final),completed_steps=5000,native_gate_passed=True,primary=True,
                   status="loadable_experimental_candidate",robot_test_performed=False,time=time.time())
    write_json(output/"candidate.json",candidate)
    write_json(root/"candidate_ready.json",candidate)
    phase("final_role_and_memory_diagnostics",checkpoint=str(final))
    checked([sys.executable,"-m","torch.distributed.run","--standalone","--nproc-per-node=8",
             "scripts/evaluate_reach_arm_candidate.py","--checkpoint",str(final),"--cache",str(args.cache),
             "--output",str(root/"final_diagnostics.json")],root/"final_diagnostics.log")
    phase("complete",checkpoint=str(final))
    write_json(root/"complete.json",dict(completed=True,checkpoint=str(final),steps=5000,time=time.time()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("logs/pi05_reach_arm_20260910"))
    p.add_argument("--cache",type=Path,default=Path(".stage1_staging/piper_rgb224_reach_arm_v1"))
    p.add_argument("--stage",choices=["gates","formal"],required=True)
    p.add_argument("--managed",action="store_true")
    args=p.parse_args();args.root.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    if args.managed:
        try:pipeline(args)
        except BaseException as error:
            write_json(args.root/(args.stage+"_failure.json"),dict(error=repr(error),traceback=traceback.format_exc(),time=time.time()))
            raise
        return
    with (args.root/"sequence.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.stage=="gates":
            path=args.root/"source_manifest.json"
            if path.exists():raise FileExistsError("Refusing to rewrite dispatched source fingerprints")
            files=sorted(Path("src/openpi").rglob("*.py"))+sorted(Path("scripts").glob("*reach_arm*.py"))
            files += [Path("scripts/gpu_reservation.py"),Path("scripts/train_subtask_pytorch.py"),
                      Path("scripts/visualize_subtask_step.py"),Path("packages/openpi-client/src/openpi_client/image_tools.py")]
            write_json(path,{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
        verify_sources(args.root)
        code=run_concurrent(Path("logs/pi05_subtask_stage1/gpu_reservation"),args.root/(args.stage+"_managed.log"),
              [sys.executable,__file__,"--root",str(args.root),"--cache",str(args.cache),"--stage",args.stage,"--managed"])
        write_json(args.root/(args.stage+"_exit.json"),dict(code=code,time=time.time()))
        raise SystemExit(code)


if __name__=="__main__":main()
