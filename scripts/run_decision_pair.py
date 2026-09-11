"""Finite exit-event handoff: finish active run, then gate and train the two confirmed experiments."""
import argparse
import ctypes
import platform
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import statistics
import sys
import time
import traceback

import psutil

from gpu_reservation import run_concurrent
from run_reach_arm_candidate import checked,write_json

EXPERIMENTS=("decision_prefix","decision_grounded")
OLD=Path("logs/pi05_semantic_pair_20260910")


def open_pidfd(pid):
    if hasattr(os,"pidfd_open"):return os.pidfd_open(pid)
    # Some Python builds omit this wrapper although the Linux kernel supports it.
    libc=ctypes.CDLL(None,use_errno=True)
    function=getattr(libc,"pidfd_open",None)
    if function is not None:
        function.argtypes=[ctypes.c_int,ctypes.c_uint];function.restype=ctypes.c_int
        result=function(pid,0)
    else:
        if platform.system()!="Linux" or platform.machine()!="x86_64":
            raise RuntimeError("Unsupported pidfd ABI; refusing to fall back to progress polling")
        libc.syscall.restype=ctypes.c_long
        result=libc.syscall(ctypes.c_long(434),ctypes.c_int(pid),ctypes.c_uint(0))
    if result<0:
        error=ctypes.get_errno();raise OSError(error,os.strerror(error))
    return result


def identity(record):
    try:
        process=psutil.Process(record["pid"])
        if abs(process.create_time()-record["created"])>.1:raise RuntimeError("PID was reused")
        return process
    except psutil.NoSuchProcess:return None


def verify_sources(root):
    for name,digest in json.loads((root/"source_manifest.json").read_text()).items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise RuntimeError("Dispatched source changed: "+name)


def train_command(args,experiment,output,steps):
    return [sys.executable,"-m","torch.distributed.run","--standalone","--nproc-per-node=8",
        "scripts/train_decision_recurrent.py","--experiment",experiment,"--output",str(output),
        "--decoded-cache",str(args.cache),"--steps",str(steps),"--batch-size","32","--unroll","4",
        "--accumulation","1","--workers","4"]


def completed_train(output):
    final=output/"step_005000"
    metadata=json.loads((final/"metadata.json").read_text())
    assert metadata["completed_steps"]==5000 and metadata["counters"]==dict(subtask=5000,action=5000,backbone=5000)
    assert metadata["config"]["engineering_condition_fixture"] is False
    rows=[json.loads(s) for s in (output/"metrics.jsonl").read_text().splitlines()]
    assert rows[-1]["event"]=="complete" and rows[-1]["completed_steps"]==5000
    assert len({r["step"] for r in rows if r.get("event")=="train"})==5000
    return final


def pipeline(args):
    root=args.root
    def phase(name,**fields):
        verify_sources(root);write_json(root/"phase.json",dict(phase=name,time=time.time(),**fields))
    handoff=json.loads((root/"queue_handoff_requested.json").read_text())
    current=identity(handoff["current_train"])
    assert current is None or current.status()==psutil.STATUS_ZOMBIE,"Active training must finish before handoff"
    old=identity(handoff["old_scheduler"])
    if old:
        assert old.status()==psutil.STATUS_STOPPED
        assert "scripts/run_semantic_pair.py" in old.cmdline() and "--managed" in old.cmdline()
        children=old.children(recursive=True)
        assert all(p.status()==psutil.STATUS_ZOMBIE for p in children),"Do not signal a parent of active training"
        old.kill()  # Exact stopped scheduler only, after its training child exited.
    retired=dict(time=time.time(),intentional=True,cancelled="semantic_s_actionrank",old_scheduler=handoff["old_scheduler"],
                 reason="User replaced the unstarted experiment; active train already exited; no training rank signalled")
    write_json(root/"old_scheduler_retired.json",retired)
    write_json(OLD/"queue_replaced.json",retired)
    phase("finish_original_candidate")
    output=Path("checkpoints/pi05_piper_semantic/semantic_s_seed42_v1")
    final=completed_train(output)
    # Preserve the first experiment's native gate and final diagnostics.
    checked([sys.executable,"scripts/check_semantic_recurrent.py","--checkpoint",str(final),
             "--output",str(output/"policy_load_gate.json")],root/"original_final_native.log")
    candidate=dict(checkpoint=str(final),completed_steps=5000,native_gate_passed=True,primary=True,
                   experiment="semantic_s",robot_test_performed=False,time=time.time())
    write_json(OLD/"semantic_s_candidate_ready.json",candidate);write_json(output/"candidate.json",candidate)
    checked([sys.executable,"-m","torch.distributed.run","--standalone","--nproc-per-node=8",
             "scripts/evaluate_semantic_recurrent.py","--checkpoint",str(final),"--cache",str(args.cache),
             "--output",str(OLD/"semantic_s_final_diagnostics.json")],root/"original_final_diagnostics.log")
    write_json(OLD/"semantic_s_complete.json",dict(completed=True,checkpoint=str(final),time=time.time(),handoff_root=str(root)))
    assert json.loads((root/"cpu_gate.json").read_text())["passed"]
    inits=[]
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
        checked([sys.executable,"scripts/check_decision_recurrent.py","--checkpoint",str(output/"step_000008"),
                 "--allow-engineering","--output",str(root/(experiment+"_engineering_native.json"))],
                 root/(experiment+"_engineering_native.log"))
        updates=[json.loads(s) for s in (output/"metrics.jsonl").read_text().splitlines() if json.loads(s).get("event")=="train"]
        assert len(updates)==8 and all(r["rank_arm_pairs"]==16 and r["rank_object_pairs"]==16 for r in updates)
        if experiment=="decision_grounded":assert all(r["grounding_ce"]>0 for r in updates)
        seconds=[r["seconds"] for r in updates if r["step"] in (2,3,4,6,7,8)]
        write_json(root/(experiment+"_engineering_passed.json"),dict(passed=True,mean_update_seconds=statistics.mean(seconds),
             resume=restored,both_ranking_paths_at_maximum=True,grounding_enabled=experiment=="decision_grounded",time=time.time()))
        inits.append(json.loads((output/"initialization.json").read_text()))
    assert len({r["common_recurrent_decoder_sha256"] for r in inits})==1
    write_json(root/"engineering_passed.json",dict(passed=True,shared_common_fresh_S=True,time=time.time()))
    for number,experiment in enumerate(EXPERIMENTS):
        output=Path("checkpoints/pi05_piper_decision")/(experiment+"_seed42_v1")
        phase("formal_training",experiment=experiment,output=str(output),steps=5000)
        checked(train_command(args,experiment,output,5000)+["--wandb"],root/(experiment+"_formal.log"))
        final=completed_train(output)
        assert json.loads((output/"startup_verified.json").read_text())["passed"]
        initial=json.loads((output/"initialization.json").read_text())
        assert initial["complete_decoder_sha256"]==inits[number]["complete_decoder_sha256"]
        phase("final_native_gate",experiment=experiment)
        checked([sys.executable,"scripts/check_decision_recurrent.py","--checkpoint",str(final),
                 "--output",str(output/"policy_load_gate.json")],root/(experiment+"_final_native.log"))
        candidate=dict(checkpoint=str(final),completed_steps=5000,native_gate_passed=True,primary=True,
                       experiment=experiment,robot_test_performed=False,time=time.time())
        write_json(root/(experiment+"_candidate_ready.json"),candidate);write_json(output/"candidate.json",candidate)
        phase("final_diagnostics",experiment=experiment)
        checked([sys.executable,"-m","torch.distributed.run","--standalone","--nproc-per-node=8",
                 "scripts/evaluate_decision_recurrent.py","--checkpoint",str(final),"--cache",str(args.cache),
                 "--output",str(root/(experiment+"_final_diagnostics.json"))],root/(experiment+"_final_diagnostics.log"))
        write_json(root/(experiment+"_complete.json"),dict(completed=True,checkpoint=str(final),time=time.time()))
    phase("complete");write_json(root/"complete.json",dict(completed=True,experiments=EXPERIMENTS,time=time.time()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("logs/rtc_grounding_pair_20260910"))
    p.add_argument("--cache",type=Path,default=Path(".stage1_staging/piper_rgb224_reach_arm_v1"))
    p.add_argument("--managed",action="store_true")
    args=p.parse_args();args.root.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        if args.managed:pipeline(args);return
        with (args.root/"sequence.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            path=args.root/"source_manifest.json"
            if path.exists():raise FileExistsError("Refusing duplicate dispatch")
            verify_sources(OLD)
            assert json.loads((args.root/"cpu_gate.json").read_text())["passed"]
            files=sorted(Path("src/openpi").rglob("*.py"))+sorted(Path("scripts").glob("*decision*.py"))
            files+=sorted(Path("scripts").glob("*semantic*.py"))
            files += [Path(s) for s in ["scripts/run_reach_arm_candidate.py","scripts/gpu_reservation.py", "scripts/train_subtask_pytorch.py",
                "scripts/visualize_subtask_step.py","packages/openpi-client/src/openpi_client/image_tools.py"]]
            write_json(path,{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
            write_json(args.root/"launcher.process.json",dict(pid=os.getpid(),created=psutil.Process().create_time(),command=sys.argv))
            handoff=json.loads((args.root/"queue_handoff_requested.json").read_text())
            old=identity(handoff["old_scheduler"])
            assert old is not None and old.status()==psutil.STATUS_STOPPED
            current=identity(handoff["current_train"])
            write_json(args.root/"phase.json",dict(phase="waiting_current_train_exit",time=time.time(),current_train=handoff["current_train"],
                mechanism="pidfd exit event; no intermediate progress or GPU polling"))
            if current is not None and current.status()!=psutil.STATUS_ZOMBIE:
                descriptor=open_pidfd(current.pid)
                assert identity(handoff["current_train"]) is not None
                try:select.select([descriptor],[],[])
                finally:os.close(descriptor)
            verify_sources(args.root)
            code=run_concurrent(Path("logs/pi05_subtask_stage1/gpu_reservation"),args.root/"managed.log",
                 [sys.executable,__file__,"--root",str(args.root),"--cache",str(args.cache),"--managed"])
            write_json(args.root/"exit.json",dict(code=code,time=time.time()))
            raise SystemExit(code)
    except BaseException as error:
        if isinstance(error,SystemExit) and error.code==0:raise
        write_json(args.root/"failure.json",dict(error=repr(error),traceback=traceback.format_exc(),time=time.time()))
        raise


if __name__=="__main__":main()
