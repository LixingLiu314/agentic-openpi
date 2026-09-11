"""Let limited finish unchanged, then hand off to the observed full-B trainer."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import psutil

from gpu_reservation import run_concurrent
from run_official_gradient_trio import checked, training_command, verify_sources, write_json


def identified(record):
    process = psutil.Process(record["pid"])
    if abs(process.create_time() - record["created"]) > .1:
        raise RuntimeError("Process identity changed")
    return process


def zombie_exit(process):
    if process.status() != psutil.STATUS_ZOMBIE:
        return None
    # The paused parent cannot reap the child. Field52 preserves its wait status.
    fields = Path(f"/proc/{process.pid}/stat").read_text().rpartition(")")[2].split()
    return os.waitstatus_to_exitcode(int(fields[49]))


def completed_limited(output):
    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    if not any(x.get("event") == "complete" and x.get("completed_steps") == 5000 for x in rows):
        raise RuntimeError("Limited did not finish all5000 updates")
    latest = json.loads((output / "latest.json").read_text())
    assert latest["completed_steps"] == 5000
    metadata = json.loads((output / latest["checkpoint"] / "metadata.json").read_text())
    assert metadata["config"]["mode"] == "limited"
    assert metadata["config"]["initialization"] == "official_pi05_base"
    assert metadata["config"]["inherited_training_updates"] == 0
    initialization = json.loads((output / "initialization.json").read_text())
    assert initialization["decoder_sha256"] == "bab5a0bce442eb7e5759a15deea98c1ff0e75b326941e9abff2e3eacc7eb53ee"
    return output / json.loads((output / "best.json").read_text())["checkpoint"]


def stop_finished_controller(armed):
    limited = identified(armed["limited"])
    if zombie_exit(limited) != 0:
        raise RuntimeError("Refusing handoff before limited exits successfully")
    managed = identified(armed["old_managed"])
    assert managed.status() == psutil.STATUS_STOPPED
    managed.send_signal(signal.SIGTERM)
    managed.send_signal(signal.SIGCONT)
    try:
        identified(armed["old_owner"]).wait(timeout=45)
    except psutil.NoSuchProcess:
        pass


def run_full(args, output):
    command = training_command("full", output, args.cache)
    index = command.index("scripts/train_official_backbone_gradient.py")
    command[index] = "scripts/train_official_backbone_gradient_observed.py"
    command += ["--wandb"]
    with (args.root / "full_formal.log").open("x") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(args.root / "full_formal.process.json", {"pid":process.pid,
                   "created":psutil.Process(process.pid).create_time(), "command":command, "started":time.time()})
        try:
            # Finite startup verification; it never monitors later training updates.
            checked([sys.executable, "scripts/verify_official_full_wandb_startup.py", "--output", str(output),
                     "--identity", str(args.root / "full_formal.process.json"),
                     "--report", str(args.root / "full_wandb_startup.json")], args.root / "full_wandb_startup.log")
            code = process.wait()
            if code:
                raise RuntimeError(f"Full training exited {code}")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def managed_pipeline(args):
    verify_sources(args.root)
    armed = json.loads((args.root / "armed.json").read_text())
    limited_output = Path("checkpoints/pi05_piper_official_grad/limited_seed42_v1")
    selected = completed_limited(limited_output)
    write_json(args.old_root / "intentional_handoff.json", {"new_root":str(args.root),
               "reason":"User requests full-B logging fix; limited training completed unchanged", "time":time.time()})
    stop_finished_controller(armed)
    write_json(args.root / "takeover_complete.json", {"limited_exit_code":0, "time":time.time()})
    write_json(args.root / "phase.json", {"phase":"final_native_gate", "mode":"limited", "time":time.time()})
    checked([sys.executable, "scripts/check_official_gradient_checkpoint.py", "--checkpoint", str(selected),
             "--output", str(limited_output / "policy_load_gate.json")], args.root / "limited_final_native.log")
    write_json(limited_output / "candidate.json", {"checkpoint":selected.name, "mode":"limited",
               "initialization":"official_pi05_base", "policy_load_gate_passed":True,
               "status":"loadable_experimental_candidate", "robot_test_performed":False})
    write_json(args.root / "limited_complete.json", {"completed":True,"output":str(limited_output),
               "candidate":str(selected), "steps":5000,"global_batch":256,"time":time.time()})
    verify_sources(args.root)
    full_output = Path("checkpoints/pi05_piper_official_grad/full_seed42_v1")
    assert not full_output.exists(), "Never overwrite or duplicate a full experiment"
    write_json(args.root / "phase.json", {"phase":"formal_training", "mode":"full", "output":str(full_output),
               "observability_version":2,"time":time.time()})
    write_json(args.root / "full_formal_dispatch.json", {"time":time.time(),"mode":"full","output":str(full_output),
               "initialization":"official_pi05_base","inherited_training_updates":0,"steps":5000,"observability_version":2})
    run_full(args, full_output)
    selected = full_output / json.loads((full_output / "best.json").read_text())["checkpoint"]
    write_json(args.root / "phase.json", {"phase":"final_native_gate","mode":"full","time":time.time()})
    checked([sys.executable, "scripts/check_official_gradient_checkpoint.py", "--checkpoint", str(selected),
             "--output", str(full_output / "policy_load_gate.json")], args.root / "full_final_native.log")
    write_json(full_output / "candidate.json", {"checkpoint":selected.name,"mode":"full",
               "initialization":"official_pi05_base","policy_load_gate_passed":True,
               "status":"loadable_experimental_candidate","robot_test_performed":False})
    write_json(args.root / "full_complete.json", {"completed":True,"output":str(full_output),"candidate":str(selected),
               "steps":5000,"global_batch":256,"time":time.time()})
    write_json(args.root / "phase.json", {"phase":"complete","time":time.time()})
    write_json(args.root / "complete.json", {"completed":True,"modes":["frozen","limited","full"],"time":time.time()})


def arm_and_wait(args):
    verify_sources(args.root)
    verify_sources(args.old_root)
    phase = json.loads((args.old_root / "phase.json").read_text())
    assert phase["phase"] == "formal_training" and phase["mode"] == "limited"
    assert not (args.old_root / "full_formal_dispatch.json").exists()
    owner = json.loads((args.old_root / "sequence.process.json").read_text())
    parent = identified(owner)
    children = [p for p in parent.children() if "--managed" in p.cmdline()]
    assert len(children) == 1
    managed = children[0]
    limited_record = json.loads((args.old_root / "limited_formal.process.json").read_text())
    limited = identified(limited_record)
    assert limited.ppid() == managed.pid
    assert limited.status() not in {psutil.STATUS_STOPPED, psutil.STATUS_ZOMBIE}
    armed = {"old_owner":owner,"old_managed":{"pid":managed.pid,"created":managed.create_time()},
             "limited":limited_record,"time":time.time(),"old_root":str(args.old_root)}
    write_json(args.root / "handoff_plan.json", armed)
    managed.send_signal(signal.SIGSTOP)  # Pause only the scheduler, never its training child.
    try:
        deadline = time.monotonic() + 5
        while managed.status() != psutil.STATUS_STOPPED:
            if time.monotonic() > deadline:raise TimeoutError("Scheduler did not pause")
            time.sleep(.05)
        assert identified(limited_record).status() != psutil.STATUS_STOPPED
        write_json(args.root / "frozen_complete.json", json.loads((args.old_root / "frozen_complete.json").read_text()))
        write_json(args.root / "limited_formal.process.json", limited_record)
        write_json(args.root / "phase.json", {"phase":"formal_training","mode":"limited",
                   "handoff_pending":True,"next_trainer":"train_official_backbone_gradient_observed.py","time":time.time()})
        write_json(args.root / "armed.json", {**armed,"limited_child_continues":True})
        deadline = time.monotonic() + 24*3600
        while True:
            process = identified(limited_record)
            code = zombie_exit(process)
            if code is not None:
                if code:raise RuntimeError(f"Limited training exited {code}")
                break
            if time.monotonic() > deadline:raise TimeoutError("Limited completion wait exceeded24h")
            # Process-exit supervision only; no repeated training-step/checkpoint reads.
            time.sleep(10)
        command = [sys.executable,str(Path(__file__).resolve()),"--root",str(args.root),
                   "--old-root",str(args.old_root),"--cache",str(args.cache),"--managed"]
        code = run_concurrent(Path("logs/pi05_subtask_stage1/gpu_reservation"),args.root / "managed.log",command)
        write_json(args.root / "supervisor_exit.json", {"code":code,"time":time.time()})
        if code:raise RuntimeError(f"Observed full continuation exited {code}")
    finally:
        # Restore the old scheduler if setup fails before ownership is transferred.
        try:
            process = identified(armed["old_managed"])
            if process.status() == psutil.STATUS_STOPPED:process.send_signal(signal.SIGCONT)
        except psutil.NoSuchProcess:
            pass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--old-root",type=Path,required=True)
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--managed",action="store_true")
    args=parser.parse_args()
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        if args.managed:managed_pipeline(args)
        else:
            with (args.root / "queue.lock").open("a") as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                arm_and_wait(args)
    except BaseException as error:
        write_json(args.root / "failure.json",{"error":repr(error),"traceback":traceback.format_exc(),"time":time.time()})
        raise


if __name__ == "__main__":
    main()
