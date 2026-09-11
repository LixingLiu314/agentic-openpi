"""Finite cached-input batch32 checks, followed by the user-authorized full-B restart."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil
import gpu_reservation


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def checked(command, log):
    with log.open("x") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(log.with_suffix(".process.json"), {"pid":process.pid,
                    "created":psutil.Process(process.pid).create_time(),"command":command,"started":time.time()})
        try:
            code = process.wait()
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        if code:
            raise RuntimeError(f"Command exited {code}: {log}")


def train_command(output, cache):
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=8",
            "scripts/train_backbone_gradient_largebatch.py", "--mode", "full", "--output", str(output),
            "--decoded-cache", str(cache), "--batch-size", "32", "--accumulation", "1", "--workers", "4",
            "--memory-fraction", "0.9", "--eval-batch-size", "2"]


def pipeline(root, cache):
    evidence = json.loads((root / "data_equivalence.json").read_text())
    if not evidence["passed"]:
        raise ValueError("Decoded input equivalence has not passed")
    gate = root / "full_b32_cache_gate"
    common = train_command(gate, cache) + ["--steps", "8", "--warmup", "1", "--checkpoint-every", "8",
             "--eval-samples", "8", "--eval-draws", "1", "--engineering-smoke"]
    checked(common + ["--stop-after", "4"], root / "cached_gate_initial.log")
    checked(common + ["--resume"], root / "cached_gate_resume.log")
    raw_meta = json.loads((root / "full_b32_a1_gate/step_000008/metadata.json").read_text())
    cache_meta = json.loads((gate / "step_000008/metadata.json").read_text())
    exact = raw_meta["weights_sha256"] == cache_meta["weights_sha256"]
    write_json(root / "raw_cached_training_parity.json", {"passed":exact,
               "original_weights_sha256":raw_meta["weights_sha256"], "cached_resumed_weights_sha256":cache_meta["weights_sha256"],
               "scope":"same batch32/global256, 8 real updates; original video path vs cached pinned path with save/resume at4"})
    if not exact:
        raise ValueError("Cached resumed training differs from original video-path control; inspect before research launch")
    checked([sys.executable,"scripts/check_backbone_gradient_checkpoint.py","--checkpoint",str(gate / "step_000008"),
             "--output",str(root / "cached_native_policy_gate.json"),"--allow-engineering"], root / "cached_policy.log")
    write_json(root / "engineering_passed.json", {"passed":True,"world_size":8,"microbatch":32,"accumulation":1,
               "global_batch":256,"data_equivalence":True,"raw_cached_resumed_weights_exact":True,"native_policy":True})
    output = Path("checkpoints/pi05_piper_backbone_grad/full_seed42_b32_cache_v1")
    write_json(root / "formal_dispatch.json", {"time":time.time(),"output":str(output),"mode":"full",
               "initialization":"original M3 step3500; fresh optimizers; old limited completed result retained"})
    checked(train_command(output, cache) + ["--wandb"], root / "full_formal.log")
    selected = output / json.loads((output / "best.json").read_text())["checkpoint"]
    checked([sys.executable,"scripts/check_backbone_gradient_checkpoint.py","--checkpoint",str(selected),
             "--output",str(output / "policy_load_gate.json")], root / "full_policy.log")
    write_json(output / "candidate.json", {"checkpoint":selected.name,"mode":"full","policy_load_gate_passed":True,
               "status":"loadable_experimental_candidate","robot_test_performed":False})
    write_json(root / "complete.json", {"completed":True,"output":str(output),"candidate":str(selected),
               "steps":5000,"global_batch":256,"time":time.time()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--managed", action="store_true")
    args = parser.parse_args()
    if args.managed:
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        try:
            pipeline(args.root, args.cache)
        except BaseException as error:
            write_json(args.root / "retry_failure.json", {"error":str(error),"type":type(error).__name__,"time":time.time()})
            raise
    else:
        command = [sys.executable,__file__,"--root",str(args.root),"--cache",str(args.cache),"--managed"]
        code = gpu_reservation.run_concurrent(Path("logs/pi05_subtask_stage1/gpu_reservation"),
                                               args.root / "retry_managed.log", command)
        write_json(args.root / "retry_supervisor_exit.json", {"code":code,"time":time.time()})
        raise SystemExit(code)


if __name__ == "__main__":
    main()
