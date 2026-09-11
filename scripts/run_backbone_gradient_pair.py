"""Finite eight-GPU gates and two research runs, under the reservation supervisor."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import gpu_reservation


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def checked(command, log):
    print(json.dumps({"event":"launch", "command":command, "log":str(log)}), flush=True)
    with log.open("x") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Exit {result.returncode}: {log}")


def training_command(mode, output):
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=8",
            "scripts/train_backbone_gradient.py", "--mode", mode, "--output", str(output)]


def run_formal(mode, output, root):
    """Validate the first saved model on CPU while the finite GPU run continues."""
    command = training_command(mode, output) + ["--wandb"]
    print(json.dumps({"event":"formal_launch", "mode":mode, "command":command}), flush=True)
    gate_process, gate_stream = None, None
    gate_report = output / "first_checkpoint_policy_gate.json"
    with (root / f"{mode}_formal.log").open("x") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(root / f"{mode}_formal.process.json", {"pid":process.pid, "command":command, "started":time.time()})
        try:
            while process.poll() is None:
                first = output / "step_000500"
                if gate_process is None and (first / "metadata.json").exists():
                    gate_command = [sys.executable, "scripts/check_backbone_gradient_checkpoint.py", "--checkpoint", str(first),
                                    "--output", str(gate_report), "--device", "cpu"]
                    gate_stream = (root / f"{mode}_first_checkpoint_policy.log").open("x")
                    gate_process = subprocess.Popen(gate_command, stdout=gate_stream, stderr=subprocess.STDOUT,
                                                     env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
                if gate_process is not None and gate_process.poll() is not None:
                    if gate_process.returncode:
                        raise RuntimeError(f"First checkpoint policy gate failed: {gate_report}")
                    candidate = output / "first_deployable.json"
                    if not candidate.exists():
                        write_json(candidate, {"checkpoint":"step_000500", "mode":mode,
                                               "status":"loadable_experimental_candidate", "policy_load_gate_passed":True,
                                               "robot_test_performed":False, "time":time.time()})
                time.sleep(10)
            if process.returncode:
                raise RuntimeError(f"Formal {mode} training failed with exit {process.returncode}")
            if gate_process is not None:
                if gate_process.wait():
                    raise RuntimeError(f"First checkpoint policy gate failed: {gate_report}")
                if not (output / "first_deployable.json").exists():
                    write_json(output / "first_deployable.json", {"checkpoint":"step_000500", "mode":mode,
                                                                 "status":"loadable_experimental_candidate",
                                                                 "policy_load_gate_passed":True, "robot_test_performed":False})
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            if gate_process is not None and gate_process.poll() is None:
                gate_process.terminate()
                gate_process.wait(timeout=60)
            if gate_stream is not None:
                gate_stream.close()


def pipeline(root):
    root.mkdir(parents=True, exist_ok=True)
    for mode in ("limited", "full"):
        gate = root / (mode + "_engineering")
        common = training_command(mode, gate) + ["--steps", "2", "--warmup", "1", "--global-batch", "32",
                  "--batch-size", "2", "--accumulation", "2", "--checkpoint-every", "2", "--eval-samples", "8",
                  "--eval-draws", "1", "--workers", "0", "--engineering-smoke"]
        checked(common + ["--stop-after", "1"], root / f"{mode}_gate_initial.log")
        checked(common + ["--resume"], root / f"{mode}_gate_resume.log")
        checked([sys.executable, "scripts/check_backbone_gradient_checkpoint.py", "--checkpoint", str(gate / "step_000002"),
                 "--output", str(root / f"{mode}_engineering_policy.json"), "--allow-engineering"],
                root / f"{mode}_gate_policy.log")
    write_json(root / "engineering_passed.json", {"passed":True, "world_size":8, "modes":["limited", "full"],
                                                 "scope":"distributed update, optimizer/RNG resume, native policy loading"})
    for mode in ("limited", "full"):
        output = Path("checkpoints/pi05_piper_backbone_grad") / f"{mode}_seed42"
        run_formal(mode, output, root)
        selected = output / json.loads((output / "best.json").read_text())["checkpoint"]
        checked([sys.executable, "scripts/check_backbone_gradient_checkpoint.py", "--checkpoint", str(selected),
                 "--output", str(output / "policy_load_gate.json")], root / f"{mode}_policy.log")
        write_json(output / "candidate.json", {"checkpoint":selected.name, "mode":mode, "policy_load_gate_passed":True,
                                               "status":"loadable_experimental_candidate", "robot_test_performed":False})
    write_json(root / "complete.json", {"completed":True, "seed":42, "steps_per_model":5000, "global_batch":256})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("logs/pi05_piper_backbone_grad/pair_seed42"))
    p.add_argument("--managed", action="store_true")
    p.add_argument("--concurrent", action="store_true", help="Explicit user-authorized sharing with the existing job")
    args = p.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.managed:
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        try:
            pipeline(args.root)
        except Exception as error:
            write_json(args.root / "failure.json", {"error":str(error), "time":time.time()})
            raise
        return
    directory = Path("logs/pi05_subtask_stage1/gpu_reservation")
    command = [sys.executable, "scripts/run_backbone_gradient_pair.py", "--root", str(args.root), "--managed"]
    if args.concurrent:
        write_json(args.root / "dispatch.json", {"status":"concurrent_supervisor_dispatch", "command":command,
                                                "existing_job_preserved":True, "time":time.time(), "seed":42})
        code = gpu_reservation.run_concurrent(directory, args.root / "managed.log", command)
        write_json(args.root / "supervisor_exit.json", {"exit_code":code, "time":time.time()})
        raise SystemExit(code)
    write_json(args.root / "queued.json", {"status":"waiting_for_existing_reservation", "command":command,
                                          "time":time.time(), "seed":42})
    while True:
        with (directory / "job.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            fcntl.flock(lock, fcntl.LOCK_UN)
        try:
            code = gpu_reservation.run(directory, args.root / "managed.log", command)
            write_json(args.root / "supervisor_exit.json", {"exit_code":code, "time":time.time()})
            raise SystemExit(code)
        except BlockingIOError:
            time.sleep(.5)


if __name__ == "__main__":
    main()
