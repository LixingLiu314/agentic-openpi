"""Finite supervised queue: real capacity/resume gates, then matched S arms."""
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


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def checked(command, log):
    with log.open("x") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(log.with_suffix(".process.json"), dict(pid=process.pid,
                   created=psutil.Process(process.pid).create_time(), command=command, started=time.time()))
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
            raise RuntimeError(f"Child exited {code}: {log}")


def command(arm, output, cache, steps):
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=8",
            "scripts/train_recurrent_subtask.py", "--arm", arm, "--output", str(output),
            "--decoded-cache", str(cache), "--steps", str(steps), "--batch-size", "32", "--unroll", "4",
            "--accumulation", "1", "--workers", "4", "--eval-batch-size", "2"]


def verify_sources(root):
    for name, digest in json.loads((root / "source_manifest.json").read_text()).items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source changed since dispatch: {name}")


def pipeline(args):
    root = args.root
    def phase(name, **fields):
        verify_sources(root)
        write_json(root / "phase.json", dict(phase=name, time=time.time(), **fields))
    if args.stage == "gates":
        records = []
        for arm in ("stateless", "recurrent"):
            phase("capacity_save_resume_gate", arm=arm)
            output = root / (arm + "_engineering")
            cmd = command(arm, output, args.cache, 8) + ["--engineering-smoke", "--warmup", "2",
                  "--checkpoint-every", "4", "--eval-samples", "8", "--eval-draws", "1"]
            checked(cmd + ["--stop-after", "4"], root / f"{arm}_gate_first.log")
            checked(cmd + ["--resume"], root / f"{arm}_gate_resume.log")
            checkpoint = output / "step_000008"
            checked([sys.executable, "scripts/check_recurrent_checkpoint.py", "--checkpoint", str(checkpoint),
                     "--allow-engineering", "--output", str(root / f"{arm}_native.json")],
                    root / f"{arm}_native.log")
            metadata = json.loads((checkpoint / "metadata.json").read_text())
            assert metadata["counters"] == dict(subtask=8, action=8, backbone=0)
            assert len(list(checkpoint.glob("training_rank_*.pt"))) == 8
            initialization = json.loads((output / "initialization.json").read_text())
            rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
            seconds = [r["seconds"] for r in rows if r.get("event") == "train" and r["step"] in (2, 3, 4, 6, 7, 8)]
            records.append(dict(arm=arm, mean_update_seconds=statistics.mean(seconds),
                                initial_decoder_sha256=initialization["decoder_sha256"],
                                native_gate=json.loads((root / f"{arm}_native.json").read_text())))
        assert len({r["initial_decoder_sha256"] for r in records}) == 1
        write_json(root / "engineering_passed.json", dict(passed=True, arms=records, time=time.time()))
        phase("gates_complete")
        return
    gate = json.loads((root / "engineering_passed.json").read_text())
    assert gate["passed"]
    for arm in ("stateless", "recurrent"):
        output = Path("checkpoints/pi05_piper_recurrent_s") / f"{arm}_seed42_v1"
        phase("formal_training", arm=arm, output=str(output), steps=args.steps)
        checked(command(arm, output, args.cache, args.steps) + ["--wandb"], root / f"{arm}_formal.log")
        chosen = output / json.loads((output / "best.json").read_text())["checkpoint"]
        phase("final_native_gate", arm=arm, checkpoint=str(chosen))
        checked([sys.executable, "scripts/check_recurrent_checkpoint.py", "--checkpoint", str(chosen),
                 "--output", str(output / "policy_load_gate.json")], root / f"{arm}_final_native.log")
        initialization = json.loads((output / "initialization.json").read_text())
        assert initialization["decoder_sha256"] == gate["arms"][0]["initial_decoder_sha256"]
        candidate = dict(checkpoint=str(chosen), arm=arm, completed_steps=args.steps, native_gate_passed=True,
                         status="loadable_experimental_candidate", robot_test_performed=False, time=time.time())
        write_json(output / "candidate.json", candidate)
        write_json(root / f"{arm}_complete.json", candidate)
    phase("complete")
    write_json(root / "complete.json", dict(completed=True, steps=args.steps, time=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--stage", choices=["gates", "formal"], required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--managed", action="store_true")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    if args.managed:
        try:
            pipeline(args)
        except BaseException as error:
            write_json(args.root / (args.stage + "_failure.json"),
                       dict(error=repr(error), traceback=traceback.format_exc(), time=time.time()))
            raise
        return
    with (args.root / "sequence.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        files = ["scripts/run_recurrent_pair.py", "scripts/check_recurrent_checkpoint.py",
                 "scripts/train_recurrent_subtask.py", "scripts/serve_recurrent_subtask_piper.py",
                 "src/openpi/models_pytorch/recurrent_subtask.py", "src/openpi/training/recurrent_sequence.py",
                 "src/openpi/training/recurrent_evaluation.py", "src/openpi/policies/recurrent_subtask_policy.py"]
        if args.stage == "gates":
            write_json(args.root / "source_manifest.json", {name:hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                                            for name in files})
        verify_sources(args.root)
        code = run_concurrent(Path("logs/pi05_subtask_stage1/gpu_reservation"),
                              args.root / (args.stage + "_managed.log"),
                              [sys.executable, __file__, "--root", str(args.root), "--cache", str(args.cache),
                               "--stage", args.stage, "--steps", str(args.steps), "--managed"])
        write_json(args.root / (args.stage + "_exit.json"), dict(code=code, time=time.time()))
        raise SystemExit(code)


if __name__ == "__main__":
    main()
