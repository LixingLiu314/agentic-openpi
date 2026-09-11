"""Finite official-init gradient comparison: checks, then three independent runs."""

import argparse
import fcntl
import hashlib
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


MODES = ("frozen", "limited", "full")


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def checked(command, log):
    with log.open("x") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(log.with_suffix(".process.json"), {"pid":process.pid,
                   "created":psutil.Process(process.pid).create_time(), "command":command, "started":time.time()})
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
            raise RuntimeError(f"Child exited {code}; see {log}")


def training_command(mode, output, cache):
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=8",
            "scripts/train_official_backbone_gradient.py", "--mode", mode, "--output", str(output),
            "--initialize-from", "checkpoints/pi05_base_pytorch", "--decoded-cache", str(cache),
            "--batch-size", "32", "--accumulation", "1", "--workers", "4", "--eval-batch-size", "2"]


def verify_sources(root):
    for name, expected in json.loads((root / "source_manifest.json").read_text()).items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Source changed since dispatch: {name}")


def cpu_gates(args):
    root = args.root
    if args.reuse_from:
        previous = json.loads((args.reuse_from / "source_manifest.json").read_text())
        allowed_changes = {"scripts/check_official_gradient_checkpoint.py", "scripts/run_official_gradient_trio.py"}
        changed = [name for name, digest in previous.items() if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest]
        if set(changed) - allowed_changes:
            raise RuntimeError(f"Cannot reuse engineering evidence after training/model/data changes: {changed}")
        record = json.loads((args.reuse_from / "cpu_gates_passed.json").read_text())
        assert record["passed"] and record["same_initial_decoder"]
        write_json(root / "cpu_gates_passed.json", {**record, "reused_from":str(args.reuse_from), "changed_sources":changed})
        return
    cpu_results = []
    for mode in MODES:
        verify_sources(root)
        write_json(root / "phase.json", {"phase":"cpu_route_gate", "mode":mode, "time":time.time()})
        checked([sys.executable, "scripts/verify_official_gradient.py", "--mode", mode,
                 "--device", "cpu", "--output", str(root / f"{mode}_cpu.json")], root / f"{mode}_cpu.log")
        result = json.loads((root / f"{mode}_cpu.json").read_text())
        assert result["passed"] and result["verified_base_tensors"] == 812
        cpu_results.append(result)
    assert len({x["decoder_sha256"] for x in cpu_results}) == 1
    write_json(root / "cpu_gates_passed.json", {"passed":True, "arms":cpu_results,
               "same_initial_decoder":True, "official_direct_initialization":True})


def pipeline(args):
    root = args.root
    def phase(name, **fields):
        verify_sources(root)
        write_json(root / "phase.json", {"phase":name, "time":time.time(), **fields})
    cpu_record = json.loads((root / "cpu_gates_passed.json").read_text())
    assert cpu_record["passed"]
    cpu_results = cpu_record["arms"]
    gpu_results = []
    for mode in MODES:
        phase("eight_gpu_save_resume_gate", mode=mode)
        output = root / f"{mode}_engineering"
        command = training_command(mode, output, args.cache) + ["--engineering-smoke", "--steps", "8",
                  "--warmup", "2", "--checkpoint-every", "4", "--eval-samples", "8", "--eval-draws", "1"]
        previous_output = args.reuse_from / f"{mode}_engineering" if args.reuse_from else None
        if previous_output and (previous_output / "step_000008/metadata.json").exists():
            output = previous_output
            write_json(root / f"{mode}_reused_engineering.json", {"output":str(output),
                       "reason":"completed save/resume preserved; only native shared-alias count assertion and orchestration changed"})
        else:
            checked(command + ["--stop-after", "4"], root / f"{mode}_gate_first.log")
            checked(command + ["--resume"], root / f"{mode}_gate_resume.log")
        checkpoint = output / "step_000008"
        metadata = json.loads((checkpoint / "metadata.json").read_text())
        assert metadata["counters"] == {"subtask":8, "action":8, "backbone":0 if mode == "frozen" else 8}
        assert len(list(checkpoint.glob("training_rank_*.pt"))) == 8
        checked([sys.executable, "scripts/check_official_gradient_checkpoint.py", "--checkpoint", str(checkpoint),
                 "--allow-engineering", "--output", str(root / f"{mode}_native.json")], root / f"{mode}_native.log")
        init = json.loads((output / "initialization.json").read_text())
        assert init["decoder_sha256"] == cpu_results[0]["decoder_sha256"]
        rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        seconds = [row["seconds"] for row in rows if row.get("event") == "train" and row["step"] in (3,4,6,7,8)]
        gpu_results.append({"mode":mode, "microbatch":32, "accumulation":1, "world":8, "global_batch":256,
                            "resume_completed_steps":8, "native_gate_passed":True,
                            "mean_update_seconds":sum(seconds)/len(seconds), "initialization":init})
    write_json(root / "engineering_passed.json", {"passed":True, "arms":gpu_results,
               "scope":"actual official-init routes, equal S initialization, 8-rank save/resume and native policy; not task success"})

    for mode in MODES:
        output = Path("checkpoints/pi05_piper_official_grad") / f"{mode}_seed42_v1"
        phase("formal_training", mode=mode, output=str(output), steps=5000)
        write_json(root / f"{mode}_formal_dispatch.json", {"time":time.time(), "mode":mode, "output":str(output),
                   "initialization":"official_pi05_base", "inherited_training_updates":0, "steps":5000})
        checked(training_command(mode, output, args.cache) + ["--wandb"], root / f"{mode}_formal.log")
        init = json.loads((output / "initialization.json").read_text())
        assert init["decoder_sha256"] == cpu_results[0]["decoder_sha256"]
        selected = output / json.loads((output / "best.json").read_text())["checkpoint"]
        phase("final_native_gate", mode=mode, checkpoint=str(selected))
        checked([sys.executable, "scripts/check_official_gradient_checkpoint.py", "--checkpoint", str(selected),
                 "--output", str(output / "policy_load_gate.json")], root / f"{mode}_final_native.log")
        write_json(output / "candidate.json", {"checkpoint":selected.name, "mode":mode,
                   "initialization":"official_pi05_base", "policy_load_gate_passed":True,
                   "status":"loadable_experimental_candidate", "robot_test_performed":False})
        write_json(root / f"{mode}_complete.json", {"completed":True, "output":str(output),
                   "candidate":str(selected), "steps":5000, "global_batch":256, "time":time.time()})
    phase("complete")
    write_json(root / "complete.json", {"completed":True, "modes":MODES, "time":time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--managed", action="store_true")
    parser.add_argument("--reuse-from", type=Path,
                        help="Reuse completed engineering evidence only if training/model/data sources are identical")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.managed:
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        try:
            pipeline(args)
        except BaseException as error:
            write_json(args.root / "failure.json", {"error":repr(error), "traceback":traceback.format_exc(), "time":time.time()})
            raise
    else:
        with (args.root / "sequence.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
            try:
                cpu_gates(args)  # Keep GPU reservation workers holding during CPU checks.
            except BaseException as error:
                write_json(args.root / "failure.json", {"error":repr(error), "traceback":traceback.format_exc(), "time":time.time()})
                raise
            code = run_concurrent(Path("logs/pi05_subtask_stage1/gpu_reservation"), args.root / "managed.log",
                                  [sys.executable, str(Path(__file__).resolve()), "--root", str(args.root),
                                   "--cache", str(args.cache), "--managed"] +
                                  (["--reuse-from", str(args.reuse_from)] if args.reuse_from else []))
            write_json(args.root / "supervisor_exit.json", {"code":code, "time":time.time()})
            raise SystemExit(code)


if __name__ == "__main__":
    main()
