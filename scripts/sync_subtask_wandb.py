"""Upload durable JSONL metrics independently of training; never load CUDA models."""

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

DEFAULT_PROJECT = "agentic-openpi-pi05-subtask"


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def flatten(value, prefix):
    result = {}
    for key, item in value.items():
        name = f"{prefix}/{key}"
        if isinstance(item, dict):
            result.update(flatten(item, name))
        elif isinstance(item, float | int) and not isinstance(item, bool):
            result[name] = item
    return result


def metric_payload(row, config, source_line):
    event = row.get("event", "other")
    step = row.get("step", row.get("completed_steps", row.get("start", 0)))
    result = {"trainer/step": step, "bridge/source_line": source_line}
    if event in {"train", "validation", "performance", "visualization"}:
        result.update(flatten({k: v for k, v in row.items() if k != "step"}, event))
    if event == "train":
        result["progress/fraction"] = step / config["steps"]
        if row.get("seconds", 0) > 0:
            batch = row.get("examples", config["batch_size"] * config["accumulation"] * config["world_size"])
            result["performance/train_examples_per_second"] = batch / row["seconds"]
    return result


def is_current_job(directory):
    try:
        import psutil

        state = json.loads(Path("logs/pi05_subtask_stage1/gpu_reservation/control.json").read_text())
        command = state.get("command", [])
        if "--output" not in command:
            return False
        output = Path(command[command.index("--output") + 1]).resolve()
        return (
            output == directory.resolve()
            and abs(psutil.Process(state["job_pid"]).create_time() - state["job_created"]) < 0.1
        )
    except (OSError, KeyError, ValueError, psutil.Error):
        return False


def gpu_metrics():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"],
        text=True,
        timeout=5,
    )
    result = {}
    for row in output.splitlines():
        index, utilization, memory, power = row.split(",")
        for name, value in [("utilization_percent", utilization), ("memory_mib", memory), ("power_w", power)]:
            result[f"gpu/{index.strip()}/{name}"] = float(value)
    return result


def sync(directory, project, entity, follow):
    import wandb

    destination = directory / "observability"
    destination.mkdir(exist_ok=True)
    with (destination / "wandb.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = json.loads((directory / "run_config.json").read_text())
        identity = hashlib.sha256((str(directory.resolve()) + json.dumps(config, sort_keys=True)).encode()).hexdigest()[
            :16
        ]
        source_line = 0
        media_sha = None
        api = wandb.Api(timeout=20)
        entity = entity or api.default_entity
        try:
            previous = api.run(f"{entity}/{project}/{identity}")
            source_line = int(previous.summary.get("bridge/source_line", 0))
            media_sha = previous.summary.get("bridge/first_step_sha")
        except wandb.errors.CommError as error:
            if "not find run" not in str(error).lower() and "not found" not in str(error).lower():
                raise
        run = wandb.init(
            project=project,
            entity=entity,
            id=identity,
            resume="allow",
            name=directory.name,
            group=config.get("stage", "m0"),
            job_type="training-metrics",
            tags=[config.get("stage", "m0"), "engineering" if config.get("engineering_smoke") else "research"],
            config=config,
            dir=str(destination),
            mode="online",
            settings=wandb.Settings(
                init_timeout=45, disable_git=True, save_code=False, console="off", x_disable_stats=True
            ),
        )
        run.define_metric("trainer/step")
        for prefix in ["train", "validation", "performance", "progress", "visualization"]:
            run.define_metric(f"{prefix}/*", step_metric="trainer/step")
        run.define_metric("telemetry/wall_time")
        run.define_metric("gpu/*", step_metric="telemetry/wall_time")
        write_json(destination / "wandb_run.json", {"url": run.url, "id": run.id, "project": project, "entity": entity})
        print(json.dumps({"event": "wandb_ready", "url": run.url}), flush=True)
        last_gpu = 0
        try:
            while True:
                lines = (directory / "metrics.jsonl").read_text().splitlines(keepends=True)
                last_event = None
                for number, line in enumerate(lines, 1):
                    if not line.endswith("\n"):
                        break  # A trainer may still be writing this record.
                    row = json.loads(line)
                    last_event = row.get("event")
                    if number <= source_line:
                        continue
                    run.log(metric_payload(row, config, number))
                    run.summary["bridge/source_line"] = number
                    run.summary["trainer/status"] = last_event
                    source_line = number
                manifest = directory / "first_step" / "manifest.json"
                if manifest.exists():
                    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
                    if digest != media_sha:
                        data = json.loads(manifest.read_text())
                        payload = {
                            "visualization/checkpoint_step": data["step"],
                            "visualization/first_step": [
                                wandb.Image(str(manifest.parent / item["image"]), caption=item["caption"])
                                for item in data["samples"]
                            ],
                        }
                        run.log(payload)
                        run.summary["bridge/first_step_sha"] = digest
                        media_sha = digest
                run.summary["trainer/step"] = max(
                    (
                        json.loads(line).get("step", json.loads(line).get("completed_steps", 0))
                        for line in lines
                        if line.endswith("\n")
                    ),
                    default=0,
                )
                active = is_current_job(directory)
                if active and time.monotonic() - last_gpu > 10:
                    with contextlib.suppress(OSError, ValueError, subprocess.SubprocessError):
                        run.log({"telemetry/wall_time": time.time(), **gpu_metrics()})
                    last_gpu = time.monotonic()
                inactive = not active and time.time() - (directory / "metrics.jsonl").stat().st_mtime > 90
                if not follow or inactive or (last_event in {"complete", "stopped_at_checkpoint"} and not active):
                    break
                time.sleep(5)
        finally:
            run.finish()
        write_json(
            destination / "wandb_synced.json",
            {
                "source_line": source_line,
                "metrics_size": (directory / "metrics.jsonl").stat().st_size,
                "media_sha": media_sha,
                "finished": time.time(),
            },
        )


def watch(root, project, entity):
    logdir = Path("logs/pi05_subtask_stage1/wandb")
    logdir.mkdir(parents=True, exist_ok=True)
    with (logdir / "watch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(logdir / "watch.json", {"pid": os.getpid(), "started": time.time(), "project": project})
        children = {}
        attempted = {}
        while True:
            for config_path in root.glob("*/run_config.json"):
                directory = config_path.parent
                config = json.loads(config_path.read_text())
                if config.get("device") != "cuda" or (
                    (config.get("engineering_smoke") or "smoke" in directory.name) and "overfit" not in directory.name
                ):
                    continue
                metrics = directory / "metrics.jsonl"
                if not metrics.exists():
                    continue
                if directory in children and children[directory].poll() is None:
                    continue
                signature = metrics.stat().st_size
                manifest = directory / "first_step" / "manifest.json"
                media_sha = hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None
                saved = directory / "observability" / "wandb_synced.json"
                state = json.loads(saved.read_text()) if saved.exists() else {}
                if state.get("metrics_size") == signature and state.get("media_sha") == media_sha:
                    continue
                if time.monotonic() - attempted.get(directory, -1e9) < 60:
                    continue
                lock_path = directory / "observability" / "wandb.lock"
                if lock_path.exists():
                    with lock_path.open("a") as run_lock:
                        try:
                            fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            continue
                        fcntl.flock(run_lock, fcntl.LOCK_UN)
                command = [sys.executable, __file__, "--run", str(directory), "--project", project, "--follow"]
                if entity:
                    command += ["--entity", entity]
                with (logdir / f"{directory.name}.log").open("a") as stream:
                    children[directory] = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                attempted[directory] = time.monotonic()
            time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", type=Path)
    mode.add_argument("--watch-root", type=Path)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--entity")
    parser.add_argument("--follow", action="store_true")
    args = parser.parse_args()
    if args.watch_root:
        watch(args.watch_root, args.project, args.entity)
    else:
        sync(args.run, args.project, args.entity, args.follow)


if __name__ == "__main__":
    main()
