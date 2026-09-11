"""Read-only W&B display bridge for the existing finite backbone pair.

Does not import training code, load weights, inspect GPUs, or edit source logs.
One SDK writer per presentation run; original training run IDs are never written.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import time

from openpi.training.wandb_standard import AXIS, SCHEMA_VERSION, class_rows, configure_run, event_payload

ENTITY = "xiahy23-tsinghua-university"
PROJECT = "agentic-openpi-pi05-subtask"
DISPLAY_SET = "backbone-grad-m3-s42"


def read_complete_rows(path, next_line):
    with path.open("rb") as stream:
        for line, raw in enumerate(stream, 1):
            if not raw.endswith(b"\n"):
                break  # A writer may still be appending this last row.
            if line < next_line:
                continue
            try:
                value = json.loads(raw)
            except (ValueError, UnicodeDecodeError) as exc:
                raise ValueError(f"Malformed completed JSONL row {path}:{line}") from exc
            yield line, value


def identity_alive(record):
    import psutil
    try:
        proc = psutil.Process(record["pid"])
        return abs(proc.create_time() - record["created"]) < 0.02 and proc.status() != psutil.STATUS_ZOMBIE
    except (psutil.Error, KeyError):
        return False


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


class Mirror:
    def __init__(self, directory, state_dir):
        import wandb
        self.directory = directory
        config_bytes = (directory / "run_config.json").read_bytes()
        self.config = json.loads(config_bytes)
        if self.config.get("engineering_smoke") or self.config.get("seed") != 42:
            raise ValueError("This bounded migration accepts only the authorized formal seed42 pair")
        identity = hashlib.sha256(str(directory.resolve()).encode() + config_bytes).hexdigest()[:16]
        self.run_id = "viewv1-" + identity
        self.run = wandb.init(
            entity=ENTITY, project=PROJECT, id=self.run_id,
            name=f"B-{self.config['mode']} | s42", group=DISPLAY_SET, job_type="metrics-view",
            tags=["display-v1", "view-only", "research", "backbone-grad", self.config["mode"], "seed42"],
            resume="allow", reinit="create_new", dir=str(state_dir),
            config={"display_schema": SCHEMA_VERSION, "display_set": DISPLAY_SET,
                    "view_only": True, "experiment_family": "backbone-grad", "variant": self.config["mode"],
                    "dataset": "eggplant_potato_gripper_binary", "seed": self.config["seed"],
                    "global_batch_size": self.config["global_batch_size"], "num_train_steps": self.config["steps"],
                    "parent": "m3/step_003500", "parent_sha256": self.config["parent_weights_sha256"],
                    "split_sha256": self.config["split_sha256"], "norm_sha256": self.config["norm_sha256"],
                    "source_run_url": f"https://wandb.ai/{ENTITY}/{PROJECT}/runs/{directory.name}",
                    "source_metrics": str(directory / "metrics.jsonl"),
                    "source_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                    "source_config": self.config},
            notes="Presentation-only mirror of durable training logs. No additional training. Optimizer-step axis; raw source run preserved.",
            settings=wandb.Settings(x_disable_stats=True, disable_git=True, save_code=False, console="off", init_timeout=60))
        configure_run(self.run)
        # SDK's resumed transport step is authoritative: replay cannot duplicate old history.
        self.next_line = self.run.step
        self.done = False
        self.media_uploaded = bool(self.run.summary.get("first_update_media_uploaded", False))
        self.sync()

    def sync(self):
        import wandb
        if self.done:
            return
        for line, row in read_complete_rows(self.directory / "metrics.jsonl", self.next_line):
            payload = event_payload(row, self.config)
            if payload:
                payload["_view/source_line"] = line
                if row.get("per_class"):
                    payload["media/subtask_per_class"] = wandb.Table(
                        columns=["optimizer_step", "subtask", "f1", "recall", "support"], data=class_rows(row))
                media = self.directory / "first_update"
                if not self.media_uploaded and row["step"] >= 1 and (media / "sample_0_inputs.png").exists() and (media / "sample_0_actions.png").exists():
                    for kind in ("inputs", "actions"):
                        payload[f"media/first_update_{kind}"] = wandb.Image(
                            str(media / f"sample_0_{kind}.png"), caption="First actual update (step 1); diagnostic sample, not a robot rollout")
                    self.media_uploaded = True
                    self.run.summary["first_update_media_uploaded"] = True
                self.run.log(payload, step=line)
                self.run.summary["last_optimizer_step"] = row["step"]
            event = row.get("event")
            if event == "ready":
                self.run.summary["parameter_counts"] = row.get("trainable_parameters", {})
                self.run.summary["evaluation"] = {"frames": self.config["eval_samples"], "draws": self.config["eval_draws"], "split": "validation"}
            if event == "checkpoint":
                self.run.summary["latest_checkpoint"] = row["path"]
                self.run.summary["selection"] = {"criterion": "val/action_flow_mse_native14", **row.get("best", {})}
            self.run.summary["source_status"] = "complete" if event == "complete" else "stopped_at_checkpoint" if event == "stopped_at_checkpoint" else "training"
            self.next_line = line + 1
            if event in {"complete", "stopped_at_checkpoint"}:
                self.close(0)
                break

    def close(self, exit_code):
        if not self.done:
            self.run.finish(exit_code=exit_code)
            self.done = True


def main():
    import psutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--once", action="store_true", help="Backfill a snapshot and exit; watch resumes the same presentation IDs")
    args = parser.parse_args()
    root = args.root.resolve()
    state_dir = root / "logs/wandb_display_v1"
    state_dir.mkdir(parents=True, exist_ok=True)
    lock = (state_dir / "bridge.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Display bridge already running; do not start a second writer")
    write_json(state_dir / "bridge.process.json", {"pid": os.getpid(), "created": psutil.Process().create_time(), "command": "sync_wandb_standard.py"})
    pair_record = json.loads((root / "logs/pi05_piper_backbone_grad/pair_seed42.process.json").read_text())
    mirrors = {}
    def request_stop(_signal, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, request_stop)
    try:
        while True:
            added_mode = False
            for mode in ("limited", "full"):
                directory = root / f"checkpoints/pi05_piper_backbone_grad/{mode}_seed42"
                if mode not in mirrors and (directory / "run_config.json").exists() and (directory / "metrics.jsonl").exists():
                    mirrors[mode] = Mirror(directory, state_dir)
                    added_mode = True
                elif mode in mirrors:
                    mirrors[mode].sync()
            write_json(state_dir / "bridge_state.json", {
                "schema": SCHEMA_VERSION, "updated_at": time.time(), "display_set": DISPLAY_SET,
                "runs": {mode: {"id": item.run_id, "url": item.run.url, "next_source_line": item.next_line,
                                 "source": str(item.directory), "finished": item.done} for mode, item in mirrors.items()}})
            # Refresh the existing saved view only when a source run appears.
            # This also assigns the full run its fixed color when it starts later.
            if added_mode and (state_dir / "workspace.json").exists():
                try:
                    from configure_wandb_workspace import save_workspace
                    save_workspace(root)
                except Exception as exc:
                    print(json.dumps({"event": "workspace_refresh_error", "error": str(exc)}), flush=True)
            pair_done = all(m in mirrors and mirrors[m].done for m in ("limited", "full"))
            if args.once or pair_done:
                break
            if not identity_alive(pair_record):
                for item in mirrors.values():
                    item.sync()  # Drain durable final rows once after owner exits.
                unfinished = [item for item in mirrors.values() if not item.done]
                for item in unfinished:
                    item.run.summary["source_status"] = "owner_exited_before_complete"
                if unfinished or len(mirrors) < 2:
                    raise RuntimeError("Finite training owner exited before both complete events; inspect original training records")
                break
            time.sleep(30)
    except KeyboardInterrupt:
        print("Display sync stopped gracefully; same presentation IDs can resume", flush=True)
    except BaseException as exc:
        write_json(state_dir / "bridge_error.json", {"time": time.time(), "error": str(exc)})
        for item in mirrors.values(): item.close(1)
        raise
    finally:
        for item in mirrors.values(): item.close(0)


if __name__ == "__main__":
    main()
