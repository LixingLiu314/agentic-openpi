"""W&B display schema v1. Pure event mapping; no model or GPU imports."""
from __future__ import annotations

import math
from numbers import Real

SCHEMA_VERSION = 1
AXIS = "trainer/step"
TRAIN = {
    "loss_subtask": "train/subtask_ce",
    "loss_action": "train/action_flow_mse_global_only_all32",
    "lr": "optim/lr",
    "seconds": "perf/update_seconds",
}
VALIDATION = {
    "loss_subtask": "val/subtask_ce",
    "exact_match": "val/subtask_exact_match",
    "macro_f1": "val/subtask_macro_f1",
    "unknown_rate": "val/unknown_rate",
    "invalid_generation_rate": "val/invalid_generation_rate",
    "flow_global_only_native14_normalized": "val/action_flow_mse_global_only_native14",
    "flow_global_only_all32": "val/action_flow_mse_global_only_all32",
}


def number(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"Expected a finite numeric metric, got {value!r}")
    return float(value)


def event_payload(row, config):
    """Explicit allowlist: train and validation can never overwrite each other.

    Each point carries its own optimizer step. No inferred microbatch/log-line axis.
    Unknown events are retained locally, not recursively expanded into charts.
    """
    event = row.get("event")
    mapping = TRAIN if event == "train" else VALIDATION if event == "validation" else None
    if mapping is None:
        return {}
    step = row.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("Metric events require a nonnegative integer optimizer step")
    out = {AXIS: step}
    for source, target in mapping.items():
        if source in row:
            out[target] = number(row[source])
    for key, value in out.items():
        if key.endswith(("_rate", "_exact_match", "_macro_f1")) and not 0 <= value <= 1:
            raise ValueError(f"Rates use fractions in [0,1]: {key}={value}")
    if event == "train":
        if config.get("steps"):
            out["progress/fraction"] = step / config["steps"]
        for source, target in {"action": "optim/grad_norm_a", "backbone": "optim/grad_norm_b", "backbone_sampled_relative_update":"optim/update_sampled_relative_b", "action_sampled_relative_update":"optim/update_sampled_relative_a", "backbone_clip_scale":"optim/clip_scale_b", "action_clip_scale":"optim/clip_scale_a"}.items():
            if source in row.get("grad_norms", {}):
                out[target] = number(row["grad_norms"][source])
        n = row.get("generated_count", 0)
        if n > 0:
            for source, target in {"invalid_generation_count": "train/invalid_generation_rate", "empty_condition_count": "train/empty_condition_rate"}.items():
                if source in row:
                    if not 0 <= row[source] <= n:
                        raise ValueError(f"Invalid count: {source}")
                    out[target] = number(row[source]) / n
        seconds = row.get("seconds", 0)
        batch = config.get("global_batch_size", config.get("global_batch"))
        if seconds > 0 and batch:
            out["perf/examples_per_second"] = batch / seconds
    return out


def class_rows(row):
    """One table, not three automatically generated plots for every label."""
    return [[row["step"], label, number(v["f1"]), number(v["recall"]), int(v["support"])]
            for label, v in sorted(row.get("per_class", {}).items())]


def configure_run(run):
    """Call immediately after wandb.init, before any log call, on rank zero only."""
    run.define_metric(AXIS, hidden=True)
    run.define_metric("_view/source_line", hidden=True)
    for namespace in ("train", "val", "optim", "perf", "progress", "media"):
        run.define_metric(f"{namespace}/*", step_metric=AXIS, step_sync=False)
    for metric in ("val/action_flow_mse_global_only_native14", "val/action_flow_mse_global_only_all32", "val/subtask_ce"):
        run.define_metric(metric, step_metric=AXIS, step_sync=False, summary="min,last")
    for metric in ("val/subtask_exact_match", "val/subtask_macro_f1"):
        run.define_metric(metric, step_metric=AXIS, step_sync=False, summary="max,last")


def log_event(run, row, config, *, upload_step=None):
    """Future trainers use this after writing the original durable JSONL event.

    upload_step is an optional monotonic transport index, never the plotted axis.
    """
    payload = event_payload(row, config)
    if not payload:
        return False
    if row.get("per_class"):
        import wandb
        payload["media/subtask_per_class"] = wandb.Table(
            columns=["optimizer_step", "subtask", "f1", "recall", "support"], data=class_rows(row))
    run.log(payload, step=upload_step)
    return True
