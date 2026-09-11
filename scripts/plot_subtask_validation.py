"""Plot complete observed validation curves at matched action update counts."""

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def validations(root, run, key):
    path = root / run / "metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return {row["step"]: row[key] for row in rows if row.get("event") == "validation"}


def main(args):
    root = Path("checkpoints/pi05_piper_stage1")
    generated = validations(root, f"m2_pilot_seed{args.seed}", "flow_generated_native14_normalized")
    generated.update(
        {
            step + 500: value
            for step, value in validations(
                root, f"m3_pilot_seed{args.seed}", "flow_generated_native14_normalized"
            ).items()
        }
    )
    baseline = validations(root, f"b1_pilot_seed{args.seed}", "flow_native14_normalized")
    common = sorted(set(generated) & set(baseline))
    delta = np.asarray([generated[step] - baseline[step] for step in common])
    args.output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.18)
        ax.set_xlabel("Cumulative action optimizer updates")
    for values, label, color in [
        (generated, "G: generated subtask", "#2166ac"),
        (baseline, "B1: no subtask", "#cc6677"),
    ]:
        steps = sorted(values)
        axes[0].plot(steps, [values[step] for step in steps], label=label, color=color, linewidth=1.8)
    axes[0].axvline(500, color="#888888", linewidth=1, linestyle="--")
    axes[0].set_ylabel("Validation flow MSE (normalized native 14 dimensions)")
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].set_title(f"Observed curves; latest B1 validation at {max(baseline):,}")
    axes[1].axhline(0, color="#555555", linewidth=1, linestyle="--")
    axes[1].plot(common, delta, color="#665191", linewidth=1.5)
    axes[1].fill_between(common, delta, 0, where=delta < 0, color="#2166ac", alpha=0.15, interpolate=True)
    axes[1].fill_between(common, delta, 0, where=delta >= 0, color="#cc6677", alpha=0.15, interpolate=True)
    axes[1].set_ylabel("G minus B1 validation flow MSE")
    axes[1].set_title("Matched updates; negative values favor G")
    fig.suptitle(f"Pi0.5 subtask pilot | seed {args.seed} | validation", fontsize=14)
    fig.text(
        0.07,
        0.02,
        "128 fixed validation frames x 2 fixed noise draws. G uses 500 M2 + 3,500 M3 action updates.\nOnly observed B1 points are shown; this single-seed training curve is not a final significance result.",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0.10, 1, 0.93])
    fig.savefig(args.output / "matched_validation.png", dpi=160)
    plt.close(fig)
    report = {
        "seed": args.seed,
        "latest_b1_validation_step": max(baseline),
        "scope": "Observed trainer validation values at matched action-update counts; no extrapolation or model selection from this plot",
        "generated_curve": generated,
        "baseline_curve": baseline,
        "paired": [
            {
                "action_updates": step,
                "G": generated[step],
                "B1": baseline[step],
                "G_minus_B1": generated[step] - baseline[step],
            }
            for step in common
        ],
    }
    (args.output / "curves.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "latest_b1_step": max(baseline)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
