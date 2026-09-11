"""Paired native-action comparison with episode-clustered uncertainty."""

import json
from pathlib import Path

import numpy as np

from openpi.policies.piper_policy import JOINT_MASK

METRICS = (
    "flow_native14_normalized",
    "joint_rmse_h50",
    "gripper_rmse_h50",
    "joint_rmse_execution",
    "gripper_rmse_execution",
    "gripper_accuracy_execution",
    "gripper_excursion_gt_1e_3_execution",
)
ROOT_METRICS = np.asarray([False, True, True, True, True, False, False])


def load_windows(directory):
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    windows = {}
    for path in sorted(directory.glob("native_rank_*.npz")):
        rank = path.stem.rsplit("_", 1)[1]
        rows = json.loads((directory / f"actions_rank_{rank}.json").read_text())
        with np.load(path, allow_pickle=False) as arrays:
            if arrays["predicted"].shape != arrays["target"].shape or arrays["predicted"].shape != (len(rows), 50, 14):
                raise ValueError("Native action archive shape mismatch")
            if not np.isfinite(arrays["predicted"]).all() or not np.isfinite(arrays["target"]).all():
                raise ValueError("Nonfinite native action archive")
            for i, row in enumerate(rows):
                key = (row["index"], row["draw"], row["condition_mode"])
                archived = (int(arrays["index"][i]), int(arrays["draw"][i]), str(arrays["condition"][i]))
                if key != archived or row["valid_horizon"] != int(arrays["valid_horizon"][i]) or key in windows:
                    raise ValueError("Native action archive identity mismatch")
                windows[key] = (row, arrays["predicted"][i].copy(), arrays["target"][i].copy())
    if not windows:
        raise ValueError("No archived native action windows")
    return report, windows


def window_values(row, predicted, target, execution_frames):
    valid = min(execution_frames, row["valid_horizon"])
    if not 1 <= valid <= 50:
        raise ValueError("Invalid execution horizon")
    error = (predicted - target) ** 2
    joints = np.asarray(JOINT_MASK)
    grippers = [6, 13]
    grip = predicted[:valid, grippers]
    excursion = np.maximum(np.maximum(-grip, grip - 0.09), 0)
    return np.asarray(
        [
            row["flow_native14_normalized"],
            error[:, joints].mean(),
            error[:, grippers].mean(),
            error[:valid, joints].mean(),
            error[:valid, grippers].mean(),
            np.mean((grip >= 0.045) == (target[:valid, grippers] >= 0.045)),
            np.mean(excursion > 1e-3),
        ],
        dtype=np.float64,
    )


def paired_arrays(experiments, execution_frames):
    if not 1 <= execution_frames <= 50:
        raise ValueError("Execution frames must be within the predicted horizon")
    names = ["G", "G-oracle", "G-drop", "G-shuffle", "B0", "B1", "O"]
    seeds, values, reference = [], [], None
    provenance = []
    for experiment in experiments:
        if experiment["seed"] in seeds:
            raise ValueError("Training seeds must be unique")
        seeds.append(experiment["seed"])
        sources = {name: load_windows(experiment[name]) for name in ["g", "b0", "b1", "o"]}
        split_ids = {
            (
                source[0]["split"],
                source[0]["split_sha256"],
                source[0]["norm_sha256"],
                source[0]["flow_steps"],
                source[0]["noise_draws"],
            )
            for source in sources.values()
        }
        if len(split_ids) != 1:
            raise ValueError("Methods use different split or normalization identities")
        if provenance and split_ids != set(map(tuple, provenance[0]["split_identity"])):
            raise ValueError("Seeds use different split or normalization identities")
        provenance.append(
            {
                "seed": experiment["seed"],
                "split_identity": list(split_ids),
                "evaluations": {name: str(experiment[name]) for name in sources},
            }
        )
        mappings = [("g", mode) for mode in ["generated", "oracle", "drop", "shuffle"]] + [
            ("b0", "none"),
            ("b1", "none"),
            ("o", "oracle"),
        ]
        method_values = []
        for source, condition in mappings:
            selected = {
                (index, draw): value for (index, draw, mode), value in sources[source][1].items() if mode == condition
            }
            if not selected:
                raise ValueError(f"Missing evaluation condition {condition}")
            keys = sorted(selected)
            if reference is None:
                reference = {key: selected[key] for key in keys}
            if keys != sorted(reference):
                raise ValueError("Methods/seeds must use identical frames and noise draws")
            rows = []
            for key in keys:
                row, predicted, target = selected[key]
                ref_row, _, ref_target = reference[key]
                for field in ["episode", "frame", "task", "valid_horizon", "crosses_subtask_boundary"]:
                    if row[field] != ref_row[field]:
                        raise ValueError(f"Paired metadata differs: {field}")
                if not np.array_equal(target, ref_target):
                    raise ValueError("Paired native targets differ")
                rows.append(window_values(row, predicted, target, execution_frames))
            method_values.append(rows)
        values.append(method_values)
    if reference is None:
        raise ValueError("No experiments")
    keys = sorted(reference)
    anchors = sorted({key[0] for key in keys})
    # Average repeated action-noise draws before sampling trajectory clusters.
    values = np.asarray(values)
    values = np.stack(
        [values[:, :, [i for i, key in enumerate(keys) if key[0] == index]].mean(axis=2) for index in anchors], axis=2
    )
    metadata = [reference[next(key for key in keys if key[0] == index)][0] for index in anchors]
    return names, seeds, values, metadata, provenance


def aggregate(values, weights=None):
    averaged = np.average(values, axis=-2, weights=weights)
    averaged[..., ROOT_METRICS] = np.sqrt(averaged[..., ROOT_METRICS])
    return averaged


def clustered_summary(names, seeds, values, metadata, *, repetitions=2000, bootstrap_seed=90210):
    if repetitions < 100:
        raise ValueError("Use at least 100 bootstrap repetitions")
    per_seed = aggregate(values)
    point = per_seed.mean(axis=0)
    rng = np.random.default_rng(bootstrap_seed)
    episode_tasks = {}
    for row in metadata:
        if row["episode"] in episode_tasks and episode_tasks[row["episode"]] != row["task"]:
            raise ValueError("Episode assigned multiple global tasks")
        episode_tasks[row["episode"]] = row["task"]
    groups = {
        task: sorted(episode for episode, label in episode_tasks.items() if label == task)
        for task in sorted(set(episode_tasks.values()))
    }
    bootstrap = []
    for _ in range(repetitions):
        counts = dict.fromkeys(episode_tasks, 0)
        for episodes in groups.values():
            for episode in rng.choice(episodes, size=len(episodes), replace=True):
                counts[int(episode)] += 1
        weights = np.asarray([counts[row["episode"]] for row in metadata])
        bootstrap.append(aggregate(values, weights).mean(axis=0))
    bootstrap = np.asarray(bootstrap)
    methods = {}
    for i, name in enumerate(names):
        methods[name] = {
            metric: {
                "mean_across_seeds": float(point[i, j]),
                "seed_standard_deviation": float(per_seed[:, i, j].std(ddof=1)) if len(seeds) > 1 else None,
                "per_seed": {str(seed): float(per_seed[k, i, j]) for k, seed in enumerate(seeds)},
                "episode_bootstrap_ci95": np.percentile(bootstrap[:, i, j], [2.5, 97.5]).tolist(),
            }
            for j, metric in enumerate(METRICS)
        }
    comparisons = {}
    for other in ["B0", "B1", "O", "G-oracle", "G-drop", "G-shuffle"]:
        i, j = names.index("G"), names.index(other)
        comparisons[f"G_minus_{other}"] = {
            metric: {
                "difference": float(point[i, k] - point[j, k]),
                "paired_episode_bootstrap_ci95": np.percentile(
                    bootstrap[:, i, k] - bootstrap[:, j, k], [2.5, 97.5]
                ).tolist(),
            }
            for k, metric in enumerate(METRICS)
        }
    return {
        "training_seeds": seeds,
        "frames": len(metadata),
        "episodes": len(episode_tasks),
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": bootstrap_seed,
        "uncertainty_scope": "Task-stratified paired episode bootstrap, identical resampled episodes across methods and seeds; conditional on the fitted checkpoints. Training-seed variability is reported separately. Frames within an episode and repeated noise draws are not independent resampling units.",
        "aggregation": "Mean across noise draws per frame; frame-weighted MSE within each seed, square root for RMSE, then mean across training seeds. Execution windows stop at episode ends. Positive G-minus-control means larger values; only gripper accuracy is better when larger.",
        "methods": methods,
        "paired_comparisons": comparisons,
    }
