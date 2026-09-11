"""Offline semantic, transition and native-action metrics with explicit denominators."""

from collections import defaultdict

import numpy as np

from openpi.policies.piper_policy import JOINT_MASK
from openpi.training.hierarchy_training import text_metrics


def canonical(text):
    return " ".join((text or "").lower().split())


def mentioned_objects(text):
    text = canonical(text)
    return {name for name in ("eggplant", "sweet potato") if name in text}


def semantic_metrics(rows, vocabulary, *, task_vocabulary=None):
    result = text_metrics(rows, vocabulary)
    result["frames"] = len(rows)
    explicit = [row for row in rows if mentioned_objects(row["label"])]
    result["target_object_labeled_frames"] = len(explicit)
    result["target_object_accuracy"] = (
        sum(mentioned_objects(row["prediction"]) == mentioned_objects(row["label"]) for row in explicit) / len(explicit)
        if explicit
        else None
    )
    result["wrong_task_object_rate"] = sum(
        bool(mentioned_objects(row["prediction"]) - mentioned_objects(row["task"])) for row in rows
    ) / max(1, len(rows))
    result["by_task"] = {
        task: text_metrics([row for row in rows if row["task"] == task], (task_vocabulary or {}).get(task, vocabulary))
        for task in sorted({row["task"] for row in rows})
    }
    confusion = defaultdict(lambda: defaultdict(int))
    classes = {canonical(label) for label in vocabulary}
    for row in rows:
        prediction = canonical(row["prediction"])
        confusion[canonical(row["label"])][prediction if prediction in classes else "<unknown>"] += 1
    result["confusion"] = {label: dict(counts) for label, counts in confusion.items()}
    return result


def transition_metrics(rows, *, tolerance=10):
    """Match direct old->new predicted changes within each GT boundary's midpoint cell.

    Only complete, dense episodes are accepted. A GT change is matched to the
    nearest exact old->new predicted change in its cell; a prediction cannot be
    reused. Extra changes (including unknown text) measure unsmoothed jitter.
    Error = predicted frame - GT frame: negative early, positive late. Missing
    switches stay missing and are included in the tolerance denominator.
    """
    episodes = defaultdict(list)
    for row in rows:
        episodes[row["episode"]].append(row)
    matched, errors, unmatched_changes, count_changes = [], [], 0, 0
    near, far = [], []
    for episode, group in sorted(episodes.items()):
        group.sort(key=lambda row: row["frame"])
        length = group[0]["episode_length"]
        if [row["frame"] for row in group] != list(range(length)):
            raise ValueError("Transition metrics require every frame of complete episodes")
        truth = [canonical(row["label"]) for row in group]
        pred = [canonical(row["prediction"]) for row in group]
        boundaries = [i for i in range(1, length) if truth[i] != truth[i - 1]]
        changes = [i for i in range(1, length) if pred[i] != pred[i - 1]]
        used = set()
        for position, boundary in enumerate(boundaries):
            left = (boundaries[position - 1] + boundary) / 2 if position else 0
            right = (boundary + boundaries[position + 1]) / 2 if position + 1 < len(boundaries) else length
            candidates = [
                i
                for i in changes
                if left <= i < right
                and i not in used
                and pred[i - 1] == truth[boundary - 1]
                and pred[i] == truth[boundary]
            ]
            chosen = min(candidates, key=lambda i: (abs(i - boundary), i)) if candidates else None
            error = chosen - boundary if chosen is not None else None
            if chosen is not None:
                used.add(chosen)
                errors.append(error)
            matched.append(
                {
                    "episode": episode,
                    "gt_frame": boundary,
                    "predicted_frame": chosen,
                    "error_frames": error,
                    "within_tolerance": error is not None and abs(error) <= tolerance,
                    "old": truth[boundary - 1],
                    "new": truth[boundary],
                }
            )
        count_changes += len(changes)
        unmatched_changes += len(changes) - len(used)
        for row in group:
            (near if any(abs(row["frame"] - b) <= tolerance for b in boundaries) else far).append(row)

    def exact(group):
        return sum(canonical(r["label"]) == canonical(r["prediction"]) for r in group) / len(group) if group else None

    return {
        "definition": "direct old->new, nearest in neighboring-GT-midpoint cell; raw predictions without smoothing",
        "tolerance_frames": tolerance,
        "episodes": len(episodes),
        "gt_switches": len(matched),
        "matched_switches": len(errors),
        "missed_switches": len(matched) - len(errors),
        "within_tolerance_fraction_all_gt": sum(row["within_tolerance"] for row in matched) / max(1, len(matched)),
        "signed_error_mean_matched": float(np.mean(errors)) if errors else None,
        "absolute_error_p50_p95_matched": np.percentile(np.abs(errors), [50, 95]).tolist() if errors else None,
        "early_switches": sum(error < 0 for error in errors),
        "late_switches": sum(error > 0 for error in errors),
        "predicted_changes": count_changes,
        "unmatched_changes": unmatched_changes,
        "unmatched_changes_per_episode": unmatched_changes / max(1, len(episodes)),
        "near_boundary": {"frames": len(near), "exact_match": exact(near)},
        "away_from_boundary": {"frames": len(far), "exact_match": exact(far)},
        "switches": matched,
    }


def shuffled_conditions(rows, *, seed=731):
    """Deterministic within-task permutation of predictions; never inspect labels.

    Among 32 random permutations choose the one changing most strings. The
    generated text histogram is preserved; report unchanged strings explicitly.
    """
    pools = defaultdict(list)
    for row in rows:
        pools[row["task"]].append(row)
    result = {}
    rng = np.random.default_rng(seed)
    for task in sorted(pools):
        group = sorted(pools[task], key=lambda row: row["index"])
        texts = np.asarray([row["prediction"] for row in group])
        choices = [rng.permutation(len(group)) for _ in range(32)]
        chosen = max(choices, key=lambda order: np.count_nonzero(texts[order] != texts))
        result.update({row["index"]: str(texts[j]) for row, j in zip(group, chosen, strict=True)})
    return result


def native_action_metrics(native, target, *, valid_horizon):
    native, target = np.asarray(native, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if native.shape != target.shape or native.shape[-1] != 14 or not 1 <= valid_horizon <= len(native):
        raise ValueError("Expected matching [H,14] native actions and valid horizon")
    if not np.isfinite(native).all() or not np.isfinite(target).all():
        raise FloatingPointError("Nonfinite action metric inputs")
    errors = (native - target) ** 2
    result = {}
    for scope, end in [("h50", len(native)), ("first10", min(10, valid_horizon)), ("valid_horizon", valid_horizon)]:
        result[f"native_mse_per_dim_{scope}"] = errors[:end].mean(axis=0).tolist()
        result[f"scalar_count_{scope}"] = end
    grip = native[:, [6, 13]]
    excursions = np.maximum(np.maximum(-grip, grip - 0.09), 0)
    result.update(
        gripper_accuracy_h50=float(((grip >= 0.045) == (target[:, [6, 13]] >= 0.045)).mean()),
        gripper_min=float(grip.min()),
        gripper_max=float(grip.max()),
        gripper_excursion_max=float(excursions.max()),
        gripper_excursion_mean=float(excursions.mean()),
        gripper_out_of_demonstrated_range=float((excursions > 0).mean()),
        gripper_excursion_gt_1e_4=float((excursions > 1e-4).mean()),
        gripper_excursion_gt_1e_3=float((excursions > 1e-3).mean()),
        gripper_excursion_gt_1e_2=float((excursions > 1e-2).mean()),
    )
    return result


def aggregate_actions(rows):
    if not rows:
        return {"draws": 0}
    result = {"draws": len(rows), "frames": len({row["index"] for row in rows})}
    for key in [
        "flow_native14_normalized",
        "flow_all32",
        "gripper_accuracy_h50",
        "gripper_excursion_mean",
        "gripper_out_of_demonstrated_range",
        "gripper_excursion_gt_1e_4",
        "gripper_excursion_gt_1e_3",
        "gripper_excursion_gt_1e_2",
    ]:
        result[key] = float(np.mean([row[key] for row in rows]))
    result["gripper_min"] = min(row["gripper_min"] for row in rows)
    result["gripper_max"] = max(row["gripper_max"] for row in rows)
    result["gripper_excursion_max"] = max(row["gripper_excursion_max"] for row in rows)
    for scope in ["h50", "first10", "valid_horizon"]:
        mse = np.mean([row[f"native_mse_per_dim_{scope}"] for row in rows], axis=0)
        result[f"native_rmse_per_dim_{scope}"] = np.sqrt(mse).tolist()
        result[f"native_joint_rmse_{scope}"] = float(np.sqrt(mse[np.asarray(JOINT_MASK)].mean()))
        result[f"native_gripper_rmse_{scope}"] = float(np.sqrt(mse[[6, 13]].mean()))
        weights = np.array([row[f"scalar_count_{scope}"] for row in rows])
        weighted = np.average([row[f"native_mse_per_dim_{scope}"] for row in rows], axis=0, weights=weights)
        result[f"native_joint_rmse_{scope}_frame_weighted"] = float(np.sqrt(weighted[np.asarray(JOINT_MASK)].mean()))
    return result
