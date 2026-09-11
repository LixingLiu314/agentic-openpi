import numpy as np
import pytest

from openpi.training.subtask_evaluation import native_action_metrics
from openpi.training.subtask_evaluation import shuffled_conditions
from openpi.training.subtask_evaluation import transition_metrics


def episode(prediction):
    return [
        {
            "index": i,
            "episode": 4,
            "frame": i,
            "episode_length": 30,
            "task": "task",
            "label": "a" if i < 10 else "b" if i < 20 else "c",
            "prediction": p,
        }
        for i, p in enumerate(prediction)
    ]


def test_transitions_early_late_and_jitter():
    rows = episode(["a"] * 8 + ["b"] * 4 + ["a", "b"] + ["b"] * 8 + ["c"] * 8)
    metrics = transition_metrics(rows, tolerance=2)
    assert [row["error_frames"] for row in metrics["switches"]] == [-2, 2]
    assert metrics["within_tolerance_fraction_all_gt"] == 1
    assert metrics["unmatched_changes"] == 2
    assert metrics["early_switches"] == metrics["late_switches"] == 1


def test_missing_switches_and_dense_requirement():
    rows = episode(["a"] * 30)
    metrics = transition_metrics(rows)
    assert metrics["missed_switches"] == 2
    assert metrics["within_tolerance_fraction_all_gt"] == 0
    assert metrics["signed_error_mean_matched"] is None
    with pytest.raises(ValueError, match="every frame"):
        transition_metrics(rows[::2])


def test_intervention_ignores_ground_truth_and_stays_within_task():
    rows = [
        {"index": i, "task": t, "prediction": p, "label": "private"}
        for i, (t, p) in enumerate([("a", "x"), ("a", "y"), ("b", "z")])
    ]
    first = shuffled_conditions(rows)
    for row in rows:
        row["label"] = "changed"
    assert shuffled_conditions(rows) == first == {0: "y", 1: "x", 2: "z"}


def test_excursion_magnitude_and_valid_tail():
    native, target = np.zeros((50, 14)), np.zeros((50, 14))
    native[:, 6], native[:, 13] = -0.002, 0.091
    native[3:, 0] = 10
    metrics = native_action_metrics(native, target, valid_horizon=3)
    assert metrics["native_mse_per_dim_first10"][0] == 0
    assert metrics["native_mse_per_dim_h50"][0] > 0
    assert metrics["gripper_excursion_max"] == pytest.approx(0.002)
    assert metrics["gripper_excursion_gt_1e_4"] == 1
    assert metrics["gripper_excursion_gt_1e_2"] == 0
