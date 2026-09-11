import numpy as np
import pytest

from openpi.training import subtask_comparison as comparison


def test_execution_metric_excludes_unexecuted_and_padded_steps():
    prediction, target = np.zeros((50, 14)), np.zeros((50, 14))
    prediction[3:, :] = 100
    row = {"valid_horizon": 3, "flow_native14_normalized": 0.5}
    values = comparison.window_values(row, prediction, target, 15)
    assert values[1] > 0
    assert values[3] == 0
    assert values[4] == 0
    assert values[5] == 1


def test_paired_bootstrap_preserves_exact_zero_difference_and_seed_variation():
    names = ["G", "B0", "B1", "O", "G-oracle", "G-drop", "G-shuffle"]
    values = np.stack([np.broadcast_to(np.asarray([1, 2, 7, 9])[:, None], (4, 7)) * factor for factor in [1, 2, 3]])
    values = np.repeat(values[:, None], len(names), axis=1)
    metadata = [{"episode": i, "task": "a" if i < 2 else "b"} for i in range(4)]
    report = comparison.clustered_summary(names, [42, 43, 44], values, metadata, repetitions=100)
    metric = report["methods"]["G"]["flow_native14_normalized"]
    assert metric["mean_across_seeds"] == pytest.approx(9.5)
    assert metric["seed_standard_deviation"] == pytest.approx(4.75)
    assert report["paired_comparisons"]["G_minus_B1"]["joint_rmse_execution"]["paired_episode_bootstrap_ci95"] == [0, 0]
    duplicated = comparison.clustered_summary(
        names, [42, 43, 44], np.repeat(values, 3, axis=2), [row for row in metadata for _ in range(3)], repetitions=100
    )
    assert duplicated["methods"] == report["methods"]


def test_pairing_rejects_changed_targets_and_denoising_budget(monkeypatch):
    report = {"split": "val", "split_sha256": "fixed", "norm_sha256": "fixed", "flow_steps": 10, "noise_draws": 1}
    row = {
        "index": 0,
        "draw": 0,
        "episode": 2,
        "frame": 0,
        "task": "a",
        "valid_horizon": 50,
        "crosses_subtask_boundary": False,
        "flow_native14_normalized": 0.1,
    }
    target = np.zeros((50, 14))
    sources = {}
    for source, modes in {
        "g": ["generated", "oracle", "drop", "shuffle"],
        "b0": ["none"],
        "b1": ["none"],
        "o": ["oracle"],
    }.items():
        sources[source] = (
            dict(report),
            {(0, 0, mode): (dict(row, condition_mode=mode), target.copy(), target.copy()) for mode in modes},
        )
    monkeypatch.setattr(comparison, "load_windows", lambda path: sources[path])
    experiments = [dict(seed=42, **{name: name for name in sources})]
    _, _, values, _, _ = comparison.paired_arrays(experiments, 15)
    assert values.shape == (1, 7, 1, 7)
    sources["b1"][1][(0, 0, "none")][2][0, 0] = 1
    with pytest.raises(ValueError, match="targets differ"):
        comparison.paired_arrays(experiments, 15)
    sources["b1"][1][(0, 0, "none")][2][0, 0] = 0
    sources["b1"][0]["flow_steps"] = 2
    with pytest.raises(ValueError, match="different split or normalization"):
        comparison.paired_arrays(experiments, 15)
