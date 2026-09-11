import numpy as np
import pytest

from openpi.training.subtask_batch import BalancedSubtaskSampler
from openpi.training.subtask_batch import predicted_condition_ratio
from openpi.training.subtask_batch import select_action_conditions


def test_balanced_sampler_rank_slices_resume_and_missing_labels():
    labels = ["frequent"] * 999 + ["rare", None, ""]
    options = {"labels": labels, "accumulation": 2, "steps": 1000, "start": 0, "seed": 17}
    global_batches = list(BalancedSubtaskSampler(batch_size=8, **options))
    rank0 = list(BalancedSubtaskSampler(batch_size=4, rank=0, world_size=2, **options))
    rank1 = list(BalancedSubtaskSampler(batch_size=4, rank=1, world_size=2, **options))
    assert global_batches == [first + second for first, second in zip(rank0, rank1, strict=True)]
    resumed = list(BalancedSubtaskSampler(batch_size=4, rank=1, world_size=2, **{**options, "start": 673}))
    assert resumed == rank1[673 * 2 :]
    drawn = np.asarray(global_batches).flatten()
    assert drawn.max() == 999
    assert 0.47 < (drawn == 999).mean() < 0.53
    with pytest.raises(ValueError, match="labeled"):
        BalancedSubtaskSampler([None, ""], 1, 1, 1, 0, 42)


def test_generated_failure_never_selects_ground_truth():
    assert select_action_conditions(
        ["GT secret", "GT B", "GT C", None],
        ["", "pred B", "pred C", "pred D"],
        [True, False, True, False],
        [False, False, True, False],
    ) == ("", "GT B", "", "")
    with pytest.raises(ValueError, match="sizes"):
        select_action_conditions(["A"], [], [True], [False])


def test_curriculum_reserves_final_quarter_for_full_predictions():
    for total in [4, 100, 101, 1000]:
        ratios = [predicted_condition_ratio(step, total) for step in range(total)]
        assert ratios[0] == 0.25
        assert ratios[-1] == 1
        assert ratios == sorted(ratios)
        assert ratios.count(1.0) >= total // 4
    with pytest.raises(ValueError, match="schedule"):
        predicted_condition_ratio(10, 10)
    with pytest.raises(ValueError, match="schedule"):
        predicted_condition_ratio(0, 3)
