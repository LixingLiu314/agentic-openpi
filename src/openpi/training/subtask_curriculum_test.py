import pytest

from openpi.training.subtask_curriculum import generated_only_updates
from openpi.training.subtask_curriculum import selection_eligible


def test_selection_excludes_checkpoints_before_full_predicted_budget():
    assert not selection_eligible("m3", 3400, 3500, 3900)
    assert selection_eligible("m3", 3407, 3500, 3907)
    assert selection_eligible("m3", 3500, 3500, 4000)
    assert generated_only_updates(3500, 3500) == 875
    assert not selection_eligible("m3", 3, 4, 4)
    assert selection_eligible("m3", 4, 4, 5)


def test_schedule_uses_ceiling_for_nondivisible_budget():
    assert generated_only_updates(4, 5) == 0
    assert generated_only_updates(5, 5) == 1
    with pytest.raises(ValueError, match="Invalid M3"):
        generated_only_updates(6, 5)
    with pytest.raises(ValueError, match="Action update counter"):
        selection_eligible("m3", 4, 4, 3)
