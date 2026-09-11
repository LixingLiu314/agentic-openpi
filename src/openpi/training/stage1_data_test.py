import json

import numpy as np
import pytest

from openpi import transforms
from openpi.policies.piper_policy import CAMERA_MAP
from openpi.policies.piper_policy import JOINT_MASK
from openpi.policies.piper_policy import PiperInputs
from openpi.policies.piper_policy import PiperOutputs
from openpi.shared.normalize import NormStats
from openpi.training.stage1_data import action_windows
from openpi.training.stage1_data import grouped_split
from openpi.training.stage1_data import load_split
from openpi.training.stage1_data import manifest_digest


def test_native_piper_roundtrip_and_no_supervision_leak():
    rng = np.random.default_rng(7)
    raw_state = rng.normal(size=14).astype(np.float32)
    raw_actions = rng.normal(size=(50, 14)).astype(np.float32)
    raw_actions[:, [6, 13]] = rng.choice([0, 0.09], size=(50, 2))
    raw = {
        "state": raw_state.copy(),
        "actions": raw_actions.copy(),
        "images": {key: rng.integers(0, 256, (3, 12, 16), dtype=np.uint8) for key in CAMERA_MAP},
        "prompt": "Put the eggplant into the box",
        "subtask": "SECRET",
        "frame_index": 77,
    }
    data = PiperInputs()(raw)
    assert set(data) == {"state", "actions", "image", "image_mask", "prompt"}
    assert all(data["image_mask"].values())
    assert all(image.shape == (12, 16, 3) for image in data["image"].values())
    delta = transforms.DeltaActions(JOINT_MASK)(data)
    np.testing.assert_array_equal(delta["actions"][:, [6, 13]], raw_actions[:, [6, 13]])
    stats = {
        key: NormStats(mean=np.zeros(14), std=np.ones(14), q01=np.full(14, -3), q99=np.full(14, 3))
        for key in ("state", "actions")
    }
    data = transforms.Normalize(stats, use_quantiles=True)(delta)
    data = transforms.PadStatesAndActions(32)(data)
    data = transforms.Unnormalize(stats, use_quantiles=True)(data)
    data = transforms.AbsoluteActions(JOINT_MASK)(data)
    result = PiperOutputs()(data)
    np.testing.assert_allclose(result["actions"], raw_actions, atol=1e-6)
    # Transforming a window must not mutate a dataset's cached absolute actions.
    np.testing.assert_array_equal(raw["actions"], raw_actions)
    np.testing.assert_array_equal(raw["state"], raw_state)


def test_missing_camera_is_explicit():
    with pytest.raises(KeyError):
        PiperInputs()({"state": np.zeros(14), "images": {}})


def test_action_windows_repeat_only_within_episode():
    actions = np.arange(3 * 14, dtype=np.float32).reshape(3, 14)
    states = np.ones((3, 14), dtype=np.float32)
    windows = action_windows(actions, states, 4)
    np.testing.assert_array_equal(windows[1, :, 0], [13, 27, 27, 27])
    np.testing.assert_array_equal(windows[1, :, 6], [20, 34, 34, 34])
    np.testing.assert_array_equal(actions[:, 0], [0, 14, 28])


def test_split_is_deterministic_balanced_and_keeps_duplicates_together():
    records = [{"episode_index": i, "task_index": i // 99, "source_path": str(i)} for i in range(198)]
    splits = grouped_split(records, 42)
    assert splits == grouped_split(records, 42)
    assert {key: len(value) for key, value in splits.items()} == {"train": 158, "val": 20, "test": 20}
    for ids in splits.values():
        assert sum(i < 99 for i in ids) == len(ids) // 2
    records[0]["source_path"] = records[1]["source_path"]
    duplicate_splits = grouped_split(records, 42)
    assert next(key for key, ids in duplicate_splits.items() if 0 in ids) == next(
        key for key, ids in duplicate_splits.items() if 1 in ids
    )


def test_split_refuses_overlap_and_tampering(tmp_path):
    manifest = {
        "schema_version": 1,
        "splits": {"train": [0], "val": [1], "test": [2]},
        "episodes": [{"episode_index": i} for i in range(3)],
    }
    path = tmp_path / "split.json"
    manifest["manifest_sha256"] = manifest_digest(manifest)
    path.write_text(json.dumps(manifest))
    assert load_split(path, "train") == [0]
    manifest["splits"]["val"] = [0]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="modified"):
        load_split(path, "train")
    manifest["manifest_sha256"] = manifest_digest(manifest)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="overlaps"):
        load_split(path, "train")


def test_sparse_episode_query_uses_local_bounds():
    import torch

    from openpi.training.local_lerobot_dataset import SplitLeRobotDataset

    dataset = SplitLeRobotDataset.__new__(SplitLeRobotDataset)
    dataset.episode_positions = {2: 0, 10: 1}
    dataset.episode_data_index = {"from": torch.tensor([0, 3]), "to": torch.tensor([3, 5])}
    dataset.delta_indices = {"action": list(range(50))}
    indices, padding = dataset._get_query_indices(4, 10)  # noqa: SLF001
    assert indices["action"] == [4] * 50
    assert padding["action_is_pad"].tolist() == [False] + [True] * 49
    indices, padding = dataset._get_query_indices(1, 2)  # noqa: SLF001
    assert indices["action"] == [1] + [2] * 49
    assert padding["action_is_pad"].tolist() == [False, False] + [True] * 48
