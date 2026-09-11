import numpy as np
import pytest
import torch

from openpi.models.subtask_tokenizer import SubtaskTextCodec
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.models_pytorch.subtask_decoder import SubtaskGeneration


def test_prompt_parity_and_only_real_state_dimensions():
    codec = SubtaskTextCodec()
    state = np.linspace(-1.1, 1.1, 14)
    padded = np.r_[state, np.full(18, 12345)]
    text = "Put the eggplant into the box"
    actual, mask = codec.prompts([text], [padded])
    expected, expected_mask = PaligemmaTokenizer(200).tokenize(text, state)
    np.testing.assert_array_equal(actual[0], expected)
    np.testing.assert_array_equal(mask[0], expected_mask)
    conditioned, conditioned_mask = codec.prompts([text], [padded], ["grasp the eggplant"])
    decoded = codec.processor.decode(conditioned[0][conditioned_mask[0]].tolist())
    assert "Subtask: grasp the eggplant" in decoded
    empty, empty_mask = codec.prompts([text], [padded], [""])
    np.testing.assert_array_equal(empty, actual)
    np.testing.assert_array_equal(empty_mask, mask)


def test_targets_keep_case_include_eos_and_handle_missing():
    codec = SubtaskTextCodec()
    text = "Grasp the handle of the lid"
    targets, mask = codec.targets([text, None, ""])
    valid = targets[0][mask[0]].tolist()
    assert valid[-1] == codec.eos_id
    assert codec.bos_id not in valid
    assert codec.processor.decode(valid[:-1]) == text
    assert mask.sum(axis=1).tolist() == [8, 0, 0]
    assert np.all(targets[~mask] == codec.pad_id)
    with pytest.raises(ValueError, match="target exceeds"):
        SubtaskTextCodec(max_subtask_tokens=2).targets([text])


def test_prompt_overflow_is_an_error():
    with pytest.raises(ValueError, match="Prompt exceeds"):
        SubtaskTextCodec(max_prompt_tokens=8).prompts(["Put the eggplant into the box"], [np.zeros(14)])


def test_invalid_decode_uses_empty_condition():
    codec = SubtaskTextCodec()
    ids, mask = codec.targets(["grasp the eggplant", "reach the eggplant", ""])
    generation = SubtaskGeneration(
        torch.from_numpy(ids), torch.from_numpy(mask), torch.tensor([True, False, True]), torch.zeros(3)
    )
    texts, statuses = codec.decode(generation)
    assert texts == ["grasp the eggplant", "", ""]
    assert statuses == ["ok", "truncated", "empty"]
