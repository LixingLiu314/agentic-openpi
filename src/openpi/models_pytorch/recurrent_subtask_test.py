import numpy as np
import torch

from openpi.models_pytorch.recurrent_subtask import RecurrentSubtaskDecoder
from openpi.models_pytorch.subtask_decoder import SubtaskDecoder, SubtaskDecoderConfig
from openpi.training.recurrent_sequence import EpisodeStreamSampler


def fixture(recurrent=True):
    torch.manual_seed(42)
    config = SubtaskDecoderConfig(memory_dim=8, embedding_dim=8, width=8, heads=2,
                                  layers=1, mlp_dim=16, max_tokens=4)
    model = RecurrentSubtaskDecoder(config, recurrent=recurrent).eval()
    memory = torch.randn(8, 6, 8, requires_grad=True)
    mask = torch.ones(8, 6, dtype=torch.bool)
    embedding = torch.randn(19, 8, requires_grad=True)
    return model, memory, mask, embedding


def test_stateless_preserves_original_logits_and_generation():
    model, memory, mask, embedding = fixture(False)
    reference = SubtaskDecoder(model.config).eval()
    reference.load_state_dict(model.state_dict(), strict=True)
    projected, valid, carry = model.compose_sequence(memory, mask, unroll=4)
    ids = torch.tensor([[2, 3, 4]] * 8)
    torch.testing.assert_close(model.logits_projected(ids, projected, valid, embedding),
                               reference(ids, memory, mask, embedding), rtol=0, atol=0)
    actual = model.generate_projected(projected, valid, embedding)
    expected = reference.generate(memory, mask, embedding)
    torch.testing.assert_close(actual.token_ids, expected.token_ids, rtol=0, atol=0)
    assert carry is None


def test_future_observations_do_not_change_past_state():
    model, memory, mask, _ = fixture()
    resets = torch.zeros(8, dtype=torch.bool)
    original, _, _ = model.compose_sequence(memory, mask, resets=resets, unroll=4)
    altered = memory.detach().clone()
    altered[4:] += 20
    changed, _, _ = model.compose_sequence(altered, mask, resets=resets, unroll=4)
    torch.testing.assert_close(original[:4], changed[:4], rtol=0, atol=0)


def test_ce_trains_memory_but_never_backbone_or_embedding():
    model, memory, mask, embedding = fixture()
    projected, valid, carry = model.compose_sequence(memory, mask, unroll=4)
    ids = torch.tensor([[3, 4, 1, 0]] * 8)
    loss = model.loss_projected(ids, ids.ne(0), projected, valid, embedding)
    loss.backward()
    assert memory.grad is None and embedding.grad is None
    assert model.memory_update.weight_hh.grad.abs().sum() > 0
    assert model.memory_attention.in_proj_weight.grad.abs().sum() > 0
    assert model.initial_memory.grad.abs().sum() > 0
    assert carry.shape == (2, 4, 8)


def test_reset_removes_previous_episode_and_isolates_streams():
    model, memory, mask, _ = fixture()
    first = torch.randn(2, 4, 8)
    changed = first.clone()
    changed[0] += 10
    resets = torch.tensor([True, False] + [False] * 6)
    a, _, _ = model.compose_sequence(memory, mask, first, resets, unroll=4)
    b, _, _ = model.compose_sequence(memory, mask, changed, resets, unroll=4)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_sequence_resume_is_independent_of_prefetch():
    rows = {i: (np.arange(i * 100, (i + 1) * 100), np.arange(100) / 30.) for i in range(5)}
    args = dict(batch_size=8, unroll=4, steps=20, seed=42, rank=0)
    full = list(EpisodeStreamSampler(rows, **args))
    resumed = list(EpisodeStreamSampler(rows, start=9, **args))
    assert full[9:] == resumed
    previous = [None, None]
    resets = 0
    for batch in full:
        for j, (index, reset) in enumerate(batch):
            stream = j % 2
            if not reset:
                assert index // 100 == previous[stream] // 100
                assert 15 <= index - previous[stream] <= 31
            else:
                resets += 1
            previous[stream] = index
    assert resets > 2
