import dataclasses

import torch

from openpi.models_pytorch.subtask_decoder import SubtaskDecoder
from openpi.models_pytorch.subtask_decoder import SubtaskDecoderConfig


def fixture():
    torch.manual_seed(19)
    config = SubtaskDecoderConfig(
        memory_dim=12, embedding_dim=10, width=16, heads=4, layers=2, mlp_dim=32, max_tokens=5
    )
    decoder = SubtaskDecoder(config)
    memory = torch.randn(2, 7, 12, requires_grad=True)
    memory_mask = torch.tensor([[True] * 7, [True] * 4 + [False] * 3])
    embedding = torch.nn.Parameter(torch.randn(23, 10))
    return decoder, memory, memory_mask, embedding


def test_ce_updates_only_decoder_and_keeps_tied_output_gradient():
    decoder, memory, memory_mask, embedding = fixture()
    target = torch.tensor([[7, 8, 1, 0], [9, 1, 0, 0]])
    before_embedding = embedding.detach().clone()
    loss = decoder.compute_loss(target, target.ne(0), memory, memory_mask, embedding)
    loss.backward()
    assert memory.grad is None
    assert embedding.grad is None
    assert decoder.output_projection.weight.grad.abs().sum() > 0
    assert decoder.memory_projection.weight.grad.abs().sum() > 0
    assert all(id(parameter) != id(embedding) for parameter in decoder.parameters())
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-3)
    optimizer.step()
    torch.testing.assert_close(embedding, before_embedding, rtol=0, atol=0)


def test_causal_attention_and_memory_padding():
    decoder, memory, mask, embedding = fixture()
    decoder.eval()
    tokens = torch.tensor([[2, 7, 8, 9], [2, 4, 5, 6]])
    output = decoder(tokens, memory, mask, embedding)
    changed = tokens.clone()
    changed[:, -1] = 12
    changed_output = decoder(changed, memory, mask, embedding)
    torch.testing.assert_close(output[:, :-1], changed_output[:, :-1], rtol=0, atol=0)
    altered_memory = memory.detach().clone()
    altered_memory[1, 4:] = 10000
    torch.testing.assert_close(output, decoder(tokens, altered_memory, mask, embedding), rtol=0, atol=0)


def test_masked_labels_and_missing_labels_do_not_contribute():
    decoder, memory, mask, embedding = fixture()
    target = torch.tensor([[7, 1, 0, 0], [0, 0, 0, 0]])
    target_mask = target.ne(0)
    first = decoder.compute_loss(target, target_mask, memory, mask, embedding)
    changed = target.clone()
    changed[~target_mask] = -999
    second = decoder.compute_loss(changed, target_mask, memory, mask, embedding)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    zero = decoder.compute_loss(target, torch.zeros_like(target_mask), memory, mask, embedding)
    zero.backward()
    assert zero.item() == 0
    assert all(parameter.grad is None or not parameter.grad.any() for parameter in decoder.parameters())


def test_batch_eos_and_truncation_restore_mode():
    class ScriptedDecoder(SubtaskDecoder):
        def forward(self, input_ids, memory, memory_mask, embedding_weight):
            logits = torch.full((2, input_ids.shape[1], 23), -50.0)
            logits[0, -1, 1] = 50  # First sample ends immediately.
            logits[1, -1, 4 if input_ids.shape[1] == 1 else 1] = 50
            return logits

    decoder, memory, mask, embedding = fixture()
    scripted = ScriptedDecoder(decoder.config)
    scripted.train()
    result = scripted.generate(memory, mask, embedding)
    assert scripted.training
    assert result.token_ids.tolist() == [[1, 0, 0, 0, 0], [4, 1, 0, 0, 0]]
    assert result.token_mask.sum(dim=1).tolist() == [1, 2]
    assert result.ended.tolist() == [True, True]
    assert not result.mean_log_probability.requires_grad
    truncated = ScriptedDecoder(dataclasses.replace(decoder.config, max_tokens=1)).generate(memory, mask, embedding)
    assert truncated.ended.tolist() == [True, False]
