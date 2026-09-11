"""Two-rank CPU check of DDP accumulation, sharded AdamW and rank RNG restore."""

import argparse
import contextlib
from pathlib import Path

import safetensors.torch
import torch
import torch.distributed as dist
from train_subtask_pytorch import restore_random_state
from train_subtask_pytorch import save_checkpoint

from openpi.training.stage1_optimizer import MixedPrecisionZeroAdamW


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
        self.precise = torch.nn.Linear(4, 2).double()
        self.unused = torch.nn.Parameter(torch.ones(3))

    def forward(self, inputs):
        return self.linear(inputs) + self.precise(inputs.double()).float()


def main(output):
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(17)
    model = Network()
    reference = Network()
    reference.load_state_dict(model.state_dict())
    ddp = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True, gradient_as_bucket_view=True)
    optimizer = MixedPrecisionZeroAdamW(ddp.parameters(), lr=0.01)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        (output / "assets").mkdir()
        (output / "assets" / "split.json").write_text("{}")
    dist.barrier()

    def update(current, optim, distributed):
        optim.zero_grad(set_to_none=True)
        inputs = torch.linspace(-1, 1, world * 4 * 4).reshape(2, world, 2, 4)
        for micro in range(2):
            sync = current.no_sync() if distributed and micro == 0 else contextlib.nullcontext()
            with sync:
                x = inputs[micro, rank] if distributed else inputs[micro].reshape(-1, 4)
                (current(x).square().mean() / 2).backward()
        torch.nn.utils.clip_grad_norm_(current.parameters(), 1.0)
        optim.step()

    update(ddp, optimizer, distributed=True)
    update(reference, reference_optimizer, distributed=False)
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    torch.manual_seed(100 + rank)
    checkpoint = save_checkpoint(ddp, optimizer, output, 1, {"world_size": world}, {"step": 1}, output / "assets")
    expected_random = torch.rand(4)
    update(ddp, optimizer, distributed=True)
    expected_state = {key: value.clone() for key, value in model.state_dict().items()}
    safetensors.torch.load_model(model, checkpoint / "model.safetensors", strict=True)
    state = torch.load(checkpoint / f"training_rank_{rank:03d}.pt", weights_only=False)
    optimizer.load_local_state_dict(state["optimizer"])
    restore_random_state(state["rng"])
    torch.testing.assert_close(torch.rand(4), expected_random, rtol=0, atol=0)
    update(ddp, optimizer, distributed=True)
    for key, actual in model.state_dict().items():
        torch.testing.assert_close(actual, expected_state[key], rtol=0, atol=0)
    print(f"rank={rank} PASS: global-batch gradient, sharded AdamW, checkpoint, rank RNG and next update", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
