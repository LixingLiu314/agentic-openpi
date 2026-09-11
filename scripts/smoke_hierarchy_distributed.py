"""Two-rank test of stage transitions, disjoint optimizers and exact resume."""

import argparse
import contextlib
from pathlib import Path

import safetensors.torch
import torch
import torch.distributed as dist

from openpi.training import hierarchy_training as training
from openpi.training.hierarchy_training_test import TinyHierarchy


def main(output):
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    rank, world = training.distributed_context()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        (output / "assets").mkdir()
        (output / "assets/split.json").write_text("{}")
    dist.barrier()
    parent = None
    for stage in ["m1", "m2", "m3"]:
        torch.manual_seed(42)
        model = TinyHierarchy(stage=stage)
        if parent:
            safetensors.torch.load_model(model, parent / "model.safetensors", strict=True)
        ddp = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=True, gradient_as_bucket_view=True
        )
        optimizers = training.BranchOptimizers(model, stage, 0.01, 0.001)
        if parent:
            state = torch.load(training.checkpoint_state_path(parent, rank, world), weights_only=False)
            optimizers.load_state_dict(state["branch_optimizers"], transition=True)
        torch.manual_seed(100 + rank)

        def update(ddp, optimizers, stage):
            optimizers.zero_grad()
            loss_value = 0.0
            for micro in range(2):
                sync = ddp.no_sync() if micro == 0 else contextlib.nullcontext()
                with sync:
                    subtask, action = ddp(torch.randn(3, 4))
                    loss = subtask + (action if stage != "m1" else 0)
                    (loss / 2).backward()
                    loss_value += float(loss.detach()) / 2
            optimizers.step()
            optimizers.zero_grad()
            return loss_value

        update(ddp, optimizers, stage)
        directory = output / stage
        if rank == 0:
            directory.mkdir()
        dist.barrier()
        checkpoint = training.save_checkpoint(
            ddp,
            optimizers,
            directory,
            1,
            {"stage": stage},
            None,
            {"subtask": 1, "action": int(stage != "m1")},
            output / "assets",
        )
        expected_loss = update(ddp, optimizers, stage)
        expected = {key: value.clone() for key, value in model.state_dict().items()}
        safetensors.torch.load_model(model, checkpoint / "model.safetensors", strict=True)
        state = torch.load(training.checkpoint_state_path(checkpoint, rank, world), weights_only=False)
        optimizers.load_state_dict(state["branch_optimizers"])
        training.restore_random_state(state["rng"])
        assert update(ddp, optimizers, stage) == expected_loss
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
        parent = checkpoint
        print(f"rank={rank} stage={stage}: branch optimizer transition and exact next-update resume passed", flush=True)
        del ddp, optimizers, model
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
