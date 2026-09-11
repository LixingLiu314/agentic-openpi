"""Sharded AdamW preserving the official model's mixed parameter dtypes."""

import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer


class MixedPrecisionZeroAdamW:
    """ZeRO requires a single dense dtype per instance; partition without casting.

    AdamW acts independently per parameter. Separate dtype instances therefore
    preserve the update rule, while their shared group dictionaries expose the
    same learning-rate control used by the training loop.
    """

    def __init__(self, parameters, **options):
        groups = {}
        for parameter in parameters:
            if parameter.requires_grad:
                groups.setdefault(str(parameter.dtype), []).append(parameter)
        if not groups:
            raise ValueError("Optimizer needs trainable parameters")
        self.dtypes = sorted(groups)
        self.optimizers = [
            ZeroRedundancyOptimizer(groups[dtype], optimizer_class=torch.optim.AdamW, **options)
            for dtype in self.dtypes
        ]
        self.param_groups = [group for optimizer in self.optimizers for group in optimizer.param_groups]

    def zero_grad(self, *, set_to_none=True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def consolidate_state_dict(self, to=0):
        for optimizer in self.optimizers:
            optimizer.consolidate_state_dict(to=to)

    def state_dict(self):
        return {
            "schema": "mixed_dtype_zero_adamw_v1",
            "dtypes": self.dtypes,
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
        }

    def load_state_dict(self, state):
        if state.get("schema") != "mixed_dtype_zero_adamw_v1" or state["dtypes"] != self.dtypes:
            raise ValueError("Optimizer schema or parameter dtype partition changed")
        for optimizer, saved in zip(self.optimizers, state["optimizers"], strict=True):
            optimizer.load_state_dict(saved)
        # Optimizer loading may replace group dictionaries.
        self.param_groups = [group for optimizer in self.optimizers for group in optimizer.param_groups]

    def local_state_dict(self):
        """Checkpoint this rank directly; avoid broadcasting multi-GB pickles."""
        return {
            "schema": "mixed_dtype_zero_adamw_local_v1",
            "dtypes": self.dtypes,
            "rank": dist.get_rank(),
            "world_size": dist.get_world_size(),
            "optimizers": [optimizer.optim.state_dict() for optimizer in self.optimizers],
        }

    def load_local_state_dict(self, state):
        if (
            state.get("schema") != "mixed_dtype_zero_adamw_local_v1"
            or state["dtypes"] != self.dtypes
            or state["rank"] != dist.get_rank()
            or state["world_size"] != dist.get_world_size()
        ):
            raise ValueError("Local optimizer dtype/rank/world-size identity changed")
        for optimizer, saved in zip(self.optimizers, state["optimizers"], strict=True):
            optimizer.optim.load_state_dict(saved)
            for outer, inner in zip(optimizer.param_groups, optimizer.optim.param_groups, strict=True):
                outer.update({key: value for key, value in inner.items() if key != "params"})
        self.param_groups = [group for optimizer in self.optimizers for group in optimizer.param_groups]
