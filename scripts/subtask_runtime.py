"""Data transfer, bounded decoded-sample cache and scoped training timers."""

from collections import OrderedDict
import dataclasses
import time

import jax
import torch

from openpi.training.subtask_batch import SubtaskTrainingBatch
from openpi.training.subtask_batch import collate_subtask


class TransferBatch(SubtaskTrainingBatch):
    def pin_memory(self):
        return dataclasses.replace(
            self,
            observation=jax.tree.map(lambda x: x.pin_memory(), self.observation),
            **{
                key: getattr(self, key).pin_memory()
                for key in ["actions", "target_ids", "target_mask", "episode_indices", "frame_indices"]
            },
        )

    def to(self, device):
        return dataclasses.replace(
            self,
            observation=jax.tree.map(lambda x: x.to(device, non_blocking=True), self.observation),
            **{
                key: getattr(self, key).to(device, non_blocking=True)
                for key in ["actions", "target_ids", "target_mask", "episode_indices", "frame_indices"]
            },
        )


def collate_transfer(samples):
    batch = collate_subtask(samples)
    return TransferBatch(**{field.name: getattr(batch, field.name) for field in dataclasses.fields(batch)})


class CachedDataset:
    """Cache transformed deterministic samples per worker; collate creates new arrays.

    Use only with the current no-augmentation training transforms. This cache is
    bounded by sample count and never caches model features or supervision in an
    observation. Repeated overfit frames avoid decoding three videos every time.
    """

    def __init__(self, dataset, capacity):
        self.dataset, self.capacity = dataset, capacity
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if index not in self.cache:
            sample = self.dataset[index]
            if self.capacity <= 0:
                return sample
            self.cache[index] = sample
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]


class StepTimer:
    def __init__(self, device, enabled):
        self.device, self.enabled = device, enabled
        self.values = {}

    def now(self):
        if self.enabled and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def add(self, key, begin):
        if self.enabled:
            self.values[key] = self.values.get(key, 0.0) + self.now() - begin


def reduce_timing_max(values, device):
    """Slowest rank for each component; maxima need not belong to the same rank."""
    import torch.distributed as dist

    keys = sorted(values)
    tensor = torch.tensor([values[key] for key in keys], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return {f"{key}_seconds_rank_max": float(value) for key, value in zip(keys, tensor.cpu(), strict=True)}
