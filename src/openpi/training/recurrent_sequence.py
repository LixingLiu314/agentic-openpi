"""Deterministic causal streams over unchanged episode/frame tables."""
import dataclasses

import numpy as np
import torch

from openpi.training.decoded_video_cache import PinnedSubtaskBatch, collate_pinned_subtask


@dataclasses.dataclass
class SequenceBatch(PinnedSubtaskBatch):
    reset_mask: torch.Tensor

    def pin_memory(self):
        result = super().pin_memory()
        result.reset_mask = self.reset_mask.pin_memory()
        return result

    def to(self, device, non_blocking=True):
        result = super().to(device, non_blocking=non_blocking)
        result.reset_mask = self.reset_mask.to(device, non_blocking=non_blocking)
        return result


class SequenceDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, request):
        index, reset = request
        return self.dataset[int(index)], bool(reset)


def collate_sequence(samples):
    batch = collate_pinned_subtask([sample for sample, _ in samples])
    return SequenceBatch(**vars(batch), reset_mask=torch.tensor([reset for _, reset in samples], dtype=torch.bool))


def episode_rows(table):
    episodes = np.asarray(table["episode_index"], dtype=np.int64)
    frames = np.asarray(table["frame_index"], dtype=np.int64)
    timestamps = np.asarray(table["timestamp"], dtype=np.float64)
    result = {}
    for ep in np.unique(episodes):
        indices = np.flatnonzero(episodes == ep)
        indices = indices[np.argsort(frames[indices], kind="stable")]
        if not (np.diff(timestamps[indices]) > 0).all():
            raise ValueError("Episode timestamps must strictly increase")
        result[int(ep)] = (indices, timestamps[indices])
    return result


class EpisodeStreamSampler:
    """Replayable state independent of worker prefetch; each yield is one update.

    Each stream starts at its episode beginning (small initial cadence jitter),
    crosses optimizer blocks, and resets only on episode replacement. Time-major
    ordering lets B and text decoding batch all observations in one forward.
    """
    def __init__(self, rows, *, batch_size=32, unroll=4, steps=5000, start=0, seed=42, rank=0,
                 min_interval=.5, max_interval=1.):
        if batch_size % unroll or not 0 < min_interval <= max_interval:
            raise ValueError("Invalid stream packing/cadence")
        self.rows, self.batch_size, self.unroll = rows, batch_size, unroll
        self.steps, self.start, self.seed, self.rank = steps, start, seed, rank
        self.min_interval, self.max_interval = min_interval, max_interval

    def __iter__(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.rank, 20260909]))
        streams = [None] * (self.batch_size // self.unroll)
        queue = []

        def begin():
            if not queue:
                queue.extend(rng.permutation(sorted(self.rows)).tolist())
            ep = queue.pop()
            indices, times = self.rows[ep]
            pos = min(int(np.searchsorted(times, times[0] + rng.uniform(0, self.min_interval))), len(indices) - 1)
            return ep, pos

        for step in range(self.steps):
            batch = []
            for _ in range(self.unroll):
                for stream, state in enumerate(streams):
                    reset = state is None
                    ep, pos = begin() if reset else state
                    indices, times = self.rows[ep]
                    batch.append((int(indices[pos]), reset))
                    next_pos = int(np.searchsorted(times, times[pos] + rng.uniform(self.min_interval, self.max_interval)))
                    streams[stream] = None if next_pos >= len(indices) else (ep, next_pos)
            if step >= self.start:
                yield batch

    def __len__(self):
        return self.steps - self.start
