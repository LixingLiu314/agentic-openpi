"""Training-only label sidecar and reproducible balanced subtask sampling."""

import dataclasses

import jax
import numpy as np
import torch

from openpi import transforms
from openpi.models.model import Observation
from openpi.models.subtask_tokenizer import SubtaskTextCodec


@dataclasses.dataclass
class SubtaskTrainingBatch:
    observation: Observation
    actions: torch.Tensor
    global_prompts: tuple[str, ...]
    labels: tuple[str | None, ...]
    target_ids: torch.Tensor
    target_mask: torch.Tensor
    episode_indices: torch.Tensor
    frame_indices: torch.Tensor

    def to(self, device):
        return dataclasses.replace(
            self,
            observation=jax.tree.map(lambda value: value.to(device), self.observation),
            **{
                key: getattr(self, key).to(device)
                for key in ["actions", "target_ids", "target_mask", "episode_indices", "frame_indices"]
            },
        )


class SubtaskTrainingDataset:
    """Extract labels before transforms and keep them outside Observation.

    Read a frame once, so video decoding and action-window lookup are shared by
    the observation and its sidecar. Only the current frame's label is exposed.
    """

    def __init__(self, raw_dataset, data_config, max_prompt_tokens=200, max_subtask_tokens=16):
        if data_config.norm_stats is None:
            raise ValueError("Audited training statistics are required")
        self.raw_dataset = raw_dataset
        self.codec = SubtaskTextCodec(max_prompt_tokens, max_subtask_tokens)
        self.transform = transforms.compose(
            [
                *data_config.repack_transforms.inputs,
                *data_config.data_transforms.inputs,
                transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.model_transforms.inputs,
            ]
        )

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, index):
        raw = self.raw_dataset[index]
        label = raw.get("subtask")
        prompt = raw["task"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A nonempty global task is required")
        if label is not None and not isinstance(label, str):
            raise ValueError("Subtask labels must be text or None")
        ids, mask = self.codec.targets([label])
        sidecar = {
            "global_prompt": prompt,
            "label": label,
            "target_ids": ids[0],
            "target_mask": mask[0],
            "episode_index": int(raw["episode_index"]),
            "frame_index": int(raw["frame_index"]),
        }
        transformed = self.transform(raw)
        if any(key in transformed for key in ["subtask", "target_ids", "target_mask", "label"]):
            raise ValueError("Subtask supervision leaked into model observations")
        return {"model": transformed, "supervision": sidecar}

    def labels_without_video(self):
        """Access the scalar table only; no image/action-window work for sampling."""
        table = self.raw_dataset.hf_dataset
        return list(table["subtask"])


def collate_subtask(samples):
    models = [sample["model"] for sample in samples]
    batch = jax.tree.map(lambda *values: torch.as_tensor(np.stack(values)), *models)
    sidecars = [sample["supervision"] for sample in samples]
    return SubtaskTrainingBatch(
        Observation.from_dict(batch),
        batch["actions"].float(),
        tuple(item["global_prompt"] for item in sidecars),
        tuple(item["label"] for item in sidecars),
        torch.as_tensor(np.stack([item["target_ids"] for item in sidecars])),
        torch.as_tensor(np.stack([item["target_mask"] for item in sidecars])),
        torch.tensor([item["episode_index"] for item in sidecars]),
        torch.tensor([item["frame_index"] for item in sidecars]),
    )


class BalancedSubtaskSampler:
    """Choose a labeled subtask uniformly, then a training frame in that bucket.

    Draw the global batch deterministically and slice by rank. Resume depends on
    completed optimizer steps, independent of DataLoader worker prefetch.
    """

    def __init__(self, labels, batch_size, accumulation, steps, start, seed, rank=0, world_size=1, *, tasks=None):
        if tasks is not None and len(tasks) != len(labels):
            raise ValueError("Task/label counts differ")
        buckets = {}
        for index, label in enumerate(labels):
            if label is not None and label.strip():
                key = label if tasks is None else (tasks[index], label)
                buckets.setdefault(key, []).append(index)
        if not buckets:
            raise ValueError("Balanced subtask training needs labeled training frames")
        self.buckets = [np.asarray(buckets[label], dtype=np.int64) for label in sorted(buckets)]
        self.batch_size, self.accumulation = batch_size, accumulation
        self.steps, self.start, self.seed = steps, start, seed
        self.rank, self.world_size = rank, world_size

    def __iter__(self):
        for step in range(self.start, self.steps):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, step, 1]))
            for _ in range(self.accumulation):
                classes = rng.integers(len(self.buckets), size=self.batch_size * self.world_size)
                indices = np.asarray([rng.choice(self.buckets[int(label)]) for label in classes])
                yield indices.reshape(self.world_size, self.batch_size)[self.rank].tolist()

    def __len__(self):
        return (self.steps - self.start) * self.accumulation


def predicted_condition_ratio(completed_m3_steps, total_m3_steps):
    """Four equal phases; the final quarter uses only generated conditions."""
    if total_m3_steps < 4 or not 0 <= completed_m3_steps < total_m3_steps:
        raise ValueError("Invalid M3 schedule position")
    return (0.25, 0.5, 0.75, 1.0)[min(3, completed_m3_steps * 4 // total_m3_steps)]


def select_action_conditions(labels, predicted, use_prediction, drop_condition):
    """A failed generation remains empty; never fall back to ground truth."""
    if not len(labels) == len(predicted) == len(use_prediction) == len(drop_condition):
        raise ValueError("Condition batch sizes differ")
    return tuple(
        "" if drop else prediction if use else (label or "")
        for label, prediction, use, drop in zip(labels, predicted, use_prediction, drop_condition, strict=True)
    )
