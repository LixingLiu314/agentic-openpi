"""Audited resized RGB cache; action windows and labels still use LeRobot."""
from __future__ import annotations

from collections import OrderedDict
import dataclasses
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
import torch
from openpi_client import image_tools as client_image_tools
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

from openpi.training.local_lerobot_dataset import SplitLeRobotDataset
from openpi.training.stage1_data import load_split
from openpi.training.subtask_batch import SubtaskTrainingBatch, collate_subtask


class CachedVideoLeRobotDataset(SplitLeRobotDataset):
    def __init__(self, *args, cache_root, split_sha256, **kwargs):
        self.cache_root = Path(cache_root)
        self.cache_manifest = json.loads((self.cache_root / "manifest.json").read_text())
        if (self.cache_manifest["schema"], self.cache_manifest["shape"], self.cache_manifest["split_sha256"]) != (
            "piper_rgb224_cache_v1", [224, 224, 3], split_sha256
        ):
            raise ValueError("Decoded image cache contract mismatch")
        if self.cache_manifest["resize_source_sha256"] != hashlib.sha256(Path(client_image_tools.__file__).read_bytes()).hexdigest():
            raise ValueError("Cached resizing must use the exact current PIL input-transform implementation")
        self._open_arrays = OrderedDict()
        super().__init__(*args, **kwargs)
        if self.image_transforms is not None:
            raise ValueError("Cache requires unaugmented raw video observations")
        for episode in self.episodes:
            for camera in self.meta.video_keys:
                record = self.cache_manifest["videos"][f"{episode}:{camera}"]
                source = self.root / self.meta.get_video_file_path(episode, camera)
                stat = source.stat()
                if [stat.st_size, stat.st_mtime_ns] != record["source_stat"]:
                    raise ValueError(f"Video source changed: {source}")
                for name in ("rgb", "pts"):
                    stat = (self.cache_root / record[name]).stat()
                    if [stat.st_size, stat.st_mtime_ns] != record[f"{name}_stat"]:
                        raise ValueError(f"Cache file changed: {record[name]}")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_open_arrays"] = OrderedDict()
        return state

    def _arrays(self, key):
        if key not in self._open_arrays:
            record = self.cache_manifest["videos"][key]
            self._open_arrays[key] = tuple(np.load(self.cache_root / record[n], mmap_mode="r") for n in ("rgb", "pts"))
            if len(self._open_arrays) > 12:
                self._open_arrays.popitem(last=False)
        self._open_arrays.move_to_end(key)
        return self._open_arrays[key]

    def _query_videos(self, query_timestamps, ep_idx):
        result = {}
        for camera, timestamps in query_timestamps.items():
            if len(timestamps) != 1:
                raise ValueError("This version caches only current-frame Piper images")
            rgb, pts = self._arrays(f"{ep_idx}:{camera}")
            # Match LeRobot's float32 torch.cdist / first-minimum timestamp rule.
            distance = np.abs(pts - np.float32(timestamps[0]))
            index = int(distance.argmin())
            if not distance[index] < self.tolerance_s:
                raise ValueError("Cached frame timestamp exceeds LeRobot tolerance")
            result[camera] = np.array(rgb[index], copy=True)
        return result


def create_cached_dataset(data_config, action_horizon, cache_root):
    if data_config.split not in ("train", "val") or data_config.prompt_from_task:
        raise ValueError("Only the existing native Piper train/val data contract is supported")
    metadata = LeRobotDatasetMetadata(data_config.repo_id, root=data_config.local_root)
    return CachedVideoLeRobotDataset(
        data_config.repo_id, root=data_config.local_root,
        episodes=load_split(data_config.split_manifest, data_config.split, data_config.local_root),
        video_backend="pyav", cache_root=cache_root,
        split_sha256=hashlib.sha256(Path(data_config.split_manifest).read_bytes()).hexdigest(),
        delta_timestamps={key: [t / metadata.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys},
    )


@dataclasses.dataclass
class PinnedSubtaskBatch(SubtaskTrainingBatch):
    def pin_memory(self):
        return dataclasses.replace(
            self,
            observation=jax.tree.map(lambda x: x.pin_memory(), self.observation),
            **{name: getattr(self, name).pin_memory() for name in (
                "actions", "target_ids", "target_mask", "episode_indices", "frame_indices")},
        )

    def to(self, device, non_blocking=True):
        return dataclasses.replace(
            self,
            observation=jax.tree.map(lambda x: x.to(device, non_blocking=non_blocking), self.observation),
            **{name: getattr(self, name).to(device, non_blocking=non_blocking) for name in (
                "actions", "target_ids", "target_mask", "episode_indices", "frame_indices")},
        )


def collate_pinned_subtask(samples):
    return PinnedSubtaskBatch(**vars(collate_subtask(samples)))


def worker_init(_):
    torch.set_num_threads(1)
