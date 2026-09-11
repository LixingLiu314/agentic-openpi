"""Versioned episode splits for the local pi05 subtask experiment."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_digest(manifest: dict) -> str:
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def grouped_split(records: list[dict], seed: int, held_out_per_task: int = 10) -> dict[str, list[int]]:
    """Keep shared source paths, identical trajectories and identical video triples together.

    Collection sessions are not present in this dataset's metadata. Grouping does
    not establish visual novelty; this limitation is recorded in the manifest.
    """
    parent = list(range(len(records)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen = {}
    for index, record in enumerate(records):
        for field in ("source_path", "trajectory_sha256", "video_triple_sha256"):
            if value := record.get(field):
                key = field, value
                if key in seen:
                    parent[find(index)] = find(seen[key])
                else:
                    seen[key] = index
    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[find(index)].append(record)
    tasks = defaultdict(list)
    for group in groups.values():
        task_ids = {record["task_index"] for record in group}
        if len(task_ids) != 1:
            raise ValueError("A duplicate/source group has conflicting tasks; audit it before splitting")
        tasks[next(iter(task_ids))].append(sorted(record["episode_index"] for record in group))
    rng = np.random.default_rng(seed)
    splits = {"train": [], "val": [], "test": []}
    for task in sorted(tasks):
        task_groups = sorted(tasks[task])
        rng.shuffle(task_groups)
        if len(task_groups) < 3:
            raise ValueError("Need at least three independent groups per task")
        counts = {"val": 0, "test": 0}
        for index, group in enumerate(task_groups):
            # Reserve at least one group for training, even if duplicates change counts.
            if index == len(task_groups) - 1:
                split = "train"
            elif counts["val"] < held_out_per_task:
                split = "val"
            elif counts["test"] < held_out_per_task:
                split = "test"
            else:
                split = "train"
            splits[split].extend(group)
            if split in counts:
                counts[split] += len(group)
    return {key: sorted(value) for key, value in splits.items()}


def load_split(path: str | Path, split: str, root: str | Path | None = None) -> list[int]:
    manifest = json.loads(Path(path).read_text())
    if manifest["schema_version"] != SCHEMA_VERSION or manifest["manifest_sha256"] != manifest_digest(manifest):
        raise ValueError("Unsupported or modified split manifest")
    all_ids = [episode for ids in manifest["splits"].values() for episode in ids]
    expected_ids = [record["episode_index"] for record in manifest["episodes"]]
    if len(set(all_ids)) != len(all_ids) or sorted(all_ids) != sorted(expected_ids):
        raise ValueError("Episode split overlaps or has missing/extra episodes")
    if split not in {"train", "val", "test"} or not manifest["splits"][split]:
        raise ValueError(f"Missing or empty split: {split}")
    if root is not None:
        root = Path(root)
        for record in manifest["episodes"]:
            if sha256_file(root / record["parquet_path"]) != record["parquet_sha256"]:
                raise ValueError(f"Dataset changed: episode {record['episode_index']}")
            for video in record["videos"]:
                if not (root / video["path"]).is_file():
                    raise FileNotFoundError(root / video["path"])
    return list(manifest["splits"][split])


def action_windows(actions: np.ndarray, states: np.ndarray, horizon: int) -> np.ndarray:
    """Match LeRobot's endpoint-repeat chunks, with current-state joint deltas."""
    if horizon < 1 or actions.shape != states.shape or actions.shape[-1] != 14:
        raise ValueError("Expected matching [T, 14] arrays and a positive horizon")
    index = np.minimum(np.arange(len(actions))[:, None] + np.arange(horizon), len(actions) - 1)
    chunks = actions[index].copy()
    mask = np.array([True] * 6 + [False] + [True] * 6 + [False])
    chunks[..., mask] -= states[:, None, mask]
    return chunks
