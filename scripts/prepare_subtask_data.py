"""Audit the local dataset, split by episode, and compute training-only statistics.

Run from the repository root. Source data is never rewritten. An existing output
directory is refused so an experiment's manifest cannot silently change.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

from openpi.policies.piper_policy import CAMERA_MAP
from openpi.shared import normalize
from openpi.training.stage1_data import SCHEMA_VERSION
from openpi.training.stage1_data import action_windows
from openpi.training.stage1_data import grouped_split
from openpi.training.stage1_data import manifest_digest
from openpi.training.stage1_data import sha256_file


def prepare(root: Path, output: Path, horizon: int = 50, seed: int = 42):
    if output.exists():
        raise FileExistsError(f"Refusing to replace experiment assets: {output}")
    info = json.loads((root / "meta/info.json").read_text())
    sources = {
        row["episode_index"]: row for row in map(json.loads, (root / "meta/episodes.jsonl").read_text().splitlines())
    }
    records = []
    arrays = {}
    labels = Counter()
    complete_windows = boundary_windows = 0
    for episode, source in sorted(sources.items()):
        chunk = episode // info["chunks_size"]
        parquet = info["data_path"].format(episode_chunk=chunk, episode_index=episode)
        table = pq.read_table(root / parquet).to_pydict()
        states = np.asarray(table["observation.state"], dtype=np.float32)
        actions = np.asarray(table["action"], dtype=np.float32)
        subtasks = table["subtask"]
        length = len(states)
        if states.shape != (length, 14) or actions.shape != states.shape:
            raise ValueError(f"Invalid state/action shape in episode {episode}")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError(f"Nonfinite state/action in episode {episode}")
        if table["frame_index"] != list(range(length)) or set(table["episode_index"]) != {episode}:
            raise ValueError(f"Episode/frame alignment error: {episode}")
        if len(set(table["task_index"])) != 1 or len(set(table["task"])) != 1:
            raise ValueError(f"Inconsistent task in episode {episode}")
        if not all(isinstance(label, str) and label.strip() for label in subtasks):
            raise ValueError(f"Missing subtask in episode {episode}")
        if not np.allclose(table["timestamp"], np.arange(length) / info["fps"], atol=1e-4):
            raise ValueError(f"Timestamp mismatch: {episode}")
        videos = []
        for camera in CAMERA_MAP:
            video_path = info["video_path"].format(
                episode_chunk=chunk, episode_index=episode, video_key=f"observation.images.{camera}"
            )
            with av.open(str(root / video_path)) as video:
                stream = video.streams.video[0]
                if stream.frames != length or float(stream.average_rate) != info["fps"]:
                    raise ValueError(f"Video header/frame count does not align: {video_path}")
                next(video.decode(stream))  # Confirm the stream is decodable, beyond file existence.
            videos.append({"path": video_path, "sha256": sha256_file(root / video_path)})
        trajectory = hashlib.sha256(states.tobytes() + actions.tobytes()).hexdigest()
        video_hash = hashlib.sha256("".join(row["sha256"] for row in videos).encode()).hexdigest()
        records.append(
            {
                "episode_index": episode,
                "task_index": table["task_index"][0],
                "task": table["task"][0],
                "length": length,
                "source_path": source.get("source_path"),
                "parquet_path": parquet,
                "parquet_sha256": sha256_file(root / parquet),
                "trajectory_sha256": trajectory,
                "video_triple_sha256": video_hash,
                "videos": videos,
            }
        )
        arrays[episode] = states, actions
        labels.update(subtasks)
        changes = np.r_[0, np.cumsum(np.array(subtasks[1:]) != np.array(subtasks[:-1]))]
        count = max(0, length - horizon + 1)
        complete_windows += count
        boundary_windows += int(np.sum(changes[horizon - 1 :] != changes[:count])) if count else 0
    splits = grouped_split(records, seed)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "fps": info["fps"],
        "action_horizon": horizon,
        "action_encoding": "12 joint deltas from current state; grippers absolute native values",
        "tail_policy": "repeat final action within the same episode (LeRobot default)",
        "grouping": ["source_path", "exact state/action trajectory", "exact three-camera video hashes"],
        "limitations": [
            "No collection-session IDs: session separation cannot be established.",
            "Exact duplicates are grouped; near-duplicate scenes still require visual review.",
            "Native controller units must be verified before a physical robot rollout.",
        ],
        "splits": splits,
        "episodes": records,
    }
    manifest["manifest_sha256"] = manifest_digest(manifest)
    state_stats, action_stats = normalize.RunningStats(), normalize.RunningStats()
    for episode in splits["train"]:
        states, actions = arrays[episode]
        state_stats.update(states)
        action_stats.update(action_windows(actions, states, horizon))
    stats = {"state": state_stats.get_statistics(), "actions": action_stats.get_statistics()}
    output.mkdir(parents=True)
    (output / "split.json").write_text(json.dumps(manifest, indent=2) + "\n")
    normalize.save(output, stats)
    report = {
        "manifest_sha256": manifest["manifest_sha256"],
        "norm_stats_sha256": sha256_file(output / "norm_stats.json"),
        "episodes": len(records),
        "frames": sum(row["length"] for row in records),
        "videos": 3 * len(records),
        "split_episodes": {key: len(ids) for key, ids in splits.items()},
        "split_frames": {key: sum(len(arrays[episode][0]) for episode in ids) for key, ids in splits.items()},
        "labels": dict(labels),
        "complete_action_windows": complete_windows,
        "boundary_windows": boundary_windows,
        "boundary_fraction": boundary_windows / complete_windows,
        "norm_training_episodes": splits["train"],
        "normalization": "OpenPI RunningStats, 5000 histogram bins; all training frames and H-step endpoint-repeat action chunks",
        "degenerate_quantile_dims": {
            key: np.where(value.q99 - value.q01 < 1e-6)[0].tolist() for key, value in stats.items()
        },
        "limitations": manifest["limitations"],
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in {"norm_training_episodes", "labels"}}, indent=2
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.root, args.output, args.horizon, args.seed)
