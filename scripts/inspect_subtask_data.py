"""Create stage contact sheets and measure prompt/target lengths on train+val."""

import argparse
from collections import Counter
import itertools
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image
from PIL import ImageDraw
import pyarrow.parquet as pq

from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.shared import normalize


def inspect(root: Path, assets: Path, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((assets / "split.json").read_text())
    stats = normalize.load(assets)["state"]
    tokenizer = PaligemmaTokenizer(1000)
    selected = set()
    for split in ("train", "val"):
        for task in (0, 1):
            selected.add(
                next(
                    row["episode_index"]
                    for row in manifest["episodes"]
                    if row["episode_index"] in manifest["splits"][split] and row["task_index"] == task
                )
            )
    prompt_lengths, conditional_lengths, target_lengths = Counter(), Counter(), {}
    inspected = []
    for record in manifest["episodes"]:
        episode = record["episode_index"]
        if episode not in manifest["splits"]["train"] + manifest["splits"]["val"]:
            continue
        table = pq.read_table(root / record["parquet_path"], columns=["observation.state", "subtask"]).to_pydict()
        states = np.asarray(table["observation.state"], dtype=np.float32)
        states = (states - stats.q01) / (stats.q99 - stats.q01 + 1e-6) * 2 - 1
        for state, label in zip(states, table["subtask"], strict=True):
            prompt_lengths[int(tokenizer.tokenize(record["task"], state)[1].sum())] += 1
            conditional_lengths[int(tokenizer.tokenize(record["task"] + ", Subtask: " + label, state)[1].sum())] += 1
            if label not in target_lengths:
                target_lengths[label] = len(tokenizer._tokenizer.encode(label)) + 1  # noqa: SLF001
        if episode not in selected:
            continue
        labels = table["subtask"]
        changes = [0] + [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]] + [len(labels)]
        # Midpoints check coarse phase alignment; +/-3-frame pairs examine transitions.
        views = {
            "stages": [(begin + end - 1) // 2 for begin, end in itertools.pairwise(changes)],
            "boundaries": [
                index
                for boundary in changes[1:-1]
                for index in (max(0, boundary - 3), min(len(labels) - 1, boundary + 3))
            ],
        }
        wanted = {index for indices in views.values() for index in indices}
        frames = []
        for video in record["videos"]:
            camera_frames = {}
            with av.open(str(root / video["path"])) as container:
                for index, frame in enumerate(container.decode(video=0)):
                    if index in wanted:
                        camera_frames[index] = frame.to_image().resize((224, 168))
                    if index >= max(wanted):
                        break
            if set(camera_frames) != wanted:
                raise ValueError(f"Could not decode all selected frames: {video['path']}")
            frames.append(camera_frames)
        for view, indices in views.items():
            sheet = Image.new("RGB", (672, 28 + 192 * len(indices)), "white")
            draw = ImageDraw.Draw(sheet)
            draw.text((6, 6), f"Episode {episode} | {record['task']} | {view}", fill="black")
            for row, index in enumerate(indices):
                top = 28 + row * 192
                draw.text((6, top + 4), f"frame {index:04d} | {labels[index]}", fill="black")
                for camera, camera_frames in enumerate(frames):
                    sheet.paste(camera_frames[index], (224 * camera, top + 24))
            sheet.save(output / f"episode_{episode:06d}_{view}.jpg", quality=90)
        inspected.append(
            {"episode": episode, "boundaries": changes, "stage_labels": [labels[index] for index in changes[:-1]]}
        )
    report = {
        "split_sha256": manifest["manifest_sha256"],
        "splits": ["train", "val"],
        "frames": sum(prompt_lengths.values()),
        "prompt_max_tokens": max(prompt_lengths),
        "conditional_prompt_max_tokens": max(conditional_lengths),
        "prompt_budget": 200,
        "prompt_overflow_count": sum(count for length, count in prompt_lengths.items() if length > 200),
        "conditional_overflow_count": sum(count for length, count in conditional_lengths.items() if length > 200),
        "target_tokens_including_eos": target_lengths,
        "contact_sheets": inspected,
        "duplicate_source_groups": sum(
            count > 1 for count in Counter(row["source_path"] for row in manifest["episodes"]).values()
        ),
        "duplicate_trajectory_groups": sum(
            count > 1 for count in Counter(row["trajectory_sha256"] for row in manifest["episodes"]).values()
        ),
        "duplicate_video_triple_groups": sum(
            count > 1 for count in Counter(row["video_triple_sha256"] for row in manifest["episodes"]).values()
        ),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("Datasets/eggplant_potato_gripper_binary"))
    parser.add_argument("--assets", type=Path, default=Path("assets/pi05_piper_stage1/eggplant_potato"))
    parser.add_argument("--output", type=Path, default=Path("logs/pi05_subtask_stage1/data_inspection"))
    args = parser.parse_args()
    inspect(args.root, args.assets, args.output)
