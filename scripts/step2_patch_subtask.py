"""
Step 2 — Patch subtask labels into parquet from source HDF5 files.
Patch aloha_object_lerobot dataset in-place:
  1. Add 'subtask' column to every parquet from observations/subtask in the source HDF5.
  2. Update meta/info.json to register the new subtask feature.

Single task dataset — no label replacement or task renaming needed.
"""

import json
import pathlib

import h5py
import pandas as pd

DATASET_DIR = pathlib.Path("/media/raid/workspace/surongpeng/ws_lixing/agentic-openpi/Datasets/avoid_obstable/aloha_banana_obstacle")
HDF5_ROOT   = pathlib.Path("/media/raid/workspace/surongpeng/ws_lixing/agentic-openpi/Datasets/avoid_obstable/put_the_banana_on_the_plate")


def hdf5_path(source_path: str) -> pathlib.Path:
    # source_path: /home/agilex/data/aloha_pipeline/put_the_correct_object_into_the_hole/episode_N.hdf5
    # HDF5 files are directly in HDF5_ROOT (not in a task subdirectory)
    return HDF5_ROOT / pathlib.Path(source_path).name


def main() -> None:
    meta_dir = DATASET_DIR / "meta"

    # --- load episodes.jsonl ---
    episodes = []
    with open(meta_dir / "episodes.jsonl") as f:
        for line in f:
            episodes.append(json.loads(line))

    errors = []
    for ep in episodes:
        ep_idx  = ep["episode_index"]
        chunk   = ep_idx // 1000
        parquet = DATASET_DIR / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
        hdf5    = hdf5_path(ep["source_path"])

        if not hdf5.exists():
            errors.append(f"  Episode {ep_idx}: HDF5 not found: {hdf5}")
            continue

        # read subtask from HDF5
        with h5py.File(hdf5, "r") as f:
            raw = f["observations/subtask"][:]
        subtasks = [s.decode() if isinstance(s, bytes) else str(s) for s in raw]

        # load parquet
        df = pd.read_parquet(parquet)

        if len(df) != len(subtasks):
            errors.append(
                f"  Episode {ep_idx}: parquet {len(df)} frames != HDF5 {len(subtasks)} frames"
            )
            continue

        df["subtask"] = subtasks
        df.to_parquet(parquet, index=False)

        if ep_idx % 50 == 0:
            print(f"  [{ep_idx:3d}/{len(episodes)-1}] patched")

    if errors:
        print("\nERRORS:")
        for e in errors:
            print(e)
        return

    print(f"\nAll {len(episodes)} parquet files patched.")

    # --- update meta/info.json ---
    info_path = meta_dir / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["subtask"] = {
        "dtype": "string",
        "shape": [1],
        "names": ["subtask"],
    }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print("Updated meta/info.json (added subtask feature)")

    print("\nDone.")


if __name__ == "__main__":
    main()
