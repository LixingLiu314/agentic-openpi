"""
Step 3 — Create aloha_object_lerobot_gripper_binary from aloha_object_lerobot.
Changes to action only:
  - Joint  6 (left  gripper) zeroed out: 0.00 (left arm is inactive in this task)
  - Joint 13 (right gripper) binarized: < 0.05 → 0.00, >= 0.05 → 0.09
  - All other joints (0-5, 7-12) unchanged

Bug fixes applied during copy:
  - Video path columns dropped from parquets (lerobot decodes from files directly;
    keeping them causes path strings to overwrite decoded tensors at load time)
  - episodes_stats.jsonl regenerated in correct lerobot format:
    {"episode_index": N, "stats": {"feature": {min,max,mean,std,count}, ...}}
    (source file has a flat layout that lerobot rejects with KeyError: 'stats')

videos/       → symlink to original (unchanged)
data/         → copied parquets with patched action, video columns dropped
meta/         → copied and updated (repo_id + regenerated episodes_stats.jsonl)
data_quality/ → symlink to original
trajectory_data/ → symlink to original
"""

import json
import pathlib
import shutil

import numpy as np
import pandas as pd

SRC = pathlib.Path(__file__).parent / "correct_object" / "aloha_object_lerobot"
DST = pathlib.Path(__file__).parent / "correct_object" / "aloha_object_lerobot_gripper_binary"

LEFT_GRIPPER_JOINT = 6    # left gripper zeroed out (left arm inactive)
GRIPPER_JOINT      = 13   # right gripper binarized
CLOSE_THRESH       = 0.05
OPEN_VAL           = 0.09
CLOSE_VAL          = 0.00

NEW_REPO_ID = "local/aloha_object_gripper_binary"

# Video path columns stored in parquet — must be dropped so lerobot
# decodes frames from the actual video files rather than returning path strings.
VIDEO_COLS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]


def binarize_action(action: np.ndarray) -> np.ndarray:
    action = action.copy()
    action[:, LEFT_GRIPPER_JOINT] = 0.0
    action[:, GRIPPER_JOINT] = np.where(
        action[:, GRIPPER_JOINT] < CLOSE_THRESH, CLOSE_VAL, OPEN_VAL
    )
    return action


def _vec_stat(arr: np.ndarray) -> dict:
    return {
        "min":   arr.min(axis=0).tolist(),
        "max":   arr.max(axis=0).tolist(),
        "mean":  arr.mean(axis=0).tolist(),
        "std":   arr.std(axis=0).tolist(),
        "count": [int(len(arr))],
    }


def _scalar_stat(arr: np.ndarray) -> dict:
    return {
        "min":   [float(arr.min())],
        "max":   [float(arr.max())],
        "mean":  [float(arr.mean())],
        "std":   [float(arr.std())],
        "count": [int(len(arr))],
    }


def _image_placeholder(shape: list) -> dict:
    # shape = [H, W, C]; stats use C channel placeholders
    C = shape[2]
    return {
        "min":   [[[0.0]] for _ in range(C)],
        "max":   [[[1.0]] for _ in range(C)],
        "mean":  [[[0.5]] for _ in range(C)],
        "std":   [[[0.25]] for _ in range(C)],
        "count": [1],
    }


def _build_episode_stats(df: pd.DataFrame, info_features: dict) -> dict:
    """Compute per-episode stats in the format lerobot expects."""
    scalar_keys = {"timestamp", "frame_index", "episode_index", "task_index"}
    stats = {}
    for feat, meta in info_features.items():
        if meta["dtype"] == "video":
            stats[feat] = _image_placeholder(meta["shape"])
        elif feat in scalar_keys and feat in df.columns:
            stats[feat] = _scalar_stat(df[feat].to_numpy(dtype=float))
        elif meta["dtype"] == "float32" and feat in df.columns:
            arr = np.array(df[feat].tolist(), dtype=np.float32)
            stats[feat] = _vec_stat(arr)
    return stats


def main() -> None:
    if DST.exists():
        print(f"[WARN] {DST} already exists, removing...")
        shutil.rmtree(DST)
    DST.mkdir(parents=True)

    # symlinks for unchanged directories
    for name in ("videos", "data_quality", "trajectory_data"):
        src_path = SRC / name
        if src_path.exists():
            (DST / name).symlink_to(src_path)
            print(f"Symlinked {name}/")

    # load info.json for stats generation
    info = json.loads((SRC / "meta" / "info.json").read_text())
    info_features = info["features"]

    # copy + patch data/
    src_data = SRC / "data"
    dst_data = DST / "data"
    parquet_files = sorted(src_data.glob("chunk-*/episode_*.parquet"))
    total = len(parquet_files)
    print(f"\nPatching {total} parquet files...")

    episode_stats_lines = []
    for i, src_pf in enumerate(parquet_files):
        rel    = src_pf.relative_to(src_data)
        dst_pf = dst_data / rel
        dst_pf.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_parquet(src_pf)

        # patch action
        action = np.array(df["action"].tolist())
        df["action"] = list(binarize_action(action))

        # drop video path columns (bug fix: prevents path strings overwriting decoded tensors)
        df = df.drop(columns=[c for c in VIDEO_COLS if c in df.columns])

        # collect stats before writing
        ep_idx = int(df["episode_index"].iloc[0])
        stats  = _build_episode_stats(df, info_features)
        episode_stats_lines.append(
            json.dumps({"episode_index": ep_idx, "stats": stats}, ensure_ascii=False)
        )

        df.to_parquet(dst_pf, index=False)

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1:3d}/{total}]")

    # copy + update meta/
    dst_meta = DST / "meta"
    dst_meta.mkdir()
    for src_f in sorted((SRC / "meta").iterdir()):
        dst_f = dst_meta / src_f.name

        if src_f.name == "episodes_stats.jsonl":
            # regenerate in correct lerobot format (bug fix: source has flat layout)
            dst_f.write_text("\n".join(episode_stats_lines) + "\n")
            print("Regenerated meta/episodes_stats.jsonl")
        elif src_f.suffix == ".json":
            data = json.loads(src_f.read_text())
            if src_f.name == "info.json":
                data["repo_id"] = NEW_REPO_ID
            dst_f.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            shutil.copy2(src_f, dst_f)
    print("Copied meta/")

    # sanity check
    sample   = dst_data / "chunk-000" / "episode_000000.parquet"
    df_check = pd.read_parquet(sample)
    assert not any(c in df_check.columns for c in VIDEO_COLS), "video cols still present!"
    vals = np.unique(np.array(df_check["action"].tolist())[:, GRIPPER_JOINT].round(4))
    print(f"\nSanity check ep0 — joint {GRIPPER_JOINT} unique: {vals}, video cols dropped: OK")

    print(f"\nDone → {DST}")
    print(f"  episodes : {total}")
    print(f"  videos   : symlink → {SRC / 'videos'}")


if __name__ == "__main__":
    main()
