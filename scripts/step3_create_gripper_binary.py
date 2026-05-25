"""
Step 3 — Copy lerobot dataset with optional gripper binarization.

Always applied (bug fixes):
  - Video path columns dropped from parquets (lerobot decodes from files directly;
    keeping them causes path strings to overwrite decoded tensors at load time)
  - episodes_stats.jsonl regenerated in correct lerobot format:
    {"episode_index": N, "stats": {"feature": {min,max,mean,std,count}, ...}}
    (source file has a flat layout that lerobot rejects with KeyError: 'stats')

With --binarize (optional):
  - Joint  6 (left  gripper): < CLOSE_THRESH → 0.00, >= CLOSE_THRESH → OPEN_VAL
  - Joint 13 (right gripper): < CLOSE_THRESH → 0.00, >= CLOSE_THRESH → OPEN_VAL
  - All other joints unchanged

Usage:
  # Without binarization (raw action kept):
  python step3_create_gripper_binary.py

  # With binarization:
  python step3_create_gripper_binary.py --binarize

videos/          → symlink to original (unchanged)
data/            → copied parquets with video columns dropped (action patched if --binarize)
meta/            → copied and updated (repo_id + regenerated episodes_stats.jsonl)
data_quality/    → symlink to original
trajectory_data/ → symlink to original
"""

import argparse
import json
import pathlib
import shutil

import numpy as np
import pandas as pd

SRC = pathlib.Path("/media/raid/workspace/surongpeng/ws_lixing/agentic-openpi/Datasets/three_object/aloha_shape")
DST_BINARIZE    = pathlib.Path("/media/raid/workspace/surongpeng/ws_lixing/agentic-openpi/Datasets/three_object/three_object_lerobot_binary")
DST_NO_BINARIZE = pathlib.Path("/media/raid/workspace/surongpeng/ws_lixing/agentic-openpi/Datasets/three_object/three_object_lerobot")

LEFT_GRIPPER_JOINT = 6
RIGHT_GRIPPER_JOINT = 13
CLOSE_THRESH = 0.05
OPEN_VAL     = 0.09
CLOSE_VAL    = 0.00

VIDEO_COLS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]


def binarize_action(action: np.ndarray) -> np.ndarray:
    action = action.copy()
    for j in (LEFT_GRIPPER_JOINT, RIGHT_GRIPPER_JOINT):
        action[:, j] = np.where(action[:, j] < CLOSE_THRESH, CLOSE_VAL, OPEN_VAL)
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
    C = shape[2]
    return {
        "min":   [[[0.0]] for _ in range(C)],
        "max":   [[[1.0]] for _ in range(C)],
        "mean":  [[[0.5]] for _ in range(C)],
        "std":   [[[0.25]] for _ in range(C)],
        "count": [1],
    }


def _build_episode_stats(df: pd.DataFrame, info_features: dict) -> dict:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--binarize", action="store_true",
                        help="Binarize gripper joints in action (default: keep raw action)")
    args = parser.parse_args()

    dst = DST_BINARIZE if args.binarize else DST_NO_BINARIZE
    new_repo_id = "local/" + dst.name

    print(f"Mode     : {'binarize' if args.binarize else 'no binarize (raw action)'}")
    print(f"SRC      : {SRC}")
    print(f"DST      : {dst}")

    if dst.exists():
        print(f"[WARN] {dst} already exists, removing...")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    for name in ("videos", "data_quality", "trajectory_data"):
        src_path = SRC / name
        if src_path.exists():
            (dst / name).symlink_to(src_path)
            print(f"Symlinked {name}/")

    info = json.loads((SRC / "meta" / "info.json").read_text())
    info_features = info["features"]

    src_data = SRC / "data"
    dst_data = dst / "data"
    parquet_files = sorted(src_data.glob("chunk-*/episode_*.parquet"))
    total = len(parquet_files)
    print(f"\nCopying {total} parquet files...")

    episode_stats_lines = []
    for i, src_pf in enumerate(parquet_files):
        rel    = src_pf.relative_to(src_data)
        dst_pf = dst_data / rel
        dst_pf.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_parquet(src_pf)

        if args.binarize:
            action = np.array(df["action"].tolist())
            df["action"] = list(binarize_action(action))

        df = df.drop(columns=[c for c in VIDEO_COLS if c in df.columns])

        ep_idx = int(df["episode_index"].iloc[0])
        stats  = _build_episode_stats(df, info_features)
        episode_stats_lines.append(
            json.dumps({"episode_index": ep_idx, "stats": stats}, ensure_ascii=False)
        )

        df.to_parquet(dst_pf, index=False)

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1:3d}/{total}]")

    dst_meta = dst / "meta"
    dst_meta.mkdir()
    for src_f in sorted((SRC / "meta").iterdir()):
        dst_f = dst_meta / src_f.name
        if src_f.name == "episodes_stats.jsonl":
            dst_f.write_text("\n".join(episode_stats_lines) + "\n")
            print("Regenerated meta/episodes_stats.jsonl")
        elif src_f.suffix == ".json":
            data = json.loads(src_f.read_text())
            if src_f.name == "info.json":
                data["repo_id"] = new_repo_id
            dst_f.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            shutil.copy2(src_f, dst_f)
    print("Copied meta/")

    sample   = dst_data / "chunk-000" / "episode_000000.parquet"
    df_check = pd.read_parquet(sample)
    assert not any(c in df_check.columns for c in VIDEO_COLS), "video cols still present!"
    if args.binarize:
        vals = np.unique(np.array(df_check["action"].tolist())[:, RIGHT_GRIPPER_JOINT].round(4))
        print(f"\nSanity check ep0 — joint {RIGHT_GRIPPER_JOINT} unique: {vals}, video cols dropped: OK")
    else:
        print(f"\nSanity check ep0 — video cols dropped: OK")

    print(f"\nDone → {dst}")
    print(f"  episodes : {total}")
    print(f"  videos   : symlink → {SRC / 'videos'}")


if __name__ == "__main__":
    main()
