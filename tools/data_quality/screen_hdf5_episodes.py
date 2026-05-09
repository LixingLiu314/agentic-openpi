#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STATE_CANDIDATES = [
    "observations/qpos",
    "observations/state",
    "observation.state",
    "qpos",
    "state",
]
ACTION_CANDIDATES = [
    "action",
    "actions",
]
IMAGE_PREFIXES = [
    "observations/images",
    "observation/images",
    "images",
]
EXCLUDED_DIR_NAMES = {
    "_removed_bad_episodes",
    "removed_bad_episodes",
    "bad_episodes",
    ".trash",
}


@dataclass
class Thresholds:
    min_frames: int
    min_joint_range: float
    min_cumulative_joint_movement: float
    min_image_std: float
    image_sample_count: int
    active_arms: str
    check_images: bool


def import_hdf5_deps():
    try:
        import h5py  # type: ignore
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency for HDF5 screening: h5py/numpy. "
            "Install in the data-processing environment, for example: "
            "pip install h5py numpy"
        ) from exc
    return h5py, np


def hdf5_paths(root: Path, tasks: set[str] | None) -> list[Path]:
    files = []
    for path in root.expanduser().rglob("*.hdf5"):
        rel_parts = path.relative_to(root).parts
        if any(part in EXCLUDED_DIR_NAMES for part in rel_parts):
            continue
        if tasks and rel_parts and rel_parts[0] not in tasks:
            continue
        files.append(path)
    return sorted(files)


def dataset_or_none(h5_file, candidates: list[str]):
    for key in candidates:
        if key in h5_file:
            obj = h5_file[key]
            if hasattr(obj, "shape"):
                return key, obj
    return None, None


def iter_datasets(h5_file):
    datasets = []

    def visit(name, obj):
        if hasattr(obj, "shape"):
            datasets.append((name, obj))

    h5_file.visititems(visit)
    return datasets


def image_datasets(h5_file) -> list[tuple[str, Any]]:
    out = []
    for name, obj in iter_datasets(h5_file):
        if any(name.startswith(prefix + "/") or name == prefix for prefix in IMAGE_PREFIXES):
            if len(getattr(obj, "shape", ())) >= 3:
                out.append((name, obj))
    return out


def finite_array(np, array) -> bool:
    try:
        return bool(np.isfinite(array).all())
    except TypeError:
        return False


def arm_metrics(np, array, dims: slice) -> dict[str, float] | None:
    if array is None or array.ndim < 2:
        return None
    width = array.shape[1]
    start = dims.start or 0
    stop = dims.stop or width
    if width < stop:
        return None
    sub = array[:, start:stop].astype("float64")
    if sub.shape[0] < 2:
        return {
            "max_range": 0.0,
            "mean_range": 0.0,
            "cumulative_movement": 0.0,
            "mean_abs_step": 0.0,
        }
    ranges = np.nanmax(sub, axis=0) - np.nanmin(sub, axis=0)
    diffs = np.diff(sub, axis=0)
    return {
        "max_range": float(np.nanmax(ranges)),
        "mean_range": float(np.nanmean(ranges)),
        "cumulative_movement": float(np.nansum(np.linalg.norm(diffs, axis=1))),
        "mean_abs_step": float(np.nanmean(np.abs(diffs))),
    }


def is_active(metrics: dict[str, float] | None, thresholds: Thresholds) -> bool:
    if metrics is None:
        return False
    return (
        metrics["max_range"] >= thresholds.min_joint_range
        or metrics["cumulative_movement"] >= thresholds.min_cumulative_joint_movement
    )


def sampled_image_std(np, dataset, sample_count: int) -> dict[str, float]:
    length = int(dataset.shape[0])
    if length <= 0:
        return {"mean_std": 0.0, "min_std": 0.0, "max_std": 0.0}
    count = min(sample_count, length)
    if count <= 1:
        indices = [0]
    else:
        indices = sorted(set(int(round(i * (length - 1) / (count - 1))) for i in range(count)))
    stds = []
    for idx in indices:
        frame = dataset[idx]
        stds.append(float(np.asarray(frame).std()))
    return {
        "mean_std": float(np.mean(stds)),
        "min_std": float(np.min(stds)),
        "max_std": float(np.max(stds)),
    }


def analyze_one(path_text: str, root_text: str, thresholds_dict: dict[str, Any]) -> dict[str, Any]:
    h5py, np = import_hdf5_deps()
    path = Path(path_text)
    root = Path(root_text)
    thresholds = Thresholds(**thresholds_dict)
    rel_path = str(path.relative_to(root))

    result: dict[str, Any] = {
        "path": str(path),
        "relative_path": rel_path,
        "bad": False,
        "reasons": [],
        "metrics": {},
    }

    try:
        with h5py.File(path, "r") as f:
            state_key, state_ds = dataset_or_none(f, STATE_CANDIDATES)
            action_key, action_ds = dataset_or_none(f, ACTION_CANDIDATES)

            if state_ds is None and action_ds is None:
                result["bad"] = True
                result["reasons"].append("missing_state_and_action")
                result["available_datasets"] = [name for name, _ in iter_datasets(f)]
                return result

            state = state_ds[()] if state_ds is not None else None
            action = action_ds[()] if action_ds is not None else None
            motion_source = state if state is not None else action
            motion_key = state_key if state is not None else action_key
            length = int(motion_source.shape[0]) if hasattr(motion_source, "shape") else 0

            result["metrics"]["state_key"] = state_key
            result["metrics"]["action_key"] = action_key
            result["metrics"]["motion_source"] = motion_key
            result["metrics"]["length"] = length

            if length < thresholds.min_frames:
                result["bad"] = True
                result["reasons"].append("too_short")

            if state is not None and not finite_array(np, state):
                result["bad"] = True
                result["reasons"].append("state_has_nan_or_inf")
            if action is not None and not finite_array(np, action):
                result["bad"] = True
                result["reasons"].append("action_has_nan_or_inf")

            if state is not None and action is not None and int(state.shape[0]) != int(action.shape[0]):
                result["bad"] = True
                result["reasons"].append("state_action_length_mismatch")

            left_metrics = arm_metrics(np, motion_source, slice(0, 6))
            right_metrics = arm_metrics(np, motion_source, slice(7, 13))
            result["metrics"]["left_arm"] = left_metrics
            result["metrics"]["right_arm"] = right_metrics
            left_active = is_active(left_metrics, thresholds)
            right_active = is_active(right_metrics, thresholds)
            result["metrics"]["left_active"] = left_active
            result["metrics"]["right_active"] = right_active

            if thresholds.active_arms == "any":
                no_motion = not (left_active or right_active)
            elif thresholds.active_arms == "left":
                no_motion = not left_active
            elif thresholds.active_arms == "right":
                no_motion = not right_active
            else:
                no_motion = not (left_active and right_active)
            if no_motion:
                result["bad"] = True
                result["reasons"].append(f"insufficient_{thresholds.active_arms}_arm_motion")

            images = image_datasets(f)
            result["metrics"]["image_keys"] = [name for name, _ in images]
            if thresholds.check_images:
                if not images:
                    result["bad"] = True
                    result["reasons"].append("missing_images")
                image_stats = {}
                for name, ds in images:
                    if int(ds.shape[0]) != length:
                        result["bad"] = True
                        result["reasons"].append(f"image_length_mismatch:{name}")
                    stats = sampled_image_std(np, ds, thresholds.image_sample_count)
                    image_stats[name] = stats
                result["metrics"]["image_stats"] = image_stats
                if image_stats and all(stats["mean_std"] < thresholds.min_image_std for stats in image_stats.values()):
                    result["bad"] = True
                    result["reasons"].append("all_sampled_images_low_variance")

    except OSError as exc:
        result["bad"] = True
        result["reasons"].append("unreadable_hdf5")
        result["error"] = str(exc)
    except Exception as exc:
        result["bad"] = True
        result["reasons"].append("analysis_exception")
        result["error"] = repr(exc)

    return result


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for idx in range(1, 10_000):
        candidate = path.with_name(f"{stem}.dup{idx}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find a free destination for {path}")


def remove_bad_files(root: Path, bad_results: list[dict[str, Any]], quarantine_dir: Path, delete: bool) -> list[dict[str, Any]]:
    actions = []
    for item in bad_results:
        src = Path(item["path"])
        if not src.exists():
            actions.append({"source": str(src), "action": "missing_at_removal_time"})
            continue
        if delete:
            src.unlink()
            actions.append({"source": str(src), "action": "deleted"})
            continue
        rel = src.relative_to(root)
        dst = unique_destination(quarantine_dir / rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        actions.append({"source": str(src), "destination": str(dst), "action": "moved_to_quarantine"})
    return actions


def parse_tasks(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    tasks = set()
    for value in values:
        for part in value.split(","):
            if part.strip():
                tasks.add(part.strip())
    return tasks or None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Screen raw Aloha HDF5 episodes and optionally remove bad ones.")
    parser.add_argument("--root", type=Path, default=Path("~/data/aloha_pipeline"), help="Root containing task directories with episode_*.hdf5 files.")
    parser.add_argument("--tasks", nargs="*", default=None, help="Optional task directory names to include.")
    parser.add_argument("--output", type=Path, default=Path("tools/data_quality/hdf5_screen_report.json"))
    parser.add_argument("--bad-list", type=Path, default=Path("tools/data_quality/bad_hdf5_episodes.txt"))
    parser.add_argument("--min-frames", type=int, default=20)
    parser.add_argument("--min-joint-range", type=float, default=0.03, help="Minimum per-joint range in radians/meters to count an arm as active.")
    parser.add_argument("--min-cumulative-joint-movement", type=float, default=0.15, help="Minimum cumulative 6D arm movement to count an arm as active.")
    parser.add_argument("--active-arms", choices=["any", "left", "right", "both"], default="any")
    parser.add_argument("--check-images", action="store_true", help="Also sample image datasets for length and low-variance checks.")
    parser.add_argument("--image-sample-count", type=int, default=5)
    parser.add_argument("--min-image-std", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--apply", action="store_true", help="Actually remove bad episodes. Without this, only writes a report.")
    parser.add_argument("--delete", action="store_true", help="Permanently delete bad files instead of moving to quarantine. Requires --apply.")
    parser.add_argument("--quarantine-dir", type=Path, default=None, help="Where bad files are moved when --apply is used without --delete.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    tasks = parse_tasks(args.tasks)
    paths = hdf5_paths(root, tasks)
    if not paths:
        raise SystemExit(f"No HDF5 files found under {root}")

    thresholds = Thresholds(
        min_frames=args.min_frames,
        min_joint_range=args.min_joint_range,
        min_cumulative_joint_movement=args.min_cumulative_joint_movement,
        min_image_std=args.min_image_std,
        image_sample_count=args.image_sample_count,
        active_arms=args.active_arms,
        check_images=args.check_images,
    )
    threshold_dict = thresholds.__dict__.copy()

    print(f"Scanning {len(paths)} HDF5 episodes under {root}")
    if args.num_workers <= 1:
        results = [analyze_one(str(path), str(root), threshold_dict) for path in paths]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(analyze_one, str(path), str(root), threshold_dict): path
                for path in paths
            }
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda item: item["relative_path"])

    bad = [item for item in results if item["bad"]]
    report = {
        "root": str(root),
        "tasks": sorted(tasks) if tasks else None,
        "num_files": len(results),
        "num_bad": len(bad),
        "thresholds": threshold_dict,
        "bad_episodes": bad,
        "all_episodes": results,
        "removal_actions": [],
    }

    if args.apply:
        if args.delete:
            print("Deleting bad HDF5 episodes permanently.")
            quarantine_dir = root / "_deleted_bad_episodes_not_used"
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            quarantine_dir = args.quarantine_dir.expanduser().resolve() if args.quarantine_dir else root / "_removed_bad_episodes" / timestamp
            print(f"Moving bad HDF5 episodes to {quarantine_dir}")
        report["removal_actions"] = remove_bad_files(root, bad, quarantine_dir, args.delete)
    else:
        print("Dry run only. Re-run with --apply to remove bad episodes.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.bad_list.parent.mkdir(parents=True, exist_ok=True)
    args.bad_list.write_text("\n".join(item["path"] for item in bad) + ("\n" if bad else ""), encoding="utf-8")

    print(f"Bad episodes: {len(bad)} / {len(results)}")
    print(f"Report: {args.output}")
    print(f"Bad list: {args.bad_list}")
    if bad:
        print("First bad episodes:")
        for item in bad[:20]:
            print(f"  {item['relative_path']}: {', '.join(item['reasons'])}")
    if args.apply:
        print(f"Removal actions: {len(report['removal_actions'])}")


if __name__ == "__main__":
    main()
