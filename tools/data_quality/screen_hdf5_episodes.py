#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import time
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
    constant_joint_range_epsilon: float
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
        if (
            any(name.startswith(prefix + "/") or name == prefix for prefix in IMAGE_PREFIXES)
            and len(getattr(obj, "shape", ())) >= 3
        ):
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


def arm_joint_dimensions(width: int) -> list[tuple[int, str, str, int]]:
    dimensions = [(global_dim, "left", f"left_j{global_dim + 1}", global_dim) for global_dim in range(min(6, width))]
    if width >= 13:
        dimensions.extend(
            (global_dim, "right", f"right_j{global_dim - 6}", global_dim - 7) for global_dim in range(7, 13)
        )
    return dimensions


def constant_joint_warnings(np, array, motion_key: str | None, epsilon: float) -> list[dict[str, Any]]:
    if array is None or array.ndim != 2 or array.shape[0] < 2:
        return []
    try:
        data = array.astype("float64")
    except (TypeError, ValueError):
        return []

    warnings = []
    for dim, arm, joint, local_joint_index in arm_joint_dimensions(data.shape[1]):
        values = data[:, dim]
        if not np.isfinite(values).all():
            continue
        min_value = float(np.min(values))
        max_value = float(np.max(values))
        range_value = max_value - min_value
        if range_value <= epsilon:
            warnings.append(
                {
                    "type": "constant_joint",
                    "motion_source": motion_key,
                    "global_dim": dim,
                    "arm": arm,
                    "joint": joint,
                    "local_joint_index": local_joint_index,
                    "range": float(range_value),
                    "value": float(values[0]),
                }
            )
    return warnings


def sampled_image_std(np, dataset, sample_count: int) -> dict[str, float]:
    length = int(dataset.shape[0])
    if length <= 0:
        return {"mean_std": 0.0, "min_std": 0.0, "max_std": 0.0}
    count = min(sample_count, length)
    indices = [0] if count <= 1 else sorted({round(i * (length - 1) / (count - 1)) for i in range(count)})
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
        "warnings": [],
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
            result["warnings"].extend(
                constant_joint_warnings(np, motion_source, motion_key, thresholds.constant_joint_range_epsilon)
            )
            result["metrics"]["num_constant_joints"] = len(result["warnings"])

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


def remove_bad_files(
    root: Path, bad_results: list[dict[str, Any]], quarantine_dir: Path, *, delete: bool
) -> list[dict[str, Any]]:
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


def source_episode_index(path: Path) -> int | None:
    match = None
    for pattern in (r"episode[_-](\d+)\.hdf5$", r"(\d+)\.hdf5$"):
        match = re.search(pattern, path.name)
        if match:
            return int(match.group(1))
    return None


def load_manual_bad_tokens(values: list[str] | None, list_path: Path | None) -> list[str]:
    tokens: list[str] = []
    for value in values or []:
        for part in value.replace(",", " ").split():
            part = part.strip()
            if part:
                tokens.append(part)
    if list_path is not None:
        for line in list_path.expanduser().read_text(encoding="utf-8").splitlines():
            text = line.split("#", 1)[0].strip()
            if text:
                tokens.extend(text.replace(",", " ").split())
    return tokens


def _parse_episode_range(text: str) -> set[int]:
    if ":" not in text:
        return {int(text)}
    start_text, stop_text = text.split(":", 1)
    if not start_text or not stop_text:
        raise ValueError(f"Invalid episode range {text!r}; expected start:stop")
    start = int(start_text)
    stop = int(stop_text)
    if stop < start:
        raise ValueError(f"Invalid episode range {text!r}; stop must be >= start")
    return set(range(start, stop))


def resolve_manual_bad_paths(root: Path, paths: list[Path], tokens: list[str]) -> tuple[set[str], list[dict[str, Any]]]:
    """Resolve operator-provided bad episode tokens to absolute HDF5 paths.

    Supported token forms:
      - absolute or root-relative path: /data/task/episode_000001.hdf5, task/episode_000001.hdf5
      - task-specific source ids: task_name:1, task_name:1:5
      - bare source ids: 1, 1:5 (matches all selected tasks with those source ids)
    """
    if not tokens:
        return set(), []

    path_by_abs = {str(path.expanduser().resolve()): path for path in paths}
    path_by_rel = {path.relative_to(root).as_posix(): path for path in paths}
    task_index: dict[tuple[str, int], list[Path]] = {}
    id_index: dict[int, list[Path]] = {}
    for path in paths:
        rel_parts = path.relative_to(root).parts
        task = rel_parts[0] if rel_parts else ""
        ep_idx = source_episode_index(path)
        if ep_idx is None:
            continue
        task_index.setdefault((task, ep_idx), []).append(path)
        id_index.setdefault(ep_idx, []).append(path)

    resolved: set[str] = set()
    unmatched: list[dict[str, Any]] = []

    def add_path(path: Path, token: str) -> None:
        del token
        resolved.add(str(path.expanduser().resolve()))

    for token in tokens:
        token = token.strip()
        if not token:
            continue

        candidate = Path(token).expanduser()
        if candidate.is_absolute():
            abs_key = str(candidate.resolve())
            if abs_key in path_by_abs:
                add_path(path_by_abs[abs_key], token)
            elif candidate.exists():
                resolved.add(abs_key)
            else:
                unmatched.append({"token": token, "reason": "absolute_path_not_found"})
            continue

        if token.endswith(".hdf5") or "/" in token:
            rel_key = candidate.as_posix()
            if rel_key in path_by_rel:
                add_path(path_by_rel[rel_key], token)
                continue
            abs_candidate = (root / candidate).resolve()
            abs_key = str(abs_candidate)
            if abs_key in path_by_abs:
                add_path(path_by_abs[abs_key], token)
            elif abs_candidate.exists():
                resolved.add(abs_key)
            else:
                unmatched.append({"token": token, "reason": "relative_path_not_found"})
            continue

        if ":" in token and not token.replace(":", "").isdigit():
            task, episode_text = token.split(":", 1)
            matches: list[Path] = []
            for ep_idx in _parse_episode_range(episode_text):
                matches.extend(task_index.get((task, ep_idx), []))
            if matches:
                for path in matches:
                    add_path(path, token)
            else:
                unmatched.append({"token": token, "reason": "task_episode_not_found"})
            continue

        try:
            episode_ids = _parse_episode_range(token)
        except ValueError as exc:
            unmatched.append({"token": token, "reason": str(exc)})
            continue

        matches = []
        for ep_idx in episode_ids:
            matches.extend(id_index.get(ep_idx, []))
        if matches:
            for path in matches:
                add_path(path, token)
        else:
            unmatched.append({"token": token, "reason": "episode_id_not_found"})

    return resolved, unmatched


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Screen raw Aloha HDF5 episodes and optionally remove bad ones.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("~/data/aloha_pipeline"),
        help="Root containing task directories with episode_*.hdf5 files.",
    )
    parser.add_argument("--tasks", nargs="*", default=None, help="Optional task directory names to include.")
    parser.add_argument("--output", type=Path, default=Path("tools/data_quality/hdf5_screen_report.json"))
    parser.add_argument("--bad-list", type=Path, default=Path("tools/data_quality/bad_hdf5_episodes.txt"))
    parser.add_argument("--min-frames", type=int, default=20)
    parser.add_argument(
        "--min-joint-range",
        type=float,
        default=0.03,
        help="Minimum per-joint range in radians/meters to count an arm as active.",
    )
    parser.add_argument(
        "--min-cumulative-joint-movement",
        type=float,
        default=0.15,
        help="Minimum cumulative 6D arm movement to count an arm as active.",
    )
    parser.add_argument(
        "--constant-joint-range-epsilon",
        type=float,
        default=1e-12,
        help="Warn when any left/right arm joint in the motion source has range less than or equal to this value.",
    )
    parser.add_argument("--active-arms", choices=["any", "left", "right", "both"], default="any")
    parser.add_argument(
        "--check-images", action="store_true", help="Also sample image datasets for length and low-variance checks."
    )
    parser.add_argument("--image-sample-count", type=int, default=5)
    parser.add_argument("--min-image-std", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--manual-bad-episodes",
        nargs="*",
        default=None,
        help=(
            "Operator-provided bad episodes to add on top of automatic screening. "
            "Accepts paths, task:episode_id, task:start:stop, episode_id, or start:stop."
        ),
    )
    parser.add_argument(
        "--manual-bad-list",
        type=Path,
        default=None,
        help="Text file with one manual bad episode token per line; # comments are ignored.",
    )
    parser.add_argument(
        "--apply", action="store_true", help="Actually remove bad episodes. Without this, only writes a report."
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Permanently delete bad files instead of moving to quarantine. Requires --apply.",
    )
    parser.add_argument(
        "--quarantine-dir",
        type=Path,
        default=None,
        help="Where bad files are moved when --apply is used without --delete.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    tasks = parse_tasks(args.tasks)
    paths = hdf5_paths(root, tasks)
    if not paths:
        raise SystemExit(f"No HDF5 files found under {root}")
    manual_tokens = load_manual_bad_tokens(args.manual_bad_episodes, args.manual_bad_list)
    manual_bad_paths, manual_unmatched = resolve_manual_bad_paths(root, paths, manual_tokens)

    thresholds = Thresholds(
        min_frames=args.min_frames,
        min_joint_range=args.min_joint_range,
        min_cumulative_joint_movement=args.min_cumulative_joint_movement,
        constant_joint_range_epsilon=args.constant_joint_range_epsilon,
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
            futures = {executor.submit(analyze_one, str(path), str(root), threshold_dict): path for path in paths}
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda item: item["relative_path"])

    for item in results:
        if str(Path(item["path"]).expanduser().resolve()) in manual_bad_paths:
            item["bad"] = True
            if "manual_bad_episode" not in item["reasons"]:
                item["reasons"].append("manual_bad_episode")

    bad = [item for item in results if item["bad"]]
    warning_episodes = [item for item in results if item.get("warnings")]
    num_warnings = sum(len(item.get("warnings", [])) for item in warning_episodes)
    report = {
        "root": str(root),
        "tasks": sorted(tasks) if tasks else None,
        "num_files": len(results),
        "num_bad": len(bad),
        "num_warning_episodes": len(warning_episodes),
        "num_warnings": num_warnings,
        "thresholds": threshold_dict,
        "manual_bad_episode_tokens": manual_tokens,
        "manual_bad_episode_paths": sorted(manual_bad_paths),
        "manual_bad_unmatched": manual_unmatched,
        "bad_episodes": bad,
        "warning_episodes": [
            {
                "path": item["path"],
                "relative_path": item["relative_path"],
                "warnings": item.get("warnings", []),
            }
            for item in warning_episodes
        ],
        "all_episodes": results,
        "removal_actions": [],
    }

    if args.apply:
        if args.delete:
            print("Deleting bad HDF5 episodes permanently.")
            quarantine_dir = root / "_deleted_bad_episodes_not_used"
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            quarantine_dir = (
                args.quarantine_dir.expanduser().resolve()
                if args.quarantine_dir
                else root / "_removed_bad_episodes" / timestamp
            )
            print(f"Moving bad HDF5 episodes to {quarantine_dir}")
        report["removal_actions"] = remove_bad_files(root, bad, quarantine_dir, delete=args.delete)
    else:
        print("Dry run only. Re-run with --apply to remove bad episodes.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.bad_list.parent.mkdir(parents=True, exist_ok=True)
    args.bad_list.write_text("\n".join(item["path"] for item in bad) + ("\n" if bad else ""), encoding="utf-8")

    print(f"Bad episodes: {len(bad)} / {len(results)}")
    print(f"Warnings: {num_warnings} constant arm joints in {len(warning_episodes)} episodes")
    print(f"Report: {args.output}")
    print(f"Bad list: {args.bad_list}")
    if bad:
        print("First bad episodes:")
        for item in bad[:20]:
            print(f"  {item['relative_path']}: {', '.join(item['reasons'])}")
    if warning_episodes:
        print("First warning episodes:")
        for item in warning_episodes[:20]:
            warnings = item.get("warnings", [])
            preview = ", ".join(
                f"{warning['joint']}(dim={warning['global_dim']}, value={warning['value']:.6g})"
                for warning in warnings[:8]
            )
            if len(warnings) > 8:
                preview += f", ... +{len(warnings) - 8} more"
            print(f"  {item['relative_path']}: {preview}")
    if args.apply:
        print(f"Removal actions: {len(report['removal_actions'])}")


if __name__ == "__main__":
    main()
