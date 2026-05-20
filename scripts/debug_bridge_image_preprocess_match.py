#!/usr/bin/env python3
"""Compare Bridge dataset images with SimplerEnv policy-image preprocess modes.

This script is meant to answer one concrete question: which preprocessing of
SimplerEnv's 640x480 WidowX camera observations best matches the 256x256 Bridge
training images after OpenPI's model resize-to-224 transform?
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path("/media/raid/workspace/xiahongyu/.cache/huggingface/lerobot/bridge_orig_lerobot")
DEFAULT_SIMPLER_ENV_ROOT = Path("/media/raid/workspace/xiahongyu/SimplerEnv")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results/debug_bridge_image_preprocess_match"


def _insert_repo_paths() -> None:
    for path in [REPO_ROOT, REPO_ROOT / "packages/openpi-client/src", REPO_ROOT / "src"]:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _task_matches(task: str, spec: Any) -> bool:
    normalized = " ".join(task.lower().strip().split())
    if any(normalized == exact.lower() for exact in spec.exact):
        return True
    return any(all(keyword in normalized for keyword in keywords) for keywords in spec.all_keywords)


def _video_path(dataset_root: Path, episode_index: int, video_key: str) -> Path:
    chunk = episode_index // 1000
    return dataset_root / "videos" / f"chunk-{chunk:03d}" / video_key / f"episode_{episode_index:06d}.mp4"


def _decode_frame(video_path: Path, frame_index: int) -> np.ndarray:
    image = iio.imread(video_path, index=frame_index)
    if image.ndim == 4:
        image = image[0]
    if image.dtype != np.uint8:
        image = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    if image.shape[-1] == 4:
        image = image[..., :3]
    return np.ascontiguousarray(image)


def _collect_dataset_images(
    *,
    dataset_root: Path,
    task_specs: Iterable[Any],
    samples_per_task: int,
    frame_fractions: tuple[float, ...],
    seed: int,
    video_key: str,
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[dict[str, Any]]]]:
    rng = random.Random(seed)
    episodes = _read_jsonl(dataset_root / "meta/episodes.jsonl")

    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        tasks = [str(task) for task in episode.get("tasks", [])]
        for spec in task_specs:
            matched_task = next((task for task in tasks if _task_matches(task, spec)), None)
            if matched_task is None:
                continue
            candidates[spec.key].append(
                {
                    "episode_index": int(episode["episode_index"]),
                    "length": int(episode["length"]),
                    "task": matched_task,
                }
            )

    images_by_task: dict[str, list[np.ndarray]] = {}
    metadata_by_task: dict[str, list[dict[str, Any]]] = {}
    for spec in task_specs:
        matches = candidates[spec.key]
        rng.shuffle(matches)
        images: list[np.ndarray] = []
        metadata: list[dict[str, Any]] = []
        for match in matches:
            if len(images) >= samples_per_task:
                break
            video_path = _video_path(dataset_root, match["episode_index"], video_key)
            if not video_path.exists():
                continue
            for fraction in frame_fractions:
                if len(images) >= samples_per_task:
                    break
                frame_index = int(round((match["length"] - 1) * fraction))
                try:
                    image = _decode_frame(video_path, frame_index)
                except Exception as exc:  # noqa: BLE001
                    metadata.append({**match, "frame_index": frame_index, "error": repr(exc)})
                    continue
                images.append(image)
                metadata.append({**match, "frame_index": frame_index, "video_path": str(video_path)})
        images_by_task[spec.key] = images
        metadata_by_task[spec.key] = metadata
    return images_by_task, metadata_by_task


def _collect_simpler_images(
    *,
    simpler_env_root: Path,
    task_keys: list[str],
    episodes_per_task: int,
    renderer_device: str,
    enable_raytracing: bool,
) -> dict[str, list[np.ndarray]]:
    from scripts.eval_pi05_bridge_simpler import _insert_paths, bridge_task_specs

    _insert_paths(simpler_env_root)
    os.chdir(simpler_env_root)
    os.environ["DISPLAY"] = ""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

    specs = {spec.task_key: spec for spec in bridge_task_specs(simpler_env_root)}
    robot_init_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    images_by_task: dict[str, list[np.ndarray]] = {}

    for task_key in task_keys:
        spec = specs[task_key]
        control_mode = get_robot_control_mode(spec.robot, "pi05")
        additional_env_build_kwargs = {}
        if enable_raytracing:
            additional_env_build_kwargs["shader_dir"] = "rt"

        renderer_kwargs = {"offscreen_only": True}
        if renderer_device:
            renderer_kwargs["device"] = renderer_device

        env = build_maniskill2_env(
            spec.env_name,
            obs_mode="rgbd",
            robot=spec.robot,
            sim_freq=500,
            control_mode=control_mode,
            control_freq=5,
            max_episode_steps=spec.max_episode_steps,
            scene_name=spec.scene_name,
            renderer_kwargs=renderer_kwargs,
            camera_cfgs={"add_segmentation": True},
            rgb_overlay_path=spec.rgb_overlay_path,
            **additional_env_build_kwargs,
        )

        task_images: list[np.ndarray] = []
        try:
            for obj_episode_id in range(episodes_per_task):
                obs, _ = env.reset(
                    options={
                        "robot_init_options": {
                            "init_xy": np.array([spec.robot_init_x, spec.robot_init_y]),
                            "init_rot_quat": robot_init_quat,
                        },
                        "obj_init_options": {"episode_id": obj_episode_id},
                    }
                )
                image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=None)
                task_images.append(np.ascontiguousarray(image))
        finally:
            close = getattr(env, "close", None)
            if close is not None:
                close()
        images_by_task[task_key] = task_images
    return images_by_task


def _resize_with_pad_np(image: np.ndarray, height: int = 224, width: int = 224) -> np.ndarray:
    import cv2

    cur_height, cur_width = image.shape[:2]
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    out = np.zeros((height, width, image.shape[2]), dtype=image.dtype)
    pad_top = (height - resized_height) // 2
    pad_left = (width - resized_width) // 2
    out[pad_top : pad_top + resized_height, pad_left : pad_left + resized_width] = resized
    return out


def _policy_inputs(images: list[np.ndarray], mode: str) -> list[np.ndarray]:
    from scripts.eval_pi05_bridge_simpler import preprocess_policy_image

    outputs: list[np.ndarray] = []
    for image in images:
        preprocessed = preprocess_policy_image(image, mode)
        outputs.append(_resize_with_pad_np(preprocessed, 224, 224))
    return outputs


def _image_features(images: list[np.ndarray]) -> dict[str, Any]:
    import cv2

    if not images:
        return {}

    arr = np.stack(images).astype(np.float32)
    flat = arr.reshape(-1, 3)
    black_mask = np.all(arr <= 2, axis=-1)

    border = np.zeros(arr.shape[:3], dtype=bool)
    border[:, :28, :] = True
    border[:, -28:, :] = True
    border[:, :, :28] = True
    border[:, :, -28:] = True
    border_black_mask = black_mask & border

    hist_rgb = []
    for channel in range(3):
        hist, _ = np.histogram(flat[:, channel], bins=32, range=(0, 256), density=True)
        hist_rgb.extend(hist.tolist())

    hsv_values = []
    edge_density = []
    for image in images:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv_values.append(hsv.reshape(-1, 3))
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edge_density.append(float((cv2.Canny(gray, 50, 150) > 0).mean()))
    hsv_flat = np.concatenate(hsv_values, axis=0)

    return {
        "num_images": len(images),
        "mean_rgb": flat.mean(axis=0).round(4).tolist(),
        "std_rgb": flat.std(axis=0).round(4).tolist(),
        "mean_hsv": hsv_flat.mean(axis=0).round(4).tolist(),
        "std_hsv": hsv_flat.std(axis=0).round(4).tolist(),
        "black_fraction": float(black_mask.mean()),
        "border_black_fraction": float(border_black_mask.sum() / border.sum()),
        "edge_density": float(np.mean(edge_density)),
        "hist_rgb_32": hist_rgb,
    }


def _l2(left: list[float], right: list[float]) -> float:
    return float(np.linalg.norm(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)))


def _hist_chi2(left: list[float], right: list[float]) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    return float(0.5 * np.sum((a - b) ** 2 / (a + b + 1e-12)))


def _compare_features(dataset: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    return {
        "mean_rgb_l2": _l2(dataset["mean_rgb"], candidate["mean_rgb"]),
        "std_rgb_l2": _l2(dataset["std_rgb"], candidate["std_rgb"]),
        "mean_hsv_l2": _l2(dataset["mean_hsv"], candidate["mean_hsv"]),
        "std_hsv_l2": _l2(dataset["std_hsv"], candidate["std_hsv"]),
        "hist_rgb_chi2": _hist_chi2(dataset["hist_rgb_32"], candidate["hist_rgb_32"]),
        "black_fraction_abs": abs(float(dataset["black_fraction"]) - float(candidate["black_fraction"])),
        "border_black_fraction_abs": abs(
            float(dataset["border_black_fraction"]) - float(candidate["border_black_fraction"])
        ),
        "edge_density_abs": abs(float(dataset["edge_density"]) - float(candidate["edge_density"])),
    }


def _rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_keys = [
        "mean_rgb_l2",
        "std_rgb_l2",
        "hist_rgb_chi2",
        "black_fraction_abs",
        "border_black_fraction_abs",
        "edge_density_abs",
    ]
    values = {key: np.asarray([row["distances"][key] for row in rows], dtype=np.float64) for key in metric_keys}
    for row_index, row in enumerate(rows):
        score = 0.0
        for key in metric_keys:
            metric_values = values[key]
            std = float(metric_values.std())
            if std < 1e-12:
                z = 0.0
            else:
                z = float((metric_values[row_index] - metric_values.mean()) / std)
            score += z
        row["rank_score"] = score
    return sorted(rows, key=lambda item: float(item["rank_score"]))


def _make_contact_sheet(
    *,
    title: str,
    rows: list[tuple[str, list[np.ndarray]]],
    save_path: Path,
    max_images: int,
    cell_size: int = 112,
) -> None:
    import cv2

    if not rows:
        return
    row_label_width = 220
    title_height = 32
    row_height = cell_size + 24
    width = row_label_width + max_images * cell_size
    height = title_height + len(rows) * row_height
    canvas = Image.new("RGB", (width, height), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill=(0, 0, 0))

    y = title_height
    for label, images in rows:
        draw.text((8, y + 8), label[:36], fill=(0, 0, 0))
        for index, image in enumerate(images[:max_images]):
            resized = cv2.resize(image, (cell_size, cell_size), interpolation=cv2.INTER_AREA)
            canvas.paste(Image.fromarray(resized), (row_label_width + index * cell_size, y))
        y += row_height

    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--simpler-env-root", type=Path, default=DEFAULT_SIMPLER_ENV_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--dataset-samples-per-task", type=int, default=64)
    parser.add_argument("--dataset-frame-fractions", default="0.0,0.25,0.5")
    parser.add_argument("--sim-episodes-per-task", type=int, default=16)
    parser.add_argument("--modes", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--renderer-device", default="")
    parser.add_argument("--enable-raytracing", action="store_true")
    parser.add_argument("--skip-sim", action="store_true", help="Only analyze cached sim images from --sim-cache-dir.")
    parser.add_argument("--sim-cache-dir", type=Path, default=None)
    parser.add_argument("--max-contact-images", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _insert_repo_paths()

    from scripts.debug_bridge_offline_imitation import DEFAULT_TASK_SPECS
    from scripts.eval_pi05_bridge_simpler import IMAGE_PREPROCESS_MODES

    selected_keys = args.tasks or [spec.key for spec in DEFAULT_TASK_SPECS]
    specs = [spec for spec in DEFAULT_TASK_SPECS if spec.key in selected_keys]
    missing = sorted(set(selected_keys) - {spec.key for spec in specs})
    if missing:
        raise ValueError(f"Unknown task keys: {missing}")

    modes = args.modes or list(IMAGE_PREPROCESS_MODES)
    invalid_modes = sorted(set(modes) - set(IMAGE_PREPROCESS_MODES))
    if invalid_modes:
        raise ValueError(f"Invalid preprocess modes: {invalid_modes}; valid={IMAGE_PREPROCESS_MODES}")

    frame_fractions = tuple(float(item) for item in args.dataset_frame_fractions.split(",") if item)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_images_by_task, dataset_metadata = _collect_dataset_images(
        dataset_root=args.dataset_root,
        task_specs=specs,
        samples_per_task=args.dataset_samples_per_task,
        frame_fractions=frame_fractions,
        seed=args.seed,
        video_key="observation.images.image_0",
    )

    sim_cache_dir = args.sim_cache_dir or output_dir / "sim_raw"
    sim_images_by_task: dict[str, list[np.ndarray]] = {}
    if args.skip_sim:
        for spec in specs:
            task_dir = sim_cache_dir / spec.key
            sim_images_by_task[spec.key] = [
                np.asarray(Image.open(path).convert("RGB")) for path in sorted(task_dir.glob("*.png"))
            ]
    else:
        sim_images_by_task = _collect_simpler_images(
            simpler_env_root=args.simpler_env_root.resolve(),
            task_keys=[spec.key for spec in specs],
            episodes_per_task=args.sim_episodes_per_task,
            renderer_device=args.renderer_device,
            enable_raytracing=args.enable_raytracing,
        )
        for task_key, images in sim_images_by_task.items():
            task_dir = sim_cache_dir / task_key
            task_dir.mkdir(parents=True, exist_ok=True)
            for index, image in enumerate(images):
                Image.fromarray(image).save(task_dir / f"{index:03d}.png")

    summary: dict[str, Any] = {
        "dataset_root": str(args.dataset_root),
        "simpler_env_root": str(args.simpler_env_root),
        "modes": modes,
        "dataset_samples_per_task": args.dataset_samples_per_task,
        "sim_episodes_per_task": args.sim_episodes_per_task,
        "dataset_frame_fractions": frame_fractions,
        "tasks": {},
    }

    for spec in specs:
        task_key = spec.key
        if not dataset_images_by_task[task_key]:
            summary["tasks"][task_key] = {
                "error": "no_matching_dataset_images",
                "dataset_metadata_sample": dataset_metadata[task_key][:10],
            }
            print(f"[{task_key}] skipped: no matching Bridge dataset images")
            continue
        if not sim_images_by_task[task_key]:
            summary["tasks"][task_key] = {
                "error": "no_simpler_images",
                "dataset_metadata_sample": dataset_metadata[task_key][:10],
            }
            print(f"[{task_key}] skipped: no SimplerEnv images")
            continue

        dataset_policy_images = _policy_inputs(dataset_images_by_task[task_key], "none")
        dataset_features = _image_features(dataset_policy_images)

        rows: list[dict[str, Any]] = []
        contact_rows: list[tuple[str, list[np.ndarray]]] = [("dataset none->224", dataset_policy_images)]
        for mode in modes:
            sim_policy_images = _policy_inputs(sim_images_by_task[task_key], mode)
            sim_features = _image_features(sim_policy_images)
            distances = _compare_features(dataset_features, sim_features)
            rows.append(
                {
                    "mode": mode,
                    "sim_preprocess": sim_features,
                    "distances": distances,
                }
            )
            contact_rows.append((f"sim {mode}", sim_policy_images))

        ranked = _rank_rows(rows)
        summary["tasks"][task_key] = {
            "dataset": dataset_features,
            "dataset_metadata_sample": dataset_metadata[task_key][:10],
            "ranked_modes": ranked,
        }

        _make_contact_sheet(
            title=f"{task_key}: Bridge dataset vs SimplerEnv policy inputs",
            rows=contact_rows,
            save_path=output_dir / f"{task_key}_contact_sheet.jpg",
            max_images=args.max_contact_images,
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {summary_path}")
    for task_key, task_summary in summary["tasks"].items():
        print(f"\n[{task_key}]")
        for row in task_summary["ranked_modes"]:
            distances = row["distances"]
            print(
                f"{row['mode']:24s} score={row['rank_score']:+.3f} "
                f"hist={distances['hist_rgb_chi2']:.6f} "
                f"black={row['sim_preprocess']['black_fraction']:.4f} "
                f"border_black={row['sim_preprocess']['border_black_fraction']:.4f} "
                f"edge_delta={distances['edge_density_abs']:.4f}"
            )


if __name__ == "__main__":
    main()
