#!/usr/bin/env python3
"""Offline imitation sanity check for the pi0.5 Bridge policy.

This script samples Bridge LeRobot frames matching SimplerEnv eval tasks, runs a
trained OpenPI checkpoint on the raw dataset observations, and compares the
predicted action chunk against the dataset action chunk.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import random
import sys
import types
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints/pi05_bridge/bridge_reproduce/30000"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results/debug_bridge_offline_imitation"


def _insert_repo_paths() -> None:
    for path in [
        REPO_ROOT / "packages/openpi-client/src",
        REPO_ROOT / "src",
        REPO_ROOT,
    ]:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _install_unused_jax_extension_stubs() -> None:
    """Stub modules only needed by JAX checkpoint/training paths.

    The PyTorch checkpoint path does not use augmax/orbax, but importing the
    OpenPI policy stack can still import these modules.
    """

    class _StubMeta(type):
        def __getattr__(cls, attr: str) -> type:
            if attr.startswith("__") and attr.endswith("__"):
                raise AttributeError(attr)
            child = _StubMeta(f"{cls.__name__}.{attr}", (object,), {})
            setattr(cls, attr, child)
            return child

        def __call__(cls, *args, **kwargs):  # noqa: ANN002, ANN003
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            if args or kwargs:
                return lambda fn: fn
            return super().__call__()

    class _StubModule(types.ModuleType):
        def __getattr__(self, attr: str) -> type:
            if attr.startswith("__") and attr.endswith("__"):
                raise AttributeError(attr)
            child = _StubMeta(f"{self.__name__}.{attr}", (object,), {})
            setattr(self, attr, child)
            return child

    for name in [
        "augmax",
        "orbax",
        "orbax.checkpoint",
        "orbax.checkpoint.future",
        "orbax.checkpoint.utils",
    ]:
        sys.modules.setdefault(name, _StubModule(name))


@dataclasses.dataclass(frozen=True)
class TaskMatchSpec:
    key: str
    eval_prompt: str
    exact: tuple[str, ...] = ()
    all_keywords: tuple[tuple[str, ...], ...] = ()


DEFAULT_TASK_SPECS = (
    TaskMatchSpec(
        key="stack_green_cube_on_yellow_cube",
        eval_prompt="stack the green block on the yellow block",
        exact=(
            "put the green block on top of the yellow block",
            "put the green cube on top of the yellow cube",
        ),
        all_keywords=(
            ("green block", "yellow block", "top"),
            ("green cube", "yellow cube", "top"),
        ),
    ),
    TaskMatchSpec(
        key="put_carrot_on_plate",
        eval_prompt="put carrot on plate",
        exact=("put carrot on plate",),
        all_keywords=(("carrot", "plate"),),
    ),
    TaskMatchSpec(
        key="put_spoon_on_tablecloth",
        eval_prompt="put the spoon on the towel",
        all_keywords=(
            ("spoon", "towel"),
            ("spoon", "table cloth"),
            ("spoon", "tablecloth"),
        ),
    ),
    TaskMatchSpec(
        key="put_eggplant_in_basket",
        eval_prompt="put eggplant into yellow basket",
        exact=("put eggplant into yellow basket", "put eggplant in yellow basket"),
        all_keywords=(("eggplant", "basket"),),
    ),
)


def _task_matches(task: str, spec: TaskMatchSpec) -> bool:
    normalized = " ".join(task.lower().strip().split())
    if any(normalized == exact.lower() for exact in spec.exact):
        return True
    return any(all(keyword in normalized for keyword in keywords) for keywords in spec.all_keywords)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _resolve_dataset_root(repo_id: str, local_root: str | None, explicit_root: str | None) -> Path:
    candidates: list[Path] = []
    if explicit_root:
        candidates.append(Path(explicit_root).expanduser())
    if local_root:
        root = Path(local_root).expanduser()
        candidates.append(root if root.is_absolute() else (REPO_ROOT / root))

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")).expanduser()
    candidates.extend(
        [
            hf_home / "lerobot" / repo_id,
            Path.home() / ".cache/huggingface/lerobot" / repo_id,
            Path("/media/raid/workspace/xiahongyu/.cache/huggingface/lerobot") / repo_id,
            Path("/media/raid/workspace/xiahongyu/Agent-VLA/playground/Datasets") / repo_id,
            Path("/media/raid/workspace/xiahongyu/LEO-VLA/Datasets") / repo_id,
        ]
    )

    for candidate in candidates:
        if (candidate / "meta/info.json").exists() and (candidate / "meta/episodes.jsonl").exists():
            return candidate.resolve()

    checked = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Could not find LeRobot dataset {repo_id!r}. Checked:\n{checked}")


def _episode_start_indices(episodes: list[dict[str, Any]]) -> dict[int, int]:
    starts: dict[int, int] = {}
    cursor = 0
    for episode in sorted(episodes, key=lambda item: int(item["episode_index"])):
        episode_index = int(episode["episode_index"])
        starts[episode_index] = cursor
        cursor += int(episode["length"])
    return starts


def _candidate_indices_for_spec(
    episodes: list[dict[str, Any]],
    starts: dict[int, int],
    spec: TaskMatchSpec,
    *,
    action_horizon: int,
    max_frames_per_episode: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    indices: list[int] = []
    matched_episodes: list[dict[str, Any]] = []
    for episode in episodes:
        tasks = [str(task) for task in episode.get("tasks", [])]
        matched_task = next((task for task in tasks if _task_matches(task, spec)), None)
        if matched_task is None:
            continue

        episode_index = int(episode["episode_index"])
        length = int(episode["length"])
        max_offset = max(0, length - action_horizon)
        if max_offset <= 0:
            continue

        per_episode = min(max_frames_per_episode, max_offset)
        if per_episode <= 1:
            offsets = [0]
        else:
            offsets = [round(i * (max_offset - 1) / (per_episode - 1)) for i in range(per_episode)]
        indices.extend(starts[episode_index] + int(offset) for offset in offsets)
        matched_episodes.append(
            {
                "episode_index": episode_index,
                "length": length,
                "task": matched_task,
            }
        )
    return indices, matched_episodes


def _tensor_to_numpy(value: Any):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def _summarize_errors(rows: list[dict[str, Any]], action_horizon: int) -> dict[str, Any]:
    import numpy as np

    if not rows:
        return {}

    pred = np.stack([row["pred_actions"] for row in rows], axis=0)
    gt = np.stack([row["gt_actions"] for row in rows], axis=0)
    valid = np.stack([row["valid_mask"] for row in rows], axis=0).astype(bool)

    err = pred - gt
    valid_3d = valid[..., None]
    abs_err = np.abs(err)
    mse = np.where(valid_3d, err**2, np.nan)
    mae = np.where(valid_3d, abs_err, np.nan)

    first = valid[:, 0]
    first_abs = np.abs(pred[:, 0, :] - gt[:, 0, :])
    first_abs = first_abs[first]

    gripper_pred = pred[..., 6] > 0.5
    gripper_gt = gt[..., 6] > 0.5
    gripper_acc = float((gripper_pred[valid] == gripper_gt[valid]).mean()) if valid.any() else float("nan")

    return {
        "num_samples": len(rows),
        "num_valid_action_steps": int(valid.sum()),
        "action_horizon": action_horizon,
        "mae_per_dim": np.nanmean(mae, axis=(0, 1)).round(6).tolist(),
        "rmse_per_dim": np.sqrt(np.nanmean(mse, axis=(0, 1))).round(6).tolist(),
        "mae_all": float(np.nanmean(mae)),
        "rmse_all": float(np.sqrt(np.nanmean(mse))),
        "first_step_mae_per_dim": first_abs.mean(axis=0).round(6).tolist() if len(first_abs) else [],
        "first_step_mae_all": float(first_abs.mean()) if len(first_abs) else float("nan"),
        "gripper_accuracy": gripper_acc,
        "pred_mean_per_dim": np.nanmean(np.where(valid_3d, pred, np.nan), axis=(0, 1)).round(6).tolist(),
        "gt_mean_per_dim": np.nanmean(np.where(valid_3d, gt, np.nan), axis=(0, 1)).round(6).tolist(),
    }


def _to_jsonable(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05_bridge")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", default=None, help="Optional explicit LeRobot dataset root.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--max-frames-per-episode", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tasks", nargs="*", default=[spec.key for spec in DEFAULT_TASK_SPECS])
    parser.add_argument("--custom-task-key", default=None, help="Optional key for one custom task query.")
    parser.add_argument("--custom-task-query", default=None, help="Match episodes containing all words in this query.")
    parser.add_argument("--custom-eval-prompt", default=None, help="Prompt passed to the policy for the custom task.")
    parser.add_argument("--keep-compile", action="store_true", help="Keep torch.compile enabled from the config.")
    parser.add_argument("--fixed-noise", action="store_true", help="Use one fixed diffusion noise tensor per sample.")
    parser.add_argument("--num-action-sampling-steps", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf-datasets-cache-openpi")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    _insert_repo_paths()
    _install_unused_jax_extension_stubs()

    import numpy as np
    import torch
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    from openpi.policies import policy_config
    from openpi.training import config as _config

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_config = _config.get_config(args.config)
    if not args.keep_compile:
        train_config = dataclasses.replace(
            train_config,
            model=dataclasses.replace(train_config.model, pytorch_compile_mode=None),
        )
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.repo_id is None:
        raise ValueError("Config data repo_id is None")

    dataset_root = _resolve_dataset_root(data_config.repo_id, data_config.local_root, args.dataset_root)
    meta = LeRobotDatasetMetadata(data_config.repo_id, root=str(dataset_root))
    episodes = _read_jsonl(dataset_root / "meta/episodes.jsonl")
    starts = _episode_start_indices(episodes)

    specs_by_key = {spec.key: spec for spec in DEFAULT_TASK_SPECS}
    if args.custom_task_query:
        key = args.custom_task_key or "custom"
        query_words = tuple(args.custom_task_query.lower().split())
        specs_by_key[key] = TaskMatchSpec(
            key=key,
            eval_prompt=args.custom_eval_prompt or args.custom_task_query,
            all_keywords=(query_words,),
        )
        if args.tasks == [spec.key for spec in DEFAULT_TASK_SPECS]:
            args.tasks = [key]

    selected_specs = [specs_by_key[key] for key in args.tasks if key in specs_by_key]
    unknown = [key for key in args.tasks if key not in specs_by_key]
    if unknown:
        raise ValueError(f"Unknown task keys: {unknown}; valid keys: {sorted(specs_by_key)}")

    print(f"dataset_root={dataset_root}", flush=True)
    print(f"dataset_fps={meta.fps} total_episodes={len(episodes)}", flush=True)
    print(f"checkpoint={args.checkpoint}", flush=True)
    print(f"device={args.device}", flush=True)

    dataset = LeRobotDataset(
        data_config.repo_id,
        root=str(dataset_root),
        delta_timestamps={
            key: [t / meta.fps for t in range(train_config.model.action_horizon)]
            for key in data_config.action_sequence_keys
        },
        video_backend="pyav",
    )

    policy = policy_config.create_trained_policy(
        train_config,
        args.checkpoint,
        pytorch_device=args.device,
        sample_kwargs={"num_steps": args.num_action_sampling_steps},
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summary: dict[str, Any] = {
        "config": args.config,
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(dataset_root),
        "device": args.device,
        "num_action_sampling_steps": args.num_action_sampling_steps,
        "fixed_noise": args.fixed_noise,
        "tasks": {},
    }

    action_horizon = train_config.model.action_horizon
    action_dim = train_config.model.action_dim
    fixed_noise = (
        np.random.default_rng(args.seed).standard_normal((action_horizon, action_dim)).astype(np.float32)
        if args.fixed_noise
        else None
    )

    for spec in selected_specs:
        candidate_indices, matched_episodes = _candidate_indices_for_spec(
            episodes,
            starts,
            spec,
            action_horizon=action_horizon,
            max_frames_per_episode=args.max_frames_per_episode,
        )
        rng.shuffle(candidate_indices)
        candidate_indices = candidate_indices[: args.num_samples]
        print(
            f"[{spec.key}] matched_episodes={len(matched_episodes)} sampled_frames={len(candidate_indices)} "
            f"policy_prompt={spec.eval_prompt!r}",
            flush=True,
        )

        rows: list[dict[str, Any]] = []
        jsonl_path = output_dir / f"{spec.key}_samples.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as sample_file:
            for sample_index, dataset_index in enumerate(candidate_indices):
                sample = dataset[dataset_index]
                image = _tensor_to_numpy(sample["observation.images.image_0"])
                state = _tensor_to_numpy(sample["observation.state"]).astype(np.float32)
                gt_actions = _tensor_to_numpy(sample["action"]).astype(np.float64)[:, :7]
                valid_mask = ~_tensor_to_numpy(sample.get("action_is_pad", np.zeros(action_horizon, dtype=bool))).astype(
                    bool
                )

                obs = {
                    "observation/image": image,
                    "observation/state": state,
                    "prompt": spec.eval_prompt,
                }
                response = policy.infer(obs, noise=fixed_noise)
                pred_actions = np.asarray(response["actions"], dtype=np.float64)[:action_horizon, :7]

                row = {
                    "sample_index": sample_index,
                    "dataset_index": int(dataset_index),
                    "episode_index": int(_tensor_to_numpy(sample["episode_index"])),
                    "frame_index": int(_tensor_to_numpy(sample["frame_index"])),
                    "dataset_task": str(sample["task"]),
                    "policy_prompt": spec.eval_prompt,
                    "valid_mask": valid_mask,
                    "gt_actions": gt_actions,
                    "pred_actions": pred_actions,
                    "first_step_abs_error": np.abs(pred_actions[0] - gt_actions[0]),
                }
                rows.append(row)
                sample_file.write(json.dumps(_to_jsonable(row), ensure_ascii=False) + "\n")
                sample_file.flush()

                first_mae = float(np.mean(row["first_step_abs_error"]))
                print(
                    f"  sample={sample_index:03d} dataset_index={dataset_index} episode={row['episode_index']} "
                    f"frame={row['frame_index']} first_step_mae={first_mae:.5f}",
                    flush=True,
                )

        summary = _summarize_errors(rows, action_horizon)
        summary["matched_episode_count"] = len(matched_episodes)
        summary["matched_episode_examples"] = matched_episodes[:10]
        summary["sample_jsonl"] = str(jsonl_path)
        all_summary["tasks"][spec.key] = summary

        if rows:
            print(
                f"[{spec.key}] first_step_mae_all={summary['first_step_mae_all']:.5f} "
                f"mae_all={summary['mae_all']:.5f} gripper_acc={summary['gripper_accuracy']:.3f}",
                flush=True,
            )
        else:
            print(f"[{spec.key}] no matching samples found", flush=True)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(_to_jsonable(all_summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved_summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
