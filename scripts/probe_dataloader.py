#!/usr/bin/env python3
"""Read-only dataloader probe for OpenPI training configs.

This script intentionally does not instantiate the model or write checkpoints.
It builds the same PyTorch data-loading path used by scripts/train_pytorch.py,
fetches one batch, and prints enough metadata to catch silent data-format
failures before or during long-running experiments.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from openpi import transforms as _transforms
from openpi.training import config as _config
from openpi.training import data_loader as _data


def _shape_dtype(value: Any) -> str:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    return f"shape={tuple(shape) if shape is not None else 'unknown'} dtype={dtype}"


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _load_info_json(local_root: str | None) -> dict[str, Any]:
    if local_root is None:
        return {}
    info_path = Path(local_root) / "meta" / "info.json"
    if not info_path.exists():
        return {}
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _unwrap_dataset(dataset: Any) -> list[Any]:
    chain = [dataset]
    while hasattr(dataset, "_dataset"):
        dataset = dataset._dataset
        chain.append(dataset)
    return chain


def _find_sentencepiece(data_config: _config.DataConfig):
    for transform in data_config.model_transforms.inputs:
        tokenizer = getattr(transform, "tokenizer", None)
        sentencepiece = getattr(tokenizer, "_tokenizer", None)
        if sentencepiece is not None:
            return sentencepiece
    return None


def _decode_prompts(observation: Any, data_config: _config.DataConfig, limit: int) -> None:
    tokens = observation.tokenized_prompt
    masks = observation.tokenized_prompt_mask
    if tokens is None:
        print("Decoded prompts: tokenized_prompt is absent")
        return

    sentencepiece = _find_sentencepiece(data_config)
    if sentencepiece is None:
        print("Decoded prompts: tokenizer decoder not found")
        return

    token_arr = _to_numpy(tokens)
    mask_arr = _to_numpy(masks).astype(bool) if masks is not None else np.ones_like(token_arr, dtype=bool)
    count = min(limit, token_arr.shape[0])

    print(f"\nDecoded tokenized prompts from the fetched batch (first {count}):")
    for i in range(count):
        ids = token_arr[i][mask_arr[i]].astype(int).tolist()
        decoded = sentencepiece.decode(ids)
        print(f"[{i}] {decoded}")


def _raw_prompt_preview(config: _config.TrainConfig, data_config: _config.DataConfig, limit: int) -> None:
    """Print full pre-tokenization prompts for deterministic samples.

    The actual batch prompt is tokenized by the training dataloader, so this
    preview runs the same transform chain only up to TokenizePrompt. It is meant
    to expose the full condition string when token decoding is truncated by the
    model's max token length.
    """

    model_transforms = []
    for transform in data_config.model_transforms.inputs:
        if transform.__class__.__name__.startswith("Tokenize"):
            break
        model_transforms.append(transform)

    raw_dataset = _data.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    preview_dataset = _data.TransformedDataset(
        raw_dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *model_transforms,
        ],
    )

    count = min(limit, len(preview_dataset))
    print(f"\nFull pre-tokenization prompt preview from dataset order (first {count}):")
    for i in range(count):
        sample = preview_dataset[i]
        prompt = sample.get("prompt", "<missing prompt>")
        if not isinstance(prompt, str):
            prompt = str(prompt.item() if hasattr(prompt, "item") else prompt)
        print(f"[{i}] {prompt}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True, help="OpenPI training config name, e.g. pi05_bridge_traj")
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the same shuffle=True behavior as train_pytorch.py by default.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional dataloader worker override. Omit for the exact config value.",
    )
    parser.add_argument("--num-prompts", type=int, default=3, help="Number of prompts/images to print from one batch.")
    parser.add_argument(
        "--skip-raw-preview",
        action="store_true",
        help="Skip the extra full prompt preview before tokenization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = _config.get_config(args.config_name)
    if args.num_workers is not None:
        config = dataclasses.replace(config, num_workers=args.num_workers)

    data_config = config.data.create(config.assets_dirs, config.model)
    info = _load_info_json(data_config.local_root)

    print("=== Config ===")
    print(f"config_name: {config.name}")
    print(f"repo_id:     {data_config.repo_id}")
    print(f"local_root:  {data_config.local_root}")
    print(f"asset_id:    {data_config.asset_id}")
    print(f"batch_size:  {config.batch_size}")
    print(f"shuffle:     {args.shuffle}")
    print(f"num_workers: {config.num_workers}")

    loader = _data.create_data_loader(config, framework="pytorch", shuffle=args.shuffle)
    torch_wrapper = loader._data_loader  # noqa: SLF001 - probe script intentionally introspects loader internals.
    torch_loader = torch_wrapper.torch_loader
    dataset = torch_loader.dataset
    dataset_chain = _unwrap_dataset(dataset)

    dataset_len = len(dataset)
    batch_size = torch_loader.batch_size
    batches_per_epoch = len(torch_loader)
    global_batch = config.batch_size
    global_batches_per_epoch = math.floor(dataset_len / global_batch) if global_batch else None

    print("\n=== Data Volume ===")
    print(f"metadata_total_episodes: {info.get('total_episodes', 'unknown')}")
    print(f"metadata_total_frames:   {info.get('total_frames', 'unknown')}")
    print(f"dataloader_samples:      {dataset_len}")
    print(f"torch_batch_size:        {batch_size}")
    print(f"drop_last:               {torch_loader.drop_last}")
    print(f"batches_per_epoch:       {batches_per_epoch}")
    print(f"global_batch_size:       {global_batch}")
    print(f"global_batches_per_epoch_estimate: {global_batches_per_epoch}")
    print("dataset_chain:")
    for depth, item in enumerate(dataset_chain):
        print(f"  {depth}: {type(item).__module__}.{type(item).__name__} len={len(item)}")

    print("\n=== One Batch ===")
    observation, actions = next(iter(loader))
    print(f"state:   {_shape_dtype(observation.state)}")
    print(f"actions: {_shape_dtype(actions)}")

    print("\nImages:")
    if not observation.images:
        print("  <no image keys>")
    for key, image in observation.images.items():
        mask = observation.image_masks.get(key)
        mask_summary = ""
        if mask is not None:
            mask_np = _to_numpy(mask).astype(bool)
            mask_summary = f" valid={int(mask_np.sum())}/{mask_np.size}"
        print(f"  {key}: {_shape_dtype(image)}{mask_summary}")

    _decode_prompts(observation, data_config, args.num_prompts)
    if not args.skip_raw_preview:
        _raw_prompt_preview(config, data_config, args.num_prompts)


if __name__ == "__main__":
    main()
