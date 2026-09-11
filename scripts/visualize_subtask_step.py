"""First-update observations and genuine BOS-only subtask/action inference."""

import argparse
import dataclasses
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import jax
import numpy as np
import torch


def first_batch_indices(config, labels, tasks):
    from train_subtask_pytorch import StepBatchSampler

    from openpi.training.subtask_batch import BalancedSubtaskSampler

    options = {
        "batch_size": config["batch_size"],
        "accumulation": config["accumulation"],
        "steps": 1,
        "start": 0,
        "seed": config["seed"],
        "rank": 0,
        "world_size": config["world_size"],
    }
    sampler = (
        BalancedSubtaskSampler(labels, tasks=tasks, **options)
        if config["stage"] == "m1"
        else StepBatchSampler(len(labels), **options)
    )
    return next(iter(sampler))


@torch.no_grad()
def render_first_step(model, batch, data_config, output, *, step=1, origin="live first update", limit=2):
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    from openpi import transforms
    from openpi.policies.piper_policy import JOINT_MASK
    from openpi.training.hierarchy_training import random_state
    from openpi.training.hierarchy_training import restore_random_state

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    saved_rng = random_state()
    previous_training = model.training
    manifest = {
        "step": step,
        "origin": origin,
        "stage": model.stage,
        "action_note": "M1 action expert is frozen; actions are diagnostic inference."
        if model.stage == "m1"
        else "Generated subtask conditions action inference.",
        "flow_steps": 10,
        "noise_seed": 73001,
        "samples": [],
    }
    try:
        count = min(limit, len(batch.labels))
        small = dataclasses.replace(
            batch,
            observation=jax.tree.map(lambda x: x[:count], batch.observation),
            **{
                key: getattr(batch, key)[:count]
                for key in [
                    "actions",
                    "global_prompts",
                    "labels",
                    "target_ids",
                    "target_mask",
                    "episode_indices",
                    "frame_indices",
                ]
            },
        )
        noise = torch.randn(small.actions.shape, generator=torch.Generator().manual_seed(73001)).to(
            small.actions.device
        )
        # Labels/targets are deliberately absent from this call.
        predicted = model.infer(small.observation, small.global_prompts, noise=noise, num_steps=10)
        inverse = transforms.Unnormalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm)
        absolute = transforms.AbsoluteActions(JOINT_MASK)
        for index in range(count):
            state = small.observation.state[index].float().cpu().numpy()
            target = small.actions[index].float().cpu().numpy()
            actions = predicted["actions"][index].float().cpu().numpy()
            truth = absolute(inverse({"state": state.copy(), "actions": target.copy()}))
            result = absolute(inverse({"state": state.copy(), "actions": actions.copy()}))
            episode, frame = int(small.episode_indices[index]), int(small.frame_indices[index])
            caption = f"Step {step} | episode {episode}, frame {frame} | {origin}"
            text = f"Task: {small.global_prompts[index]}\nGT subtask: {small.labels[index]}\nGenerated: {predicted['subtasks'][index]!r} | status: {predicted['subtask_status'][index]}\n{manifest['action_note']}"
            figure, axes = plt.subplots(2, 3, figsize=(15, 7), gridspec_kw={"height_ratios": [2, 1]})
            for axis, (name, images) in zip(axes[0], small.observation.images.items(), strict=True):
                image = images[index].float().cpu().numpy()
                if image.shape[0] == 3:
                    image = image.transpose(1, 2, 0)
                if image.min() < 0:
                    image = (image + 1) / 2
                elif image.max() > 1:
                    image = image / 255
                axis.imshow(np.clip(image, 0, 1))
                axis.set_title(name)
                axis.axis("off")
            for axis in axes[1]:
                axis.axis("off")
            axes[1, 0].text(0, 1, text, va="top", fontsize=10, transform=axes[1, 0].transAxes)
            axes[1, 2].axis("on")
            axes[1, 2].plot(truth["state"][:14], "o-")
            axes[1, 2].set_title("Native state (14 dimensions)")
            axes[1, 2].set_xlabel("Dimension")
            figure.suptitle(caption, fontsize=12)
            figure.tight_layout()
            input_name = f"sample_{index}_inputs.png"
            figure.savefig(output / input_name, dpi=140)
            plt.close(figure)
            figure, axes = plt.subplots(7, 2, figsize=(14, 15), sharex=True)
            for dimension, axis in enumerate(axes.flat):
                axis.plot(truth["actions"][:, dimension], label="Ground truth", linewidth=1.7)
                axis.plot(result["actions"][:, dimension], label="Predicted (10 flow steps)", linewidth=1.4)
                axis.set_title(
                    f"{'Gripper' if dimension in [6, 13] else 'Joint'} {dimension} — native dataset units", fontsize=10
                )
                axis.grid(alpha=0.2)
            axes[0, 0].legend(fontsize=8)
            for axis in axes[-1]:
                axis.set_xlabel("Future frame offset (30 Hz; endpoint padding follows dataset)")
            figure.suptitle(caption + "\n" + manifest["action_note"], fontsize=11)
            figure.tight_layout(rect=(0, 0, 1, 0.96))
            action_name = f"sample_{index}_actions.png"
            figure.savefig(output / action_name, dpi=120)
            plt.close(figure)
            np.savez_compressed(
                output / f"sample_{index}.npz",
                native_state=truth["state"][:14],
                target_native=truth["actions"][:, :14],
                predicted_native=result["actions"][:, :14],
                target_normalized=target,
                predicted_normalized=actions,
            )
            details = {
                "episode": episode,
                "frame": frame,
                "task": small.global_prompts[index],
                "label": small.labels[index],
                "prediction": predicted["subtasks"][index],
                "status": predicted["subtask_status"][index],
            }
            for filename in [input_name, action_name]:
                manifest["samples"].append({"image": filename, "caption": caption + " | " + text, **details})
        temporary = output / ".manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(output / "manifest.json")
        return manifest
    finally:
        model.train(previous_training)
        restore_random_state(saved_rng)


def main():
    import safetensors.torch

    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.models_pytorch.pi05_subtask_pytorch import Pi05SubtaskPytorch
    from openpi.models_pytorch.subtask_decoder import SubtaskDecoderConfig
    from openpi.shared import normalize
    from openpi.training import config as config_lib
    from openpi.training import data_loader
    from openpi.training.stage1_data import sha256_file
    from openpi.training.subtask_batch import SubtaskTrainingDataset
    from openpi.training.subtask_batch import collate_subtask

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(8)
    metadata = json.loads((args.checkpoint / "metadata.json").read_text())
    if metadata["stage"] not in {"m1", "m2", "m3"} or metadata["completed_steps"] != 1:
        raise ValueError("First-step replay requires a hierarchy step-1 checkpoint")
    config = metadata["config"]
    model_config = Pi0Config(**config["model"])
    if args.device == "cpu":
        model_config = dataclasses.replace(model_config, dtype="float32")
    cfg = config_lib.get_config("pi05_piper_stage1")
    dc = cfg.data.create(cfg.assets_dirs, model_config)
    assets = args.checkpoint / "assets" / "eggplant_potato"
    if sha256_file(assets / "norm_stats.json") != config["norm_sha256"]:
        raise ValueError("Normalization provenance mismatch")
    if sha256_file(assets / "split.json") != sha256_file(Path(dc.split_manifest)):
        raise ValueError("Split provenance mismatch")
    dc = dataclasses.replace(dc, norm_stats=normalize.load(assets))
    raw = data_loader.create_torch_dataset(dc, model_config.action_horizon, model_config)
    dataset = SubtaskTrainingDataset(raw, dc)
    labels, tasks = dataset.labels_without_video(), list(raw.hf_dataset["task"])
    if config.get("overfit_samples"):
        selected = json.loads((args.checkpoint.parent / "overfit_indices.json").read_text())
        dataset = torch.utils.data.Subset(dataset, selected)
        labels, tasks = [labels[i] for i in selected], [tasks[i] for i in selected]
    indices = first_batch_indices(config, labels, tasks)[: args.limit]
    batch = collate_subtask([dataset[i] for i in indices]).to(args.device)
    model = Pi05SubtaskPytorch(PI0Pytorch(model_config).to(args.device), SubtaskDecoderConfig(**config["decoder"]))
    safetensors.torch.load_model(model, args.checkpoint / "model.safetensors", strict=True)
    model.set_stage(metadata["stage"])
    origin = f"checkpoint replay; rank-0 first microbatch; compute={model_config.dtype}"
    result = render_first_step(model, batch, dc, args.checkpoint.parent / "first_step", origin=origin, limit=args.limit)
    print(
        json.dumps(
            {
                "event": "visualized",
                "samples": len(indices),
                "origin": origin,
                "files": [x["image"] for x in result["samples"]],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
