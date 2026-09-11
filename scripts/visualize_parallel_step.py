"""Parallel first-update media; S text is display-only, never action conditioning."""

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


@torch.no_grad()
def render_first_step(model, batch, data_config, output, *, step=1, origin="live first update; cold-start S diagnostic", limit=2):
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
        "action_note": "Parallel Action-stop: global task/state/images condition A; S text is display-only.",
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
