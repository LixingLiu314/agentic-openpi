"""
Debug utilities for inspecting training inputs.

Activated via --debug_steps N in train_pytorch.py.
For each of the first N steps, saves to <checkpoint_dir>/debug/step_XXXXXX/:
  summary.json        — decoded prompts, state/action shapes, token counts
  images/             — one PNG per camera per sample
  state.npy           — state tensor  [samples, state_dim]
  actions.npy         — actions tensor [samples, horizon, action_dim]
"""

import json
import pathlib

import numpy as np
import torch


class DebugInputSaver:
    def __init__(self, save_dir: pathlib.Path, num_samples: int = 4):
        self.save_dir = pathlib.Path(save_dir)
        self.num_samples = num_samples
        self._tokenizer = None

    def _tokenizer_(self):
        if self._tokenizer is None:
            from openpi.models.tokenizer import PaligemmaTokenizer
            self._tokenizer = PaligemmaTokenizer(max_len=200)
        return self._tokenizer

    def _decode_prompts(self, tokenized_prompt, tokenized_prompt_mask, n: int) -> list[str]:
        if tokenized_prompt is None:
            return ["<no prompt>"] * n
        tok = self._tokenizer_()
        prompts = []
        for i in range(n):
            tokens = tokenized_prompt[i].cpu().numpy()
            mask = tokenized_prompt_mask[i].cpu().numpy().astype(bool)
            text = tok._tokenizer.decode(tokens[mask].tolist())
            prompts.append(text)
        return prompts

    def _save_images(self, images: dict, step_dir: pathlib.Path, n: int):
        """Save images as PNG. Handles float32 [-1,1] and uint8 [0,255]."""
        try:
            from PIL import Image
        except ImportError:
            return

        img_dir = step_dir / "images"
        img_dir.mkdir(exist_ok=True)

        for cam_name, img_tensor in images.items():
            # img_tensor: [B, H, W, C] or [B, C, H, W]
            if img_tensor.ndim == 4 and img_tensor.shape[1] in (1, 3):
                # CHW → HWC
                img_tensor = img_tensor.permute(0, 2, 3, 1)

            img_np = img_tensor.cpu().float().numpy()

            # Float [-1, 1] → uint8 [0, 255]
            if img_np.min() < 0 or img_np.max() <= 1.0:
                img_np = ((img_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

            for i in range(min(n, img_np.shape[0])):
                frame = img_np[i]
                if frame.shape[-1] == 1:
                    frame = frame[..., 0]
                Image.fromarray(frame).save(img_dir / f"{cam_name}_s{i}.png")

    def save(self, step: int, observation, actions: torch.Tensor, loss: float | None = None):
        """Save debug info for one training step."""
        step_dir = self.save_dir / f"step_{step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        n = min(self.num_samples, observation.state.shape[0])

        # Decode prompts
        prompts = self._decode_prompts(
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            n,
        )

        # Compute token counts
        token_counts = []
        if observation.tokenized_prompt_mask is not None:
            for i in range(n):
                token_counts.append(int(observation.tokenized_prompt_mask[i].sum().item()))

        # Save summary JSON
        summary = {
            "step": step,
            "loss": loss,
            "batch_size": int(observation.state.shape[0]),
            "samples_saved": n,
            "state_shape": list(observation.state.shape),
            "actions_shape": list(actions.shape),
            "image_keys": list(observation.images.keys()),
            "token_counts": token_counts,
            "prompts": prompts,
        }
        (step_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

        # Save images
        self._save_images(observation.images, step_dir, n)

        # Save state and actions as numpy
        np.save(step_dir / "state.npy", observation.state[:n].cpu().float().numpy())
        np.save(step_dir / "actions.npy", actions[:n].cpu().float().numpy())
