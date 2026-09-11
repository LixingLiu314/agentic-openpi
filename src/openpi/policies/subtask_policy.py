"""Native Piper policy: observations and global task in; subtask and actions out."""

import hashlib
import json
from pathlib import Path
import time

import jax
import numpy as np
from openpi_client.base_policy import BasePolicy
import safetensors.torch
import torch

from openpi import transforms
from openpi.models.model import Observation
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pi05_subtask_pytorch import Pi05SubtaskPytorch
from openpi.models_pytorch.subtask_decoder import SubtaskDecoderConfig
from openpi.policies.piper_policy import JOINT_MASK
from openpi.policies.piper_policy import PiperInputs
from openpi.policies.piper_policy import PiperOutputs
from openpi.shared import normalize
from openpi.training.config import ModelTransformFactory
from openpi.training.stage1_data import manifest_digest
from openpi.training.stage1_data import sha256_file
from openpi.training.subtask_curriculum import selection_eligible


class SubtaskPolicy(BasePolicy):
    def __init__(self, model, norm_stats, *, device="cpu", num_steps=10, metadata=None):
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.num_steps = num_steps
        self._metadata = metadata or {}
        model_transforms = ModelTransformFactory()(model.base.config)
        self.input_transform = transforms.compose(
            [
                PiperInputs(),
                transforms.Normalize(norm_stats, use_quantiles=True),
                *model_transforms.inputs,
            ]
        )
        self.output_transform = transforms.compose(
            [
                *model_transforms.outputs,
                transforms.Unnormalize(norm_stats, use_quantiles=True),
                transforms.AbsoluteActions(JOINT_MASK),
                PiperOutputs(),
            ]
        )

    def _timestamp(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    @torch.no_grad()
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        forbidden = {
            "action",
            "actions",
            "subtask",
            "subtask_target_ids",
            "subtask_target_mask",
            "target_ids",
            "target_mask",
            "labels",
        }
        if forbidden & obs.keys():
            raise ValueError("Deployment observations must not contain subtask supervision")
        prompt = obs.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Provide a nonempty global task in prompt")
        begun = self._timestamp()
        # PiperInputs copies state and rebuilds the image dictionary; transforms cannot mutate caller state.
        inputs = self.input_transform(obs)
        tensors = jax.tree.map(lambda value: torch.as_tensor(np.asarray(value), device=self.device)[None], inputs)
        observation = Observation.from_dict(tensors)
        if noise is not None:
            noise = torch.as_tensor(noise, device=self.device, dtype=torch.float32)
            if noise.ndim == 2:
                noise = noise[None]
            expected = (1, self.model.base.config.action_horizon, self.model.base.config.action_dim)
            if noise.shape != expected or not torch.isfinite(noise).all():
                raise ValueError(f"Expected finite flow noise with shape {expected}")
        after_inputs = self._timestamp()
        timings = {"input_ms": (after_inputs - begun) * 1000}
        context = self.model.prepare_context(observation, [prompt], timings=timings)
        start_text = self._timestamp()
        texts, statuses, generation = self.model.generate_subtask(context)
        after_text = self._timestamp()
        prefix = self.model.action_prefix(context, texts)
        after_prefix = self._timestamp()
        actions = self.model.sample_actions_from_prefix(context, prefix, noise=noise, num_steps=self.num_steps)
        after_actions = self._timestamp()
        result = self.output_transform(
            {
                "state": tensors["state"][0].cpu().numpy(),
                "actions": actions[0].float().cpu().numpy(),
            }
        )
        result.update(
            subtask=texts[0],
            subtask_status=statuses[0],
            subtask_score=float(generation.mean_log_probability[0]),
        )
        ended = self._timestamp()
        timings.update(
            subtask_ms=(after_text - start_text) * 1000,
            prefix2_ms=(after_prefix - after_text) * 1000,
            flow_ms=(after_actions - after_prefix) * 1000,
            output_ms=(ended - after_actions) * 1000,
            infer_ms=(ended - begun) * 1000,
        )
        result["policy_timing"] = timings
        return result

    @property
    def metadata(self):
        return dict(self._metadata)


def create_subtask_policy(checkpoint, *, device="cpu", num_steps=10, allow_engineering=False):
    checkpoint = Path(checkpoint)
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    if metadata["schema_version"] != 3 or metadata["stage"] not in {"m1", "m2", "m3"}:
        raise ValueError("Expected a hierarchical pi05 checkpoint")
    config = metadata["config"]
    if not allow_engineering and (metadata["stage"] != "m3" or config["engineering_smoke"]):
        raise ValueError("Normal deployment requires a research M3 checkpoint")
    if not allow_engineering and not selection_eligible(
        metadata["stage"], metadata["completed_steps"], config["steps"], metadata["counters"]["action"]
    ):
        raise ValueError("Deployment checkpoint has not completed the final generated-only action budget")
    if not config.get("use_quantile_norm", True):
        raise ValueError("Piper stage-one policy requires the audited quantile normalization")
    model_config = Pi0Config(**config["model"])
    base = PI0Pytorch(model_config).to(device)
    model = Pi05SubtaskPytorch(base, SubtaskDecoderConfig(**config["decoder"]))
    actual_tokenizer = hashlib.sha256(model.codec.processor.serialized_model_proto()).hexdigest()
    if actual_tokenizer != config["tokenizer_model_sha256"]:
        raise ValueError("Checkpoint tokenizer vocabulary changed")
    assets = checkpoint / "assets" / "eggplant_potato"
    if sha256_file(assets / "norm_stats.json") != config["norm_sha256"]:
        raise ValueError("Checkpoint normalization fingerprint changed")
    manifest = json.loads((assets / "split.json").read_text())
    if manifest_digest(manifest) != config["split_sha256"]:
        raise ValueError("Checkpoint split fingerprint changed")
    safetensors.torch.load_model(model, checkpoint / "model.safetensors", strict=True)
    return SubtaskPolicy(
        model,
        normalize.load(assets),
        device=device,
        num_steps=num_steps,
        metadata={
            "stage": metadata["stage"],
            "checkpoint": str(checkpoint),
            "c0_weights_sha256": config["c0_weights_sha256"],
            "subtask_score_kind": "uncalibrated mean token log probability",
            "action_convention": "native Piper absolute joints and grippers; 14 dimensions",
        },
    )
