"""Session-local recurrent S inference with the unchanged native Piper action path."""
import copy
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
import safetensors.torch
import torch

from openpi.models.model import Observation
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.subtask_decoder import SubtaskDecoderConfig
from openpi.models_pytorch.official_backbone_gradient import OFFICIAL_SHA256
from openpi.models_pytorch.recurrent_subtask import RecurrentSubtaskModel, VARIANT
from openpi.policies.subtask_policy import SubtaskPolicy
from openpi.shared import normalize
from openpi.training.stage1_data import manifest_digest, sha256_file


class RecurrentSubtaskPolicy(SubtaskPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()

    def reset(self):
        self._memory = None
        self._task = None

    def new_session(self):
        policy = copy.copy(self)
        policy.reset()
        return policy

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
        if prompt != self._task:
            self.reset()
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
        texts, statuses, generation, next_memory = self.model.generate_with_memory(context, self._memory)
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
        if result["actions"].shape != (50, 14) or not np.isfinite(result["actions"]).all():
            raise FloatingPointError("Invalid native action output")
        self._memory = None if next_memory is None else next_memory.detach()
        self._task = prompt
        return result


def create_recurrent_subtask_policy(checkpoint, *, device="cpu", num_steps=10, allow_engineering=False):
    checkpoint = Path(checkpoint)
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    config = metadata["config"]
    if (metadata.get("schema_version"), metadata.get("stage"), metadata.get("variant")) != (6, "recurrent_subtask", VARIANT):
        raise ValueError("Expected official-initialized gradient checkpoint")
    if (config.get("initialization") != "official_pi05_base" or config.get("official_weights_sha256") != OFFICIAL_SHA256
            or config.get("parent_weights_sha256") != OFFICIAL_SHA256 or config.get("inherited_training_updates") != 0):
        raise ValueError("Checkpoint does not have the required official initialization provenance")
    if config["engineering_smoke"] and not allow_engineering:
        raise ValueError("Engineering weights are not a research policy")
    if (config["mode"] != "frozen" or config.get("arm") not in {"stateless", "recurrent"}) or metadata["completed_steps"] <= 0:
        raise ValueError("Invalid mode or empty checkpoint")
    if not config["use_quantile_norm"]:
        raise ValueError("Native Piper policy requires audited quantile normalization")
    if sha256_file(checkpoint / "model.safetensors") != metadata["weights_sha256"]:
        raise ValueError("Checkpoint weight fingerprint mismatch")
    assets = checkpoint / "assets/eggplant_potato"
    if sha256_file(assets / "norm_stats.json") != config["norm_sha256"]:
        raise ValueError("Normalization fingerprint mismatch")
    if manifest_digest(json.loads((assets / "split.json").read_text())) != config["split_sha256"]:
        raise ValueError("Episode split fingerprint mismatch")
    options = dict(config["model"])
    if str(device) == "cpu":
        options["dtype"] = "float32"
    model = RecurrentSubtaskModel(PI0Pytorch(Pi0Config(**options)).to(device),
                                   SubtaskDecoderConfig(**config["decoder"]),
                                   recurrent=config["arm"] == "recurrent", seed=config["seed"], unroll=config["unroll"])
    if hashlib.sha256(model.codec.processor.serialized_model_proto()).hexdigest() != config["tokenizer_model_sha256"]:
        raise ValueError("Tokenizer changed")
    safetensors.torch.load_model(model, checkpoint / "model.safetensors", strict=True)
    return RecurrentSubtaskPolicy(model, normalize.load(assets), device=device, num_steps=num_steps,
                         metadata={"stage":"recurrent_subtask", "variant":VARIANT,
                                   "mode":config["mode"], "arm":config["arm"], "memory_scope":"websocket_session", "checkpoint":str(checkpoint.resolve()),
                                   "weights_sha256":metadata["weights_sha256"],
                                   "parent_weights_sha256":config["parent_weights_sha256"],
                                   "initialization":"official_pi05_base", "inherited_training_updates":0,
                                   "subtask_score_kind":"uncalibrated mean token log probability",
                                   "action_convention":"native Piper absolute joints and grippers; 14 dimensions",
                                   "experimental":True, "state_dim":14, "action_horizon":50,
                                   "subtask_input_required":False, "gripper_unit":"metres"})
