"""Strict loading of complete action-backbone ablation checkpoints."""

import hashlib
import json
from pathlib import Path

import safetensors.torch

from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pi05_subtask_pytorch import Pi05SubtaskPytorch
from openpi.models_pytorch.subtask_decoder import SubtaskDecoderConfig
from openpi.policies.subtask_policy import SubtaskPolicy
from openpi.shared import normalize
from openpi.training.stage1_data import manifest_digest, sha256_file


def create_backbone_gradient_policy(checkpoint, *, device="cpu", num_steps=10, allow_engineering=False):
    checkpoint = Path(checkpoint)
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    config = metadata["config"]
    if (metadata.get("schema_version"), metadata.get("stage"), metadata.get("variant")) != (4, "backbone_grad", "action_backbone_v1"):
        raise ValueError("Expected action_backbone_v1 checkpoint")
    if config["engineering_smoke"] and not allow_engineering:
        raise ValueError("Engineering weights are not a research policy")
    if config["mode"] not in {"limited", "full"} or metadata["completed_steps"] <= 0:
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
    model = Pi05SubtaskPytorch(PI0Pytorch(Pi0Config(**options)).to(device), SubtaskDecoderConfig(**config["decoder"]))
    if hashlib.sha256(model.codec.processor.serialized_model_proto()).hexdigest() != config["tokenizer_model_sha256"]:
        raise ValueError("Tokenizer changed")
    safetensors.torch.load_model(model, checkpoint / "model.safetensors", strict=True)
    return SubtaskPolicy(model, normalize.load(assets), device=device, num_steps=num_steps,
                         metadata={"stage":"backbone_grad", "variant":"action_backbone_v1",
                                   "mode":config["mode"], "checkpoint":str(checkpoint.resolve()),
                                   "weights_sha256":metadata["weights_sha256"],
                                   "parent_weights_sha256":config["parent_weights_sha256"],
                                   "subtask_score_kind":"uncalibrated mean token log probability",
                                   "action_convention":"native Piper absolute joints and grippers; 14 dimensions",
                                   "experimental":True, "state_dim":14, "action_horizon":50,
                                   "subtask_input_required":False, "gripper_unit":"metres"})
