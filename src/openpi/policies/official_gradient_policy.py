"""Strict loading of complete action-backbone ablation checkpoints."""

import hashlib
import json
from pathlib import Path

import safetensors.torch

from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pi05_subtask_pytorch import Pi05SubtaskPytorch
from openpi.models_pytorch.subtask_decoder import SubtaskDecoderConfig
from openpi.models_pytorch.official_backbone_gradient import OFFICIAL_SHA256, VARIANT
from openpi.policies.subtask_policy import SubtaskPolicy
from openpi.shared import normalize
from openpi.training.stage1_data import manifest_digest, sha256_file


def create_official_gradient_policy(checkpoint, *, device="cpu", num_steps=10, allow_engineering=False):
    checkpoint = Path(checkpoint)
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    config = metadata["config"]
    if (metadata.get("schema_version"), metadata.get("stage"), metadata.get("variant")) != (5, "official_backbone_grad", VARIANT):
        raise ValueError("Expected official-initialized gradient checkpoint")
    if (config.get("initialization") != "official_pi05_base" or config.get("official_weights_sha256") != OFFICIAL_SHA256
            or config.get("parent_weights_sha256") != OFFICIAL_SHA256 or config.get("inherited_training_updates") != 0):
        raise ValueError("Checkpoint does not have the required official initialization provenance")
    if config["engineering_smoke"] and not allow_engineering:
        raise ValueError("Engineering weights are not a research policy")
    if config["mode"] not in {"frozen", "limited", "full"} or metadata["completed_steps"] <= 0:
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
                         metadata={"stage":"official_backbone_grad", "variant":VARIANT,
                                   "mode":config["mode"], "checkpoint":str(checkpoint.resolve()),
                                   "weights_sha256":metadata["weights_sha256"],
                                   "parent_weights_sha256":config["parent_weights_sha256"],
                                   "initialization":"official_pi05_base", "inherited_training_updates":0,
                                   "subtask_score_kind":"uncalibrated mean token log probability",
                                   "action_convention":"native Piper absolute joints and grippers; 14 dimensions",
                                   "experimental":True, "state_dim":14, "action_horizon":50,
                                   "subtask_input_required":False, "gripper_unit":"metres"})
