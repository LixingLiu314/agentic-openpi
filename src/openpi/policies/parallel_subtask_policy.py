"""Strict loader for the limited/recurrent reach-actor candidate; ordinary Piper A."""
import hashlib
import json
from pathlib import Path

import safetensors.torch

from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.parallel_subtask import ParallelSubtaskModel, VARIANT, SCHEMA_VERSION
from openpi.models_pytorch.subtask_decoder import SubtaskDecoderConfig
from openpi.models_pytorch.official_backbone_gradient import OFFICIAL_SHA256
from openpi.policies.recurrent_subtask_policy import RecurrentSubtaskPolicy
from openpi.shared import normalize
from openpi.training.reach_arm_data import ANNOTATIONS_SHA256, NORM_SHA256, SPLIT_FILE_SHA256
from openpi.training.stage1_data import manifest_digest, sha256_file


def create_parallel_policy(checkpoint, *, device="cpu", num_steps=10, allow_engineering=False):
    checkpoint=Path(checkpoint)
    metadata=json.loads((checkpoint/"metadata.json").read_text())
    config=metadata["config"]
    if (metadata.get("schema_version"),metadata.get("stage"),metadata.get("variant")) != (SCHEMA_VERSION,"parallel_subtask",VARIANT):
        raise ValueError("Expected parallel action-stop checkpoint")
    if (config.get("initialization"),config.get("official_weights_sha256"),config.get("parent_weights_sha256"),
            config.get("inherited_training_updates")) != ("official_pi05_base",OFFICIAL_SHA256,OFFICIAL_SHA256,0):
        raise ValueError("Official fresh initialization provenance is required")
    if config["engineering_smoke"] and not allow_engineering:
        raise ValueError("Engineering weights are not a research policy")
    if (config.get("mode"),config.get("arm"),config.get("label_version"),config.get("annotations_sha256"),
            config.get("memory_tokens"),config.get("unroll"),config.get("condition_dropout")) != (
            "action_stop","recurrent","reach_arm_v1",ANNOTATIONS_SHA256,4,4,0.0):
        raise ValueError("Parallel architecture or label contract differs")
    if config.get("gradient_contract") != "CE->S+B; flow->A only; every prefix K/V detached for A":
        raise ValueError("Incorrect gradient contract")
    if metadata["completed_steps"] <= 0 or not config["use_quantile_norm"]:
        raise ValueError("Empty checkpoint or incorrect normalization")
    if not config["engineering_smoke"] and config.get("steps") != 5000:
        raise ValueError("Formal candidate must retain its 5000-update recipe")
    if sha256_file(checkpoint/"model.safetensors") != metadata["weights_sha256"]:
        raise ValueError("Deployment weight fingerprint mismatch")
    assets=checkpoint/"assets/eggplant_potato"
    if sha256_file(assets/"norm_stats.json") != NORM_SHA256 or config["norm_sha256"] != NORM_SHA256:
        raise ValueError("Normalization fingerprint mismatch")
    if sha256_file(assets/"split.json") != SPLIT_FILE_SHA256:
        raise ValueError("Wrong episode split file")
    if manifest_digest(json.loads((assets/"split.json").read_text())) != config["split_sha256"]:
        raise ValueError("Episode manifest fingerprint mismatch")
    options=dict(config["model"])
    if str(device)=="cpu": options["dtype"]="float32"
    model=ParallelSubtaskModel(PI0Pytorch(Pi0Config(**options)).to(device),
                                SubtaskDecoderConfig(**config["decoder"]),recurrent=True,seed=config["seed"],unroll=4)
    # This export has no LoRA adapters and uses the ordinary global-only action prefix.
    if hashlib.sha256(model.codec.processor.serialized_model_proto()).hexdigest() != config["tokenizer_model_sha256"]:
        raise ValueError("Tokenizer fingerprint mismatch")
    safetensors.torch.load_model(model,checkpoint/"model.safetensors",strict=True)
    return RecurrentSubtaskPolicy(model,normalize.load(assets),device=device,num_steps=num_steps,
        metadata=dict(stage="parallel_subtask",variant=VARIANT,mode="action_stop",arm="recurrent",
                      label_version="reach_arm_v1",memory_scope="websocket_session",checkpoint=str(checkpoint.resolve()),
                      weights_sha256=metadata["weights_sha256"],parent_weights_sha256=OFFICIAL_SHA256,
                      initialization="official_pi05_base",inherited_training_updates=0,
                      subtask_score_kind="uncalibrated mean token log probability",
                      action_convention="native Piper absolute joints and grippers; 14 dimensions",
                      experimental=True,state_dim=14,action_horizon=50,subtask_input_required=False,gripper_unit="metres"))
