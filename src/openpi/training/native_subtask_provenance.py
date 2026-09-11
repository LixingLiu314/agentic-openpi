"""N1 native-language provenance; no independent subtask network."""

import dataclasses
import hashlib
import json
from pathlib import Path

import torch

from openpi.models.subtask_tokenizer import SubtaskTextCodec
from openpi.models_pytorch.official_backbone_gradient import OFFICIAL_SHA256
from openpi.models_pytorch.native_subtask import VARIANT, DISPLAY_SET, CONTRACT
from openpi.training.runtime_provenance import capture_runtime
from openpi.training.stage1_data import manifest_digest, sha256_file


def build_run_config(args, cfg, dc, world):
    assets = Path(dc.split_manifest).parent
    manifest = json.loads((assets / "split.json").read_text())
    audit = json.loads((assets / "audit.json").read_text())
    norm_sha = sha256_file(assets / "norm_stats.json")
    if manifest["manifest_sha256"] != manifest_digest(manifest) or audit["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("Split/audit provenance mismatch")
    if audit["norm_stats_sha256"] != norm_sha or manifest["action_horizon"] != cfg.model.action_horizon:
        raise ValueError("Normalization/horizon provenance mismatch")
    official_audit_path = Path("logs/pi05_subtask_stage1/official_base_weight_audit.json")
    official_audit = json.loads(official_audit_path.read_text())
    if sha256_file(args.initialize_from / "model.safetensors") != OFFICIAL_SHA256:
        raise ValueError("Only the audited official pi05_base weights may initialize this experiment")
    if (official_audit["local_weights_sha256"] != OFFICIAL_SHA256 or
            official_audit["verified_tensor_count"] != 811 or official_audit["mismatched_tensors"]):
        raise ValueError("Official JAX conversion audit did not pass")
    model_sources = [
        "models/model.py", "models/pi0_config.py", "models/gemma.py",
        "models/subtask_tokenizer.py", "models/tokenizer.py",
        "models_pytorch/pi0_pytorch.py", "models_pytorch/gemma_pytorch.py",
        "models_pytorch/pi05_subtask_pytorch.py", "models_pytorch/subtask_decoder.py",
        "models_pytorch/preprocessing_pytorch.py", "models_pytorch/backbone_gradient.py",
        "models_pytorch/official_backbone_gradient.py",
        "shared/image_tools.py", "shared/normalize.py", "shared/download.py", "transforms.py",
        "policies/piper_policy.py", "policies/subtask_policy.py", "policies/official_gradient_policy.py",
        "training/runtime_provenance.py", "training/subtask_batch.py", "training/subtask_curriculum.py",
        "training/hierarchy_training.py", "training/stage1_optimizer.py", "training/stage1_data.py",
        "training/local_lerobot_dataset.py", "training/config.py", "training/data_loader.py",
        "training/decoded_video_cache.py", "training/wandb_standard.py", "training/official_gradient_provenance.py",
    ]
    sources = [Path("src/openpi") / name for name in model_sources] + [
        Path(name) for name in ["scripts/train_official_backbone_gradient.py", "scripts/train_subtask_pytorch.py",
                               "scripts/visualize_subtask_step.py", "scripts/check_official_gradient_checkpoint.py",
                               "scripts/run_official_gradient_trio.py", "scripts/verify_official_gradient.py",
                               "packages/openpi-client/src/openpi_client/image_tools.py"]
    ]
    result = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
              if key not in {"resume", "stop_after"}}
    result.update(
        world_size=world, runtime=capture_runtime(), model=dataclasses.asdict(cfg.model),
        use_quantile_norm=dc.use_quantile_norm,
        tokenizer_model_sha256=hashlib.sha256(SubtaskTextCodec().processor.serialized_model_proto()).hexdigest(),
        split_sha256=manifest["manifest_sha256"], norm_sha256=norm_sha,
        official_weights_sha256=OFFICIAL_SHA256, parent_weights_sha256=OFFICIAL_SHA256,
        initialization="official_pi05_base", inherited_training_updates=0, native_head="pretrained tied embedding/lm_head",
        official_audit_sha256=sha256_file(official_audit_path), official_verified_jax_tensors=811,
        official_unused_constructor_tensor="paligemma_with_expert.gemma_expert.lm_head.weight",
        sources={str(path): sha256_file(path) for path in sources},
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        variant=VARIANT, global_batch_size=args.global_batch,
        optimizer_transition="fresh B/A optimizers; zero inherited updates",
        gradient_contract=CONTRACT,
        action_conditions="global task and state only; no subtask condition",
        lr_schedule={"warmup_steps":args.warmup, "peak_lr":args.peak_lr, "decay_steps":args.steps,
                     "decay_lr":args.decay_lr, "applies_to":"all trainable groups"},
        optimizer={"name":"AdamW", "betas":[0.9,0.95], "eps":1e-8, "weight_decay":1e-10,
                   "clip_gradient_norm":1.0, "clipping_groups":["B", "A"]},
        training_source="audited official pi05_base B/A including native tied head; no trained parent or new text network",
        evaluation_source="validation_natural", prompt_contract="ordinary global task plus normalized native14 state",
        display_schema=1, display_set=DISPLAY_SET, experiment_family="native-n1-action-stop",
        num_train_steps=args.steps, dataset="eggplant_potato_reach_arm_v1", view_only=False,
        data_pipeline="audited PIL RGB224 cache; persistent workers; pinned/nonblocking transport",
        cache_manifest_sha256=sha256_file(args.decoded_cache / "manifest.json"),
        validation_batch_size=args.eval_batch_size, wandb_train_every=10,
    )
    return json.loads(json.dumps(result))
