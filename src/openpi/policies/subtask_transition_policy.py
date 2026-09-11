"""Strict decoder-delta R1 loader; immutable M3 parent remains the action owner."""
import json
from pathlib import Path

import safetensors.torch

from openpi.policies.subtask_policy import create_subtask_policy
from openpi.training.stage1_data import sha256_file


def create_transition_policy(checkpoint, *, device="cpu", num_steps=10, parent_checkpoint=None,
                             allow_engineering=False, require_candidate=True):
    checkpoint=Path(checkpoint)
    meta=json.loads((checkpoint/"metadata.json").read_text())
    if meta.get("schema_version")!=1 or meta.get("variant")!="r1_boundary_ce_v1":
        raise ValueError("Expected an R1 decoder-delta checkpoint")
    if meta["config"]["seed"]!=42 or (meta["config"]["engineering_smoke"] and not allow_engineering):
        raise ValueError("Expected a seed42 research checkpoint")
    if not meta["frozen_base_equal"] or meta["base_tensor_hash"]!=meta["initial_base_tensor_hash"]:
        raise ValueError("Frozen B/A identity changed")
    if sha256_file(checkpoint/"decoder.safetensors")!=meta["decoder_sha256"]:
        raise ValueError("Decoder checksum mismatch")
    if require_candidate:
        candidate=json.loads((checkpoint.parent/"candidate.json").read_text())
        if candidate.get("checkpoint")!=checkpoint.name or candidate.get("decoder_sha256")!=meta["decoder_sha256"] or not candidate.get("proxy_gates_passed"):
            raise ValueError("This checkpoint did not pass the R1 candidate gates")
    parent=Path(parent_checkpoint or meta["config"]["initialize_from"])
    if sha256_file(parent/"model.safetensors")!=meta["config"]["parent_weights_sha256"]:
        raise ValueError("R1 parent weights changed")
    parent_meta=json.loads((parent/"metadata.json").read_text())
    for key in ["split_sha256","norm_sha256"]:
        if parent_meta["config"][key]!=meta["config"][key]:
            raise ValueError("R1 and parent data identities differ")
    policy=create_subtask_policy(parent,device=device,num_steps=num_steps)
    safetensors.torch.load_model(policy.model.decoder,checkpoint/"decoder.safetensors",strict=True)
    policy.model.eval()
    policy._metadata.update(variant=meta["variant"],checkpoint=str(checkpoint),parent_checkpoint=str(parent),
                            decoder_sha256=meta["decoder_sha256"],physical_readiness_reviewed=False)
    return policy
