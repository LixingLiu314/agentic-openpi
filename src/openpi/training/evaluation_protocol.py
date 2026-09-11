"""Require fixed checkpoints and evaluation options before opening the test split."""

import json

import torch.distributed as dist

from openpi.training.research_checkpoint import require_research_checkpoint
from openpi.training.stage1_data import sha256_file


def validate_evaluation_request(args, metadata):
    if args.split == "val":
        if args.test_protocol is not None:
            raise ValueError("Test protocol may only be supplied for test evaluation")
        return None
    if args.split != "test" or args.test_protocol is None:
        raise ValueError("Test evaluation requires a sealed test protocol")
    protocol = json.loads(args.test_protocol.read_text())
    if protocol.get("status") != "sealed" or protocol.get("split") != "test":
        raise ValueError("Test protocol is not sealed")
    if not 1 <= protocol.get("execution_frames", 0) <= 50:
        raise ValueError("Common execution length must be fixed before test evaluation")
    require_research_checkpoint(args.checkpoint, metadata)
    for field in ["split_sha256", "norm_sha256"]:
        if protocol.get(field) != metadata["config"][field]:
            raise ValueError(f"Test protocol {field} differs from checkpoint")
    checkpoint = args.checkpoint.resolve()
    entry = protocol.get("checkpoints", {}).get(str(checkpoint))
    if entry is None:
        raise ValueError("Checkpoint was not selected in the sealed test protocol")
    if sha256_file(checkpoint / "metadata.json") != entry["metadata_sha256"]:
        raise ValueError("Selected checkpoint metadata changed")
    required = {"draws", "num_steps", "latency_samples", "device", "cpu_threads"}
    if hasattr(args, "action_samples"):
        required |= {"action_samples", "semantic_samples", "no_image", "batch_size", "workers"}
    else:
        required.add("samples")
    if set(entry["evaluation_options"]) != required:
        raise ValueError("Sealed evaluation options are incomplete")
    for key, value in entry["evaluation_options"].items():
        if getattr(args, key) != value:
            raise ValueError(f"Test evaluation option differs from protocol: {key}")
    world = dist.get_world_size() if dist.is_initialized() else 1
    if world != entry["world_size"]:
        raise ValueError("Test evaluation world size differs from protocol")
    rank = dist.get_rank() if dist.is_initialized() else 0
    error = [None]
    if rank == 0:
        try:
            if sha256_file(checkpoint / "model.safetensors") != entry["weights_sha256"]:
                error[0] = "Selected checkpoint weights changed"
        except OSError as exception:
            error[0] = str(exception)
    if world > 1:
        dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise ValueError(error[0])
    return {
        "path": str(args.test_protocol),
        "sha256": sha256_file(args.test_protocol),
        "execution_frames": protocol["execution_frames"],
        "selected_weights_sha256": entry["weights_sha256"],
    }
