"""Recognize research checkpoints, including the explicitly selected legacy C0."""

import json
from pathlib import Path

from openpi.training.stage1_data import sha256_file

RESEARCH_PROTOCOL_PATH = Path("assets/pi05_piper_stage1/eggplant_potato/research_protocol_v1.json")


def require_research_checkpoint(checkpoint, metadata):
    """Reject engineering/unknown metadata; admit legacy M0 only by its fixed hash."""
    config = metadata.get("config", {})
    flag = config.get("engineering_smoke")
    if flag is False:
        return
    if flag is not None or metadata.get("stage") != "m0" or metadata.get("schema_version") != 2:
        raise ValueError("Engineering or unidentified checkpoint cannot be used as research weights")
    protocol = json.loads(RESEARCH_PROTOCOL_PATH.read_text())
    checkpoint = Path(checkpoint)
    if checkpoint.resolve() != Path(protocol["c0_shared_across_seeds"]).resolve():
        raise ValueError("Legacy M0 is not the C0 selected in the research protocol")
    if config.get("stage") != "m0" or config.get("model", {}).get("pi05") is not True:
        raise ValueError("Legacy C0 metadata is incompatible")
    if metadata.get("completed_steps", 0) < 1 or metadata["completed_steps"] != config.get("steps"):
        raise ValueError("Selected legacy C0 training is incomplete")
    for field in ["split_sha256", "norm_sha256"]:
        if config.get(field) != protocol[field]:
            raise ValueError(f"Selected legacy C0 {field} differs from protocol")
    if sha256_file(checkpoint / "model.safetensors") != protocol["c0_weights_sha256"]:
        raise ValueError("Selected legacy C0 weights differ from protocol")
