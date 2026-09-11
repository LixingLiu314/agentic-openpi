import copy
import json

import pytest

from openpi.training import research_checkpoint
from openpi.training.stage1_data import sha256_file


def legacy_fixture(tmp_path, monkeypatch):
    checkpoint = tmp_path / "selected_c0"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"selected legacy C0")
    metadata = {
        "schema_version": 2,
        "stage": "m0",
        "completed_steps": 1000,
        "config": {
            "stage": "m0",
            "steps": 1000,
            "model": {"pi05": True},
            "split_sha256": "split",
            "norm_sha256": "norm",
        },
    }
    protocol = {
        "c0_shared_across_seeds": str(checkpoint),
        "c0_weights_sha256": sha256_file(checkpoint / "model.safetensors"),
        "split_sha256": "split",
        "norm_sha256": "norm",
    }
    path = tmp_path / "research_protocol.json"
    path.write_text(json.dumps(protocol))
    monkeypatch.setattr(research_checkpoint, "RESEARCH_PROTOCOL_PATH", path)
    return checkpoint, metadata


def test_only_selected_complete_legacy_c0_is_admitted(tmp_path, monkeypatch):
    checkpoint, metadata = legacy_fixture(tmp_path, monkeypatch)
    research_checkpoint.require_research_checkpoint(checkpoint, metadata)
    with pytest.raises(ValueError, match="not the C0 selected"):
        research_checkpoint.require_research_checkpoint(tmp_path / "another_c0", metadata)
    incomplete = copy.deepcopy(metadata)
    incomplete["completed_steps"] = 999
    with pytest.raises(ValueError, match="incomplete"):
        research_checkpoint.require_research_checkpoint(checkpoint, incomplete)
    (checkpoint / "model.safetensors").write_bytes(b"modified weights")
    with pytest.raises(ValueError, match="weights differ"):
        research_checkpoint.require_research_checkpoint(checkpoint, metadata)


def test_legacy_data_identity_and_engineering_remain_rejected(tmp_path, monkeypatch):
    checkpoint, metadata = legacy_fixture(tmp_path, monkeypatch)
    for field in ["split_sha256", "norm_sha256"]:
        changed = copy.deepcopy(metadata)
        changed["config"][field] = "wrong"
        with pytest.raises(ValueError, match="differs from protocol"):
            research_checkpoint.require_research_checkpoint(checkpoint, changed)
    metadata["config"]["engineering_smoke"] = True
    with pytest.raises(ValueError, match="Engineering"):
        research_checkpoint.require_research_checkpoint(checkpoint, metadata)


def test_missing_flags_are_not_generally_treated_as_research(tmp_path):
    with pytest.raises(ValueError, match="unidentified"):
        research_checkpoint.require_research_checkpoint(tmp_path, {"stage": "m3", "config": {}})
    research_checkpoint.require_research_checkpoint(tmp_path, {"stage": "m3", "config": {"engineering_smoke": False}})
