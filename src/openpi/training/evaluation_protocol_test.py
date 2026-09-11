import argparse
import json

import pytest

from openpi.training.evaluation_protocol import validate_evaluation_request
from openpi.training.stage1_data import sha256_file


def fixture(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    metadata = {"stage": "m3", "config": {"engineering_smoke": False, "split_sha256": "split", "norm_sha256": "norm"}}
    (checkpoint / "metadata.json").write_text(json.dumps(metadata))
    (checkpoint / "model.safetensors").write_bytes(b"selected checkpoint")
    options = {"draws": 2, "num_steps": 10, "latency_samples": 0, "device": "cpu", "cpu_threads": 4, "samples": 128}
    protocol = {
        "status": "sealed",
        "split": "test",
        "execution_frames": 15,
        "split_sha256": "split",
        "norm_sha256": "norm",
        "checkpoints": {
            str(checkpoint.resolve()): {
                "metadata_sha256": sha256_file(checkpoint / "metadata.json"),
                "weights_sha256": sha256_file(checkpoint / "model.safetensors"),
                "world_size": 1,
                "evaluation_options": options,
            }
        },
    }
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    args = argparse.Namespace(checkpoint=checkpoint, split="test", test_protocol=path, **options)
    return args, metadata, protocol


def test_validation_needs_no_sealed_protocol(tmp_path):
    args, metadata, _ = fixture(tmp_path)
    args.split, args.test_protocol = "val", None
    assert validate_evaluation_request(args, metadata) is None
    args.split = "test"
    with pytest.raises(ValueError, match="requires a sealed"):
        validate_evaluation_request(args, metadata)


def test_test_options_and_weights_are_immutable(tmp_path):
    args, metadata, _ = fixture(tmp_path)
    assert validate_evaluation_request(args, metadata)["execution_frames"] == 15
    args.num_steps = 2
    with pytest.raises(ValueError, match="option differs"):
        validate_evaluation_request(args, metadata)
    args.num_steps = 10
    (args.checkpoint / "model.safetensors").write_bytes(b"different weights")
    with pytest.raises(ValueError, match="weights changed"):
        validate_evaluation_request(args, metadata)


def test_unsealed_and_engineering_models_rejected(tmp_path):
    args, metadata, protocol = fixture(tmp_path)
    protocol["status"] = "draft"
    args.test_protocol.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="not sealed"):
        validate_evaluation_request(args, metadata)
    protocol["status"] = "sealed"
    args.test_protocol.write_text(json.dumps(protocol))
    metadata["config"]["engineering_smoke"] = True
    with pytest.raises(ValueError, match="Engineering"):
        validate_evaluation_request(args, metadata)


def test_selected_legacy_c0_passes_the_test_request_gate(tmp_path, monkeypatch):
    from openpi.training import research_checkpoint

    args, metadata, protocol = fixture(tmp_path)
    metadata.update(schema_version=2, stage="m0", completed_steps=1000)
    metadata["config"].pop("engineering_smoke")
    metadata["config"].update(stage="m0", steps=1000, model={"pi05": True})
    (args.checkpoint / "metadata.json").write_text(json.dumps(metadata))
    protocol["checkpoints"][str(args.checkpoint.resolve())]["metadata_sha256"] = sha256_file(
        args.checkpoint / "metadata.json"
    )
    args.test_protocol.write_text(json.dumps(protocol))
    research = tmp_path / "research_protocol.json"
    research.write_text(
        json.dumps(
            {
                "c0_shared_across_seeds": str(args.checkpoint),
                "c0_weights_sha256": sha256_file(args.checkpoint / "model.safetensors"),
                "split_sha256": "split",
                "norm_sha256": "norm",
            }
        )
    )
    monkeypatch.setattr(research_checkpoint, "RESEARCH_PROTOCOL_PATH", research)
    assert validate_evaluation_request(args, metadata)["execution_frames"] == 15
    (args.checkpoint / "model.safetensors").write_bytes(b"tampered selected C0")
    with pytest.raises(ValueError, match="weights differ"):
        validate_evaluation_request(args, metadata)
