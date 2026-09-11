"""Verify local converted pi05 weights against generation-pinned official GCS objects."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import urllib.parse
import urllib.request

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")


def digest(path, algorithm):
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value


def download_object(item, root):
    prefix = "checkpoints/pi05_base/"
    if not item["name"].startswith(prefix):
        raise ValueError("Unexpected official object prefix")
    relative = Path(item["name"][len(prefix) :])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe object path")
    target = root / relative
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Object escapes audit directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(item["size"])

    def valid(path):
        if path.stat().st_size != expected_size:
            return False
        if "md5Hash" in item:
            return base64.b64encode(digest(path, "md5").digest()).decode() == item["md5Hash"]
        import google_crc32c

        checksum = google_crc32c.Checksum()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b""):
                checksum.update(block)
        return base64.b64encode(checksum.digest()).decode() == item["crc32c"]

    if target.exists():
        if not valid(target):
            raise ValueError(f"Existing audit object has a different hash: {target}")
    else:
        temporary = target.with_name(target.name + ".partial")
        url = (
            "https://storage.googleapis.com/openpi-assets/"
            + urllib.parse.quote(item["name"], safe="/")
            + "?generation="
            + item["generation"]
        )
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
            while block := response.read(8 * 1024**2):
                stream.write(block)
        if not valid(temporary):
            raise ValueError(f"Official object size/checksum mismatch: {relative}")
        temporary.rename(target)
    result = {
        "name": item["name"],
        "generation": item["generation"],
        "bytes": expected_size,
        "checksum_verified": True,
        "checksum_algorithm": "MD5" if "md5Hash" in item else "CRC32C",
    }
    print(json.dumps({"event": "official_object_verified", **result}), flush=True)
    return result


def audit(args):
    import numpy as np
    from safetensors import safe_open
    import torch

    import openpi.models.gemma

    torch.set_num_threads(8)
    manifest = json.loads(args.manifest.read_text())
    if manifest["manifest"].get("nextPageToken"):
        raise ValueError("Official object manifest is incomplete")
    objects = [
        item for item in manifest["manifest"]["items"] if item["name"].startswith("checkpoints/pi05_base/params/")
    ]
    args.official_root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        verified = list(pool.map(lambda item: download_object(item, args.official_root), objects))
    converter_path = Path("examples/convert_jax_model_to_pytorch.py")
    specification = importlib.util.spec_from_file_location("official_pi05_converter", converter_path)
    converter = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(converter)
    print(json.dumps({"event": "restore_official_arrays"}), flush=True)
    initial = converter.slice_initial_orbax_checkpoint(str(args.official_root), restore_precision="float32")
    config = SimpleNamespace(
        vision_config=SimpleNamespace(hidden_size=1152, num_hidden_layers=27),
        text_config=SimpleNamespace(hidden_size=2048, num_hidden_layers=18, num_attention_heads=8, head_dim=256),
    )
    paligemma, expert = converter.slice_paligemma_state_dict(initial["paligemma_params"], config)
    gemma = converter.slice_gemma_state_dict(
        expert,
        openpi.models.gemma.get_config("gemma_300m"),
        num_expert=1,
        checkpoint_dir=str(args.official_root),
        pi05=True,
    )
    converted = {**paligemma, **gemma}
    for name in ["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"]:
        parameters = initial["projection_params"][name]
        kernel, bias = parameters["kernel"], parameters["bias"]
        if isinstance(kernel, dict):
            kernel, bias = kernel["value"], bias["value"]
        converted[name + ".weight"] = torch.from_numpy(np.array(kernel)).T
        converted[name + ".bias"] = torch.from_numpy(np.array(bias))
    rows = []
    with safe_open(args.local_weights, framework="pt", device="cpu") as saved:
        aliases = saved.metadata() or {}
        saved_keys = set(saved.keys())
        unused_key = "paligemma_with_expert.gemma_expert.lm_head.weight"
        # The official converter loads only JAX-provided tensors with strict=False.
        # This unused action-expert vocabulary head remains constructor-initialized.
        unused_behavior_path = Path("logs/pi05_subtask_stage1/unused_expert_head_behavior.json")
        unused_behavior = json.loads(unused_behavior_path.read_text())
        if unused_behavior.get("passed") is not True or unused_behavior["tensor"] != unused_key:
            raise ValueError("Unused-head behavior proof is missing")
        for path, expected_hash in unused_behavior["sources"].items():
            if digest(Path(path), "sha256").hexdigest() != expected_hash:
                raise ValueError("Behavior proof source changed")
        expected_aliases = {
            "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight": "paligemma_with_expert.paligemma.lm_head.weight"
        }
        if aliases != expected_aliases:
            raise ValueError("Unexpected shared-weight aliases")
        for alias, target in aliases.items():
            if alias not in converted or target in converted:
                raise ValueError("Unexpected converted tied-weight coverage")
            converted[target] = converted[alias]
        missing = sorted(saved_keys - set(converted) - {unused_key})
        unexpected = sorted(set(converted) - saved_keys - set(aliases))
        if missing or unexpected:
            raise ValueError(f"Conversion key coverage mismatch: missing={missing}, unexpected={unexpected}")
        for key in sorted(saved_keys):
            actual = saved.get_tensor(key)
            if key == unused_key:
                if tuple(actual.shape) != (257152, 1024):
                    raise ValueError("Unexpected unused expert-head shape")
                rows.append(
                    {
                        "key": key,
                        "shape": list(actual.shape),
                        "dtype": str(actual.dtype),
                        "elements": actual.numel(),
                        "exact_match": None,
                        "status": "constructor-initialized, no official JAX source, unused by verified VLA inference/action loss",
                    }
                )
                continue
            expected = converted[key]
            expected = torch.as_tensor(expected).to(dtype=actual.dtype)
            same = actual.shape == expected.shape and torch.equal(actual, expected)
            rows.append(
                {
                    "key": key,
                    "shape": list(actual.shape),
                    "dtype": str(actual.dtype),
                    "elements": actual.numel(),
                    "exact_match": same,
                }
            )
        for alias, target in aliases.items():
            if (
                alias not in converted
                or target not in converted
                or not torch.equal(torch.as_tensor(converted[alias]), torch.as_tensor(converted[target]))
            ):
                raise ValueError(f"Official tied-weight alias is inconsistent: {alias}")
    report = {
        "official_source": "gs://openpi-assets/checkpoints/pi05_base",
        "object_manifest": str(args.manifest),
        "verified_official_objects": verified,
        "local_weights": str(args.local_weights),
        "local_weights_sha256": digest(args.local_weights, "sha256").hexdigest(),
        "converter_source_sha256": digest(converter_path, "sha256").hexdigest(),
        "script_source_sha256": digest(Path(__file__), "sha256").hexdigest(),
        "comparison": "official float32 restore, original converter tensor transforms, cast to each saved tensor dtype, exact equality for all JAX-derived tensors; tied aliases resolved and verified; constructor-only unused expert LM head explicitly excluded",
        "tensor_count": len(rows),
        "elements": sum(row["elements"] for row in rows),
        "mismatched_tensors": [row["key"] for row in rows if row["exact_match"] is False],
        "verified_tensor_count": sum(row["exact_match"] is True for row in rows),
        "unverified_unused_tensors": [row["key"] for row in rows if row["exact_match"] is None],
        "unused_head_behavior_proof": str(unused_behavior_path),
        "unused_head_behavior_proof_sha256": digest(unused_behavior_path, "sha256").hexdigest(),
        "tensors": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "event": "complete",
                "report": str(args.output),
                "tensor_count": len(rows),
                "mismatched_tensors": report["mismatched_tensors"],
            }
        ),
        flush=True,
    )
    if report["mismatched_tensors"]:
        raise AssertionError("The supplied converted base does not exactly match the official pi05_base conversion")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("assets/pi05_piper_stage1/provenance/official_pi05_base_gcs_manifest.json"),
    )
    parser.add_argument("--official-root", type=Path, default=Path("checkpoints/pi05_base_official_jax_audit"))
    parser.add_argument("--local-weights", type=Path, default=Path("checkpoints/pi05_base_pytorch/model.safetensors"))
    parser.add_argument("--output", type=Path, default=Path("logs/pi05_subtask_stage1/official_base_weight_audit.json"))
    audit(parser.parse_args())
