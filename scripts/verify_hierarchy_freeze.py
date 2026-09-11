"""Compare every saved tensor across a real hierarchy optimizer transition."""

import argparse
import json
from pathlib import Path

from safetensors import safe_open
import torch


def branch(key):
    if key.startswith("decoder."):
        return "subtask"
    if key.startswith(("base.paligemma_with_expert.gemma_expert.model.", "base.action_", "base.time_mlp_")):
        return "action"
    return "frozen"


def main(parent, child, output):
    torch.set_num_threads(16)
    parent_metadata = json.loads((parent / "metadata.json").read_text())
    child_metadata = json.loads((child / "metadata.json").read_text())
    report = {
        "parent": str(parent),
        "child": str(child),
        "stage": child_metadata["stage"],
        "groups": {
            name: {"tensors": 0, "changed_tensors": 0, "elements": 0} for name in ["frozen", "action", "subtask"]
        },
    }
    with (
        safe_open(parent / "model.safetensors", framework="pt", device="cpu") as before,
        safe_open(child / "model.safetensors", framework="pt", device="cpu") as after,
    ):
        before_keys = set(before.keys())
        for key in after.keys():  # noqa: SIM118 - safetensors handle, not a dictionary
            source_key = key.removeprefix("base.") if parent_metadata["stage"] == "m0" else key
            if source_key not in before_keys:
                if child_metadata["stage"] == "m1" and key.startswith("decoder."):
                    continue
                raise AssertionError(f"Unexpected new tensor {key}")
            expected, actual = before.get_tensor(source_key), after.get_tensor(key)
            group = branch(key)
            changed = not torch.equal(expected, actual)
            report["groups"][group]["tensors"] += 1
            report["groups"][group]["elements"] += actual.numel()
            report["groups"][group]["changed_tensors"] += int(changed)
            if group == "frozen" or (child_metadata["stage"] == "m1" and group == "action"):
                assert not changed, f"Frozen tensor changed: {key}"
        mapped_before = {f"base.{key}" if parent_metadata["stage"] == "m0" else key for key in before_keys}
        assert mapped_before <= set(after.keys()), "Saved parent tensors are missing in child"
    if child_metadata["stage"] != "m1":
        assert report["groups"]["action"]["changed_tensors"] > 0
        assert report["groups"]["subtask"]["changed_tensors"] > 0
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--child", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.parent, args.child, args.output)
