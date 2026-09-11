"""Verify the resumed transfer, publish weights atomically, and launch inference."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path.home() / "agentic-openpi"
STAGING = Path.home() / ".cache/openpi-deploy/m3_seed42_20260907"
CHECKPOINT = ROOT / "checkpoints/pi05_piper_stage1/m3_pilot_seed42/step_003500"
LOGS = ROOT / "logs/robot_m3_seed42_20260907"
EXPECTED = "64600bed0b60e0529ac2721edb55717ac1c80fba2a607292857668a35e7a9275"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    manifest = json.loads((STAGING / "weight_tail_manifest.json").read_text())
    prefix = CHECKPOINT / "model.safetensors.transfer_prefix"
    assert prefix.stat().st_size == manifest["prefix_bytes"]
    started = time.monotonic()
    while True:
        progress = [{"file": item["file"], "bytes": (STAGING / item["file"]).stat().st_size
                     if (STAGING / item["file"]).exists() else 0, "expected": item["bytes"]}
                    for item in manifest["parts"]]
        (LOGS / "transfer_progress.json").write_text(json.dumps({"updated": time.time(), "parts": progress,
                                                               "prefix_bytes": manifest["prefix_bytes"]}, indent=2))
        if all(item["bytes"] == item["expected"] for item in progress):
            break
        if time.monotonic() - started > 3600:
            raise TimeoutError("Transfer did not complete in one hour")
        time.sleep(5)
    for item in manifest["parts"]:
        assert digest(STAGING / item["file"]) == item["sha256"], item["file"]
    temporary = CHECKPOINT / "model.safetensors.assembling"
    sha = hashlib.sha256()
    with temporary.open("xb") as out:
        for path in [prefix] + [STAGING / item["file"] for item in manifest["parts"]]:
            with path.open("rb") as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    out.write(block)
                    sha.update(block)
        out.flush()
        os.fsync(out.fileno())
    assert temporary.stat().st_size == manifest["total_bytes"]
    assert sha.hexdigest() == EXPECTED, sha.hexdigest()
    target = CHECKPOINT / "model.safetensors"
    assert not target.exists()
    temporary.rename(target)
    assert digest(CHECKPOINT / "metadata.json") == "291a6667ff818c955568ae5c084d912b3842722f19f612d5e61f92d84ffcfd19"
    metadata = json.loads((CHECKPOINT / "metadata.json").read_text())
    source_checks = {}
    for name, expected in metadata["config"]["sources"].items():
        if name.startswith("/"):
            continue
        if name.startswith("src/"):
            actual = digest(ROOT / name)
            assert actual == expected, name
            source_checks[name] = actual
    runtime_checks = {}
    for name, item in metadata["config"]["runtime"]["sources"].items():
        if not name.startswith("transformers."):
            continue
        path = ROOT / ".venv/lib/python3.11/site-packages" / (name.replace(".", "/") + ".py")
        actual = digest(path)
        assert actual == item["sha256"], name
        runtime_checks[name] = actual
    (LOGS / "checkpoint_verification.json").write_text(json.dumps({"weights_sha256": EXPECTED,
        "weights_bytes": target.stat().st_size, "project_sources": source_checks,
        "transformers_sources": runtime_checks, "verified_at": time.time()}, indent=2))
    setup = ("source /opt/ros/noetic/setup.bash; source /home/agilex/miniconda3/etc/profile.d/conda.sh; "
             "conda activate aloha; source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash; "
             "export ROS_MASTER_URI=http://localhost:11311 ROS_HOSTNAME=localhost; "
             "export PYTHONPATH=/home/agilex/agentic-openpi/packages/openpi-client/src:${PYTHONPATH:-}; ")
    subprocess.run(["/bin/bash", "-c", setup + "exec python scripts/run_subtask_piper.py --observe-only "
                    "--output logs/robot_m3_seed42_20260907/preflight_at_launch"], cwd=ROOT, check=True, timeout=40)
    command = [str(ROOT / ".venv/bin/python"), "scripts/serve_subtask_policy.py", "--checkpoint", str(CHECKPOINT),
               "--port", "8000", "--log-dir", str(LOGS), "--warmup-observation",
               str(LOGS / "preflight_at_launch/observation.npz")]
    with (LOGS / "policy_server.log").open("ab") as out:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out,
                                   stderr=subprocess.STDOUT, start_new_session=True,
                                   env=dict(os.environ, JAX_PLATFORMS="cpu", HF_HUB_OFFLINE="1",
                                            OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1"))
    (LOGS / "policy_server.process.json").write_text(json.dumps({"pid": process.pid, "cmd": command,
        "proc_stat": Path("/proc/%d/stat" % process.pid).read_text(), "started": time.time()}, indent=2))
    print("Verified exact checkpoint and launched policy server PID %d" % process.pid, flush=True)


if __name__ == "__main__":
    main()
