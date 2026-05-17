from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np

SCRIPT = Path(__file__).with_name("screen_hdf5_episodes.py")


def test_warns_when_one_arm_joint_is_constant(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    task_dir = root / "task"
    task_dir.mkdir(parents=True)
    episode = task_dir / "episode_000000.hdf5"

    qpos = np.zeros((24, 14), dtype=np.float64)
    qpos[:, 0] = np.linspace(0.0, 0.2, qpos.shape[0])
    qpos[:, 1] = 0.25
    qpos[:, 7] = np.linspace(0.0, -0.2, qpos.shape[0])

    with h5py.File(episode, "w") as h5_file:
        h5_file.create_dataset("observations/qpos", data=qpos)
        h5_file.create_dataset("action", data=qpos)

    report = tmp_path / "report.json"
    bad_list = tmp_path / "bad.txt"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(root),
            "--tasks",
            "task",
            "--output",
            str(report),
            "--bad-list",
            str(bad_list),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["num_bad"] == 0
    assert payload["num_warning_episodes"] == 1
    assert "Warnings:" in result.stdout
    assert "left_j2" in result.stdout

    warnings = payload["warning_episodes"][0]["warnings"]
    assert any(warning["joint"] == "left_j2" for warning in warnings)
