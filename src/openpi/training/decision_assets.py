"""Lightweight integrity contract for reviewed decision assets."""
import json
from pathlib import Path
from openpi.training.stage1_data import sha256_file

ASSETS = Path("assets/pi05_piper_decision_v1")


def load_assets(root=ASSETS):
    root=Path(root)
    ready=json.loads((root/"READY.json").read_text())
    if ready.get("status")!="reviewed_and_verified":raise ValueError("Grounding review is incomplete")
    for name,digest in ready["files"].items():
        if sha256_file(root/name)!=digest:raise ValueError("Changed decision asset: "+name)
    decisions=json.loads((root/"decisions.json").read_text())
    points=json.loads((root/"reviewed_points.json").read_text())
    return decisions,points,ready

