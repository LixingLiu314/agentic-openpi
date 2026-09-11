"""Pinned reach-only actor dataset, shared by candidate training and evaluation."""
import dataclasses
import json
from pathlib import Path

from openpi.training import config as config_lib
from openpi.training.stage1_data import sha256_file

VARIANT = "official_pi05_recurrent_reach_arm_v1"
SCHEMA_VERSION = 7
DISPLAY_SET = "official-pi05-reach-arm-s42-v1"
DATASET_ROOT = Path("Datasets/eggplant_potato_reach_arm_v1")
ASSETS_ROOT = Path("assets/pi05_piper_reach_arm_v1/eggplant_potato")
REPO_ID = "local/eggplant_potato_reach_arm_v1"
SPLIT_FILE_SHA256 = "e86b8d7f91a60c6b120cfbe821b882980b56f8dc99dfe701e80124f7581c2bab"
NORM_SHA256 = "e42de696d2a699111c2bb1022c9de74acd47706a147a5cfa9c8fe9d7ff1f366d"
ANNOTATIONS_SHA256 = "b95c4a31ac5bb39f56a7702ef9d270bfb8c1f5ce74447c8725cbfeadaf83146b"


def data_config(model_config, *, split="train"):
    if split not in {"train", "val"}:
        raise ValueError("The reach-only experiment has train/val and no test split")
    for path, expected in [(ASSETS_ROOT/"split.json", SPLIT_FILE_SHA256),
                           (ASSETS_ROOT/"norm_stats.json", NORM_SHA256),
                           (DATASET_ROOT/"meta/reach_arm_annotations.jsonl", ANNOTATIONS_SHA256)]:
        if sha256_file(path) != expected:
            raise ValueError(f"Changed reach-arm data contract: {path}")
    ready = json.loads((ASSETS_ROOT/"READY.json").read_text())
    if ready["status"] != "ready_for_training_configuration":
        raise ValueError("Dataset preparation has not passed its native loader check")
    cfg = config_lib.get_config("pi05_piper_stage1")
    factory = dataclasses.replace(cfg.data, repo_id=REPO_ID, local_root=str(DATASET_ROOT),
                                  split_manifest=str(ASSETS_ROOT/"split.json"), split=split,
                                  assets=config_lib.AssetsConfig(assets_dir=str(ASSETS_ROOT.parent),
                                                                 asset_id=ASSETS_ROOT.name))
    return factory.create(cfg.assets_dirs, model_config)
