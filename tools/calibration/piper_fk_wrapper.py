#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


_DEFAULT_FK_PATH = Path("~/cobot_magic/collect_data/piper_sdk_demo/eval_fk_error.py").expanduser()


def _load_module():
    if not _DEFAULT_FK_PATH.exists():
        raise FileNotFoundError(f"FK file not found: {_DEFAULT_FK_PATH}")
    spec = importlib.util.spec_from_file_location(_DEFAULT_FK_PATH.stem, _DEFAULT_FK_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to import FK module from {_DEFAULT_FK_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_FK_MODULE = _load_module()


def fk_full(joints_rad):
    joints_rad = np.asarray(joints_rad, dtype=np.float64).reshape(6)
    return _FK_MODULE.fk(joints_rad)

