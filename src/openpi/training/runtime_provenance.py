"""Fingerprint and archive the installed model/data code used by an experiment."""

import importlib.metadata
import importlib.util
from pathlib import Path
import platform
import shutil

import torch

from openpi.training.stage1_data import sha256_file

PACKAGES = (
    "torch",
    "torchvision",
    "transformers",
    "jax",
    "jaxlib",
    "numpy",
    "sentencepiece",
    "safetensors",
    "lerobot",
    "av",
    "datasets",
    "flax",
)
MODULES = (
    "transformers.models.gemma.modeling_gemma",
    "transformers.models.gemma.configuration_gemma",
    "transformers.models.paligemma.modeling_paligemma",
    "transformers.models.paligemma.configuration_paligemma",
    "transformers.models.siglip.modeling_siglip",
    "transformers.models.siglip.configuration_siglip",
    "lerobot.common.datasets.lerobot_dataset",
    "lerobot.common.datasets.video_utils",
    "lerobot.common.datasets.utils",
)


def capture_runtime():
    sources = {}
    for module in MODULES:
        spec = importlib.util.find_spec(module)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"Cannot fingerprint runtime module {module}")
        path = Path(spec.origin)
        sources[module] = {"path": str(path), "sha256": sha256_file(path)}
    return {
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
        "sources": sources,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "cuda_capability": list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None,
    }


def archive_runtime(runtime, directory):
    for module, source in runtime["sources"].items():
        destination = directory / (module.replace(".", "/") + ".py")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source["path"], destination)
        if sha256_file(destination) != source["sha256"]:
            raise ValueError(f"Runtime source changed during archiving: {module}")
