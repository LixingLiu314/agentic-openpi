"""Three gradient arms with an audited official pi05 starting point."""

import hashlib
from pathlib import Path

import safetensors.torch
import torch

from openpi.models_pytorch.backbone_gradient import BackboneGradientModel
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


OFFICIAL_SHA256 = "be5b2233cf302a8fd7097e239e0e2c4680fc6f3696a55084983f2225e2d7044e"
VARIANT = "official_pi05_backbone_v1"


def tensor_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}\n".encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class OfficialBackboneGradientModel(BackboneGradientModel):
    def enable_backbone(self, mode, **kwargs):
        if mode == "frozen":
            self.backbone_mode = mode
            self.set_stage("m3")  # S/A trainability only; this does not load M3 weights.
            for parameter in self.base.paligemma_with_expert.paligemma.parameters():
                parameter.requires_grad_(False)
            self.train()
        else:
            super().enable_backbone(mode, **kwargs)


def initialize_official(model_config, decoder_config, weights, *, seed=42, device="cpu"):
    """Strictly load the complete official base, then independently initialize S.

    The caller verifies the pinned input file hash before entering this function.
    No trained hierarchical checkpoint is accepted by the base tensor layout.
    """
    base = PI0Pytorch(model_config).to(device)
    safetensors.torch.load_model(base, Path(weights), strict=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = OfficialBackboneGradientModel(base, decoder_config)
    return model


def verify_official_tensors(base, weights):
    """Check every stored base tensor after the intended precision conversion."""
    state = base.state_dict()
    checked = 0
    with safetensors.safe_open(weights, framework="pt", device="cpu") as source:
        for key in source.keys():
            actual = state[key].detach().cpu()
            expected = source.get_tensor(key).to(actual.dtype)
            if not torch.equal(actual, expected):
                raise AssertionError(f"Official initialization differs at {key}")
            checked += 1
    if checked != 812:
        raise AssertionError(f"Unexpected official tensor count: {checked}")
    return checked
