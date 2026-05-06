"""
PyTorch-only policy server — no JAX native extensions required.

Functionally identical to serve_policy.py but stubs out augmax / orbax
before they are imported, so the process does not segfault on machines
where those JAX packages have incompatible native libraries.

The stubs are safe because:
  - augmax is only used in the JAX training pipeline (data augmentation).
  - orbax is only used when loading JAX (non-PyTorch) checkpoints.
  - For PyTorch checkpoints (model.safetensors), neither is ever called.

Usage (same as serve_policy.py):
  python scripts/serve_policy_pytorch.py \\
      --env ALOHA \\
      --default-prompt "put banana in the green plate" \\
      policy:checkpoint \\
      --policy.config pi05_aloha_banana \\
      --policy.dir checkpoints/pi05_aloha_banana/banana_baseline/5000
"""

# ── Stub problematic JAX extensions BEFORE any openpi imports ────────────────
import sys
import types


def _make_stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    # Return a harmless object for any attribute access.
    mod.__getattr__ = lambda self, _: _make_stub(f"{name}.<attr>")  # type: ignore[method-assign]
    return mod


for _mod_name in [
    "augmax",
    "orbax",
    "orbax.checkpoint",
    "orbax.checkpoint.future",
    "orbax.checkpoint.utils",
]:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = _make_stub(_mod_name)

# Also give orbax.checkpoint a usable CheckpointManager stub so that
# any isinstance() checks or type annotations don't blow up.
import orbax.checkpoint as _ocp_stub  # noqa: E402  (already a stub)

_ocp_stub.CheckpointManager = type("CheckpointManager", (), {})  # type: ignore[attr-defined]
_ocp_stub.PyTreeCheckpointer = type("PyTreeCheckpointer", (), {})  # type: ignore[attr-defined]
_ocp_stub.Checkpointer = type("Checkpointer", (), {})  # type: ignore[attr-defined]

# ── Now safe to import openpi ─────────────────────────────────────────────────
# Re-use serve_policy.main() directly — all logic lives there.
import logging  # noqa: E402

import tyro  # noqa: E402

from scripts.serve_policy import Args, main  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
