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


class _Stub(types.ModuleType):
    """Module stub that returns a new Stub for any attribute access."""
    def __getattr__(self, attr: str) -> "_Stub":
        child = _Stub(f"{self.__name__}.{attr}")
        setattr(self, attr, child)
        return child
    def __call__(self, *a, **kw):  # noqa: ANN002,ANN003
        return _Stub(f"{self.__name__}()")
    def __repr__(self) -> str:
        return f"<Stub '{self.__name__}'>"


for _mod_name in [
    "augmax",
    "orbax",
    "orbax.checkpoint",
    "orbax.checkpoint.future",
    "orbax.checkpoint.utils",
]:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = _Stub(_mod_name)

# ── Now safe to import openpi ─────────────────────────────────────────────────
# Re-use serve_policy.main() directly — all logic lives there.
import logging  # noqa: E402

import tyro  # noqa: E402

import importlib.util as _ilu
import pathlib as _pl

_spec = _ilu.spec_from_file_location(
    "serve_policy",
    _pl.Path(__file__).parent / "serve_policy.py",
)
_serve = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_serve)
Args = _serve.Args
main = _serve.main

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
