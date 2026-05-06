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


# ── Stub design ──────────────────────────────────────────────────────────────
# Problem: checkpoints.py does things like:
#
#   class CallbackHandler(ocp.AsyncCheckpointHandler): ...
#   @ocp.args.register_with_handler(CallbackHandler, for_save=True)
#   class CallbackSave(ocp.args.CheckpointArgs): ...
#
# Attribute access on a stub must therefore return a real *class* (a type),
# not a module instance, so that Python can use it as a base class.
# Calling the stub with kwargs (factory pattern) must return an identity
# decorator so that @stub(args) leaves the decorated class unchanged.

class _StubMeta(type):
    """Metaclass for stub classes — supports attribute access and decorator use."""

    def __getattr__(cls, attr: str) -> type:
        # Raise AttributeError for dunders so Python's dataclass machinery,
        # inspect, and other introspection tools see a normal empty class.
        # e.g. __dataclass_fields__ must be absent (not a stub) so that
        # @dataclasses.dataclass can process subclasses of stub base classes.
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        child = _StubMeta(f"{cls.__name__}.{attr}", (object,), {})
        setattr(cls, attr, child)
        return child

    def __call__(cls, *args, **kwargs):  # noqa: ANN002,ANN003
        # @stub  →  return the decorated object unchanged
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        # @stub(args)  →  return identity decorator
        if kwargs or len(args) > 0:
            return lambda fn: fn
        return super().__call__()

    def __repr__(cls) -> str:
        return f"<StubClass '{cls.__name__}'>"


class _StubModule(types.ModuleType):
    """Module-level stub — safe entry in sys.modules (has no __file__)."""

    def __getattr__(self, attr: str) -> type:
        # Raise AttributeError for dunder attrs so inspect.getmodule() works.
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        # Return a real class stub (not a module instance) so it can be
        # used as a base class or decorator.
        child = _StubMeta(f"{self.__name__}.{attr}", (object,), {})
        setattr(self, attr, child)
        return child

    def __repr__(self) -> str:
        return f"<StubModule '{self.__name__}'>"


for _mod_name in [
    "augmax",
    "orbax",
    "orbax.checkpoint",
    "orbax.checkpoint.future",
    "orbax.checkpoint.utils",
]:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = _StubModule(_mod_name)

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
