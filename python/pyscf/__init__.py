"""pyscf-rs overlay package — Phase 3 BIND-02.

Plan 03-07 ships the `_native` cdylib so this import is unconditional.
The maturin `python-source = "python"` config puts this directory on
sys.path ahead of the upstream `pyscf/` tree, so `import pyscf` resolves
here, and `from pyscf import scf` resolves to the overlay submodule which
re-exports `pyscf._native.scf` (RHF/UHF/GHF).
"""
import pkgutil as _pkgutil

__path__ = _pkgutil.extend_path(__path__, __name__)
del _pkgutil

# 20-19 A: bind the overlay PACKAGES (python/pyscf/{scf,dft,gto}), not the flat
# `pyscf._native.{scf,dft,gto}` modules. Before, `from pyscf import gto` returned the
# native module (no `__path__`, no upstream fallthrough) until something imported
# `pyscf.gto` and rebound the attribute — so upstream code saw an import-order-
# dependent `gto`. The packages re-export the same native classes
# (`pyscf.scf.RHF is pyscf._native.scf.RHF`).
from pyscf import scf  # noqa: E402
from pyscf import dft  # noqa: E402
from pyscf import gto  # noqa: E402
from pyscf._native import PyscfRsRuntimeError as _PyscfRsBase  # type: ignore[attr-defined]  # noqa: E402


class PyscfRsError(_PyscfRsBase):  # type: ignore[misc, valid-type]
    """Phase 3 BIND-09 panic→exception with .kind and .source_chain attrs.

    The Rust side raises the bare PyException subclass `PyscfRsRuntimeError`
    with positional args `(msg, kind, source_chain)`; this overlay grafts
    them onto Python `.kind: str` and `.source_chain: list[str]` properties.

    Attributes:
        kind: Rust error variant name (e.g., 'ConvergenceFailure').
        source_chain: list of `str(err.source())` walking the Rust error tree.
    """

    @property
    def kind(self) -> str:
        return self.args[1] if len(self.args) > 1 else "Unknown"

    @property
    def source_chain(self) -> list[str]:
        return self.args[2] if len(self.args) > 2 else []


def M(**kwargs):
    """Main driver to create Molecule object (mol) or Material crystal object (cell).

    Port of upstream ``pyscf/__init__.py:106-112`` (2.12.1): a lattice ``a`` →
    ``pyscf.pbc.gto.M`` (the native periodic ``Cell``, plan 20-09); otherwise the
    molecular ``gto.M``. Upstream imports ``pyscf.__all__`` first to register every
    module; the overlay imports only the ``pbc.gto`` it dispatches to (20-18).
    """
    if kwargs.get("a") is not None:  # a is crystal lattice parameter
        from pyscf.pbc import gto as _pbcgto

        return _pbcgto.M(**kwargs)
    else:  # Molecule
        return gto.M(**kwargs)


__all__ = ["scf", "dft", "gto", "M", "PyscfRsError"]


def __getattr__(name):
    """20-19 A: upstream submodules (`pyscf.lib`, `pyscf.ao2mo`, ...) as attributes, and
    upstream's `pyscf.DEBUG`, without executing upstream's `pyscf/__init__.py`."""
    import importlib
    import os

    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name == "DEBUG":
        return importlib.import_module("pyscf.__config__").DEBUG
    here = os.path.dirname(os.path.abspath(__file__))
    for entry in __path__:
        if os.path.abspath(entry) == here:
            continue
        if os.path.isfile(os.path.join(entry, name + ".py")) or \
                os.path.isfile(os.path.join(entry, name, "__init__.py")):
            try:
                return importlib.import_module(f"{__name__}.{name}")
            except Exception as exc:  # noqa: BLE001 — keep hasattr() safe
                raise AttributeError(
                    f"module {__name__!r} has no attribute {name!r}: upstream submodule "
                    f"failed to import: {type(exc).__name__}: {exc}") from exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
