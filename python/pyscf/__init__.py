"""pyscf-rs overlay package — Phase 3 BIND-02.

Plan 03-07 ships the `_native` cdylib so this import is unconditional.
The maturin `python-source = "python"` config puts this directory on
sys.path ahead of the upstream `pyscf/` tree, so `import pyscf` resolves
here, and `from pyscf import scf` resolves to the overlay submodule which
re-exports `pyscf._native.scf` (RHF/UHF/GHF).
"""
import pkgutil as _pkgutil

from pyscf._native import scf  # type: ignore[attr-defined]
from pyscf._native import dft  # type: ignore[attr-defined]
from pyscf._native import PyscfRsRuntimeError as _PyscfRsBase  # type: ignore[attr-defined]

__path__ = _pkgutil.extend_path(__path__, __name__)
del _pkgutil


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


__all__ = ["scf", "dft", "PyscfRsError"]
