"""Backward-compat shim: `from pyscf.scf.hf import RHF` works as in upstream PySCF.

BIND-02 (plan 03-07). The `hf` submodule in upstream PySCF is the canonical
home of `RHF`; this shim re-exports the Rust cdylib symbol so existing
PySCF scripts that do `from pyscf.scf.hf import RHF` continue to work
verbatim against pyscf-rs.
"""
from pyscf._native.scf import RHF  # type: ignore[attr-defined]

__all__ = ["RHF"]
