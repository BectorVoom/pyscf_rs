"""Backward-compat shim: `from pyscf.scf.ghf import GHF` works as in upstream PySCF.

BIND-02 (plan 03-07). Re-exports the Rust cdylib `GHF` symbol so existing
PySCF scripts that do `from pyscf.scf.ghf import GHF` continue to work
verbatim against pyscf-rs.
"""
from pyscf._native.scf import GHF  # type: ignore[attr-defined]

__all__ = ["GHF"]
