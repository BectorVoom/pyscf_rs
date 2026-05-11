"""Backward-compat shim: `from pyscf.scf.uhf import UHF` works as in upstream PySCF.

BIND-02 (plan 03-07). Re-exports the Rust cdylib `UHF` symbol so existing
PySCF scripts that do `from pyscf.scf.uhf import UHF` continue to work
verbatim against pyscf-rs.
"""
from pyscf._native.scf import UHF  # type: ignore[attr-defined]

__all__ = ["UHF"]
