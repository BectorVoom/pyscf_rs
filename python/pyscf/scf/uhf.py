"""Backward-compat shim: `from pyscf.scf.uhf import UHF` works as in upstream PySCF.

BIND-02 (plan 03-07). Re-exports the Rust cdylib `UHF` symbol so existing
PySCF scripts that do `from pyscf.scf.uhf import UHF` continue to work
verbatim against pyscf-rs.
"""
from pyscf._native.scf import UHF  # type: ignore[attr-defined]

__all__ = ["UHF"]

# 20-19 A: names this shim does not bind come from the upstream `scf/uhf.py` it
# shadows, executed as `pyscf.scf._upstream_uhf` (silent; pyscf/_passthrough.py).
from pyscf._passthrough import module_getattr as _module_getattr  # noqa: E402

__getattr__ = _module_getattr(__name__)
