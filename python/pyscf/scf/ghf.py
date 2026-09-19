"""Backward-compat shim: `from pyscf.scf.ghf import GHF` works as in upstream PySCF.

BIND-02 (plan 03-07). Re-exports the Rust cdylib `GHF` symbol so existing
PySCF scripts that do `from pyscf.scf.ghf import GHF` continue to work
verbatim against pyscf-rs.
"""
from pyscf._native.scf import GHF  # type: ignore[attr-defined]

__all__ = ["GHF"]

# 20-19 A: names this shim does not bind come from the upstream `scf/ghf.py` it
# shadows, executed as `pyscf.scf._upstream_ghf` (silent; pyscf/_passthrough.py).
from pyscf._passthrough import module_getattr as _module_getattr  # noqa: E402

__getattr__ = _module_getattr(__name__)
