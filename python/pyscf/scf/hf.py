"""Backward-compat shim: `from pyscf.scf.hf import RHF` works as in upstream PySCF.

BIND-02 (plan 03-07). The `hf` submodule in upstream PySCF is the canonical
home of `RHF`; this shim re-exports the Rust cdylib symbol so existing
PySCF scripts that do `from pyscf.scf.hf import RHF` continue to work
verbatim against pyscf-rs.
"""
from pyscf._native.scf import RHF  # type: ignore[attr-defined]

__all__ = ["RHF"]

# 20-19 A: names this shim does not bind come from the upstream `scf/hf.py` it
# shadows, executed as `pyscf.scf._upstream_hf` (silent; pyscf/_passthrough.py).
from pyscf._passthrough import module_getattr as _module_getattr  # noqa: E402

__getattr__ = _module_getattr(__name__)
