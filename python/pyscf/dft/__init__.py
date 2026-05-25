"""pyscf.dft overlay — re-exports from pyscf._native.dft (BIND-02 / DFT-08).

Plan 04-09 ships the `_native.dft` PyO3 submodule (PyRKS/PyUKS), so this
import is unconditional. The maturin `python-source = "python"` config puts
this directory on sys.path ahead of the upstream `pyscf/` tree, so
`from pyscf import dft` resolves here, and `dft.RKS(mol, xc='b3lyp').run()`
drives the Rust Kohn-Sham SCF.

`xc` defaults to `'LDA,VWN'` (the upstream RKS class default). A Python
subclass overriding `get_veff` / `define_xc_` has those overrides invoked
every SCF cycle via the `slf.call_method1` dispatch seam (Pitfall 7).

D-08: the active precision is readable read-only via `mf.dtype` (and
`mf._numint.dtype`), returning "f32"/"f64". There is intentionally NO
precision setter — `PYSCF_DTYPE` is the single source of truth for switching.
"""
from pyscf._native.dft import RKS, UKS  # type: ignore[attr-defined]

__all__ = ["RKS", "UKS"]
