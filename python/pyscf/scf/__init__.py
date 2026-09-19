"""pyscf.scf overlay — re-exports from pyscf._native.scf (BIND-02).

Plan 03-07 makes this import unconditional. `density_fit` is a method on
RHF/UHF (`mf.density_fit(...)`), not a top-level function — the BIND-03
idiom `mf.density_fit().run()` covers the user-facing surface.
"""
from pyscf._native.scf import RHF, UHF, GHF  # type: ignore[attr-defined]

__all__ = ["RHF", "UHF", "GHF"]

# 20-19 A: fall through to upstream PySCF for submodules (`extend_path`) and for
# top-level names this overlay does not bind (`__getattr__`, pyscf/_passthrough.py).
# Silent (not a PBC family); the native names above stay module globals, so
# `pyscf.scf.<native name> is pyscf._native.scf.<native name>` is unchanged.
import pkgutil as _pkgutil  # noqa: E402

__path__ = _pkgutil.extend_path(__path__, __name__)
del _pkgutil

from pyscf._passthrough import package_getattr as _package_getattr  # noqa: E402

__getattr__ = _package_getattr(globals(), ("pyscf._native.scf",))
