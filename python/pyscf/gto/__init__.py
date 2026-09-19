"""Native pyscf-rs GTO overlay."""

from pyscf._native.gto import M, Mole  # type: ignore[attr-defined]

__all__ = ["M", "Mole"]

# 20-19 A: fall through to upstream PySCF for submodules (`extend_path`) and for
# top-level names this overlay does not bind (`__getattr__`, pyscf/_passthrough.py).
# Silent (not a PBC family); the native names above stay module globals, so
# `pyscf.gto.<native name> is pyscf._native.gto.<native name>` is unchanged.
import pkgutil as _pkgutil  # noqa: E402

__path__ = _pkgutil.extend_path(__path__, __name__)
del _pkgutil

from pyscf._passthrough import package_getattr as _package_getattr  # noqa: E402

__getattr__ = _package_getattr(globals(), ("pyscf._native.gto",))
