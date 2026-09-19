"""pyscf.pbc.dft.multigrid overlay shim (plan 20-18).

Upstream 2.12.1 ``pyscf/pbc/dft/multigrid/__init__.py:16-17`` exports exactly two
names::

    from .multigrid import MultiGridNumInt
    from .multigrid_pair import MultiGridNumInt as MultiGridNumInt2

Both are the SAME objects as ``pyscf._native.pbc.dft.MultiGridNumInt`` /
``MultiGridNumInt2`` (plan 20-13: ``numint::KsNumInt::multigrid`` /
``::multigrid2``), so ``from pyscf.pbc.dft import multigrid;
multigrid.MultiGridNumInt(cell)`` is native.

The upstream submodules (``multigrid``, ``multigrid_pair``, ``utils``, ``pp``,
``_backend_c``) are not ported as Python modules; ``__path__`` is extended so
they resolve to upstream Python (announced once per family, D-PBC-35), and a
failed upstream import surfaces as ``AttributeError`` so ``hasattr`` stays safe.
Upstream's ``__init__`` defines no other names.
"""

import pkgutil as _pkgutil

__path__ = _pkgutil.extend_path(__path__, __name__)
del _pkgutil

from pyscf._native.pbc.dft import MultiGridNumInt, MultiGridNumInt2  # type: ignore[attr-defined]  # noqa: E402,E501

_PYSCF_RS_IMPL = "partial"

__all__ = ["MultiGridNumInt", "MultiGridNumInt2"]

_UPSTREAM_SUBMODULES = ("multigrid", "multigrid_pair", "utils", "pp", "_backend_c")


def __getattr__(name):
    if name in _UPSTREAM_SUBMODULES:
        import importlib

        try:
            return importlib.import_module(f"{__name__}.{name}")
        except ImportError as exc:
            raise AttributeError(
                f"module {__name__!r} has no native attribute {name!r}, and the upstream "
                f"submodule failed to import: {type(exc).__name__}: {exc}"
            ) from exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
