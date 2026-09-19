"""pyscf.pbc.dft overlay — re-exports the native periodic Kohn-Sham drivers (plan 20-13).

``KRKS``, ``KUKS``, ``KROKS``, ``KGKS``, ``KRKSpU``, ``KUKSpU``, the
``KsymAdapted*`` classes, ``KohnShamDFT``, ``UniformGrids`` / ``BeckeGrids``
and the numint selectors are the SAME objects as ``pyscf._native.pbc.dft.*``
(the identity contract of ``python/pyscf/tests/test_pbc_identity_gate.py``).

Upstream makes ``KRKS``, ``KUKS``, ``KRKSpU`` and ``KUKSpU`` *functions* that
branch on ``isinstance(kpts, KPoints)`` (``pyscf/pbc/dft/__init__.py:37-73``);
here the native constructors apply the same rule so the identity holds:
a built ``pyscf.pbc.symm.KPoints`` runs the Rust ``KsymAdaptedKrks`` /
``KsymAdaptedKuks`` / ``KsymAdaptedKrkspu`` / ``KsymAdaptedKukspu``. ``KROKS`` /
``KGKS`` with a ``KPoints`` raise (upstream has no k-symmetry class for them).

This file deliberately does NOT import upstream's ``pbc/dft/__init__.py``:
that module imports ``kuks_ksymm``, which subclasses ``pyscf.pbc.scf.
kuhf_ksymm.KUHF`` — a 20-12 stub that raises — and ``gen_grid``, which needs
the molecular ``pyscf.dft.radi`` the molecular overlay does not export. The
function-shaped upstream names (``RKS``, ``UKS``, ``ROKS``, ``GKS``, ``KS``,
``KKS``) keep their Python-level dispatch below; the gamma-point forms are the
k-point drivers at one k-point (``pyscf-pbc-dft/src/gamma.rs``).

Upstream submodules (``krks``, ``numint``, ``gen_grid``, ``multigrid``, ...)
fall through lazily to the vendored tree via ``pkgutil.extend_path``; whether
that is announced is plan 20-17's decision.
"""

import pkgutil as _pkgutil

__path__ = _pkgutil.extend_path(__path__, __name__)
del _pkgutil

from pyscf._native.pbc.dft import (  # type: ignore[attr-defined]  # noqa: E402
    KGKS,
    KROKS,
    KRKS,
    KRKSpU,
    KUKS,
    KUKSpU,
    BeckeGrids,
    KNumInt,
    KohnShamDFT,
    KsymAdaptedKRKS,
    KsymAdaptedKRKSpU,
    KsymAdaptedKUKS,
    KsymAdaptedKUKSpU,
    MultiGridNumInt,
    MultiGridNumInt2,
    UniformGrids,
)
from pyscf._native.pbc.dft import GKS as _gamma_GKS  # noqa: E402
from pyscf._native.pbc.dft import RKS as _gamma_RKS  # noqa: E402
from pyscf._native.pbc.dft import ROKS as _gamma_ROKS  # noqa: E402
from pyscf._native.pbc.dft import UKS as _gamma_UKS  # noqa: E402

__all__ = [
    "KohnShamDFT", "KRKS", "KUKS", "KROKS", "KGKS", "KRKSpU", "KUKSpU",
    "KsymAdaptedKRKS", "KsymAdaptedKUKS", "KsymAdaptedKRKSpU", "KsymAdaptedKUKSpU",
    "UniformGrids", "BeckeGrids", "KNumInt", "MultiGridNumInt", "MultiGridNumInt2",
    "RKS", "UKS", "GKS", "ROKS", "KS", "KKS",
]


def _cell_spin(cell):
    spin = getattr(cell, "spin", 0)
    return spin() if callable(spin) else spin


def RKS(cell, *args, **kwargs):
    """`pbc/dft/__init__.py:76`: `kpts=` → KRKS; spin 0 → gamma RKS; else ROKS."""
    if "kpts" in kwargs:
        return KRKS(cell, *args, **kwargs)
    if _cell_spin(cell) == 0:
        return _gamma_RKS(cell, *args, **kwargs)
    return _gamma_ROKS(cell, *args, **kwargs)


def UKS(cell, *args, **kwargs):
    """`pbc/dft/__init__.py:85`: `kpts=` → KUKS; else gamma UKS."""
    if "kpts" in kwargs:
        return KUKS(cell, *args, **kwargs)
    return _gamma_UKS(cell, *args, **kwargs)


def GKS(cell, *args, **kwargs):
    """`pbc/dft/__init__.py:91`: `kpts=` → KGKS; else gamma GKS."""
    if "kpts" in kwargs:
        return KGKS(cell, *args, **kwargs)
    return _gamma_GKS(cell, *args, **kwargs)


def ROKS(cell, *args, **kwargs):
    """`pbc/dft/__init__.py:97`: `kpts=` → KROKS; else gamma ROKS."""
    if "kpts" in kwargs:
        return KROKS(cell, *args, **kwargs)
    return _gamma_ROKS(cell, *args, **kwargs)


def KS(cell, *args, **kwargs):
    """`pbc/dft/__init__.py:103`: RKS for spin 0, else UKS."""
    if _cell_spin(cell) == 0:
        return RKS(cell, *args, **kwargs)
    return UKS(cell, *args, **kwargs)


def KKS(cell, *args, **kwargs):
    """`pbc/dft/__init__.py:112`: KRKS for spin 0, else KUKS."""
    if _cell_spin(cell) == 0:
        return KRKS(cell, *args, **kwargs)
    return KUKS(cell, *args, **kwargs)


# 20-17: upstream's __init__ imports gen_grid, so `dft.gen_grid.BeckeGrids` works
# without an explicit submodule import; the overlay shim re-exports the native grids.
from pyscf.pbc.dft import gen_grid  # noqa: E402

# 20-18: upstream's __init__ imports krks/rks, which import `multigrid`
# (pyscf/pbc/dft/krks.py:33, rks.py:39), so `dft.multigrid.MultiGridNumInt` works without
# an explicit submodule import; the overlay shim re-exports the native classes.
from pyscf.pbc.dft import multigrid  # noqa: E402
