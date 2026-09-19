"""pyscf.pbc.dft.gen_grid overlay shim (plan 20-17; upstream names 20-19 C).

``UniformGrids`` and ``BeckeGrids`` (alias ``AtomicGrids``) are the SAME objects as
``pyscf._native.pbc.dft.*`` (plan 20-13), so ``dft.gen_grid.BeckeGrids(cell)`` —
the spelling of ``examples/pbc/20-k_points_scf.py`` — is native.

Upstream ``pyscf/pbc/dft/gen_grid.py`` also exports names with NO native binding
(20-19 C, measured against the vendored 2.12.1 file and its importers):

* ``gen_uniform_grids`` / ``get_uniform_grids`` — upstream re-exports them from
  ``pyscf.pbc.gto.cell``; resolved lazily from THERE (the same function object
  upstream's ``gen_grid.gen_uniform_grids`` is), without loading upstream
  ``gen_grid.py`` and its ``libpbc`` / molecular-``dft`` imports;
* ``make_mask``, ``BLKSIZE``, ``NBINS``, ``CUTOFF``, ``ALIGNMENT_UNIT``,
  ``get_becke_grids`` / ``gen_becke_grids``, ``libpbc``, the molecular prune /
  becke functions, ... — fall through lazily to the upstream ``gen_grid.py`` this
  file shadows (``_unported.fallthrough_getattr``).

Both routes are announced once per family (D-PBC-35, ``PbcUpstreamFallthroughWarning``).
No name here is native except ``__all__``.
"""

import importlib as _importlib

from pyscf._native.pbc.dft import BeckeGrids, UniformGrids  # type: ignore[attr-defined]
from pyscf.pbc._unported import fallthrough_getattr as _fallthrough

AtomicGrids = BeckeGrids

_PYSCF_RS_IMPL = "partial"

__all__ = ["UniformGrids", "BeckeGrids", "AtomicGrids"]

# upstream `from pyscf.pbc.gto.cell import get_uniform_grids, gen_uniform_grids`
_FROM_UPSTREAM_CELL = ("get_uniform_grids", "gen_uniform_grids")

_upstream_getattr = _fallthrough(__name__, __name__)


def __getattr__(name):
    if name in _FROM_UPSTREAM_CELL:
        try:
            value = getattr(_importlib.import_module("pyscf.pbc.gto.cell"), name)
        except Exception as exc:  # noqa: BLE001 — keep hasattr() safe (20-17 convention)
            raise AttributeError(
                f"module {__name__!r} has no native attribute {name!r}, and the upstream "
                f"fallthrough pyscf.pbc.gto.cell failed to import: {type(exc).__name__}: {exc}"
            ) from exc
        globals()[name] = value
        return value
    return _upstream_getattr(name)
