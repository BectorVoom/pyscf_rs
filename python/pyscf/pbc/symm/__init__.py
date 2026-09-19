"""pyscf.pbc.symm overlay — re-exports the native k-point symmetry types (plan 20-14).

``KPoints`` and ``SpaceGroup`` below are the SAME objects as
``pyscf._native.pbc.symm.*`` (the identity contract of
``python/pyscf/tests/test_pbc_identity_gate.py``). ``KPoints`` is also
``pyscf._native.pbc.lib.kpts.KPoints``: upstream's ``pbc.scf`` / ``pbc.dft``
dispatch on ``isinstance(kpts, KPoints)``.

``Symmetry`` is deliberately NOT re-exported here, although it is bound
(``pyscf._native.pbc.symm.Symmetry``): upstream's own
``Cell.build_lattice_symmetry`` (``pyscf/pbc/gto/cell.py:1567-1579``) constructs
``pyscf.pbc.symm.Symmetry`` on an upstream cell and then deletes attributes off
it, so this name keeps resolving to upstream's class. The native ``Cell.build``
builds its lattice symmetry in Rust and never reaches that code.

The port DELIBERATELY differs from upstream k-symmetry where Phase 17 found
upstream defects (``.planning/phases/17-ksymm-multigrid/17-VERIFICATION.md`` §6).

Every other upstream ``pyscf.pbc.symm`` name (``geom``, ``group``, ``basis``,
``space_group``, ``symmetry``, ...) is NOT ported here; ``__path__`` is extended
so the upstream submodules stay importable and ``__getattr__`` falls through to
them lazily. Whether that fallthrough is announced is plan 20-17's decision.
"""

import pkgutil as _pkgutil

__path__ = _pkgutil.extend_path(__path__, __name__)
del _pkgutil

from pyscf._native.pbc.symm import (  # type: ignore[attr-defined]  # noqa: E402
    SPGElement,
    KPoints,
    SpaceGroup,
    get_crystal_class,
    make_kpts,
)

__all__ = ["KPoints", "SPGElement", "SpaceGroup", "get_crystal_class", "make_kpts"]


def __getattr__(name):
    import importlib

    if name == "Symmetry":
        return importlib.import_module(f"{__name__}.symmetry").Symmetry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
