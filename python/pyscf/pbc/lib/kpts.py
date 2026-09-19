"""pyscf.pbc.lib.kpts overlay shim (plan 20-17).

``KPoints`` and ``make_kpts`` are the SAME objects as
``pyscf._native.pbc.lib.kpts.*`` — and ``KPoints`` is also
``pyscf.pbc.symm.KPoints`` — so upstream's ``isinstance(kpts, libkpts.KPoints)``
tests the native type. Upstream's module-level functions (``transform_mo_coeff``,
``symmetrize_density``, ``make_k4_ibz``, …) are bound as ``KPoints`` methods;
the free functions, ``KQuartets`` and ``MORotationMatrix`` fall through lazily
to the upstream module this file shadows (which currently fails to import under
the overlay — see ``docs/pbc-status.md``).
"""

from pyscf._native.pbc.lib.kpts import KPT_DIFF_TOL, KPoints, make_kpts  # type: ignore[attr-defined]
from pyscf.pbc._unported import fallthrough_getattr as _fallthrough

_PYSCF_RS_IMPL = "partial"

__all__ = ["KPT_DIFF_TOL", "KPoints", "make_kpts"]

__getattr__ = _fallthrough(__name__, __name__)
