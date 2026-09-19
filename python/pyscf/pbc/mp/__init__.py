"""pyscf.pbc.mp overlay — re-exports the native periodic MP2 drivers (plan 20-15).

``KMP2`` (``KRMP2``), ``KsymAdaptedKMP2``, ``KUMP2`` and ``KMP2_stagger`` are the
SAME objects as ``pyscf._native.pbc.mp.*`` (identity contract,
``python/pyscf/tests/test_pbc_identity_gate.py``). Upstream makes ``KRMP2`` a
function that branches on ``isinstance(mf.kpts, KPoints)``
(``pyscf/pbc/mp/__init__.py``); here the native constructor applies the same
rule, so ``KMP2(KRHF(cell, kpts=KPoints))`` runs the k-symmetric route.

Every driver takes an already-converged ``pyscf.pbc.scf`` K-point driver and
never re-runs SCF. ``KUMP2.kernel`` raises ``NotImplementedError`` as upstream's
does.

Not ported: the gamma-point ``RMP2``/``MP2``/``UMP2``/``GMP2`` (upstream wraps the
MOLECULAR MP2 around a gamma-point PBC mean field) — they raise
``NotImplementedError`` here rather than handing a native mean field to upstream
Python. Upstream submodules (``kmp2``, ``kmp2_ksymm``, ``kump2``,
``kmp2_stagger``, ``mp2``) fall through lazily; that policy is plan 20-17's.
"""

import pkgutil as _pkgutil

__path__ = _pkgutil.extend_path(__path__, __name__)
del _pkgutil

from pyscf._native.pbc.mp import (  # type: ignore[attr-defined]  # noqa: E402
    KMP2,
    KMP2_stagger,
    KRMP2,
    KUMP2,
    KsymAdaptedKMP2,
)

__all__ = ["KMP2", "KRMP2", "KsymAdaptedKMP2", "KUMP2", "KMP2_stagger",
           "RMP2", "MP2", "UMP2", "GMP2"]


def _gamma_unported(name):
    def make(mf, *args, **kwargs):
        raise NotImplementedError(
            f"pyscf.pbc.mp.{name}: the gamma-point PBC MP2 (upstream pbc/mp/mp2.py over the "
            "molecular MP2) is not implemented in pyscf-rs; use KMP2 at one k-point"
        )

    make.__name__ = name
    make.__qualname__ = name
    return make


RMP2 = _gamma_unported("RMP2")
MP2 = RMP2
UMP2 = _gamma_unported("UMP2")
GMP2 = _gamma_unported("GMP2")

_UPSTREAM_SUBMODULES = ("kmp2", "kmp2_ksymm", "kump2", "kmp2_stagger", "mp2")


def __getattr__(name):
    import importlib

    if name in _UPSTREAM_SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
