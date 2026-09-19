"""pyscf.pbc.ci overlay — re-exports the native k-point CIS (plan 20-15).

``KCIS`` (``CIS``) is the SAME object as ``pyscf._native.pbc.ci.KCIS``; it takes a
converged full-BZ ``KRHF``. ``kernel(nroots=1, eris=None, kptlist=None)`` returns
``(e, None)`` — the Rust ``kernel_at_kshift`` returns eigenvalues only.

``RCISD``/``CISD``/``UCISD``/``GCISD`` (upstream ``pbc/ci/cisd.py``) are DEFERRED BY
DESIGN: they are gamma-point shims over the molecular CISD, and this port has no
molecular CI crate (``crates/pyscf-pbc-ci/src/lib.rs:3-22``). They raise
``NotImplementedError``. The upstream ``kcis_rhf`` module falls through lazily.
"""

import pkgutil as _pkgutil

__path__ = _pkgutil.extend_path(__path__, __name__)
del _pkgutil

from pyscf._native.pbc.ci import CIS, KCIS  # type: ignore[attr-defined]  # noqa: E402

__all__ = ["KCIS", "CIS", "RCISD", "CISD", "UCISD", "GCISD"]


def _cisd_deferred(name):
    def make(mf, *args, **kwargs):
        raise NotImplementedError(
            f"pyscf.pbc.ci.{name}: pbc/ci/cisd.py is deferred by design in pyscf-rs — it wraps "
            "the molecular CISD, which is not ported (crates/pyscf-pbc-ci/src/lib.rs:3-22)"
        )

    make.__name__ = name
    make.__qualname__ = name
    return make


RCISD = _cisd_deferred("RCISD")
CISD = RCISD
UCISD = _cisd_deferred("UCISD")
GCISD = _cisd_deferred("GCISD")


def __getattr__(name):
    import importlib

    if name == "kcis_rhf":
        return importlib.import_module(f"{__name__}.kcis_rhf")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
