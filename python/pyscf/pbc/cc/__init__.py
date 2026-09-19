"""pyscf.pbc.cc overlay — re-exports the native periodic coupled-cluster drivers (plan 20-15).

``KRCCSD`` (``KCCSD``), ``KUCCSD``, ``KGCCSD`` and ``KsymAdaptedRCCSD``, and the EOM
solvers ``EOMIP``, ``EOMEA``, ``EOMEESinglet`` (KRCCSD) and ``EOMEE`` (KGCCSD), are
the SAME objects as ``pyscf._native.pbc.cc.*`` (identity contract).

Differences from upstream's function-shaped entry points
(``pyscf/pbc/cc/__init__.py``):

* ``KRCCSD``/``KUCCSD``/``KGCCSD`` require a converged ``KRHF``/``KUHF``/``KGHF``
  respectively; no ``mf.to_rhf()``/``to_uhf()``/``to_ghf()`` conversion is done.
  A ``KRHF(cell, kpts=KPoints)`` mean field needs ``KsymAdaptedRCCSD``.
* The DF route is the mean field's ``with_df`` (FFTDF/AFTDF/MDF vs GDF/RSDF);
  upstream's two route pairs differ by 9.22e-4 Ha on diamond, so compare
  energies only within a route.
* ``frozen=`` raises ``NotImplementedError`` (the Rust drivers do not pad with a
  frozen spec). EOM ``partition='mp'``/``'full'`` raise ``NotImplementedError``,
  as upstream's ``ipccsd``/``eaccsd`` do (``eom_kccsd_ghf.py:618``).
* Option defaults are the Rust ones (``conv_tol=1e-9``, ``conv_tol_normt=1e-7``).

Not ported here: the gamma-point ``RCCSD``/``CCSD``/``UCCSD``/``GCCSD`` shims (they
raise ``NotImplementedError``). Upstream submodules (``kccsd_rhf``, ``kccsd_uhf``,
``kccsd``, ``eom_kccsd_*``, ``kccsd_t_rhf``, ...) fall through lazily (20-17).
"""

import pkgutil as _pkgutil

__path__ = _pkgutil.extend_path(__path__, __name__)
del _pkgutil

from pyscf._native.pbc.cc import (  # type: ignore[attr-defined]  # noqa: E402
    EOMEA,
    EOMEE,
    EOMIP,
    KCCSD,
    KGCCSD,
    KRCCSD,
    KUCCSD,
    EOMEESinglet,
    KsymAdaptedRCCSD,
)

__all__ = ["KRCCSD", "KCCSD", "KUCCSD", "KGCCSD", "KsymAdaptedRCCSD",
           "EOMIP", "EOMEA", "EOMEESinglet", "EOMEE",
           "RCCSD", "CCSD", "UCCSD", "GCCSD"]


def _gamma_unported(name):
    def make(mf, *args, **kwargs):
        raise NotImplementedError(
            f"pyscf.pbc.cc.{name}: the gamma-point PBC CCSD shim (upstream pbc/cc/ccsd.py) is "
            "not bound in pyscf-rs; use KRCCSD/KUCCSD/KGCCSD at one k-point"
        )

    make.__name__ = name
    make.__qualname__ = name
    return make


RCCSD = _gamma_unported("RCCSD")
CCSD = RCCSD
UCCSD = _gamma_unported("UCCSD")
GCCSD = _gamma_unported("GCCSD")

_UPSTREAM_SUBMODULES = ("ccsd", "kccsd_rhf", "kccsd_uhf", "kccsd", "eom_kccsd_rhf",
                        "eom_kccsd_uhf", "eom_kccsd_ghf", "kccsd_rhf_ksymm", "kccsd_t",
                        "kccsd_t_rhf", "kccsd_t_rhf_slow", "kintermediates",
                        "kintermediates_rhf", "kintermediates_uhf", "kuccsd_rdm")


def __getattr__(name):
    import importlib

    if name in _UPSTREAM_SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    alias = {"krccsd": "kccsd_rhf", "kuccsd": "kccsd_uhf", "kgccsd": "kccsd"}
    if name in alias:
        return importlib.import_module(f"{__name__}.{alias[name]}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
