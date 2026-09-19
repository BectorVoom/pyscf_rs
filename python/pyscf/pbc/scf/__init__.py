"""pyscf.pbc.scf overlay — re-exports the native periodic SCF drivers (plan 20-12).

``KRHF``, ``KUHF``, ``KROHF``, ``KGHF`` (and ``KSCF``, ``KsymAdaptedKRHF``) are
the SAME objects as ``pyscf._native.pbc.scf.*`` — the identity contract of
``python/pyscf/tests/test_pbc_identity_gate.py``. Upstream makes ``KRHF``,
``KUHF`` and ``KGHF`` *functions* that branch on ``isinstance(kpts, KPoints)``
(``pyscf/pbc/scf/__init__.py:40-106``); here the native constructors apply the
same rule so the identity holds:

* ``KRHF(cell, kpts=<KPoints>)`` runs the k-point-symmetry adapted driver;
* ``KUHF`` / ``KGHF`` with a ``KPoints`` RAISE ``NotImplementedError`` — this
  port has no ``kuhf_ksymm`` / ``kghf_ksymm``, and they must not fall through
  to upstream Python. ``pyscf.pbc.scf.kuhf_ksymm`` and ``.kghf_ksymm`` are
  stubs that raise on use, for the same reason.

The function-shaped upstream entry points (``RHF``, ``UHF``, ``ROHF``, ``GHF``,
``HF``, ``KHF``) keep their Python-level dispatch below. The gamma-point
``RHF``/``UHF``/``ROHF``/``GHF`` are the k-point drivers at one k-point
(``pyscf-pbc-scf/src/gamma.rs``); their results are per-k lists of length 1.

Upstream submodules (``khf``, ``kuhf``, ``hf``, ``addons``, ``chkfile``,
``smearing``, ``newton_ah``, ...) still fall through lazily to the vendored
upstream tree; whether that is announced is plan 20-17's decision. The native
add-ons are re-exported at the top level: ``smearing_``, ``project_mo_nr2nr``,
``load_scf``.
"""

import pkgutil as _pkgutil
import sys as _sys
import types as _types

__path__ = _pkgutil.extend_path(__path__, __name__)
del _pkgutil

from pyscf._native.pbc.scf import (  # type: ignore[attr-defined]  # noqa: E402
    KGHF,
    KRHF,
    KROHF,
    KSCF,
    KUHF,
    KsymAdaptedKRHF,
    load_scf,
    project_mo_nr2nr,
    smearing_,
)
from pyscf._native.pbc.scf import GHF as _gamma_GHF  # noqa: E402
from pyscf._native.pbc.scf import RHF as _gamma_RHF  # noqa: E402
from pyscf._native.pbc.scf import ROHF as _gamma_ROHF  # noqa: E402
from pyscf._native.pbc.scf import UHF as _gamma_UHF  # noqa: E402

__all__ = [
    "KSCF", "KRHF", "KUHF", "KROHF", "KGHF", "KsymAdaptedKRHF",
    "RHF", "UHF", "ROHF", "GHF", "HF", "KHF",
    "KS", "KKS", "RKS", "UKS", "ROKS", "KRKS", "KUKS", "KROKS",
    "smearing_", "project_mo_nr2nr", "load_scf",
]


def _cell_spin(cell):
    spin = getattr(cell, "spin", 0)
    return spin() if callable(spin) else spin


def RHF(cell, *args, **kwargs):
    """`pbc/scf/__init__.py:40`: `kpts=` → KRHF; spin 0 → gamma RHF; else ROHF."""
    if "kpts" in kwargs:
        return KRHF(cell, *args, **kwargs)
    if _cell_spin(cell) == 0:
        return _gamma_RHF(cell, *args, **kwargs)
    return _gamma_ROHF(cell, *args, **kwargs)


def UHF(cell, *args, **kwargs):
    """`pbc/scf/__init__.py:48`: `kpts=` → KUHF; else gamma UHF."""
    if "kpts" in kwargs:
        return KUHF(cell, *args, **kwargs)  # a KPoints raises: no kuhf_ksymm
    return _gamma_UHF(cell, *args, **kwargs)


def GHF(cell, *args, **kwargs):
    """`pbc/scf/__init__.py:54`: `kpts=` → KGHF; else gamma GHF."""
    if "kpts" in kwargs:
        return KGHF(cell, *args, **kwargs)
    return _gamma_GHF(cell, *args, **kwargs)


def ROHF(cell, *args, **kwargs):
    """`pbc/scf/__init__.py:60`: `kpts=` → KROHF; else gamma ROHF."""
    if "kpts" in kwargs:
        return KROHF(cell, *args, **kwargs)
    return _gamma_ROHF(cell, *args, **kwargs)


def HF(cell, *args, **kwargs):
    if _cell_spin(cell) == 0:
        return RHF(cell, *args, **kwargs)
    return UHF(cell, *args, **kwargs)


def KHF(cell, *args, **kwargs):
    if _cell_spin(cell) == 0:
        return KRHF(cell, *args, **kwargs)
    return KUHF(cell, *args, **kwargs)


def _dft(name):
    def make(cell, *args, **kwargs):
        from pyscf.pbc import dft

        return getattr(dft, name)(cell, *args, **kwargs)

    make.__name__ = name
    make.__qualname__ = name
    return make


KS = _dft("KS")
KKS = _dft("KKS")
RKS = _dft("RKS")
ROKS = _dft("ROKS")
UKS = _dft("UKS")
KRKS = _dft("KRKS")
KUKS = _dft("KUKS")
KROKS = _dft("KROKS")


# ── k-point-symmetry variants this port does not have: raise, never fall through ──

_UNPORTED_KSYMM = {
    "kuhf_ksymm": "KUHF with k-point symmetry (upstream pyscf/pbc/scf/kuhf_ksymm.py)",
    "kghf_ksymm": "KGHF with k-point symmetry (upstream pyscf/pbc/scf/kghf_ksymm.py)",
}


class _UnportedModule(_types.ModuleType):
    """A `sys.modules` stub whose public attributes raise NotImplementedError.

    Private/dunder names raise AttributeError so introspection
    (`inspect.getmodule`, pickling, pytest) that probes every module stays safe.
    """

    def __init__(self, name, what):
        super().__init__(name)
        self.__doc__ = f"NOT PORTED: {what}."
        object.__setattr__(self, "_what", what)

    def __getattr__(self, attr):
        if attr.startswith("_"):
            raise AttributeError(attr)
        raise NotImplementedError(
            f"{self.__name__}.{attr}: {self._what} is not implemented in pyscf-rs; "
            "it is deliberately not served from the upstream Python tree"
        )


for _mod, _what in _UNPORTED_KSYMM.items():
    _full = f"{__name__}.{_mod}"
    _stub = _sys.modules.get(_full)
    if not isinstance(_stub, _UnportedModule):
        _stub = _UnportedModule(_full, _what)
        _sys.modules[_full] = _stub
    globals()[_mod] = _stub
del _mod, _what, _full, _stub

_UPSTREAM_SUBMODULES = ("hf", "uhf", "rohf", "ghf", "khf", "kuhf", "krohf", "kghf",
                        "khf_ksymm", "newton_ah", "addons", "chkfile", "cphf", "smearing",
                        "stability", "rsjk", "scfint", "_response_functions")


def __getattr__(name):
    import importlib

    if name in _UPSTREAM_SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    if name == "krhf":
        return importlib.import_module(f"{__name__}.khf")
    if name == "rhf":
        return importlib.import_module(f"{__name__}.hf")
    if name == "newton":
        return importlib.import_module(f"{__name__}.newton_ah").newton
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
