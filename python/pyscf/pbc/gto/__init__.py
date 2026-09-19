"""pyscf.pbc.gto overlay — re-exports the native periodic Cell (plan 20-09).

The names below are the SAME objects as ``pyscf._native.pbc.gto.*`` (the
identity contract of ``python/pyscf/tests/test_pbc_identity_gate.py``).

``Cell`` has no ``get_hcore``: the periodic core Hamiltonian lives on the DF
object, ``pyscf.pbc.df.FFTDF(cell, kpts).get_hcore(kpts)``.

Every other upstream ``pyscf.pbc.gto`` name (``cell``, ``basis``, ``pseudo``,
``parse``, ``load``, ...) is NOT ported here; ``__path__`` is extended so the
upstream submodules stay importable and ``__getattr__`` falls through to them
lazily. Whether that fallthrough is announced is plan 20-17's decision.
"""

import pkgutil as _pkgutil

__path__ = _pkgutil.extend_path(__path__, __name__)
del _pkgutil

from pyscf._native.pbc.gto import (  # type: ignore[attr-defined]  # noqa: E402
    Cell,
    KPath,
    M,
    band_path,
    band_path_from_segments,
    cell_plus_imgs,
    detect_lattice,
    dumps,
    get_coulG,
    get_kconserv,
    loads,
    make_kpts,
    pack,
    super_cell,
    unpack,
)

__all__ = [
    "Cell",
    "KPath",
    "M",
    "band_path",
    "band_path_from_segments",
    "cell_plus_imgs",
    "detect_lattice",
    "dumps",
    "get_coulG",
    "get_kconserv",
    "loads",
    "make_kpts",
    "pack",
    "super_cell",
    "unpack",
]

# upstream `pyscf/pbc/gto/__init__.py` star-imports these modules
_UPSTREAM_SOURCES = ("pyscf.pbc.gto.cell", "pyscf.pbc.gto.basis", "pyscf.pbc.gto.pseudo",
                     "pyscf.pbc.gto.neighborlist")


def __getattr__(name):
    import importlib

    if name in ("cell", "basis", "pseudo", "neighborlist", "ecp"):
        return importlib.import_module(f"{__name__}.{name}")
    if name.startswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    for src in _UPSTREAM_SOURCES:
        # 20-17: an upstream source that fails to import under the overlay must not
        # turn `hasattr(gto, name)` into an ImportError (the passthrough is announced
        # by `pyscf.pbc._unported`; the failure is recorded in docs/pbc-status.md).
        try:
            mod = importlib.import_module(src)
        except ImportError as exc:
            raise AttributeError(
                f"module {__name__!r} has no native attribute {name!r}, and the upstream "
                f"fallthrough {src} failed to import: {exc}"
            ) from exc
        if hasattr(mod, name):
            return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
