"""pyscf.pbc.ao2mo overlay — the native ``pbc/ao2mo/eris.py`` wrappers (plan 20-15).

``general``, ``get_mo_eri``, ``get_mo_pairs_G``, ``get_mo_pairs_invG``,
``assemble_eri``, ``get_ao_pairs_G`` and ``get_ao_eri`` are the SAME objects as
``pyscf._native.pbc.ao2mo.*`` (over ``crates/pyscf-pbc-ao2mo/src/eris.rs``). Each
builds a fresh FFTDF over ``cell``, as upstream's do; results are always
complex and never ``s2``-packed. The upstream ``eris`` module falls through lazily.
"""

import pkgutil as _pkgutil

__path__ = _pkgutil.extend_path(__path__, __name__)
del _pkgutil

from pyscf._native.pbc.ao2mo import (  # type: ignore[attr-defined]  # noqa: E402
    assemble_eri,
    general,
    get_ao_eri,
    get_ao_pairs_G,
    get_mo_eri,
    get_mo_pairs_G,
    get_mo_pairs_invG,
)

__all__ = ["general", "get_mo_eri", "get_mo_pairs_G", "get_mo_pairs_invG", "assemble_eri",
           "get_ao_pairs_G", "get_ao_eri"]


def __getattr__(name):
    import importlib

    if name == "eris":
        return importlib.import_module(f"{__name__}.eris")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
