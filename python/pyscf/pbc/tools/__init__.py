"""pyscf.pbc.tools overlay — the native FFT / Coulomb / lattice helpers
(bindings 20-09 and 20-14, shim 20-17).

The names in ``__all__`` are the SAME objects as ``pyscf._native.pbc.tools.*``
(``get_kconserv``/``get_kconserv3``/``intersection`` are
``pyscf._native.pbc.lib.kpts_helper``'s, re-exported here as upstream's
``from pyscf.pbc.tools.pbc import *`` does). ``pyscf.pbc.tools.pbc`` is an overlay
shim with the same names.

Not ported (``pyscf.pbc.which_impl('tools')`` is ``"partial"``): ``k2gamma``,
``lattice``, ``pyscf_ase``, ``pywannier90``, ``print_funcs``, ``tril``,
``make_test_cell``, and the ``pbc.py`` names ``precompute_exx``,
``get_monkhorst_pack_size``, ``get_lattice_Ls``, ``check_lattice_sum_range``,
``cutoff_to_gs``, ``gs_to_cutoff``, ``round_to_cell0`` (ported in Rust, unbound).
Those resolve lazily to upstream Python, with the one-time passthrough warning.
"""

import importlib as _importlib
import pkgutil as _pkgutil

__path__ = _pkgutil.extend_path(__path__, __name__)
del _pkgutil

from pyscf._native.pbc.lib.kpts_helper import (  # type: ignore[attr-defined]  # noqa: E402
    get_kconserv,
    get_kconserv3,
    intersection,
)
from pyscf._native.pbc.tools import (  # type: ignore[attr-defined]  # noqa: E402
    ExxDiv,
    cell_plus_imgs,
    cutoff_to_mesh,
    fft,
    fftk,
    get_coulG,
    ifft,
    ifftk,
    madelung,
    mesh_to_cutoff,
    super_cell,
)

_PYSCF_RS_IMPL = "partial"

__all__ = ["ExxDiv", "cell_plus_imgs", "cutoff_to_mesh", "fft", "fftk", "get_coulG", "ifft",
           "ifftk", "madelung", "mesh_to_cutoff", "super_cell", "get_kconserv",
           "get_kconserv3", "intersection"]

_SUBMODULES = ("pbc", "k2gamma", "lattice", "make_test_cell", "print_funcs", "pyscf_ase",
               "pywannier90", "tril")


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name in _SUBMODULES:
        return _importlib.import_module(f"{__name__}.{name}")
    # upstream's __init__ star-imports tools/pbc.py, then tools/print_funcs.py
    try:
        return getattr(_importlib.import_module(f"{__name__}.pbc"), name)
    except AttributeError:
        pass
    try:
        funcs = _importlib.import_module(f"{__name__}.print_funcs")
    except Exception as exc:  # noqa: BLE001
        raise AttributeError(
            f"module {__name__!r} has no native attribute {name!r}, and the upstream "
            f"fallthrough {__name__}.print_funcs failed to import: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        return getattr(funcs, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
