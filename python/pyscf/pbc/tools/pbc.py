"""pyscf.pbc.tools.pbc overlay shim (plan 20-17).

The names in ``__all__`` are the SAME objects as ``pyscf._native.pbc.tools.*`` /
``pyscf._native.pbc.lib.kpts_helper.*`` (so ``from pyscf.pbc.tools.pbc import
super_cell`` is native). Upstream names without a binding (``precompute_exx``,
``get_monkhorst_pack_size``, ``get_lattice_Ls``, ``check_lattice_sum_range``,
``cutoff_to_gs``, ``gs_to_cutoff``, ``round_to_cell0``, ``FFT_ENGINE``, …) fall
through lazily to the upstream ``pbc.py`` this file shadows, with the one-time
passthrough warning. That upstream module currently fails to import under the
overlay (``ATM_SLOTS`` missing from the molecular ``pyscf.gto``), so those names
raise ``AttributeError`` naming the cause.
"""

from pyscf._native.pbc.lib.kpts_helper import (  # type: ignore[attr-defined]
    get_kconserv,
    get_kconserv3,
    intersection,
)
from pyscf._native.pbc.tools import (  # type: ignore[attr-defined]
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
from pyscf.pbc._unported import fallthrough_getattr as _fallthrough

_PYSCF_RS_IMPL = "partial"

__all__ = ["ExxDiv", "cell_plus_imgs", "cutoff_to_mesh", "fft", "fftk", "get_coulG", "ifft",
           "ifftk", "madelung", "mesh_to_cutoff", "super_cell", "get_kconserv",
           "get_kconserv3", "intersection"]

__getattr__ = _fallthrough(__name__, __name__)
