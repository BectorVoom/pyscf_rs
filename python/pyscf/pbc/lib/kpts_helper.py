"""pyscf.pbc.lib.kpts_helper overlay shim (plan 20-17).

The names in ``__all__`` are the SAME objects as
``pyscf._native.pbc.lib.kpts_helper.*`` (plan 20-14; ``get_kconserv`` is 20-09's
``pyscf._native.pbc.gto.get_kconserv``). Every other upstream name
(``round_to_fbz``, ``members_with_wrap_around``, ``loop_kkk``, ``conj_mapping``,
``get_kconserv_ria``, ``check_kpt_antiperm_symmetry``, ``VectorComposer``,
``VectorSplitter``, ``KptsHelper``, ``KPointSymmetryError``) is served lazily by
the upstream module this file shadows, with the one-time passthrough warning.
"""

from pyscf._native.pbc.lib.kpts_helper import (  # type: ignore[attr-defined]
    KPT_DIFF_TOL,
    gamma_point,
    get_kconserv,
    get_kconserv3,
    group_by_conj_pairs,
    intersection,
    is_gamma_point,
    is_trim,
    is_zero,
    kk_adapted_iter,
    member,
    unique,
    unique_with_wrap_around,
)
from pyscf.pbc._unported import fallthrough_getattr as _fallthrough

_PYSCF_RS_IMPL = "partial"

__all__ = ["KPT_DIFF_TOL", "gamma_point", "get_kconserv", "get_kconserv3",
           "group_by_conj_pairs", "intersection", "is_gamma_point", "is_trim", "is_zero",
           "kk_adapted_iter", "member", "unique", "unique_with_wrap_around"]

__getattr__ = _fallthrough(__name__, __name__)
