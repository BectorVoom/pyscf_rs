"""pyscf.pbc.lib overlay — k-point helpers (bindings 20-14, shim 20-17).

``kpts_helper`` and ``kpts`` are overlay shim modules: their ported names are
the SAME objects as ``pyscf._native.pbc.lib.kpts_helper.*`` /
``pyscf._native.pbc.lib.kpts.*``; unported names (``loop_kkk``, ``KptsHelper``,
``conj_mapping``, …) fall through lazily to the upstream module they shadow,
with the one-time passthrough warning. ``arnoldi``, ``linalg_helper``,
``chkfile`` and ``ktensor`` are upstream Python (``pyscf.pbc.which_impl``).
"""

import pkgutil as _pkgutil

__path__ = _pkgutil.extend_path(__path__, __name__)
del _pkgutil

from pyscf.pbc.lib import kpts_helper  # noqa: E402  (upstream's __init__ does the same)

_PYSCF_RS_IMPL = "partial"

__all__: list[str] = []  # the submodule shims carry the native names
