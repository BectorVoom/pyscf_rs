"""pyscf.pbc overlay — native Rust bindings for the ported families, announced
upstream passthrough for everything else (plans 20-08 … 20-17, D-PBC-35).

This is the pyscf-rs ``_native`` re-export overlay (DISTINCT from the vendored
upstream ``pyscf/pbc/``). The overlay packages below re-export the SAME objects
as ``pyscf._native.pbc.*`` — never wrappers — so
``pyscf.pbc.scf.KRHF is pyscf._native.pbc.scf.KRHF`` holds
(``python/pyscf/tests/test_pbc_identity_gate.py``):

  ``gto`` (20-09) · ``df`` (20-10) · ``scf`` (20-12) · ``dft`` (20-13) ·
  ``symm`` (20-14) · ``lib`` / ``tools`` (20-14 bindings, 20-17 shims) ·
  ``mp`` / ``cc`` / ``ci`` / ``ao2mo`` (20-15)

Upstream's ``isinstance(kpts, KPoints)`` dispatch for ``KRHF``/``KUHF``/``KGHF``/
``KRKS``/``KUKS``/``KRKSpU``/``KUKSpU``/``KMP2`` lives in the native constructors
(a type check, not MRO); the function-shaped ``RHF``/``UHF``/``KS``/``KKS``/…
dispatch lives in the overlay ``scf``/``dft`` packages.

``pkgutil.extend_path`` is KEPT (D-PBC-35): every ``pyscf.pbc.*`` module the
overlay does not shadow resolves to the next ``pyscf`` distribution on
``sys.path`` — upstream PySCF *Python*. That passthrough is ANNOUNCED: the first
import per family per process that lands outside the overlay emits one
``pyscf.pbc._unported.PbcUpstreamFallthroughWarning`` (a ``UserWarning``), and
``pyscf.pbc.which_impl(name)`` answers ``"native"`` / ``"partial"`` /
``"upstream"`` at runtime. Status table and the measured import status of
every upstream module under the overlay: ``docs/pbc-status.md``.
"""

import pkgutil as _pkgutil

__path__ = _pkgutil.extend_path(__path__, __name__)
del _pkgutil

from pyscf.pbc import _unported  # noqa: E402
from pyscf.pbc._unported import FAMILIES, PbcUpstreamFallthroughWarning, which_impl  # noqa: E402

_unported.install()

__all__: list[str] = ["which_impl", "FAMILIES", "PbcUpstreamFallthroughWarning"]
