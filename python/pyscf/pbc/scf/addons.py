"""pyscf.pbc.scf.addons overlay shim (plan 20-18).

``smearing_`` and ``project_mo_nr2nr`` are the SAME objects as
``pyscf._native.pbc.scf.*`` (plan 20-12): upstream 2.12.1 re-exports
``smearing_`` from ``pyscf/pbc/scf/smearing.py`` (``addons.py:28-37``) and
defines ``project_mo_nr2nr`` at ``addons.py:39``.

Every other upstream name (``convert_to_uhf``, ``convert_to_rhf``,
``convert_to_ghf``, ``convert_to_kscf``/``convert_to_khf``, ``canonical_occ_``,
``project_dm_k2k``, ``mo_energy_with_exxdiv_none``, ``smearing``,
``SMEARING_METHOD``, ...) falls through lazily, with the one-time D-PBC-35
announcement, to the upstream ``addons.py`` this file shadows. That module
imports ``pyscf.scf.addons``, which the molecular overlay does not provide, so
today those names raise ``AttributeError`` naming the cause
(``pyscf.pbc.which_impl('scf.addons.convert_to_uhf') == 'upstream'``).
"""

from pyscf._native.pbc.scf import project_mo_nr2nr, smearing_  # type: ignore[attr-defined]
from pyscf.pbc._unported import fallthrough_getattr as _fallthrough

_PYSCF_RS_IMPL = "partial"

__all__ = ["smearing_", "project_mo_nr2nr"]

__getattr__ = _fallthrough(__name__, __name__)
