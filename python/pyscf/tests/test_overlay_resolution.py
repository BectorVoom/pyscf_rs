"""Phase 3 BIND-02 overlay resolution (`from pyscf import scf` → `_native.scf`).

Covers: BIND-02 — overlay package supersedes upstream pyscf/ tree.

Plan 03-07 ships the `_native` cdylib + the `python/pyscf/__init__.py`
overlay; this test asserts the overlay binding `scf.RHF` is the SAME
Python object as `pyscf._native.scf.RHF` (i.e. that the overlay re-export
chain in `python/pyscf/scf/__init__.py` did not accidentally wrap or
re-create the class).
"""
from pyscf import scf
from pyscf._native.scf import RHF as RHF_native
from pyscf._native.scf import UHF as UHF_native
from pyscf._native.scf import GHF as GHF_native


def test_overlay_pyscf_scf_resolves_to_native():
    assert scf.RHF is RHF_native, (
        f"overlay broken: pyscf.scf.RHF = {scf.RHF}, expected {RHF_native}"
    )
    assert scf.UHF is UHF_native, (
        f"overlay broken: pyscf.scf.UHF = {scf.UHF}, expected {UHF_native}"
    )
    assert scf.GHF is GHF_native, (
        f"overlay broken: pyscf.scf.GHF = {scf.GHF}, expected {GHF_native}"
    )
