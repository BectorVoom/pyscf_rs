"""Phase 3 BIND-02 overlay resolution (`from pyscf import scf` → `_native.scf`).

Covers: BIND-02 — overlay package supersedes upstream pyscf/ tree.
"""
import pytest


@pytest.mark.xfail(
    reason="from pyscf import scf resolves to _native.scf assertion pending — plan 03-10",
    strict=False,
)
def test_overlay_pyscf_scf_resolves_to_native():
    assert False, "Phase 3 plan 03-10 must implement BIND-02 overlay-resolution assertion"
