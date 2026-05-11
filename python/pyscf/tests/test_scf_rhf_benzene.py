"""Phase 3 SCF-01 RHF on benzene / 6-31G* vs upstream PySCF.

Covers: SCF-01 — RHF total energy on benzene/6-31G* matches upstream.
"""
import pytest


@pytest.mark.xfail(
    reason="RHF benzene/6-31G* ≤ 1 µHartree assertion pending — plan 03-10",
    strict=False,
)
def test_scf_rhf_benzene_uhartree_oracle():
    assert False, "Phase 3 plan 03-10 must implement SCF-01 RHF/benzene µHartree assertion"
