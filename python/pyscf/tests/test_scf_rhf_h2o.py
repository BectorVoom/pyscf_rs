"""Phase 3 SCF-01 RHF on H2O / cc-pVDZ vs upstream PySCF.

Covers: SCF-01 — RHF total energy ≤ 1 µHartree vs upstream.
"""
import pytest


@pytest.mark.xfail(
    reason="RHF H2O/cc-pVDZ ≤ 1 µHartree assertion pending — plan 03-10",
    strict=False,
)
def test_scf_rhf_h2o_uhartree_oracle():
    assert False, "Phase 3 plan 03-10 must implement SCF-01 RHF/H2O µHartree assertion"
