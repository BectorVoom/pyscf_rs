"""Phase 3 SCF-07 density-fit RHF (`mf.density_fit().kernel()`).

Covers: SCF-07 — DF-HF total energy ≤ 1 µHartree vs upstream.
"""
import pytest


@pytest.mark.xfail(
    reason="mf.density_fit().kernel() ≤ 1 µHartree assertion pending — plan 03-10",
    strict=False,
)
def test_scf_density_fit_uhartree_oracle():
    assert False, "Phase 3 plan 03-10 must implement SCF-07 DF-HF µHartree assertion"
