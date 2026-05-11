"""Phase 3 SCF-13 cross-platform µHartree consistency (Pitfall 12 mitigation).

Covers: SCF-13 — Linux x86_64 + macOS aarch64 µHartree assertion on H2O/cc-pVDZ.
This is the Python-side companion to the matrix-CI job; the assertion runs on
each platform and the matrix job's pass/fail is the cross-platform contract.
"""
import pytest


@pytest.mark.xfail(
    reason="Linux x86_64 + macOS aarch64 µHartree assertion pending — plan 03-10",
    strict=False,
)
def test_scf_xplat_uhartree_h2o_ccpvdz():
    assert False, "Phase 3 plan 03-10 must implement SCF-13 cross-platform µHartree assertion"
