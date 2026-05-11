"""Phase 3 SCF-01 smoke aggregator (RHF on H2O/cc-pVDZ).

Covers: SCF-01 (smoke entry — the wave-level health check).
"""
import pytest


@pytest.mark.xfail(
    reason="smoke body pending — plan 03-10 (SCF-01 H2O/cc-pVDZ RHF)",
    strict=False,
)
def test_scf_rhf_h2o_smoke():
    assert False, "Phase 3 plan 03-10 must implement RHF/H2O smoke body"
