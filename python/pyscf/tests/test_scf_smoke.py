"""Phase 3 SCF-01 smoke aggregator (RHF on H2O/cc-pVDZ).

Covers: SCF-01 (smoke entry — the wave-level health check).

This is the minimum-cost test the maturin-smoke CI job runs after
`maturin develop` to confirm the wheel imports + RHF.kernel runs to
completion. Energy assertion is a coarse range check (NOT the µHartree
oracle — that's test_scf_rhf_h2o.py); the smoke job's role is "the wheel
isn't a paperweight."
"""
from pyscf import scf


def test_scf_rhf_h2o_smoke(h2o_mol):
    mf = scf.RHF(h2o_mol).run()
    assert mf.converged, "RHF on H2O/cc-pVDZ failed to converge"
    # H2O/cc-pVDZ RHF e_tot ≈ -76.026765 Ha (NIST CCCBDB).
    # A 30 mHa window survives any reasonable basis/grid-mode drift while
    # still catching real regressions (e.g. wrong nuclear term, wrong
    # ovlp orientation).
    assert -76.05 < mf.e_tot < -76.00, f"unexpected e_tot = {mf.e_tot}"
