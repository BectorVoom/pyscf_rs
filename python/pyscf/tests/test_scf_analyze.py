"""Phase 3 SCF-09 mf.analyze / mulliken_pop / mulliken_meta / dip_moment vs upstream.

Covers: SCF-09 — analysis methods match upstream.
"""
import pytest


@pytest.mark.xfail(
    reason="analyze/mulliken_pop/mulliken_meta/dip_moment assertion pending — plan 03-10",
    strict=False,
)
def test_scf_analyze_mulliken_dipole_vs_upstream():
    assert False, "Phase 3 plan 03-10 must implement SCF-09 analyze/mulliken/dipole assertion"
