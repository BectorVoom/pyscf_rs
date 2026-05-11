"""Phase 3 SCF-11 cross-module dispatch (to_uhf / to_rhf / to_ghf works; to_uks / to_rks raises NotYetImplemented).

Covers: SCF-11 — cross-module dispatch helpers; KS targets stubbed for Phase 4.
"""
import pytest


@pytest.mark.xfail(
    reason="to_uhf/to_rhf/to_ghf assertion + to_uks/to_rks NotYetImplemented assertion pending — plan 03-10",
    strict=False,
)
def test_scf_cross_dispatch_to_methods_and_ks_stubs():
    assert False, "Phase 3 plan 03-10 must implement SCF-11 cross-dispatch + KS-stub assertion"
