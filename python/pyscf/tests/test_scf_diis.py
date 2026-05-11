"""Phase 3 SCF-04 C-DIIS iteration count vs upstream (±1 cycle).

Covers: SCF-04 — C-DIIS converges in upstream iteration count ±1.
"""
import pytest


@pytest.mark.xfail(
    reason="C-DIIS iteration count ±1 vs upstream assertion pending — plan 03-10",
    strict=False,
)
def test_scf_cdiis_iteration_count_within_one():
    assert False, "Phase 3 plan 03-10 must implement SCF-04 C-DIIS iteration-count assertion"
