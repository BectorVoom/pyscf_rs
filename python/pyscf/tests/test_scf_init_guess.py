"""Phase 3 SCF-05 init_guess modes (5 modes vs upstream first-iter density).

Covers: SCF-05 — init_guess modes minao/atom/1e/huckel/chkfile.
"""
import pytest


@pytest.mark.xfail(
    reason="5 init_guess modes (minao/atom/1e/huckel/chkfile) assertion pending — plan 03-10",
    strict=False,
)
def test_scf_init_guess_five_modes_match_upstream_first_iter():
    assert False, "Phase 3 plan 03-10 must implement SCF-05 init_guess 5-mode assertion"
