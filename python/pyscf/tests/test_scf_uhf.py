"""Phase 3 SCF-02 UHF on open-shell radical vs upstream PySCF.

Covers: SCF-02 — UHF on radical matches upstream.
"""
import pytest


@pytest.mark.xfail(
    reason="UHF open-shell assertion pending — plan 03-10",
    strict=False,
)
def test_scf_uhf_open_shell_oracle():
    assert False, "Phase 3 plan 03-10 must implement SCF-02 UHF open-shell assertion"
