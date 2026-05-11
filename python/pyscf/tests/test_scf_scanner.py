"""Phase 3 SCF-12 mf.as_scanner()(mol2) returns energy (Phase 7 geomopt prerequisite).

Covers: SCF-12 — scanner closure callable returns energy on a fresh Mole.
"""
import pytest


@pytest.mark.xfail(
    reason="mf.as_scanner()(mol2) returns energy assertion pending — plan 03-10",
    strict=False,
)
def test_scf_as_scanner_returns_energy_on_new_mole():
    assert False, "Phase 3 plan 03-10 must implement SCF-12 as_scanner assertion"
