"""Phase 3 SCF-08 / BIND-07 subclass override dispatch (`get_veff` invoked once per cycle).

Covers: SCF-08 / BIND-07 — Python subclass override of `get_veff` runs ≥ 1× per SCF cycle.
"""
import pytest


@pytest.mark.xfail(
    reason="Subclass get_veff invoked ≥ cycles assertion pending — plan 03-10",
    strict=False,
)
def test_scf_subclass_override_get_veff_invoked_per_cycle():
    assert False, "Phase 3 plan 03-10 must implement SCF-08/BIND-07 subclass-dispatch assertion"
