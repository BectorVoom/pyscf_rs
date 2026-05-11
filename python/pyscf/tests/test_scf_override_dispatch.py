"""Phase 3 SCF-08 / BIND-07 subclass override dispatch.

Covers: SCF-08 / BIND-07 — Python subclass override of `get_veff` runs
≥ 1× per SCF cycle. This is the BIND-07 contract enforced by Plan 03-07's
`PyOverrideBridge::call_method1` MRO dispatch.

Per plan 03-07 §"call_method1 Dispatch Sites": the Rust SCF kernel calls
`slf.get_veff(mol, dm)` on every cycle (via `call_method1` on the cached
`Py<PyAny>` handle), so a Python subclass override is invoked
transparently with no special wiring.
"""
from pyscf import scf


def test_scf_subclass_override_get_veff_invoked_per_cycle(h2o_mol):
    """A subclass overriding `get_veff` is invoked ≥ mf.cycles times."""
    call_count = {"n": 0}

    class CountedHF(scf.RHF):
        def get_veff(self, mol, dm):
            call_count["n"] += 1
            return super().get_veff(mol, dm)

    mf = CountedHF(h2o_mol).run()
    assert mf.converged, "subclass-driven RHF did not converge"

    # The kernel typically calls get_veff once before the loop (initial
    # V_HF) and once per cycle thereafter. The exact count is implementation
    # detail; the BIND-07 contract is "subclass override is reached at least
    # `mf.cycles` times across an SCF run".
    assert call_count["n"] >= int(mf.cycles), (
        f"subclass get_veff was invoked {call_count['n']} times, "
        f"expected ≥ mf.cycles={mf.cycles} (BIND-07 fail)"
    )
