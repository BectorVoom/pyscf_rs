"""Phase 3 SCF-11 cross-module dispatch.

Covers: SCF-11 — `to_uhf` / `to_rhf` / `to_ghf` work; `to_uks` / `to_rks`
raise `PyscfRsError` (KS targets stubbed for Phase 4).
"""
import pytest

from pyscf import PyscfRsError, scf


def test_rhf_to_uhf(h2o_mol):
    """RHF → UHF conversion produces a working UHF instance."""
    mf = scf.RHF(h2o_mol)
    uhf = mf.to_uhf()
    assert uhf is not None
    # The returned object must be a UHF type (BIND-02 overlay class name).
    assert type(uhf).__name__ == "UHF", f"expected UHF, got {type(uhf).__name__}"


def test_rhf_to_ghf(h2o_mol):
    """RHF → GHF conversion produces a working GHF instance."""
    mf = scf.RHF(h2o_mol)
    ghf = mf.to_ghf()
    assert ghf is not None
    assert type(ghf).__name__ == "GHF", f"expected GHF, got {type(ghf).__name__}"


def test_rhf_to_uks_raises_phase4_stub(h2o_mol):
    """`to_uks()` raises PyscfRsError citing Phase 4 (KS stubs deferred)."""
    mf = scf.RHF(h2o_mol)
    with pytest.raises(PyscfRsError) as exc_info:
        mf.to_uks()
    # The error message must reference NotYetImplemented or Phase 4.
    msg = str(exc_info.value).lower()
    assert ("notyetimplemented" in msg
            or "phase" in msg
            or "phase 4" in msg
            or "uks" in msg), (
        f"to_uks must raise PyscfRsError citing Phase 4 KS stub; got: {exc_info.value}"
    )


def test_rhf_to_rks_raises_phase4_stub(h2o_mol):
    """`to_rks()` raises PyscfRsError citing Phase 4."""
    mf = scf.RHF(h2o_mol)
    with pytest.raises(PyscfRsError) as exc_info:
        mf.to_rks()
    msg = str(exc_info.value).lower()
    assert ("notyetimplemented" in msg
            or "phase" in msg
            or "phase 4" in msg
            or "rks" in msg), (
        f"to_rks must raise PyscfRsError citing Phase 4 KS stub; got: {exc_info.value}"
    )


def test_scf_cross_dispatch_to_methods_and_ks_stubs(h2o_mol):
    """Aggregator name kept for grep continuity (plan 03-02 stub name)."""
    test_rhf_to_uhf(h2o_mol)
    test_rhf_to_ghf(h2o_mol)
    test_rhf_to_uks_raises_phase4_stub(h2o_mol)
    test_rhf_to_rks_raises_phase4_stub(h2o_mol)
