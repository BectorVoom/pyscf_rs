"""Phase 3 SCF-11 cross-module dispatch.

Covers: SCF-11 — `to_uhf` / `to_rhf` / `to_ghf` work; `to_uks` / `to_rks`
are wired to the real KS targets (carryover 20-molecular-python-suite-drift
item 2: wired in `fbe1e35`, 2026-05-22 — the Phase-3 stub refusal this file
asserted is long gone).
"""
from pyscf import scf


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


def test_rhf_to_uks_is_wired(h2o_mol):
    """`to_uks(xc)` returns a working UKS instance (wired, not a stub)."""
    mf = scf.RHF(h2o_mol)
    uks = mf.to_uks(xc="lda,vwn")
    assert uks is not None
    assert type(uks).__name__ == "UKS", f"expected UKS, got {type(uks).__name__}"


def test_rhf_to_rks_is_wired(h2o_mol):
    """`to_rks(xc)` returns a working RKS instance (wired, not a stub)."""
    mf = scf.RHF(h2o_mol)
    rks = mf.to_rks(xc="lda,vwn")
    assert rks is not None
    assert type(rks).__name__ == "RKS", f"expected RKS, got {type(rks).__name__}"


def test_scf_cross_dispatch_to_methods_and_ks_stubs(h2o_mol):
    """Aggregator name kept for grep continuity (plan 03-02 stub name)."""
    test_rhf_to_uhf(h2o_mol)
    test_rhf_to_ghf(h2o_mol)
    test_rhf_to_uks_is_wired(h2o_mol)
    test_rhf_to_rks_is_wired(h2o_mol)
