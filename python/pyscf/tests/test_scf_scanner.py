"""Phase 3 SCF-12 `mf.as_scanner()(mol2)` returns energy.

Covers: SCF-12 — scanner closure callable returns the converged total
energy on a perturbed Mole. Prerequisite for Phase 7 geomopt drivers
(which call `scanner(mol_t+dt)` per BFGS line-search step).
"""
from pyscf import gto, scf


def test_as_scanner_perturbed_mole(h2o_mol):
    """Scanner closure runs SCF on a perturbed geometry."""
    mf = scf.RHF(h2o_mol).run()
    scanner = mf.as_scanner()
    assert callable(scanner)

    # Tiny geometry perturbation (one bond stretched by 0.003 Å).
    mol2 = gto.M(
        atom="O 0.0 0.0 0.0; H 0.760 0.587 0.0; H -0.757 0.587 0.0",
        basis="cc-pvdz",
    )
    e2 = scanner(mol2)
    assert isinstance(e2, float)
    # A 3 mÅ geometry change moves the energy by at most a few mHa.
    delta_e = abs(e2 - mf.e_tot)
    assert delta_e < 0.01, (
        f"tiny geometry change must not change e_tot by > 10 mHa; "
        f"|Δe| = {delta_e:.6f} Ha"
    )


def test_scf_as_scanner_returns_energy_on_new_mole(h2o_mol):
    """Aggregator name kept for grep continuity (plan 03-02 stub name)."""
    test_as_scanner_perturbed_mole(h2o_mol)
