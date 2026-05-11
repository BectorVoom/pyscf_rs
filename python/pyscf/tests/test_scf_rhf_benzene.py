"""Phase 3 SCF-01 RHF on benzene / 6-31G* vs upstream PySCF.

Covers: SCF-01 — RHF total energy on benzene/6-31G* matches upstream.
"""
from pyscf import scf


def test_scf_rhf_benzene_uhartree_oracle(benzene_mol, upstream):
    mf_rs = scf.RHF(benzene_mol).run()
    assert mf_rs.converged, "pyscf-rs RHF on benzene did not converge"

    # Build a fresh upstream mol from the SAME atom string (the fixture's
    # `benzene_mol.atom` is the canonical pyscf-rs representation; pass it
    # verbatim so any cleanup-by-upstream is symmetric).
    mol_up = upstream.gto.M(atom=benzene_mol.atom, basis="6-31g*")
    mf_up = upstream.scf.RHF(mol_up).run()
    assert mf_up.converged

    diff = abs(mf_rs.e_tot - mf_up.e_tot)
    assert diff < 1e-6, (
        f"benzene/6-31G* |e_rs - e_up| = {diff:.3e} > 1 µHartree (SCF-01); "
        f"e_rs={mf_rs.e_tot:.12f} e_up={mf_up.e_tot:.12f}"
    )
