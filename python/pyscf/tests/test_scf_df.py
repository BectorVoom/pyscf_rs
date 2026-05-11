"""Phase 3 SCF-07 density-fit RHF (`mf.density_fit().kernel()`).

Covers: SCF-07 — DF-HF total energy ≤ 1 µHartree vs upstream (same DF
auxiliary basis on both sides; cc-pVDZ-JKFIT is the default).
"""
from pyscf import scf


def test_scf_density_fit_uhartree_oracle(h2o_mol, upstream):
    mf_rs = scf.RHF(h2o_mol).density_fit().run()
    assert mf_rs.converged, "pyscf-rs DF-RHF did not converge"
    assert mf_rs.with_df, "with_df flag not set after density_fit()"

    mol_up = upstream.gto.M(atom=h2o_mol.atom, basis="cc-pvdz")
    mf_up = upstream.scf.RHF(mol_up).density_fit().run()
    assert mf_up.converged

    diff = abs(mf_rs.e_tot - mf_up.e_tot)
    assert diff < 1e-6, (
        f"DF-HF H2O/cc-pVDZ |e_rs - e_up| = {diff:.3e} > 1 µHartree (SCF-07); "
        f"e_rs={mf_rs.e_tot:.12f} e_up={mf_up.e_tot:.12f}"
    )
