"""Phase 3 SCF-01 RHF on H2O / cc-pVDZ vs upstream PySCF.

Covers: SCF-01 — RHF total energy ≤ 1 µHartree vs upstream.

Per RESEARCH §Validation Architecture: pyscf-rs runs in-process alongside
upstream PySCF (loaded via importlib under the `_upstream_pyscf` name —
see conftest.py). The Rust kernel cannot mutate upstream state, so
side-by-side comparison is safe and ~10× faster than subprocess isolation.
"""
from pyscf import scf


def test_scf_rhf_h2o_uhartree_oracle(h2o_mol, upstream):
    """RHF on H2O/cc-pVDZ — |e_rs - e_up| < 1 µHartree."""
    mf_rs = scf.RHF(h2o_mol).run()
    assert mf_rs.converged, "pyscf-rs RHF did not converge"

    mol_up = upstream.gto.M(
        atom="O 0.0 0.0 0.0; H 0.757 0.587 0.0; H -0.757 0.587 0.0",
        basis="cc-pvdz",
    )
    mf_up = upstream.scf.RHF(mol_up).run()
    assert mf_up.converged, "upstream RHF did not converge"

    diff = abs(mf_rs.e_tot - mf_up.e_tot)
    assert diff < 1e-6, (
        f"|e_rs - e_up| = {diff:.3e} > 1 µHartree (SCF-01); "
        f"e_rs={mf_rs.e_tot:.12f} e_up={mf_up.e_tot:.12f}"
    )
