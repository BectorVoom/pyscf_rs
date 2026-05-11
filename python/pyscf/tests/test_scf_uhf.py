"""Phase 3 SCF-02 UHF on open-shell radical vs upstream PySCF.

Covers: SCF-02 — UHF on radical matches upstream.

Fixture: NH2 doublet (S=1/2, 2 atoms, small enough for fast CI test).
"""
from pyscf import gto, scf


NH2_ATOM = "N 0.0 0.0 0.0; H 0.0 0.766 0.587; H 0.0 -0.766 0.587"


def test_scf_uhf_open_shell_oracle(upstream):
    """UHF on NH2 doublet — |e_rs - e_up| < 1 µHartree."""
    mol_rs = gto.M(atom=NH2_ATOM, basis="cc-pvdz", spin=1, charge=0)
    mf_rs = scf.UHF(mol_rs).run()
    assert mf_rs.converged, "pyscf-rs UHF did not converge"

    mol_up = upstream.gto.M(atom=NH2_ATOM, basis="cc-pvdz", spin=1, charge=0)
    mf_up = upstream.scf.UHF(mol_up).run()
    assert mf_up.converged, "upstream UHF did not converge"

    diff = abs(mf_rs.e_tot - mf_up.e_tot)
    assert diff < 1e-6, (
        f"UHF NH2/cc-pVDZ |e_rs - e_up| = {diff:.3e} > 1 µHartree (SCF-02); "
        f"e_rs={mf_rs.e_tot:.12f} e_up={mf_up.e_tot:.12f}"
    )
