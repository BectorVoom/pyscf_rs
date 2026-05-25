"""SCF-01 RHF cc-pVDZ general-contraction µHartree gate (d-free He/H2 vs upstream).

Restores cc-pVDZ as a ≤ 1 µHartree RHF gate. Commit 52b6965 demoted the
H2O/cc-pVDZ gate to STO-3G because H2O/cc-pVDZ retains a ~0.46 mHa d-shell
(l=2) Rys residual — far above µHartree. This gate keeps cc-pVDZ coverage but
on d-free systems, so it stays a *true* µHartree gate.

cc-pVDZ generally-contracts He 4s→2s and H 4s→2s, exercising the int2e
general-contraction path (cintx `two_electron.rs`). He and H2 carry NO d-shell
(li+lj ≤ 2), isolating that path from the still-open l=2 Rys gap, so they match
upstream PySCF 2.12.1 to < 1 µHartree. This is the Python-overlay twin of the
kernel-level regression in
``crates/pyscf-scf/tests/int2e_general_contraction.rs``.

Per RESEARCH §Validation Architecture: pyscf-rs runs alongside upstream PySCF
(loaded via a separate interpreter / importlib — see conftest.py).
"""
import pytest

from pyscf import gto, scf


# (atom, label) — both d-free, both generally-contracted (4s→2s) in cc-pVDZ.
CCPVDZ_GENCONTRACT_SYSTEMS = [
    ("He 0.0 0.0 0.0", "He"),
    ("H 0.0 0.0 0.0; H 0.0 0.0 0.74", "H2"),
]


@pytest.mark.parametrize("atom,label", CCPVDZ_GENCONTRACT_SYSTEMS)
def test_scf_rhf_ccpvdz_gencontract_uhartree_oracle(atom, label, upstream_rhf_energy):
    """RHF on {He,H2}/cc-pVDZ — |e_rs - e_up| < 1 µHartree (general contraction)."""
    mol_rs = gto.M(atom=atom, basis="cc-pvdz")
    mf_rs = scf.RHF(mol_rs)
    mf_rs.run()
    assert mf_rs.converged, f"pyscf-rs RHF on {label}/cc-pVDZ did not converge"

    mf_up = upstream_rhf_energy(atom, "cc-pvdz")
    assert mf_up["converged"], f"upstream RHF on {label}/cc-pVDZ did not converge"

    diff = abs(mf_rs.e_tot - mf_up["e_tot"])
    assert diff < 1e-6, (
        f"|e_rs - e_up| = {diff:.3e} > 1 µHartree ({label}/cc-pVDZ general "
        f"contraction); e_rs={mf_rs.e_tot:.12f} e_up={mf_up['e_tot']:.12f}"
    )
