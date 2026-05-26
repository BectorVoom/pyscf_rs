"""SCF-01 RHF cc-pVDZ µHartree gate vs upstream (general contraction + d-shell).

Restores cc-pVDZ as a ≤ 1 µHartree RHF gate. Commit 52b6965 had demoted the
H2O/cc-pVDZ gate to STO-3G because H2O/cc-pVDZ carried a ~0.46 mHartree d-shell
(l=2) Rys residual. The cintx Rys-quadrature fix closed that gap by porting
libcint's missing intermediate-x branches into rys_root3/4/5 (rys.rs), so
H2O/cc-pVDZ is now a *true* µHartree gate again.

Coverage across the cc-pVDZ angular-momentum range:
  - He   — generally-contracts 4s→2s (general-contraction int2e path), nroots≤1.
  - H2   — generally-contracts 4s→2s, p-shell present, nroots≤3.
  - H2O  — O carries a d-shell (l=2); (dd|dd) needs nroots=5, exercising the
           full Rys quadrature that the d-shell fix restored.

This is the Python-overlay twin of the kernel-level regressions in
``crates/pyscf-scf/tests/int2e_general_contraction.rs`` (He/H2) and
``crates/pyscf-scf/tests/d_shell_rys.rs`` (H2O d-shell).

Per RESEARCH §Validation Architecture: pyscf-rs runs alongside upstream PySCF
(loaded via a separate interpreter / importlib — see conftest.py).
"""
import pytest

from pyscf import gto, scf


# (atom, label). He/H2 are d-free generally-contracted (4s→2s); H2O adds the
# l=2 d-shell on oxygen — the case that the Rys intermediate-x port restored.
CCPVDZ_SYSTEMS = [
    ("He 0.0 0.0 0.0", "He"),
    ("H 0.0 0.0 0.0; H 0.0 0.0 0.74", "H2"),
    ("O 0.0 0.0 0.0; H 0.757 0.587 0.0; H -0.757 0.587 0.0", "H2O"),
]


@pytest.mark.parametrize("atom,label", CCPVDZ_SYSTEMS)
def test_scf_rhf_ccpvdz_uhartree_oracle(atom, label, upstream_rhf_energy):
    """RHF on {He,H2,H2O}/cc-pVDZ — |e_rs - e_up| < 1 µHartree."""
    mol_rs = gto.M(atom=atom, basis="cc-pvdz")
    mf_rs = scf.RHF(mol_rs)
    mf_rs.run()
    assert mf_rs.converged, f"pyscf-rs RHF on {label}/cc-pVDZ did not converge"

    mf_up = upstream_rhf_energy(atom, "cc-pvdz")
    assert mf_up["converged"], f"upstream RHF on {label}/cc-pVDZ did not converge"

    diff = abs(mf_rs.e_tot - mf_up["e_tot"])
    assert diff < 1e-6, (
        f"|e_rs - e_up| = {diff:.3e} > 1 µHartree ({label}/cc-pVDZ); "
        f"e_rs={mf_rs.e_tot:.12f} e_up={mf_up['e_tot']:.12f}"
    )
