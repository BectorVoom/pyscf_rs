//! Regression: d-shell (l=2) ERIs must be correct end-to-end through RHF.
//!
//! Guards the cintx Rys-quadrature fix (`math/rys.rs` rys_root3/4/5 host port).
//! Before the fix, rys_root3/4/5 fell to the large-x asymptotic `r/(x-r)` form
//! for all x past a far-too-low cutoff (x>3 / x>1 / x>1), producing wrong (even
//! negative) Rys roots in the intermediate range that d-shell integrals land in.
//! High-angular-momentum ERIs need more roots — `(dd|dd)` needs nroots=5 — so a
//! basis with a d-shell (cc-pVDZ on O) was systematically off: H2O/cc-pVDZ RHF
//! sat ~0.46 mHartree above upstream. Lighter systems (He/H2 cc-pVDZ, max
//! nroots=3) stayed within the intact x<3 polynomial branch and were unaffected
//! — see `int2e_general_contraction.rs`.
//!
//! With the full libcint branch port (intermediate-x polynomial branches up to
//! x≈47/53/59), H2O/cc-pVDZ now matches upstream PySCF 2.12.1 to < 1 µHartree.

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};
use pyscf_scf::{KernelConfig, NoOverrides, kernel};

/// Same geometry as the Phase-3 H2O gate (`python/pyscf/tests/test_scf_rhf_h2o.py`).
const H2O: &str = "O 0.0 0.0 0.0; H 0.757 0.587 0.0; H -0.757 0.587 0.0";

#[test]
fn h2o_ccpvdz_rhf_matches_upstream_uhartree() {
    // Upstream PySCF 2.12.1: H2O / cc-pVDZ RHF = -76.026765673118 Hartree
    // (geometry above, default RHF: minao guess + DIIS + direct_scf).
    let mol = M(MoleBuildArgs {
        atom: AtomInput::String(H2O.into()),
        basis: BasisInput::Name("cc-pvdz".into()),
        unit: Unit::Ang,
        ..Default::default()
    })
    .expect("build H2O/cc-pVDZ molecule");
    let r = kernel(&mol, &NoOverrides, KernelConfig::default()).expect("H2O/cc-pVDZ RHF kernel");

    assert!(r.converged, "H2O/cc-pVDZ RHF must converge");
    let e = r.e_tot.0;
    assert!(
        (e - (-76.026_765_673_118)).abs() < 1e-6,
        "H2O/cc-pVDZ RHF = {e:.12} must match upstream -76.026765673118 to < 1 µHartree \
         — d-shell (l=2) Rys-quadrature regression (cintx math/rys.rs rys_root3/4/5)"
    );
}
