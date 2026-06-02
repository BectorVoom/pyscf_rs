//! Regression: generally-contracted (nctr>1) two-electron integrals must be
//! correct end-to-end through RHF.
//!
//! Guards the cintx `two_electron.rs` general-contraction fix (per-(ci,cj,ck,cl)
//! Cartesian blocks + contraction-major scatter, mirroring the 1e fix 9af2164).
//! Before the fix, the 2e path summed every contraction quad into ONE buffer and
//! scattered it as if `nctr==1`, leaving the inner-contraction AO blocks zero →
//! near-zero ERIs → a wrong Fock matrix → RHF diverged for generally-contracted
//! bases (cc-pVDZ etc.).
//!
//! cc-pVDZ contracts He `4s→2s` and H `4s→2s` (general contraction). He and H2
//! carry NO d-shell (`li+lj ≤ 2`), so these cases isolate the general-contraction
//! path from the separate, still-open d-shell (l=2) nuclear/2e Rys gap and match
//! upstream PySCF 2.12.1 to < 1 µHartree. (H2O/cc-pVDZ converges after the fix but
//! retains a ~0.46 mHa d-shell residual, so it is intentionally NOT asserted here.)

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};
use pyscf_scf::{KernelConfig, NoOverrides, kernel};

fn rhf_ccpvdz_etot(atom: &str) -> (f64, bool) {
    let mol = M(MoleBuildArgs {
        atom: AtomInput::String(atom.into()),
        basis: BasisInput::Name("cc-pvdz".into()),
        unit: Unit::Ang,
        ..Default::default()
    })
    .expect("build cc-pVDZ molecule");
    // Default config = minao guess + DIIS + direct_scf (the pyscf-py RHF path).
    let r = kernel(&mol, &NoOverrides, KernelConfig::default()).expect("cc-pVDZ RHF kernel");
    (r.e_tot.0, r.converged)
}

#[test]
fn he_ccpvdz_rhf_matches_upstream_uhartree() {
    // Upstream PySCF 2.12.1: He / cc-pVDZ RHF = -2.8551604772 Hartree.
    let (e, conv) = rhf_ccpvdz_etot("He 0 0 0");
    assert!(
        conv,
        "He/cc-pVDZ RHF must converge (generally-contracted s-block)"
    );
    assert!(
        (e - (-2.855_160_477_2)).abs() < 1e-6,
        "He/cc-pVDZ RHF = {e:.10} must match upstream -2.8551604772 to < 1 µHartree \
         — general-contraction int2e regression (cintx two_electron.rs)"
    );
}

#[test]
fn h2_ccpvdz_rhf_matches_upstream_uhartree() {
    // Upstream PySCF 2.12.1: H2 (0.74 Å) / cc-pVDZ RHF = -1.1287000936 Hartree.
    let (e, conv) = rhf_ccpvdz_etot("H 0 0 0; H 0 0 0.74");
    assert!(
        conv,
        "H2/cc-pVDZ RHF must converge (generally-contracted s-block)"
    );
    assert!(
        (e - (-1.128_700_093_6)).abs() < 1e-6,
        "H2/cc-pVDZ RHF = {e:.10} must match upstream -1.1287000936 to < 1 µHartree \
         — general-contraction int2e regression (cintx two_electron.rs)"
    );
}
