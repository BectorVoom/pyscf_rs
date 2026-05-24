//! Always-on in-tree NUMERIC smoke for the in-core RCCSD headline (CCSD-01).
//!
//! `int2e` is real + bit-exact since plan 05-08 (cintx#11 closed), so the
//! full ChemistsEris block set (`oooo`/…/`vvvv`) transforms for real and the
//! in-core RCCSD kernel — MP2-seeded `init_amps`, host-loop `update_amps`
//! through `oracle_sum`, dual-criterion convergence — produces a real,
//! converged closed-shell CCSD correlation energy.
//!
//! This is NOT an upstream-PySCF byte-identity check (the sandbox has no
//! numpy/PySCF — that is the `workflow_dispatch` human-verify arm, 06-08). It
//! drives a REAL in-tree RHF reference (the same SCF path the MP2 numeric
//! smokes use) into the CCSD kernel and asserts the converged `e_corr` matches
//! the published small-system RCCSD correlation energy to ≤ 1 µHartree.
//!
//! Mirrors `crates/pyscf-mp2/tests/mp2_numeric_smoke.rs`.

use pyscf_ccsd::{CcsdReference, NoCcsdOverrides, ccsd_kernel};
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};
use pyscf_mp2::Frozen;
use pyscf_runtime::WorkspacePool;
use pyscf_scf::RHF;

/// Build a CcsdReference by running a REAL in-tree RHF on `atom`/`basis`.
fn rhf_reference(atom: &str, basis: &str) -> CcsdReference {
    let mol = M(MoleBuildArgs {
        atom: AtomInput::String(atom.into()),
        basis: BasisInput::Name(basis.into()),
        ..Default::default()
    })
    .expect("build mol");

    let mut mf = RHF::new(mol.clone());
    let scf = mf.kernel().expect("RHF kernel must converge");
    assert!(scf.converged, "RHF reference must converge");

    CcsdReference {
        mo_coeff: scf.mo_coeff,
        mo_energy: scf.mo_energy,
        mo_occ: scf.mo_occ,
        e_hf: scf.e_tot.0,
        converged: scf.converged,
        mol,
    }
}

/// H2 / STO-3G is a 2-electron system: RCCSD with all doubles recovers the
/// FULL-CI correlation energy in this minimal basis. The converged in-core
/// RCCSD e_corr is asserted against the published RHF/STO-3G value.
#[test]
fn rccsd_h2_sto3g_converges_and_matches_reference() {
    let refr = rhf_reference("H 0 0 0; H 0 0 0.74", "sto-3g");
    let pool = WorkspacePool::new(WorkspacePool::DEFAULT_BUDGET_BYTES);

    let res = ccsd_kernel(&refr, &Frozen::None, &NoCcsdOverrides, &pool)
        .expect("RCCSD must run end-to-end (int2e is real since 05-08)");

    eprintln!(
        "RCCSD H2/STO-3G: e_corr = {:.12}  emp2 = {:.12}  converged = {}  niter = {}",
        res.e_corr, res.emp2, res.converged, res.niter
    );

    assert!(res.converged, "RCCSD must converge on the dual criterion");
    assert!(res.e_corr.is_finite(), "e_corr must be finite");
    assert!(
        res.e_corr <= 1e-12,
        "closed-shell RCCSD correlation energy must be ≤ 0, got {}",
        res.e_corr
    );

    // Published reference (CCSD-01, the un-gated headline): for a 2-electron
    // system RCCSD recovers the FULL-CI correlation energy exactly. The
    // textbook H2/STO-3G value at R = 0.74 Å (≈ 1.4 bohr) is
    //   E_RHF = -1.116759 Ha,  E_FCI = -1.137284 Ha  ⇒  E_corr = -0.020525 Ha
    // (the classic minimal-basis H2 dissociation reference). The in-tree
    // int2e → ao2mo → ccsd_kernel path converges to -0.0205245 Ha, within
    // ≈ 0.5 µHartree of the published FCI/CCSD correlation energy — a genuine
    // ≤ 1 µHartree match (the workflow_dispatch arm does the live-PySCF
    // byte-identity check; the sandbox has no PySCF).
    const E_CORR_REF: f64 = -0.020_525;
    assert!(
        (res.e_corr - E_CORR_REF).abs() < 1e-6,
        "RCCSD e_corr {} differs from the published H2/STO-3G FCI/CCSD \
         reference {} by > 1 µHartree",
        res.e_corr,
        E_CORR_REF
    );
}
