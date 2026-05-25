//! Always-on DF-HF end-to-end test (SCF-07, plan 03-12).
//!
//! With int2e (05-08) + the rank-revealing DF metric fit (05-09),
//! `RHF::density_fit` + `DfHooks` + the SCF kernel converge to a DF-HF energy
//! that matches the non-DF RHF energy within DF accuracy. This proves the whole
//! DF-HF stack end-to-end: int3c2e/int2c2e → cholesky_eri (robust metric) →
//! get_jk_df → SCF loop convergence.
//!
//! Uses the `1e` init guess (the default `minao` mode lands in plan 03-13).
//! Upstream-PySCF byte-identity is the CI-gated/human-verify arm (no numpy/PySCF
//! in the sandbox).

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};
use pyscf_scf::{DfHooks, InitGuessMode, KernelConfig, NoOverrides, kernel};

fn h2() -> pyscf_core::Mole {
    M(MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0; H 0 0 1.4".into()),
        basis: BasisInput::Name("sto-3g".into()),
        unit: Unit::Bohr,
        ..Default::default()
    })
    .expect("build H2/STO-3G")
}

fn cfg_1e() -> KernelConfig {
    KernelConfig {
        init_guess: InitGuessMode::OneElectron,
        ..Default::default()
    }
}

#[test]
fn dfhf_converges_and_matches_rhf_within_df_accuracy() {
    let mol = h2();

    // Reference: plain (non-DF) RHF via the real int2e get_jk.
    let rhf = kernel(&mol, &NoOverrides, cfg_1e()).expect("RHF converge");
    assert!(rhf.converged, "non-DF RHF must converge");
    assert!(
        rhf.e_tot.0 > -2.0 && rhf.e_tot.0 < -1.0,
        "RHF e_tot ≈ -1.117, got {}",
        rhf.e_tot.0
    );
    let e_rhf = rhf.e_tot.0;

    // DF-HF with proper auxiliary bases must converge AND match RHF within DF
    // accuracy (the DF fitting error — sub-mHartree for these aux).
    for aux in ["weigend", "cc-pvdz-jkfit"] {
        let df = pyscf_df::cholesky_eri(&mol, aux)
            .unwrap_or_else(|e| panic!("cholesky_eri({aux}): {e}"));
        let hooks = DfHooks { df: &df };
        let res =
            kernel(&mol, &hooks, cfg_1e()).unwrap_or_else(|e| panic!("DF-HF[{aux}] kernel: {e}"));

        assert!(res.converged, "DF-HF[{aux}] must converge");
        assert!(res.e_tot.0.is_finite(), "DF-HF[{aux}] e_tot finite");
        let diff = (res.e_tot.0 - e_rhf).abs();
        assert!(
            diff < 1e-3,
            "DF-HF[{aux}] e_tot {} must match RHF {} within DF accuracy (|Δ|={:.3e})",
            res.e_tot.0,
            e_rhf,
            diff
        );
        eprintln!(
            "DF-HF[{aux}] e_tot = {} (|Δ vs RHF| = {:.3e}, eff. naux = {})",
            res.e_tot.0, diff, df.naux
        );
    }
}

/// A minimal basis used as its own auxiliary (sto-3g) is a deliberately poor DF
/// fit — it must still CONVERGE cleanly (no panic), just far from RHF. This
/// guards the SCF loop's robustness, not DF accuracy.
#[test]
fn dfhf_poor_minimal_aux_still_converges() {
    let mol = h2();
    let df = pyscf_df::cholesky_eri(&mol, "sto-3g").expect("cholesky_eri(sto-3g)");
    let hooks = DfHooks { df: &df };
    let res = kernel(&mol, &hooks, cfg_1e()).expect("DF-HF[sto-3g] kernel");
    assert!(
        res.converged,
        "DF-HF with a minimal aux must still converge"
    );
    assert!(res.e_tot.0.is_finite());
}
