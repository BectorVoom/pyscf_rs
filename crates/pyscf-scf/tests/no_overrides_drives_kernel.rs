//! Plan 03-11 Task 1 — smoke that `NoOverrides + kernel` drives an
//! end-to-end SCF on a minimal H2 fixture.
//!
//! ALWAYS-ON since 03-12: the `int2e_sph` arity-4 dispatch gap that blocked
//! this (`NotYetImplemented{phase:2}`) is closed (05-08), so `default_get_jk`
//! builds a real Fock matrix and the kernel converges. Uses the `1e` init
//! guess — the default `minao` guess lands in plan 03-13. The kernel cycle
//! loop is a verbatim port of `pyscf/scf/hf.py:48-244`; eig/occ/rdm/energy_*/
//! init_guess_by_1e/analyze/convert/scanner are real bodies (unit-tested in
//! kernel_internals_unit.rs).

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};
use pyscf_scf::{InitGuessMode, KernelConfig, NoOverrides, kernel};

#[test]
fn h2_no_overrides_converges() {
    let mol = M(MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0; H 0 0 1.4".into()),
        basis: BasisInput::Name("sto-3g".into()),
        unit: Unit::Bohr,
        ..Default::default()
    })
    .expect("build H2");
    // 1e (hcore) init guess — the default `minao` path is exercised separately
    // by `default_minao_config_converges` (minao landed in 03-13).
    let cfg = KernelConfig {
        init_guess: InitGuessMode::OneElectron,
        ..Default::default()
    };
    let result = kernel(&mol, &NoOverrides, cfg).expect("converge");
    assert!(result.converged);
    assert!(
        result.cycles <= 30,
        "H2/STO-3G should converge in ≤30 cycles"
    );
    assert!(
        result.e_tot.0 > -2.0 && result.e_tot.0 < -1.0,
        "H2/STO-3G total energy should be ~ -1.117 Hartree, got {}",
        result.e_tot.0
    );
}

/// FLIPPED in 03-13: the DEFAULT `KernelConfig` (minao init guess) now converges
/// — both the int2e_sph gap (05-08) AND the minao init guess (03-13) are closed,
/// so the out-of-the-box `kernel(&mol, &NoOverrides, default())` runs a real SCF.
/// minao and the 1e guess must converge to the SAME energy (same molecule/basis).
#[test]
fn default_minao_config_converges() {
    let mol = M(MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0; H 0 0 1.4".into()),
        basis: BasisInput::Name("sto-3g".into()),
        unit: Unit::Bohr,
        ..Default::default()
    })
    .expect("build H2");

    // Default config = minao init guess (no longer NotYetImplemented).
    let minao = kernel(&mol, &NoOverrides, KernelConfig::default()).expect("minao converge");
    assert!(minao.converged, "default (minao) SCF must converge");
    assert!(
        minao.e_tot.0 > -2.0 && minao.e_tot.0 < -1.0,
        "minao H2/STO-3G e_tot ≈ -1.117, got {}",
        minao.e_tot.0
    );

    // 1e guess on the same system → same converged energy (within conv_tol).
    let cfg_1e = KernelConfig {
        init_guess: InitGuessMode::OneElectron,
        ..Default::default()
    };
    let one_e = kernel(&mol, &NoOverrides, cfg_1e).expect("1e converge");
    assert!(
        (minao.e_tot.0 - one_e.e_tot.0).abs() < 1e-7,
        "minao ({}) and 1e ({}) must converge to the same energy",
        minao.e_tot.0,
        one_e.e_tot.0
    );
}
