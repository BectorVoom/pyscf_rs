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
    // 1e (hcore) init guess — the default `minao` mode is NotYetImplemented
    // until plan 03-13.
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

#[test]
fn kernel_propagates_jk_not_yet_implemented() {
    // This test runs unignored — it asserts the gap closes cleanly via
    // an error rather than panicking. Once int2e_sph lands the test
    // either updates to assert success or moves under #[ignore].
    let mol = M(MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0; H 0 0 1.4".into()),
        basis: BasisInput::Name("sto-3g".into()),
        unit: Unit::Bohr,
        ..Default::default()
    })
    .expect("build H2");
    let result = kernel(&mol, &NoOverrides, KernelConfig::default());
    // Must return Err — never panic, never reach convergence (no JK builder).
    match result {
        Err(e) => {
            let msg = format!("{}", e);
            // Should be either int2e NotYetImplemented or an init_guess
            // failure (minao mode is also not yet implemented). Both are
            // acceptable: the kernel is well-formed, the gap is documented.
            assert!(
                msg.contains("not yet implemented")
                    || msg.contains("NotYetImplemented")
                    || msg.contains("int2e")
                    || msg.contains("minao")
                    || msg.contains("init_guess"),
                "expected NotYetImplemented-flavoured error, got: {}",
                msg
            );
        }
        Ok(_) => panic!(
            "expected NotYetImplemented (int2e_sph or minao); plan 03-11's surface ships an \
             error, not silent convergence. Unignore the bit-exact H2 test if this changes."
        ),
    }
}
