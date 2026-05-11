//! Plan 03-11 Task 1 RED — smoke that `NoOverrides + kernel` drives an
//! end-to-end SCF on a minimal H2 fixture.
//!
//! `#[ignore]`'d because `int2e_sph` (the arity-4 ERI dispatcher in
//! pyscf-gto) is `NotYetImplemented{phase:2}` per the Phase 2 plan
//! 02-09 verification rollup gap-closure. Plan 03-10 (oracle harness)
//! unignores this once int2e_sph lands or once DF-HF (plan 03-05)
//! provides an alternative override path.
//!
//! Until then, plan 03-11 ships:
//!   - The kernel cycle loop body (verbatim port of pyscf/scf/hf.py:48-244).
//!   - default_get_jk that propagates int2e_sph's NotYetImplemented error.
//!   - All other defaults (eig, occ, rdm, energy_*, init_guess_by_1e,
//!     analyze, convert, scanner) are real bodies — see the unit tests in
//!     kernel_internals_unit.rs.

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs, M};
use pyscf_scf::{kernel, KernelConfig, NoOverrides};

#[test]
#[ignore = "needs pyscf-gto int2e_sph arity-4 dispatch — unignore in plan 03-10"]
fn h2_no_overrides_converges() {
    let mol = M(MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0; H 0 0 1.4".into()),
        basis: BasisInput::Name("sto-3g".into()),
        unit: Unit::Bohr,
        ..Default::default()
    })
    .expect("build H2");
    let result = kernel(&mol, &NoOverrides, KernelConfig::default()).expect("converge");
    assert!(result.converged);
    assert!(result.cycles <= 30, "H2/STO-3G should converge in ≤30 cycles");
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
                msg.contains("not yet implemented") || msg.contains("NotYetImplemented")
                    || msg.contains("int2e") || msg.contains("minao")
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
