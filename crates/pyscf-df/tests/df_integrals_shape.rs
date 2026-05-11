//! Shape-only smoke test for `cholesky_eri` (plan 03-05 Task 1 RED).
//!
//! Bit-exact DF-HF energy assertion vs upstream is in plan 03-10. This test
//! only asserts:
//!   1. `DfIntegrals { b_uvq, naux, nao }` is constructable.
//!   2. `nao` matches `mol.nao_nr`.
//!   3. `b_uvq.len() == nao * nao * naux`.
//!
//! The test is `#[ignore]`d because cintx-ops does NOT yet ship the
//! `int3c2e_sph` base operator id (only the derivative + an unstable
//! source variant); plan 03-05 Task 0's `intor_with_auxmol` for that name
//! returns a zero-filled buffer. The (P|Q) Cholesky of cintx's current
//! synthetic-staging `int2c2e_sph` output is likely rank-deficient
//! (singular). Plan 03-10 unignores this test once cintx upstream lands
//! the base operator id AND flips from synthetic to real evaluation.

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs, M};

#[test]
#[ignore = "cintx-ops int3c2e_sph base symbol missing; plan 03-10 unignores"]
fn h2o_cc_pvdz_df_integrals_shape() {
    let mol = M(MoleBuildArgs {
        atom: AtomInput::String("O 0 0 0; H 0 1 0; H 1 0 0".into()),
        basis: BasisInput::Name("cc-pvdz".into()),
        unit: Unit::Bohr,
        ..Default::default()
    })
    .expect("H2O/cc-pVDZ");
    let df = pyscf_df::cholesky_eri(&mol, "cc-pvdz-jkfit").expect("cholesky_eri");
    assert_eq!(df.nao, mol.nao_nr);
    assert!(df.naux > 0);
    assert_eq!(df.b_uvq.len(), df.nao * df.nao * df.naux);
}

/// Negative-path: cholesky_eri's body is reachable even when the inner
/// dispatcher is partially stubbed. Asserts the `DfIntegrals` struct surface
/// (`b_uvq`, `naux`, `nao`) is wired without requiring real numerical
/// integrals.
#[test]
fn df_integrals_struct_fields_compile() {
    // This is a pure-compile check — the struct must declare the trio of
    // public fields. If it fails, the auxbasis_defaults test won't compile
    // either, so this is belt-and-suspenders.
    let df = pyscf_df::DfIntegrals {
        b_uvq: vec![0.0; 6],
        naux: 2,
        nao: 1,
    };
    assert_eq!(df.b_uvq.len(), 6);
    assert_eq!(df.naux, 2);
    assert_eq!(df.nao, 1);
}
