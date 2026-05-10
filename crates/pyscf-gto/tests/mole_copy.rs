//! GTO-10 — `mol.clone()` deep-copy semantics.
//!
//! `Mole` derives `Clone` (Phase 1 + 02-02). Clone deep-copies value fields
//! (`Vec<i32>`, `Vec<f64>`) and Arc-clones `basis_set` so GTO-11 zero-copy
//! is preserved across copies.

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs, M};
use std::sync::Arc;

#[test]
fn mole_clone_is_deep_for_value_fields() {
    let mut a = M(MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0; H 0 0 1.4".into()),
        basis: BasisInput::Name("sto-3g".into()),
        unit: Unit::Bohr,
        max_memory: 4000.0,
        axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ..Default::default()
    })
    .unwrap();
    let b = a.clone();
    // Mutate `a`'s _env; `b._env` must NOT change (Vec<f64> is value-cloned).
    let original_b_env = b._env.clone();
    // Pick an index that exists; _env is at least PTR_ENV_START + per-atom slots
    // long, which is well over 20 for any built mol.
    let mutate_idx = a._env.len() / 2;
    a._env[mutate_idx] = 999.0;
    assert_eq!(b._env, original_b_env, "Mole.clone deep-copies _env");
}

#[test]
fn mole_clone_arc_identity_preserved_for_basis_set() {
    let a = M(MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0".into()),
        basis: BasisInput::Name("sto-3g".into()),
        unit: Unit::Bohr,
        max_memory: 4000.0,
        axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ..Default::default()
    })
    .unwrap();
    let b = a.clone();
    // GTO-11 zero-copy preserved by clone: same Arc.
    assert!(Arc::ptr_eq(
        a.basis_set.as_ref().unwrap(),
        b.basis_set.as_ref().unwrap()
    ));
}
