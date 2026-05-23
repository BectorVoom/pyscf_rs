//! GTO-10 — `set_geom_` in-place geometry mutation with granular cache
//! invalidation per RESEARCH Pattern 5.
//!
//! Contract:
//!   - `_env` xyz slots are mutated.
//!   - `_bas`, `_basis`, `ao_loc_nr`, `nao_nr`, `basis_set` Arc identity
//!     are PRESERVED.
//!   - Atom-count and per-position symbol mismatches are rejected.
//!   - `mol.unit` is honoured for the input string conversion.

use pyscf_core::Unit;
use pyscf_core::raw_layout::{ATM_SLOTS, PTR_COORD};
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs, set_geom_};
use std::sync::Arc;

fn h2_at_1_4() -> pyscf_core::Mole {
    M(MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0; H 0 0 1.4".into()),
        basis: BasisInput::Name("sto-3g".into()),
        unit: Unit::Bohr,
        max_memory: 4000.0,
        axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ..Default::default()
    })
    .unwrap()
}

#[test]
fn set_geom_updates_env_coords_only() {
    let mut mol = h2_at_1_4();
    let bas_before = mol._bas.clone();
    let ao_loc_before = mol.ao_loc_nr.clone();
    let nao_before = mol.nao_nr;
    let basis_arc_before = mol.basis_set.as_ref().unwrap().clone();
    let env_len_before = mol._env.len();

    // Capture per-atom coord pointers BEFORE the mutation so the test reads
    // exactly the slots set_geom_ touched (avoids hard-coding _env layout).
    let ptr_a0 = mol._atm[PTR_COORD] as usize; // atom 0 row starts at offset 0
    let ptr_a1 = mol._atm[ATM_SLOTS + PTR_COORD] as usize;

    set_geom_(&mut mol, "H 0 0 0; H 0 0 2.0").unwrap();

    // _env coordinate slots updated.
    assert_eq!(mol._env[ptr_a0], 0.0);
    assert_eq!(mol._env[ptr_a0 + 1], 0.0);
    assert_eq!(mol._env[ptr_a0 + 2], 0.0);
    assert_eq!(mol._env[ptr_a1], 0.0);
    assert_eq!(mol._env[ptr_a1 + 1], 0.0);
    assert_eq!(mol._env[ptr_a1 + 2], 2.0);

    // _env length unchanged (no extra primitives appended).
    assert_eq!(mol._env.len(), env_len_before);

    // _bas, ao_loc_nr, nao_nr UNCHANGED.
    assert_eq!(mol._bas, bas_before);
    assert_eq!(mol.ao_loc_nr, ao_loc_before);
    assert_eq!(mol.nao_nr, nao_before);

    // basis_set Arc identity preserved (Pattern 5 zero-invalidation contract).
    assert!(Arc::ptr_eq(
        mol.basis_set.as_ref().unwrap(),
        &basis_arc_before
    ));
}

#[test]
fn set_geom_atom_count_mismatch_errors() {
    let mut mol = h2_at_1_4();
    let r = set_geom_(&mut mol, "H 0 0 0");
    assert!(matches!(r, Err(pyscf_core::PyscfRsError::Core(_))));
}

#[test]
fn set_geom_symbol_mismatch_errors() {
    let mut mol = h2_at_1_4();
    let r = set_geom_(&mut mol, "He 0 0 0; H 0 0 2.0");
    assert!(matches!(r, Err(pyscf_core::PyscfRsError::Core(_))));
}

#[test]
fn set_geom_honours_unit_kwarg() {
    let mut mol = M(MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0; H 0 0 0.7411".into()),
        basis: BasisInput::Name("sto-3g".into()),
        unit: Unit::Ang, // Ang → Bohr conversion
        max_memory: 4000.0,
        axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ..Default::default()
    })
    .unwrap();
    // mol.unit still Ang; set_geom_ honours it.
    set_geom_(&mut mol, "H 0 0 0; H 0 0 1.0").unwrap();
    // Second H should be at z = 1.0 * 1.8897261339213 Bohr.
    approx::assert_abs_diff_eq!(mol.atom_coords()[1][2], 1.8897261339213, epsilon = 1e-9);
}

#[test]
fn set_geom_updates_atom_coords_method_output() {
    let mut mol = h2_at_1_4();
    set_geom_(&mut mol, "H 0 0 0; H 0 0 2.0").unwrap();
    let coords = mol.atom_coords();
    assert_eq!(coords[0], [0.0, 0.0, 0.0]);
    assert_eq!(coords[1], [0.0, 0.0, 2.0]);
}
