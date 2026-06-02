//! F-11 — `pyscf_core::Mole::build()` IoC builder hook.
//!
//! `pyscf-core` is zero-compute-deps (FOUND-02): it cannot parse basis input
//! and cannot depend on `pyscf-gto`. The builder is registered at runtime via
//! an inversion-of-control hook (`fn(&mut Mole)`), so a direct `Mole::build()`
//! dispatches into `pyscf-gto` without any compile-time `core → gto` edge.
//!
//! These tests exercise the registered arm. The unregistered (cold-start /
//! NYI) arm is covered by the `pyscf-core` unit tests, where `pyscf-gto` is
//! never linked.

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs, register_mole_builder};

fn h2(basis: &str) -> MoleBuildArgs {
    MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0; H 0 0 1.4".into()),
        basis: BasisInput::Name(basis.into()),
        unit: Unit::Bohr,
        ..Default::default()
    }
}

#[test]
fn front_door_arms_the_hook() {
    // Building through `M` must register the IoC hook (FOUND-02 runtime wiring).
    let _ = M(h2("sto-3g")).unwrap();
    assert!(
        pyscf_core::mole_builder_is_registered(),
        "pyscf_gto::M must arm the pyscf-core builder hook"
    );
}

#[test]
fn explicit_register_is_idempotent() {
    register_mole_builder();
    register_mole_builder(); // second call is a no-op, must not panic
    assert!(pyscf_core::mole_builder_is_registered());
}

#[test]
fn copy_rebuild_with_new_basis_via_mole_build() {
    // Canonical upstream pattern: `auxmol = mol.copy(); auxmol.basis = aux;
    // auxmol.build()` (pyscf/df/incore.py:23-31), expressed with the native
    // `Mole::build()` entry point rather than `pyscf_gto::M`.
    let reference_small = M(h2("sto-3g")).unwrap();
    let reference_big = M(h2("cc-pvdz")).unwrap();
    assert_ne!(
        reference_small.nao_nr, reference_big.nao_nr,
        "sto-3g and cc-pvdz must differ in nao for this test to be meaningful"
    );

    let mut mol = reference_small.clone();
    mol.basis = "cc-pvdz".to_string(); // raw name assignment, upstream-style
    mol.build().expect("Mole::build() must dispatch to the gto hook");

    assert_eq!(mol.natm, 2, "atoms preserved across rebuild");
    assert_eq!(
        mol.nao_nr, reference_big.nao_nr,
        "rebuild must pick up the new basis (cc-pvdz nao)"
    );
    assert_eq!(
        mol._bas.len(),
        reference_big._bas.len(),
        "shell table must match a direct M(cc-pvdz) build"
    );
    assert!(mol._built);
}

#[test]
fn rebuild_same_basis_is_value_stable() {
    // Rebuilding without changing the basis must reproduce the same molecule
    // (no double unit conversion of the Bohr-stored _atom, echo basis name
    // round-trips through strip_name_echo).
    let reference = M(h2("sto-3g")).unwrap();

    let mut mol = reference.clone();
    mol.build().expect("Mole::build() must dispatch to the gto hook");

    assert_eq!(mol.nao_nr, reference.nao_nr);
    assert_eq!(mol.natm, reference.natm);
    assert_eq!(mol._atm, reference._atm, "_atm stable across same-basis rebuild");
    assert_eq!(mol._bas, reference._bas, "_bas stable across same-basis rebuild");
    // Coordinates must NOT drift (Bohr fed back as Bohr — no re-conversion).
    for (a, b) in mol._atom.iter().zip(reference._atom.iter()) {
        assert_eq!(a.0, b.0, "atom symbol stable");
        assert_eq!(a.1, b.1, "atom coordinates stable (no double unit conversion)");
    }
}

#[test]
fn direct_build_from_structured_atoms() {
    // A Mole carrying structured `_atom` + a basis name (but not yet built)
    // builds directly through `Mole::build()` once the hook is armed.
    register_mole_builder();
    let reference = M(h2("sto-3g")).unwrap();

    let mut fresh = pyscf_core::Mole {
        _atom: reference._atom.clone(),
        natm: reference.natm,
        basis: "sto-3g".to_string(),
        unit: Unit::Bohr,
        _built: false,
        ..Default::default()
    };
    fresh.build().expect("direct build from structured atoms");
    assert!(fresh._built);
    assert_eq!(fresh.nao_nr, reference.nao_nr);
    assert_eq!(fresh._bas, reference._bas);
}
