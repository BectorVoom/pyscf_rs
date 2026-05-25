//! GTO-11 verification: zero-copy `Arc<BasisSet>` contract.
//!
//! `mol.basis_set` stores `Option<Arc<cintx_core::BasisSet>>`. Cloning the
//! `Mole` clones the `Option<Arc<...>>`, which `Arc::clone()`s the inner
//! handle — same underlying allocation, refcount bumped. `Arc::ptr_eq`
//! across clones must be `true`; `Arc::strong_count` must be ≥ 2.
//!
//! Downstream phases (intor 02-05, eval_gto 02-06, SCF, DFT, MP2, CCSD,
//! grad) inherit the zero-copy contract by construction — they only ever
//! see `Arc<BasisSet>` clones, never raw bytes.

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};
use std::sync::Arc;

fn h2_sto3g_args() -> MoleBuildArgs {
    MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0; H 0 0 1.4".into()),
        basis: BasisInput::Name("sto-3g".into()),
        unit: Unit::Bohr,
        ..Default::default()
    }
}

fn h_only_args() -> MoleBuildArgs {
    MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0".into()),
        basis: BasisInput::Name("sto-3g".into()),
        unit: Unit::Bohr,
        ..Default::default()
    }
}

#[test]
fn arc_ptr_eq_after_mole_clone() {
    let mol_a = M(h2_sto3g_args()).expect("H2/STO-3G build must succeed");
    let mol_b = mol_a.clone();
    // GTO-11 contract: cloning the Mole must NOT deep-copy the basis set;
    // both Arc handles point to the same allocation.
    let arc_a = mol_a
        .basis_set
        .as_ref()
        .expect("mol_a.build must have populated basis_set");
    let arc_b = mol_b
        .basis_set
        .as_ref()
        .expect("mol_b.build must have populated basis_set");
    assert!(
        Arc::ptr_eq(arc_a, arc_b),
        "GTO-11: basis_set Arc must be shared after Mole::clone (zero-copy contract)"
    );
    // Strong count must be ≥ 2 (one in mol_a, one in mol_b).
    assert!(
        Arc::strong_count(arc_a) >= 2,
        "Arc strong_count = {} (expected ≥ 2 after clone)",
        Arc::strong_count(arc_a)
    );
}

#[test]
fn cintx_basis_returns_clone_with_same_ptr() {
    let mol = M(h_only_args()).expect("H/STO-3G build must succeed");
    let arc1 = mol.cintx_basis().unwrap();
    let arc2 = mol.cintx_basis().unwrap();
    assert!(
        Arc::ptr_eq(&arc1, &arc2),
        "cintx_basis() must return clones of the same Arc (no rebuild per call)"
    );
    // strong_count: one in mol.basis_set + arc1 + arc2 = 3.
    assert_eq!(Arc::strong_count(&arc1), 3, "expected 3 Arc handles total");
}

#[test]
fn basis_set_is_some_after_build() {
    let mol = M(h_only_args()).expect("H/STO-3G build must succeed");
    assert!(mol.basis_set.is_some(), "build must populate mol.basis_set");
    assert!(mol._built);
}

#[test]
fn mole_default_basis_set_is_none() {
    let mol = pyscf_core::Mole::default();
    assert!(
        mol.basis_set.is_none(),
        "Mole::default() must leave basis_set as None"
    );
    assert!(!mol._built, "Mole::default() must leave _built=false");
}

#[test]
fn cintx_basis_errors_when_not_built() {
    let mol = pyscf_core::Mole::default();
    match mol.cintx_basis() {
        Err(pyscf_core::PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(msg))) => {
            assert!(
                msg.contains("not built"),
                "expected error mentioning 'not built', got: {msg}"
            );
        }
        other => panic!("expected InvalidMolecule error, got {:?}", other.err()),
    }
}

#[test]
fn basis_set_accessor_borrowed_handle() {
    // The `basis_set()` method returns &Arc<BasisSet> without bumping the
    // refcount; `cintx_basis()` clones explicitly. This test exercises the
    // borrowed accessor for the consumer pattern that 02-05/02-06 will use.
    let mol = M(h2_sto3g_args()).expect("H2/STO-3G build must succeed");
    let borrowed = mol.basis_set().expect("built Mole must expose basis_set()");
    // Cloning explicitly through the borrow must produce the same pointer.
    let owned = borrowed.clone();
    assert!(Arc::ptr_eq(borrowed, &owned));
}
