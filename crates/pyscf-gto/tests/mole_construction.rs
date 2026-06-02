//! GTO-01 conformance tests for `pyscf_gto::M(...)` over the 4 atom-input
//! forms shipping in Phase 2 + the deferred 5th (callable) form.
//!
//! Source-of-truth for the expected behaviour: `pyscf/gto/mole.py:320-415`
//! `format_atom`. The plan 02-02 PLAN.md `<behavior>` block enumerates 9
//! test cases; this file covers all 9 + a bonus negative-electron-count
//! check for the `tot_electrons` signed math.

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};

fn h2_args(atom: AtomInput, unit: Unit) -> MoleBuildArgs {
    MoleBuildArgs {
        atom,
        basis: BasisInput::Name("sto-3g".into()),
        unit,
        ..Default::default()
    }
}

#[test]
fn h2_string_form_bohr() {
    let mol = M(h2_args(
        AtomInput::String("H 0 0 0; H 0 0 1.4".into()),
        Unit::Bohr,
    ))
    .unwrap();
    assert_eq!(mol.natm, 2);
    assert_eq!(mol._atom[0].0, "H");
    assert_eq!(mol._atom[0].1, [0.0, 0.0, 0.0]);
    assert_eq!(mol._atom[1].1, [0.0, 0.0, 1.4]);
    assert_eq!(mol.nelectron, 2);
    assert_eq!(mol.charge, 0);
}

#[test]
fn h2_string_form_angstrom_converts_to_bohr() {
    let mol = M(h2_args(
        AtomInput::String("H 0 0 0; H 0 0 1.4".into()),
        Unit::Ang,
    ))
    .unwrap();
    approx::assert_abs_diff_eq!(mol._atom[1].1[2], 1.4 * 1.8897261339213, epsilon = 1e-9);
}

#[test]
fn h2_tuples_form() {
    let atoms = vec![
        ("H".to_string(), [0.0, 0.0, 0.0]),
        ("H".to_string(), [0.0, 0.0, 1.4]),
    ];
    let mol = M(h2_args(AtomInput::Tuples(atoms), Unit::Bohr)).unwrap();
    assert_eq!(mol._atom[1].1, [0.0, 0.0, 1.4]);
    assert_eq!(mol.natm, 2);
}

#[test]
fn h2_tuple_vec_form() {
    let atoms = vec![
        ("H".to_string(), vec![0.0, 0.0, 0.0]),
        ("H".to_string(), vec![0.0, 0.0, 1.4]),
    ];
    let mol = M(h2_args(AtomInput::TupleVec(atoms), Unit::Bohr)).unwrap();
    assert_eq!(mol._atom[1].1, [0.0, 0.0, 1.4]);
}

#[test]
fn h2_file_form() {
    let dir = std::env::temp_dir();
    // Use a unique name per test invocation to avoid races with parallel
    // tests / leftover files from interrupted runs.
    let path = dir.join(format!("pyscf_rs_h2_test_{}.atoms", std::process::id()));
    std::fs::write(&path, "H 0 0 0\nH 0 0 1.4\n").unwrap();
    let mol = M(h2_args(AtomInput::FilePath(path.clone()), Unit::Bohr)).unwrap();
    assert_eq!(mol.natm, 2);
    assert_eq!(mol._atom[1].1[2], 1.4);
    let _ = std::fs::remove_file(&path);
}

/// Form 5 — callable: a closure returning a `String` spec must build the exact
/// same molecule as passing that `String` directly (congruence oracle).
#[test]
fn callable_form_returning_string_matches_direct() {
    let direct = M(h2_args(
        AtomInput::String("H 0 0 0; H 0 0 1.4".into()),
        Unit::Bohr,
    ))
    .unwrap();
    let via_callable = M(h2_args(
        AtomInput::callable(|| Ok(AtomInput::String("H 0 0 0; H 0 0 1.4".into()))),
        Unit::Bohr,
    ))
    .unwrap();
    assert_eq!(via_callable.natm, direct.natm);
    assert_eq!(via_callable.nelectron, direct.nelectron);
    assert_eq!(via_callable._atom, direct._atom);
}

/// The callable's produced form is resolved through the full `format_atom`
/// path, so unit conversion (Angstrom → Bohr) applies to its output.
#[test]
fn callable_form_honours_unit_conversion() {
    let mol = M(h2_args(
        AtomInput::callable(|| Ok(AtomInput::Tuples(vec![("H".into(), [0.0, 0.0, 1.4])]))),
        Unit::Ang,
    ))
    .unwrap();
    approx::assert_abs_diff_eq!(mol._atom[0].1[2], 1.4 * 1.8897261339213, epsilon = 1e-9);
}

/// One-level recursion guard: a callable that returns another callable is
/// rejected (not silently re-entered or infinitely recursed).
#[test]
fn callable_returning_callable_is_rejected() {
    let r = M(h2_args(
        AtomInput::callable(|| {
            Ok(AtomInput::callable(|| {
                Ok(AtomInput::String("H 0 0 0".into()))
            }))
        }),
        Unit::Bohr,
    ));
    match r {
        Err(pyscf_core::PyscfRsError::Core(_)) => {}
        _ => panic!("expected Core error for nested callable, got {:?}", r),
    }
}

/// An error returned by the closure propagates out of `format_atom` unchanged.
#[test]
fn callable_error_propagates() {
    let r = M(h2_args(
        AtomInput::callable(|| {
            Err(pyscf_core::PyscfRsError::Core(
                pyscf_core::CoreError::InvalidMolecule("boom".into()),
            ))
        }),
        Unit::Bohr,
    ));
    match r {
        Err(pyscf_core::PyscfRsError::Core(e)) => {
            assert!(format!("{e}").contains("boom"), "got {e}");
        }
        _ => panic!("expected propagated Core error, got {:?}", r),
    }
}

#[test]
fn separator_and_comment_handling() {
    // ";" and "\n" both work; "#" introduces a comment.
    let s = "H 0 0 0  # first H\n; O 0 0 1; H 0 1 0";
    let mol = M(h2_args(AtomInput::String(s.into()), Unit::Bohr)).unwrap();
    assert_eq!(mol.natm, 3);
    assert_eq!(mol._atom[1].0, "O");
    assert_eq!(mol._atom[2].0, "H");
    assert_eq!(mol.nelectron, 1 + 8 + 1); // 2 H + 1 O = 10 e
}

#[test]
fn ghost_suffix_preserved() {
    let s = "H1 0 0 0; H2 0 0 1.4";
    let mol = M(h2_args(AtomInput::String(s.into()), Unit::Bohr)).unwrap();
    assert_eq!(mol._atom[0].0, "H1");
    assert_eq!(mol._atom[1].0, "H2");
    assert_eq!(mol.nelectron, 2); // both H1/H2 are H atoms; charge = 1 each
}

#[test]
fn negative_electron_count_errs() {
    let mut args = h2_args(AtomInput::String("H 0 0 0".into()), Unit::Bohr);
    args.charge = 5; // 1 e atom with +5 charge → -4 electrons
    let r = M(args);
    assert!(matches!(r, Err(pyscf_core::PyscfRsError::Core(_))));
}
