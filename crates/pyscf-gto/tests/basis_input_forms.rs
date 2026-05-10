//! GTO-02 acceptance — the 11 upstream basis-input forms collapse to the
//! 5-arm `BasisInput` match dispatch and produce a per-element-symbol
//! `ParsedBasis` map honouring first-occurrence order (Pitfall 4).

use pyscf_core::{ParsedAtom, Unit};
use pyscf_gto::{format_basis, AtomInput, BasisInput, MoleBuildArgs, M};
use std::collections::HashMap;

fn h2o_atoms() -> Vec<ParsedAtom> {
    vec![
        ("O".to_string(), [0.0, 0.0, 0.0]),
        ("H".to_string(), [0.0, 0.7, 0.6]),
        ("H".to_string(), [0.0, -0.7, 0.6]),
    ]
}

#[test]
fn name_form_resolves_sto3g() {
    let parsed = format_basis(&BasisInput::Name("sto-3g".into()), &h2o_atoms()).unwrap();
    assert!(parsed.contains_key("O"));
    assert!(parsed.contains_key("H"));
    assert_eq!(parsed.len(), 2); // first-occurrence collapse (Pitfall 4)
    // O has multiple shells in STO-3G (1s + 2sp shared-exponent → 1s + 1s + 1p).
    assert!(
        parsed["O"].shells.len() >= 2,
        "O STO-3G has ≥ 2 shells, got {}",
        parsed["O"].shells.len()
    );
    assert_eq!(parsed["H"].shells.len(), 1, "H STO-3G has exactly 1 (1s) shell");
}

#[test]
fn name_form_resolves_ccpvdz() {
    let parsed = format_basis(&BasisInput::Name("cc-pvdz".into()), &h2o_atoms()).unwrap();
    assert!(parsed.contains_key("O"));
    // O cc-pVDZ has multiple shells (3s + 2p + 1d after segmenting).
    assert!(
        parsed["O"].shells.len() >= 4,
        "O cc-pVDZ has ≥ 4 shells, got {}",
        parsed["O"].shells.len()
    );
}

#[test]
fn per_element_form() {
    let mut m = HashMap::new();
    m.insert("O".to_string(), BasisInput::Name("cc-pvdz".into()));
    m.insert("H".to_string(), BasisInput::Name("sto-3g".into()));
    let parsed = format_basis(&BasisInput::PerElement(m), &h2o_atoms()).unwrap();
    let h_shells = parsed["H"].shells.len();
    let o_shells = parsed["O"].shells.len();
    assert_eq!(h_shells, 1, "STO-3G H = 1 shell");
    assert!(o_shells >= 4, "cc-pVDZ O ≥ 4 shells");
}

#[test]
fn per_element_default_fallback() {
    // Caller supplies "default" key; symbols not in the map fall back to it.
    let mut m = HashMap::new();
    m.insert("default".to_string(), BasisInput::Name("sto-3g".into()));
    m.insert("O".to_string(), BasisInput::Name("cc-pvdz".into()));
    let parsed = format_basis(&BasisInput::PerElement(m), &h2o_atoms()).unwrap();
    // H → default → STO-3G (1 shell); O → cc-pVDZ (≥ 4 shells).
    assert_eq!(parsed["H"].shells.len(), 1);
    assert!(parsed["O"].shells.len() >= 4);
}

#[test]
fn nwchem_text_form() {
    let text = "H    S\n      3.42525091  0.15432897\n      0.62391373  0.53532814\n      0.16885540  0.44463454\n";
    let parsed = format_basis(
        &BasisInput::NwchemText(text.into()),
        &[("H".into(), [0.0, 0.0, 0.0])],
    )
    .unwrap();
    assert_eq!(parsed["H"].shells.len(), 1);
    assert_eq!(parsed["H"].shells[0].l, 0);
    assert_eq!(parsed["H"].shells[0].exponents.len(), 3);
    approx::assert_abs_diff_eq!(parsed["H"].shells[0].exponents[0], 3.42525091, epsilon = 1e-9);
}

#[test]
fn first_occurrence_order_for_unique_symbols() {
    let atoms = vec![
        ("H".to_string(), [0.0, 0.0, 0.0]),
        ("O".to_string(), [0.0, 0.0, 1.0]),
        ("H".to_string(), [0.0, 0.0, 2.0]),
        ("O".to_string(), [0.0, 0.0, 3.0]),
    ];
    let parsed = format_basis(&BasisInput::Name("sto-3g".into()), &atoms).unwrap();
    // 2 unique symbols, regardless of HashMap iteration order.
    assert_eq!(parsed.len(), 2);
    assert!(parsed.contains_key("H"));
    assert!(parsed.contains_key("O"));
}

#[test]
fn unknown_basis_name_errors() {
    let r = format_basis(
        &BasisInput::Name("totally-fake-basis".into()),
        &h2o_atoms(),
    );
    assert!(matches!(
        r,
        Err(pyscf_core::PyscfRsError::BasisLoad(
            pyscf_core::BasisLoadError::UnknownName { .. }
        ))
    ));
}

#[test]
#[allow(non_snake_case)]
fn full_pipeline_h2_sto3g_via_M() {
    // End-to-end: M(...) populates mol._basis correctly.
    let mol = M(MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0; H 0 0 1.4".into()),
        basis: BasisInput::Name("sto-3g".into()),
        unit: Unit::Bohr,
        max_memory: 4000.0,
        axes: [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        ..Default::default()
    })
    .unwrap();
    assert_eq!(mol._basis.len(), 1, "only 1 unique symbol 'H'");
    assert_eq!(mol._basis["H"].shells.len(), 1);
    assert_eq!(mol._basis["H"].shells[0].l, 0);
}

#[test]
fn parsed_form_passthrough() {
    use pyscf_core::{ParsedBasis, ShellSpec};
    let synthetic = ParsedBasis {
        shells: vec![ShellSpec {
            l: 0,
            exponents: vec![1.0],
            coeffs: vec![vec![1.0]],
        }],
    };
    let parsed = format_basis(
        &BasisInput::Parsed(synthetic.clone()),
        &[("H".into(), [0.0, 0.0, 0.0])],
    )
    .unwrap();
    assert_eq!(parsed["H"].shells.len(), 1);
    approx::assert_abs_diff_eq!(parsed["H"].shells[0].exponents[0], 1.0, epsilon = 1e-12);
}
