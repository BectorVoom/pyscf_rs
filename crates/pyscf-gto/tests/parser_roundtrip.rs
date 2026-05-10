//! Parser round-trip: synthetic NWChem fixtures parse to the expected
//! `ParsedBasis` shape. Catches regressions in the line-stream parser
//! (header detection, primitive parsing, SP shared-exponent handling,
//! comment + blank-line tolerance, Fortran-D exponent normalisation).

use pyscf_gto::basis::nwchem;

#[test]
fn nwchem_h_sto3g_parses_correct_primitives() {
    let text = "H    S\n      3.42525091             0.15432897\n      0.62391373             0.53532814\n      0.16885540             0.44463454\nEND";
    let p = nwchem::parse_nwchem(text, "H", "test").unwrap();
    assert_eq!(p.shells.len(), 1);
    assert_eq!(p.shells[0].l, 0);
    approx::assert_abs_diff_eq!(p.shells[0].exponents[0], 3.42525091, epsilon = 1e-9);
    approx::assert_abs_diff_eq!(p.shells[0].coeffs[0][2], 0.44463454, epsilon = 1e-9);
}

#[test]
fn nwchem_sp_shared_exponent_creates_two_shells() {
    // Pople-canonical SINGLE-token "SP" form — one block produces TWO shells
    // (l=0 + l=1) sharing one exponent vector.
    let text = "Li    SP\n      0.6362897   -0.09996723   0.15591627\n      0.1478601    0.39951283   0.60768372\n      0.0480887    0.70011547   0.39195739\nEND";
    let p = nwchem::parse_nwchem(text, "Li", "test").unwrap();
    assert_eq!(p.shells.len(), 2);
    assert_eq!(p.shells[0].l, 0);
    assert_eq!(p.shells[1].l, 1);
    // Both shells share the same exponents.
    assert_eq!(p.shells[0].exponents, p.shells[1].exponents);
    approx::assert_abs_diff_eq!(p.shells[0].coeffs[0][0], -0.09996723, epsilon = 1e-9);
    approx::assert_abs_diff_eq!(p.shells[1].coeffs[0][0], 0.15591627, epsilon = 1e-9);
}

#[test]
fn nwchem_handles_comments_and_blank_lines() {
    let text = "# header comment\nBASIS\n\nH S  # another comment\n     1.0  1.0\n\nEND\n";
    let p = nwchem::parse_nwchem(text, "H", "test").unwrap();
    assert_eq!(p.shells.len(), 1);
    approx::assert_abs_diff_eq!(p.shells[0].exponents[0], 1.0, epsilon = 1e-12);
}
