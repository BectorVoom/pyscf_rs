//! F-03 task T1 verification — `Mole.nao_2c` (2-component spinor AO count).
//!
//! Oracle-free checks per the F-03 plan: an explicit hand count for H2/STO-3G
//! and the molecule-level relation `nao_2c == 2·nao_nr` for spherical bases
//! (spinors split each spherical AO into `j = l ± 1/2`, doubling the count).
//! Exercised across s-only, p- and d-containing bases so every angular
//! momentum present is walked through the `_bas` summation.

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};

fn build(atom: &str, basis: &str) -> pyscf_core::Mole {
    M(MoleBuildArgs {
        atom: AtomInput::String(atom.into()),
        basis: BasisInput::Name(basis.into()),
        unit: Unit::Bohr,
        ..Default::default()
    })
    .unwrap()
}

/// H2/STO-3G: two H atoms, one contracted s-shell each ⇒ `nao_nr = 2`,
/// `nao_2c = 2·(2·0+1)·1 per shell = 2 each ⇒ 4`.
#[test]
fn h2_sto3g_nao_2c_is_four() {
    let mol = build("H 0 0 0; H 0 0 1.4", "sto-3g");
    assert_eq!(mol.nao_nr, 2, "H2/STO-3G spherical AO count");
    assert_eq!(mol.nao_2c, 4, "H2/STO-3G 2-component AO count");
    assert_eq!(mol.nao_2c, 2 * mol.nao_nr);
}

/// `nao_2c == 2·nao_nr` must hold for any spherical basis, regardless of the
/// angular momenta present. Covers s-only, p-containing, and d-containing
/// bases (cc-pVDZ on C/O includes l = 2 shells).
#[test]
fn nao_2c_is_twice_nao_nr_for_spherical_bases() {
    let cases = [
        ("H 0 0 0; H 0 0 1.4", "sto-3g"),                 // s only
        ("O 0 0 0; H 0 1.4 1.1; H 0 -1.4 1.1", "sto-3g"), // s + p
        ("O 0 0 0; H 0 1.4 1.1; H 0 -1.4 1.1", "6-31g"),  // s + p, multi-contraction
        ("C 0 0 0; O 0 0 2.1", "cc-pvdz"),                // s + p + d
    ];
    for (atom, basis) in cases {
        let mol = build(atom, basis);
        assert!(mol.nao_nr > 0, "{basis}: nonempty basis");
        assert_eq!(
            mol.nao_2c,
            2 * mol.nao_nr,
            "{atom} / {basis}: nao_2c ({}) should equal 2·nao_nr ({})",
            mol.nao_2c,
            2 * mol.nao_nr
        );
    }
}
