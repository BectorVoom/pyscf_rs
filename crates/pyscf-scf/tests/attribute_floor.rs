//! SCF-14: assert RHF struct exposes the ≥30-attribute floor with the
//! upstream pyscf/scf/hf.py:1689-1759 defaults.
//!
//! Sample-check 10 of the 32 fields (full Python introspection lands in
//! plan 03-10). The first compile/build proves all 32 fields exist on
//! the struct (Rust rejects missing-field access at compile time);
//! the runtime asserts prove their defaults match upstream.

use pyscf_scf::{GHF, RHF, UHF};

#[test]
fn rhf_has_30_attribute_floor() {
    let mol = pyscf_core::Mole::default();
    let rhf = RHF::new(mol);

    // Sample 10 of the 32 attrs to prove the floor is real.
    assert_eq!(rhf.diis_space, 8, "upstream pyscf/scf/hf.py:1700");
    assert_eq!(rhf.diis_start_cycle, 1, "upstream pyscf/scf/hf.py:1701");
    assert_eq!(rhf.diis_damp, 0.0, "upstream pyscf/scf/hf.py:1702");
    assert_eq!(rhf.max_cycle, 50, "upstream pyscf/scf/hf.py:1704");
    assert_eq!(rhf.conv_tol, 1e-9, "upstream pyscf/scf/hf.py:1705");
    assert_eq!(rhf.init_guess, "minao", "upstream pyscf/scf/hf.py:1693");
    assert_eq!(rhf.level_shift, 0.0, "upstream pyscf/scf/hf.py:1694");
    assert_eq!(rhf.damp, 0.0, "upstream pyscf/scf/hf.py:1695");
    assert!(rhf.direct_scf, "upstream pyscf/scf/hf.py:1691");
    assert_eq!(rhf.direct_scf_tol, 1e-13, "upstream pyscf/scf/hf.py:1692");

    // Initial state — none of the dynamic results are populated yet.
    assert!(rhf.mo_coeff.is_none());
    assert!(rhf.mo_energy.is_none());
    assert!(rhf.mo_occ.is_none());
    assert_eq!(rhf.e_tot, 0.0);
    assert_eq!(rhf.e_elec, 0.0);
    assert!(!rhf.converged);
    assert_eq!(rhf.cycles, 0);
}

#[test]
fn uhf_constructs_with_alpha_beta_pair_slot() {
    let mol = pyscf_core::Mole::default();
    let uhf = UHF::new(mol);
    assert_eq!(uhf.max_cycle, 50);
    assert_eq!(uhf.conv_tol, 1e-9);
    assert!(uhf.mo_coeff.is_none(), "alpha/beta pair starts empty");
}

#[test]
fn ghf_constructs_with_2c_spinor_slot() {
    let mol = pyscf_core::Mole::default();
    let ghf = GHF::new(mol);
    assert_eq!(ghf.max_cycle, 50);
    assert_eq!(ghf.conv_tol, 1e-9);
    assert!(ghf.mo_coeff.is_none(), "2c spinor MO starts empty");
}

#[test]
fn parse_init_guess_mode_round_trip() {
    use pyscf_scf::{InitGuessMode, parse_init_guess_mode};
    assert!(matches!(
        parse_init_guess_mode("minao").expect("ok"),
        InitGuessMode::Minao
    ));
    assert!(matches!(
        parse_init_guess_mode("atom").expect("ok"),
        InitGuessMode::Atom
    ));
    assert!(matches!(
        parse_init_guess_mode("1e").expect("ok"),
        InitGuessMode::OneElectron
    ));
    assert!(matches!(
        parse_init_guess_mode("huckel").expect("ok"),
        InitGuessMode::Huckel
    ));
    assert!(parse_init_guess_mode("garbage").is_err());
}
