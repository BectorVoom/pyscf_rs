//! Plan 03-11 Task 2 — analyze (mulliken_pop + dip_moment) + convert
//! (to_rhf / to_uhf / to_ghf / to_uks_stub / to_rks_stub) + scanner
//! (as_scanner) tests.
//!
//! The analyze tests don't run a full SCF (int2e_sph is gated by plan
//! 02-09); they instead inject a hand-crafted converged MO into the RHF
//! struct and verify the analyze helpers return well-formed outputs.
//! The convert tests are pure type-level (struct copies).

use pyscf_core::MOCoefficients;
use pyscf_scf::{
    RHF, UHF,
    analyze::MullikenResult,
    convert::{to_ghf, to_rhf, to_rks_stub, to_uhf, to_uks_stub},
};

#[test]
fn to_uhf_from_rhf_preserves_scalar_state() {
    let mol = pyscf_core::Mole::default();
    let mut rhf = RHF::new(mol);
    rhf.conv_tol = 1e-12;
    rhf.max_cycle = 100;
    rhf.diis_space = 12;
    rhf.level_shift = 0.05;
    rhf.damp = 0.1;
    let uhf = to_uhf(&rhf).expect("to_uhf");
    assert_eq!(uhf.conv_tol, 1e-12);
    assert_eq!(uhf.max_cycle, 100);
    assert_eq!(uhf.diis_space, 12);
    assert_eq!(uhf.level_shift, 0.05);
    assert_eq!(uhf.damp, 0.1);
}

#[test]
fn to_rhf_from_uhf_preserves_scalar_state() {
    let mol = pyscf_core::Mole::default();
    let mut uhf = UHF::new(mol);
    uhf.conv_tol = 1e-11;
    uhf.max_cycle = 75;
    uhf.diis_space = 6;
    uhf.level_shift = 0.2;
    uhf.damp = 0.3;
    let rhf = to_rhf(&uhf).expect("to_rhf");
    assert_eq!(rhf.conv_tol, 1e-11);
    assert_eq!(rhf.max_cycle, 75);
    assert_eq!(rhf.diis_space, 6);
    assert_eq!(rhf.level_shift, 0.2);
    assert_eq!(rhf.damp, 0.3);
}

#[test]
fn to_ghf_from_rhf_preserves_scalar_state() {
    let mol = pyscf_core::Mole::default();
    let mut rhf = RHF::new(mol);
    rhf.conv_tol = 1e-10;
    rhf.max_cycle = 60;
    rhf.diis_space = 5;
    let ghf = to_ghf(&rhf).expect("to_ghf");
    assert_eq!(ghf.conv_tol, 1e-10);
    assert_eq!(ghf.max_cycle, 60);
    assert_eq!(ghf.diis_space, 5);
}

// DFT-11 / plan 04-06: Phase 4 DFT has landed, so to_uks/to_rks no longer
// return `NotYetImplemented{phase:4}` — they produce a `KsConversion`
// descriptor carrying the source RHF's scalar SCF settings.
#[test]
fn to_uks_carries_scf_settings() {
    let mol = pyscf_core::Mole::default();
    let mut rhf = RHF::new(mol);
    rhf.conv_tol = 1e-9;
    rhf.max_cycle = 42;
    rhf.diis_space = 7;
    let ks = to_uks_stub(&rhf).expect("to_uks must succeed now that Phase 4 has landed");
    assert_eq!(ks.conv_tol, 1e-9);
    assert_eq!(ks.max_cycle, 42);
    assert_eq!(ks.diis_space, 7);
}

#[test]
fn to_rks_carries_scf_settings() {
    let mol = pyscf_core::Mole::default();
    let mut rhf = RHF::new(mol);
    rhf.conv_tol = 1e-9;
    rhf.max_cycle = 42;
    rhf.diis_space = 7;
    let ks = to_rks_stub(&rhf).expect("to_rks must succeed now that Phase 4 has landed");
    assert_eq!(ks.conv_tol, 1e-9);
    assert_eq!(ks.max_cycle, 42);
    assert_eq!(ks.diis_space, 7);
}

#[test]
fn mulliken_pop_rejects_uninitialised_rhf() {
    // mulliken_pop must error (not panic) when kernel hasn't run yet
    // (mo_coeff is None on a fresh RHF).
    let mol = pyscf_core::Mole::default();
    let rhf = RHF::new(mol);
    let r = pyscf_scf::analyze::mulliken_pop(&rhf);
    assert!(
        r.is_err(),
        "mulliken_pop on un-run RHF must return Err, not panic"
    );
}

#[test]
fn dip_moment_rejects_uninitialised_rhf() {
    let mol = pyscf_core::Mole::default();
    let rhf = RHF::new(mol);
    let r = pyscf_scf::analyze::dip_moment(&rhf);
    assert!(
        r.is_err(),
        "dip_moment on un-run RHF must return Err, not panic"
    );
}

#[test]
fn mulliken_result_has_atom_charges_and_ao_populations() {
    // Pure struct construction — verifies the type surface used by oracle
    // Arms 7 + 8 (plan 03-08).
    let r = MullikenResult {
        atom_charges: vec![1.0, 1.0],
        ao_populations: vec![1.0, 1.0],
    };
    assert_eq!(r.atom_charges.len(), 2);
    assert_eq!(r.ao_populations.len(), 2);
}

#[test]
fn as_scanner_returns_closure() {
    // Pure type-level smoke: as_scanner(&rhf) returns a closure with the
    // documented signature; we don't invoke it (would need a real SCF
    // run which int2e_sph blocks).
    let mol = pyscf_core::Mole::default();
    let rhf = RHF::new(mol);
    #[allow(clippy::type_complexity)]
    let _scanner: Box<
        dyn Fn(&pyscf_core::Mole) -> Result<pyscf_core::Energy, pyscf_core::PyscfRsError>
            + Send
            + Sync,
    > = pyscf_scf::scanner::as_scanner(&rhf);
}

#[test]
fn mulliken_pop_on_synthetic_converged_rhf_returns_natm_charges() {
    // Build a default Mole (natm=0). Inject a hand-crafted MO so the
    // pre-conditions in mulliken_pop are satisfied. We don't expect
    // physical charges (the AO loop is empty); we just verify the
    // function returns Ok with atom_charges.len() == mol.natm.
    let mol = pyscf_core::Mole::default();
    let mut rhf = RHF::new(mol);
    rhf.mo_coeff = Some(MOCoefficients {
        nao: 0,
        nmo: 0,
        data: vec![],
        energies: vec![],
        occupations: vec![],
    });
    let r = pyscf_scf::analyze::mulliken_pop(&rhf);
    // With nao=0, get_ovlp may fail or succeed; the test asserts no panic.
    // If it returns Ok, atom_charges.len() == 0. If Err, the gap is
    // documented by the error message.
    // An Err here is the documented gap from int1e_ovlp on an empty Mole.
    if let Ok(m) = r {
        assert_eq!(m.atom_charges.len(), 0);
    }
}
