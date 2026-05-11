//! `InitGuessMode::Chkfile(path)` integration test — plan 03-06 wires the
//! 5th init_guess mode that plan 03-03 left as NotYetImplemented.
//!
//! Strategy: dump an ScfResult to a chkfile (Rust ↔ Rust round-trip),
//! then load via `default_get_init_guess` and assert it returns a Density
//! (not a NotYetImplemented error). The full SCF restart path is plan
//! 03-08's ORACLE-08 territory.
use pyscf_core::{Energy, MOCoefficients, Mole};
use pyscf_scf::{
    chkfile::dump_scf_to_file, default_get_init_guess, InitGuessMode, ScfResult,
};

fn sample_result(nao: usize) -> ScfResult {
    let mut data = vec![0.0_f64; nao * nao];
    // Diagonal-dominant orthonormal-ish C: I as a placeholder.
    for i in 0..nao {
        data[i + i * nao] = 1.0;
    }
    let energies = vec![-1.0_f64; nao];
    let mut occs = vec![0.0_f64; nao];
    // Closed shell: fill first nao/2 orbitals with 2.
    for i in 0..(nao / 2) {
        occs[i] = 2.0;
    }
    ScfResult {
        e_tot: Energy(-1.0),
        mo_coeff: MOCoefficients {
            nao,
            nmo: nao,
            data,
            energies: energies.clone(),
            occupations: occs.clone(),
        },
        mo_energy: energies,
        mo_occ: occs,
        converged: true,
        cycles: 1,
    }
}

#[test]
fn init_guess_by_chkfile_reads_prior_density() {
    // Use H2/sto-3g (nao=2, nelec=2).
    let mut mol = Mole::default();
    mol.atom = "H 0 0 0; H 0 0 0.74".into();
    mol.basis = "sto-3g".into();
    mol.nelectron = 2;
    mol.nao_nr = 2;
    mol.natm = 2;
    mol._built = true; // bypass build() for the Phase-3 simple-case wire

    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    let path = tmp.path().to_path_buf();
    let prior = sample_result(2);
    dump_scf_to_file(&path, r#"{"atom":"H 0 0 0; H 0 0 0.74"}"#, &prior).expect("dump");

    let mode = InitGuessMode::Chkfile(path.clone());
    let dens = default_get_init_guess(&mol, &mode)
        .expect("InitGuessMode::Chkfile must succeed with same-basis prior");
    assert_eq!(dens.nao, 2);
    // D[mu,nu] = sum_i occ_i * C[mu,i] * C[nu,i].
    // With C = I and occ = [2, 0]: D[0,0] = 2, D[1,1] = 0, D[0,1] = D[1,0] = 0.
    assert!((dens.data[0] - 2.0).abs() < 1e-12, "D[0,0] expected 2.0");
    assert!(dens.data[1].abs() < 1e-12, "D[0,1] expected 0.0");
    assert!(dens.data[2].abs() < 1e-12, "D[1,0] expected 0.0");
    assert!(dens.data[3].abs() < 1e-12, "D[1,1] expected 0.0");
}

#[test]
fn init_guess_by_chkfile_rejects_nao_mismatch() {
    let mut mol = Mole::default();
    mol.atom = "H 0 0 0".into();
    mol.basis = "cc-pvdz".into();
    mol.nelectron = 1;
    mol.nao_nr = 5; // current basis is bigger than prior chkfile
    mol._built = true;

    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    let path = tmp.path().to_path_buf();
    let prior = sample_result(2); // prior had nao=2
    dump_scf_to_file(&path, "{}", &prior).expect("dump");

    let mode = InitGuessMode::Chkfile(path);
    let result = default_get_init_guess(&mol, &mode);
    // Phase 3 ships the simple same-basis case; basis-projection is deferred.
    assert!(
        result.is_err(),
        "expected error on nao mismatch (basis projection deferred)"
    );
}
