//! `InitGuessMode::Chkfile(path)` integration test — plan 03-06 wires the
//! 5th init_guess mode that plan 03-03 left as NotYetImplemented.
//!
//! Strategy: dump an ScfResult to a chkfile (Rust ↔ Rust round-trip), then
//! load via `default_get_init_guess` and assert it returns a Density. Covers
//! the same-basis fast path, the cross-basis projection path (F-10 —
//! `project_mo_nr2nr`, reconstructing the prior Mole from the stored JSON),
//! and the graceful-error path when the prior Mole cannot be reconstructed.
//! The full SCF restart path is plan 03-08's ORACLE-08 territory.
use pyscf_core::{Energy, MOCoefficients, Mole, Unit};
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs, dumps};
use pyscf_scf::{InitGuessMode, ScfResult, chkfile::dump_scf_to_file, default_get_init_guess};

fn sample_result(nao: usize) -> ScfResult {
    let mut data = vec![0.0_f64; nao * nao];
    // Diagonal-dominant orthonormal-ish C: I as a placeholder.
    for i in 0..nao {
        data[i + i * nao] = 1.0;
    }
    let energies = vec![-1.0_f64; nao];
    let mut occs = vec![0.0_f64; nao];
    // Closed shell: fill first nao/2 orbitals with 2.
    for o in occs.iter_mut().take(nao / 2) {
        *o = 2.0;
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
    let mol = Mole {
        atom: "H 0 0 0; H 0 0 0.74".into(),
        basis: "sto-3g".into(),
        nelectron: 2,
        nao_nr: 2,
        natm: 2,
        _built: true, // bypass build() for the Phase-3 simple-case wire
        ..Default::default()
    };

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

fn h2(basis: &str) -> Mole {
    M(MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0; H 0 0 0.74".into()),
        basis: BasisInput::Name(basis.into()),
        unit: Unit::Ang,
        ..Default::default()
    })
    .expect("build H2")
}

#[test]
fn init_guess_by_chkfile_projects_across_basis() {
    // Prior SCF was in sto-3g (nao=2); the current run uses 6-31g (nao=4).
    // The cross-basis branch reconstructs the prior Mole from the chkfile JSON
    // and projects the prior MOs onto the current basis (project_mo_nr2nr).
    let prior_mol = h2("sto-3g");
    let cur_mol = h2("6-31g");
    assert_eq!(prior_mol.nao_nr, 2);
    assert!(cur_mol.nao_nr > prior_mol.nao_nr);

    let prior_json = dumps(&prior_mol).expect("serialize prior mol");
    let prior = sample_result(prior_mol.nao_nr); // nao=2, occ=[2,0]

    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    let path = tmp.path().to_path_buf();
    dump_scf_to_file(&path, &prior_json, &prior).expect("dump");

    let dens = default_get_init_guess(&cur_mol, &InitGuessMode::Chkfile(path))
        .expect("cross-basis chkfile projection must succeed");

    // Seed is shaped to the CURRENT basis, finite, and symmetric.
    assert_eq!(dens.nao, cur_mol.nao_nr);
    assert!(
        dens.data.iter().all(|x| x.is_finite()),
        "non-finite density"
    );
    let n = dens.nao;
    for i in 0..n {
        for j in 0..n {
            assert!(
                (dens.data[i * n + j] - dens.data[j * n + i]).abs() < 1e-10,
                "asymmetry at ({i},{j})"
            );
        }
    }
}

#[test]
fn init_guess_by_chkfile_cross_basis_errors_on_unreconstructable_prior() {
    // nao mismatch forces the projection path; an empty/invalid stored mol
    // JSON cannot be reconstructed, so the guess must error gracefully
    // (never panic, never silently return a wrong-shaped density).
    let mol = Mole {
        atom: "H 0 0 0".into(),
        basis: "cc-pvdz".into(),
        nelectron: 1,
        nao_nr: 5,
        _built: true,
        ..Default::default()
    };

    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    let path = tmp.path().to_path_buf();
    let prior = sample_result(2);
    dump_scf_to_file(&path, "{}", &prior).expect("dump");

    let result = default_get_init_guess(&mol, &InitGuessMode::Chkfile(path));
    assert!(
        result.is_err(),
        "expected graceful error when the prior mol cannot be reconstructed"
    );
}
