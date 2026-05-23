//! Smoke: `intor_with_auxmol` returns correctly-shaped `IntorOutput`.
//!
//! These arms assert SHAPE only. As of 05-08 (cintx#11 closed) `int3c2e_sph`
//! and `int2c2e_sph` evaluate for real; the numeric finite/non-zero/symmetry
//! gate lives in `int3c2e_auxmol.rs`, and bit-exact-vs-upstream-PySCF is the
//! CI-gated/human-verify arm.
//!
//! Source: plan 03-05 Task 0 (`<verify>` block).

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs, intor_with_auxmol};

fn h2_mole(basis: &str) -> pyscf_core::Mole {
    M(MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0; H 0 0 1.4".into()),
        basis: BasisInput::Name(basis.into()),
        unit: Unit::Bohr,
        ..Default::default()
    })
    .expect("build h2 mol")
}

#[test]
fn h2_sto3g_with_weigend_3c2e_shape() {
    let mol = h2_mole("sto-3g");
    // weigend (universal Ahlrichs aux) is mapped by upstream pyscf to
    // various aux files. For Phase 3 smoke we use any non-empty basis
    // file ALIAS resolves; sto-3g doubles as the auxiliary here since
    // the cintx int3c2e_sph base op is still a future cintx release.
    let auxmol = h2_mole("sto-3g");
    let out = intor_with_auxmol(&mol, "int3c2e_sph", &auxmol).expect("int3c2e_sph dispatch");
    assert_eq!(out.shape, vec![mol.nao_nr, mol.nao_nr, auxmol.nao_nr]);
    assert_eq!(out.values.len(), mol.nao_nr * mol.nao_nr * auxmol.nao_nr);
}

#[test]
fn h2_with_aux_2c2e_shape() {
    let mol = h2_mole("sto-3g");
    let auxmol = h2_mole("sto-3g");
    let out = intor_with_auxmol(&mol, "int2c2e_sph", &auxmol).expect("int2c2e_sph dispatch");
    assert_eq!(out.shape, vec![auxmol.nao_nr, auxmol.nao_nr]);
    assert_eq!(out.values.len(), auxmol.nao_nr * auxmol.nao_nr);
}

#[test]
fn unsupported_name_returns_not_yet_implemented() {
    let mol = h2_mole("sto-3g");
    let auxmol = h2_mole("sto-3g");
    let err = intor_with_auxmol(&mol, "int1e_ovlp_sph", &auxmol).expect_err("must reject");
    let msg = format!("{}", err);
    assert!(
        msg.contains("not yet implemented") || msg.contains("Phase 3"),
        "expected NotYetImplemented error, got: {}",
        msg
    );
}

#[test]
fn unbuilt_mol_errors_cleanly() {
    let auxmol = h2_mole("sto-3g");
    let unbuilt = pyscf_core::Mole::default();
    let err =
        intor_with_auxmol(&unbuilt, "int3c2e_sph", &auxmol).expect_err("unbuilt mol must error");
    let msg = format!("{}", err);
    assert!(
        msg.contains("not built") || msg.contains("invalid molecule"),
        "expected built check error, got: {}",
        msg
    );
}
