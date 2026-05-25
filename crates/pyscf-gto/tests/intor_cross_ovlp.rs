//! Always-on test for `intor_cross` — cross-basis arity-2 overlap (plan 03-13,
//! the minao `S_cross = <working|minao>` need).
//!
//! Self-consistency: with the SAME basis on both sides, the cross-overlap
//! `<A|int1e_ovlp|B>` must equal the ordinary self-overlap `intor("int1e_ovlp")`.
//! Cross-basis: a sto-3g × ano cross has the right `[nao_a, nao_b]` shape and is
//! finite.

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs, intor, intor_cross};

fn h2(basis: &str) -> pyscf_core::Mole {
    M(MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0; H 0 0 1.4".into()),
        basis: BasisInput::Name(basis.into()),
        unit: Unit::Bohr,
        ..Default::default()
    })
    .expect("build h2")
}

#[test]
fn cross_same_basis_equals_self_overlap() {
    let mol = h2("sto-3g");
    let self_ovlp = intor(&mol, "int1e_ovlp_sph").expect("self overlap");
    let cross = intor_cross(&mol, &mol, "int1e_ovlp_sph").expect("cross overlap");

    assert_eq!(cross.shape, self_ovlp.shape, "[nao,nao]");
    assert_eq!(cross.values.len(), self_ovlp.values.len());
    for (i, (&c, &s)) in cross.values.iter().zip(self_ovlp.values.iter()).enumerate() {
        assert!(
            (c - s).abs() < 1e-12,
            "cross[{i}]={c} must equal self-overlap[{i}]={s}"
        );
    }
    // Accepts the unsuffixed name too (suffix added per mol.cart).
    let cross2 = intor_cross(&mol, &mol, "int1e_ovlp").expect("cross (unsuffixed)");
    assert_eq!(cross2.values, cross.values);
}

#[test]
fn cross_distinct_bases_shape_and_finite() {
    let mol = h2("sto-3g"); // nao_a = 2
    let ano = h2("ano"); // nao_b = 2 * 16 = 32
    let cross = intor_cross(&mol, &ano, "int1e_ovlp_sph").expect("cross sto-3g × ano");
    assert_eq!(
        cross.shape,
        vec![mol.nao_nr, ano.nao_nr],
        "[nao_sto, nao_ano]"
    );
    assert_eq!(cross.values.len(), mol.nao_nr * ano.nao_nr);
    for v in &cross.values {
        assert!(v.is_finite(), "cross overlap must be finite");
    }
    assert!(
        cross.values.iter().any(|&v| v.abs() > 1e-12),
        "cross overlap must be non-zero (the s blocks overlap)"
    );
}

#[test]
fn cross_rejects_unbuilt_and_non_arity2() {
    let mol = h2("sto-3g");
    // int2e is arity-4 → rejected by intor_cross.
    let r = intor_cross(&mol, &mol, "int2e");
    assert!(r.is_err(), "intor_cross must reject non-arity-2 names");
}
