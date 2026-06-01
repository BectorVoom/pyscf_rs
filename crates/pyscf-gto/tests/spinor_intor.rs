//! F-03 — `intor_spinor` integration tests (route-through-cintx architecture).
//!
//! In-sandbox verification of the complex one-electron spinor integrals
//! (`int1e_{ovlp,kin,nuc}_spinor`):
//!   - shape `[n2c, n2c]` with `n2c == mol.nao_2c == 2·nao_nr`,
//!   - buffer length and finiteness,
//!   - Hermiticity `S = S†` (re symmetric, im antisymmetric),
//!   - non-trivial (nonzero) output.
//!
//! The per-shell-pair cart→spinor numerics are cintx's, validated against
//! vendor libcint at atol 1e-12 (`cintx-oracle/tests/
//! one_electron_scalar_spinor_parity.rs`). The remaining gap — global
//! shell-block ordering vs upstream PySCF `mol.intor("int1e_*_spinor")` —
//! needs the live-PySCF environment and is captured by the `#[ignore]`d
//! byte-identity placeholder at the bottom (shared blocker with F-14).

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, IntorOutputComplex, M, MoleBuildArgs, intor_spinor};

fn build(atom: &str, basis: &str) -> pyscf_core::Mole {
    M(MoleBuildArgs {
        atom: AtomInput::String(atom.into()),
        basis: BasisInput::Name(basis.into()),
        unit: Unit::Bohr,
        ..Default::default()
    })
    .unwrap()
}

/// `S[a,b]` from the F-order complex buffer: `re[a + b*n2c] + i·im[...]`.
fn at(out: &IntorOutputComplex, a: usize, b: usize, n2c: usize) -> (f64, f64) {
    let k = a + b * n2c;
    (out.re[k], out.im[k])
}

fn assert_shape_and_finite(out: &IntorOutputComplex, mol: &pyscf_core::Mole, op: &str) {
    let n2c = mol.nao_2c;
    assert!(n2c > 0, "{op}: n2c should be positive");
    assert_eq!(n2c, 2 * mol.nao_nr, "{op}: n2c == 2·nao_nr");
    assert_eq!(out.shape, vec![n2c, n2c], "{op}: shape [n2c, n2c]");
    assert_eq!(out.re.len(), n2c * n2c, "{op}: re buffer length");
    assert_eq!(out.im.len(), n2c * n2c, "{op}: im buffer length");
    assert!(
        out.re.iter().chain(out.im.iter()).all(|v| v.is_finite()),
        "{op}: all elements finite"
    );
}

/// Hermitian: re symmetric, im antisymmetric (so diagonal im == 0).
fn assert_hermitian(out: &IntorOutputComplex, n2c: usize, op: &str) {
    const ATOL: f64 = 1e-10;
    for a in 0..n2c {
        for b in 0..n2c {
            let (re_ab, im_ab) = at(out, a, b, n2c);
            let (re_ba, im_ba) = at(out, b, a, n2c);
            assert!(
                (re_ab - re_ba).abs() < ATOL,
                "{op}: re not symmetric at ({a},{b}): {re_ab} vs {re_ba}"
            );
            assert!(
                (im_ab + im_ba).abs() < ATOL,
                "{op}: im not antisymmetric at ({a},{b}): {im_ab} vs {im_ba}"
            );
        }
    }
}

fn assert_nonzero(out: &IntorOutputComplex, op: &str) {
    let nz = out
        .re
        .iter()
        .zip(out.im.iter())
        .filter(|(r, i)| r.abs() > 1e-12 || i.abs() > 1e-12)
        .count();
    assert!(
        nz > 0,
        "{op}: expected at least one nonzero complex element"
    );
}

#[test]
fn h2_sto3g_ovlp_spinor_shape_and_hermitian() {
    let mol = build("H 0 0 0; H 0 0 1.4", "sto-3g");
    let out = intor_spinor(&mol, "int1e_ovlp_spinor").unwrap();
    // H2/STO-3G: nao_nr = 2 → n2c = 4.
    assert_eq!(mol.nao_2c, 4);
    assert_shape_and_finite(&out, &mol, "ovlp");
    assert_hermitian(&out, mol.nao_2c, "ovlp");
    assert_nonzero(&out, "ovlp");
}

#[test]
fn h2o_sto3g_one_electron_spinor_families() {
    let mol = build("O 0 0 0; H 0 1.4 1.1; H 0 -1.4 1.1", "sto-3g");
    for op in ["int1e_ovlp_spinor", "int1e_kin_spinor", "int1e_nuc_spinor"] {
        let out = intor_spinor(&mol, op).unwrap();
        assert_shape_and_finite(&out, &mol, op);
        assert_hermitian(&out, mol.nao_2c, op);
        assert_nonzero(&out, op);
    }
}

#[test]
fn ovlp_spinor_diagonal_is_real_positive() {
    // The spinor overlap diagonal <i|i> must be real (im == 0) and positive.
    let mol = build("O 0 0 0; H 0 1.4 1.1; H 0 -1.4 1.1", "sto-3g");
    let out = intor_spinor(&mol, "int1e_ovlp_spinor").unwrap();
    let n2c = mol.nao_2c;
    for a in 0..n2c {
        let (re_aa, im_aa) = at(&out, a, a, n2c);
        assert!(im_aa.abs() < 1e-12, "ovlp diagonal im at {a} = {im_aa}");
        assert!(
            re_aa > 0.0,
            "ovlp diagonal re at {a} = {re_aa} (should be > 0)"
        );
    }
}

#[test]
fn name_normalisation_accepts_bare_operator() {
    // "int1e_ovlp" (no suffix) must route to the spinor path identically.
    let mol = build("H 0 0 0; H 0 0 1.4", "sto-3g");
    let a = intor_spinor(&mol, "int1e_ovlp").unwrap();
    let b = intor_spinor(&mol, "int1e_ovlp_spinor").unwrap();
    assert_eq!(a.re, b.re);
    assert_eq!(a.im, b.im);
}

#[test]
fn int2e_spinor_is_deferred() {
    // Two-electron spinor ERIs are F-03 T6 (deferred) — must error cleanly,
    // not panic or return garbage.
    let mol = build("H 0 0 0; H 0 0 1.4", "sto-3g");
    let r = intor_spinor(&mol, "int2e_spinor");
    assert!(
        r.is_err(),
        "int2e_spinor should be a clean error (deferred)"
    );
}

/// CONTRACT (live-PySCF gate, shared blocker with F-14): the assembled global
/// matrix must byte-match upstream `mol.intor("int1e_ovlp_spinor")` to
/// atol 1e-10. Per-pair numerics are cintx-vendor-validated; this checks the
/// global shell-block ORDERING, which cannot be verified without live PySCF.
#[test]
#[ignore = "F-03: global-ordering byte-identity needs live PySCF (maturin) — shared F-14 blocker"]
fn ovlp_spinor_byte_matches_upstream() {
    unimplemented!("live-PySCF oracle harness pending — see F-03 plan §6");
}
