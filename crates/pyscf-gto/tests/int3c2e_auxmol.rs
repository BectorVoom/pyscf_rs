//! Always-on in-tree gate for the real `int3c2e_sph` orbital×aux dispatch
//! (05-08, cintx#11 DF half). cintx now ships `int3c2e_sph` as a base arity-3
//! operator (api_manifest.rs:404), libcint-byte-identical via its own
//! `center_3c2e_parity`/`safe_api_arity3_parity` suites.
//!
//! Replaces the previous all-zeros stub: `intor_with_auxmol(mol,"int3c2e_sph",
//! auxmol)` must now return a real, finite, non-zero `[nao,nao,naux]` F-order
//! tensor obeying the `(μν|P) = (νμ|P)` bra symmetry. Byte-identity vs upstream
//! PySCF is the CI-gated/human-verify arm (no numpy/PySCF in the sandbox).

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs, intor, intor_with_auxmol};

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
fn int3c2e_h2_real_finite_nonzero_bra_symmetric() {
    let mol = h2_mole("sto-3g");
    let auxmol = h2_mole("sto-3g");

    let nao = mol.nao_nr;
    let naux = auxmol.nao_nr;

    let out = intor_with_auxmol(&mol, "int3c2e_sph", &auxmol)
        .expect("int3c2e_sph must evaluate via the cintx orbital×aux path (zero-stub is closed)");

    // Shape + length contract.
    assert_eq!(
        out.shape,
        vec![nao, nao, naux],
        "int3c2e shape is [nao,nao,naux]"
    );
    assert_eq!(out.values.len(), nao * nao * naux, "len == nao*nao*naux");

    // All finite.
    for (i, v) in out.values.iter().enumerate() {
        assert!(v.is_finite(), "int3c2e value [{i}] = {v} must be finite");
    }

    // Non-zero — the regression guard against the old zero-fill stub.
    assert!(
        out.values.iter().any(|&v| v.abs() > 1e-12),
        "int3c2e must produce a non-zero tensor (zero-fill stub regression?)"
    );

    // Bra symmetry (μν|P) == (νμ|P). F-order [nao,nao,naux]:
    // (μ,ν,P) at μ + ν*nao + P*nao*nao.
    let at = |mu: usize, nu: usize, p: usize| out.values[mu + nu * nao + p * nao * nao];
    let tol = 1e-12;
    for p in 0..naux {
        for mu in 0..nao {
            for nu in 0..nao {
                assert!(
                    (at(mu, nu, p) - at(nu, mu, p)).abs() < tol,
                    "(μν|P) bra symmetry broken at (μ={mu},ν={nu},P={p}): {} vs {}",
                    at(mu, nu, p),
                    at(nu, mu, p)
                );
            }
        }
    }
}

#[test]
fn int3c2e_single_mol_equals_self_auxmol() {
    // F-04: `intor(mol, "int3c2e_sph")` (single-mol arity-3, all three centers
    // over mol's own basis) is the `auxmol == mol` specialisation of
    // `intor_with_auxmol`. With mol as its own auxmol, the byte-verified DF
    // wrapper produces an [nao,nao,naux=nao] tensor that must be bit-for-bit
    // identical to the single-mol path. This is the rigorous in-tree oracle:
    // the DF wrapper is libcint-byte-identical at the cintx source, so equality
    // here pins the single-mol path to the same numbers — no live PySCF needed.
    let mol = h2_mole("sto-3g");
    let nao = mol.nao_nr;

    let single = intor(&mol, "int3c2e_sph").expect("single-mol int3c2e_sph must evaluate (F-04)");

    // Shape contract: [nao, nao, nao].
    assert_eq!(
        single.shape,
        vec![nao, nao, nao],
        "single-mol int3c2e shape is [nao,nao,nao]"
    );
    assert_eq!(single.values.len(), nao * nao * nao);
    assert!(
        single.values.iter().all(|v| v.is_finite()),
        "single-mol int3c2e values must all be finite"
    );
    assert!(
        single.values.iter().any(|&v| v.abs() > 1e-12),
        "single-mol int3c2e must be non-zero"
    );

    // Oracle: identical to the byte-verified DF wrapper with auxmol == mol.
    let auxmol = h2_mole("sto-3g");
    let viaaux = intor_with_auxmol(&mol, "int3c2e_sph", &auxmol)
        .expect("int3c2e_sph via auxmol==mol must evaluate");
    assert_eq!(
        single.shape, viaaux.shape,
        "single-mol and auxmol==mol shapes must match"
    );
    for (idx, (a, b)) in single.values.iter().zip(viaaux.values.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-12,
            "single-mol int3c2e diverges from auxmol==mol at [{idx}]: {a} vs {b}"
        );
    }
}

#[test]
fn int2c2e_aux_metric_finite_symmetric() {
    // The DF metric (P|Q) is arity-2 (already shipped); assert it is finite and
    // symmetric so the cholesky_eri path it feeds is well-formed.
    let mol = h2_mole("sto-3g");
    let auxmol = h2_mole("sto-3g");
    let naux = auxmol.nao_nr;

    let out = intor_with_auxmol(&mol, "int2c2e_sph", &auxmol).expect("int2c2e_sph dispatch");
    assert_eq!(out.shape, vec![naux, naux]);

    for v in &out.values {
        assert!(v.is_finite(), "int2c2e (P|Q) must be finite");
    }
    assert!(
        out.values.iter().any(|&v| v.abs() > 1e-12),
        "int2c2e (P|Q) must be non-zero"
    );
    // (P|Q) == (Q|P).
    let at = |p: usize, q: usize| out.values[p + q * naux];
    for p in 0..naux {
        for q in 0..naux {
            assert!(
                (at(p, q) - at(q, p)).abs() < 1e-12,
                "(P|Q) metric symmetry broken at ({p},{q})"
            );
        }
    }
}
