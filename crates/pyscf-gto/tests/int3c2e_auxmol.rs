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
