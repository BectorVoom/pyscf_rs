//! Always-on in-tree gate for the arity-4 `int2e` dispatch (05-08, cintx#11
//! numeric closure). cintx now ships `int2e_sph` as a base arity-4 operator,
//! libcint-byte-identical via its own oracle parity suite
//! (`safe_api_arity4_parity.rs`, `two_electron_parity.rs`).
//!
//! This test asserts the pyscf-gto DISPATCH + LAYOUT contract that consumes it:
//!   - shape `[nao, nao, nao, nao]`, F-order, all finite, non-zero
//!   - 8-fold permutational symmetry of the real chemist ERI `(ij|kl)`
//!   - a positive self-repulsion diagonal `(00|00) > 0`
//!
//! Byte-identity vs upstream PySCF cannot run in this sandbox (no numpy/PySCF);
//! cintx pins libcint byte-identity at source, and the upstream oracle is the
//! CI-gated/human-verify arm (02-10 precedent).

use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs, intor};

/// F-order flat index for `(i,j,k,l)` in a `[nao,nao,nao,nao]` tensor:
/// `i + j*nao + k*nao^2 + l*nao^3` (matches `pyscf_ao2mo::transform`'s
/// documented `eri_ao` convention; first index fastest).
fn idx(i: usize, j: usize, k: usize, l: usize, nao: usize) -> usize {
    i + j * nao + k * nao * nao + l * nao * nao * nao
}

#[test]
fn int2e_h2_sto3g_real_finite_nonzero_8fold_symmetric() {
    let mol = M(MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0; H 0 0 0.74".into()),
        basis: BasisInput::Name("sto-3g".into()),
        ..Default::default()
    })
    .expect("build H2/STO-3G");

    let nao = mol.nao_nr;
    assert_eq!(nao, 2, "H2/STO-3G has 2 spherical AOs");

    let out = intor(&mol, "int2e").expect(
        "int2e must evaluate via the cintx arity-4 path (NotYetImplemented{phase:2} is closed)",
    );

    // Shape + length contract.
    assert_eq!(
        out.shape,
        vec![nao, nao, nao, nao],
        "int2e shape is [nao;4]"
    );
    assert_eq!(out.values.len(), nao * nao * nao * nao, "len == nao^4 = 16");

    // All finite.
    for (i, v) in out.values.iter().enumerate() {
        assert!(v.is_finite(), "int2e value [{i}] = {v} must be finite");
    }

    // Non-zero (regression guard against a zero-fill stub).
    assert!(
        out.values.iter().any(|&v| v.abs() > 1e-12),
        "int2e must produce a non-zero tensor (zero-fill stub regression?)"
    );

    // Self-repulsion diagonal (00|00) is a positive Coulomb integral.
    assert!(
        out.values[idx(0, 0, 0, 0, nao)] > 0.0,
        "(00|00) self-repulsion must be positive, got {}",
        out.values[idx(0, 0, 0, 0, nao)]
    );

    // 8-fold permutational symmetry of the real chemist ERI (ij|kl):
    //   (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk) = (kl|ij) = (lk|ij) = (kl|ji) = (lk|ji)
    let tol = 1e-12;
    for i in 0..nao {
        for j in 0..nao {
            for k in 0..nao {
                for l in 0..nao {
                    let base = out.values[idx(i, j, k, l, nao)];
                    let perms = [
                        out.values[idx(j, i, k, l, nao)], // bra swap
                        out.values[idx(i, j, l, k, nao)], // ket swap
                        out.values[idx(j, i, l, k, nao)], // both swap
                        out.values[idx(k, l, i, j, nao)], // bra<->ket
                        out.values[idx(l, k, i, j, nao)],
                        out.values[idx(k, l, j, i, nao)],
                        out.values[idx(l, k, j, i, nao)],
                    ];
                    for (p, &val) in perms.iter().enumerate() {
                        assert!(
                            (base - val).abs() < tol,
                            "8-fold symmetry broken at (ij|kl)=({i}{j}|{k}{l}) perm {p}: \
                             {base} vs {val}"
                        );
                    }
                }
            }
        }
    }
}
