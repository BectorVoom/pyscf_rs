//! Gamma-point restricted TDA/TDHF (`pbc/tdscf/rhf.py`, 238 l).
//!
//! The smallest complete method in the phase — its driver/solver split is the
//! shape every later tdscf plan copies (19-07 k-points, 19-08 unrestricted,
//! 19-09 KS subclasses).
//!
//! Response matrices (Casida form, `get_ab`: `A[ia][jb] = δ(e_a−e_i) +
//! (ai||jb)`, `B[ia][jb] = (ai||bj)`):
//!
//! * singlet: `A = diag(e_ia) + 2·J − hyb·K`, `B = 2·J − hyb·K` with
//!   `J[ia][jb] = (ia|jb)`, `K[ia][jb] = (ij|ab)` / `(ib|ja)`;
//! * triplet: `A = diag(e_ia) − hyb·K`, `B = −hyb·K` (no Coulomb — the
//!   `gen_response(singlet=False)` convention; upstream's `get_ab` builds the
//!   singlet matrices and the vind selects, so the triplet build is verified
//!   end-to-end through triplet roots, stated in tests).
//!
//! TDA diagonalizes `A` densely (test-size exact-Davidson via
//! [`crate::davidson`]). TDHF uses the Casida reduction `M = (A−B)(A+B)`,
//! `w = √eig(M)` — exact for real symmetric `A ± B` with `A − B` positive
//! definite. A non-positive-definite `A − B` is an instability, not an
//! excitation: refused toward `pyscf_pbc_scf::stability`, never square-rooted
//! into a complex number. TDA and TDHF are distinct entry points with
//! distinct numbers (never shared).

use crate::davidson::tda_dense;
use crate::error::PbcTdscfError;
use crate::types::{TdaConfig, TdaResult};
use pyscf_algebra::{eigh_gen, oracle_sum};

/// Build the Casida `A`/`B` matrices (row-major `dim × dim`, `dim = nocc·nvir`,
/// `[ia][jb]` with `ia = i·nvir + a`).
///
/// Ports `get_ab`'s `add_hf_` einsums over the half-transformed ERI tensor
/// `eri[i][p][q][r]` (`(i q|r s)` chemist, `i` occupied, shape
/// `nocc·nmo³`, `nmo = nocc + nvir`):
///
/// * `A = diag(e_ia) + 2·(ia|bj) − hyb·(ij|ba)` with
///   `(ia|bj) = eri[i][nocc+a][nocc+b][j]` (`'iabj->iajb'`) and
///   `(ij|ba) = eri[i][j][nocc+b][nocc+a]` (`'ijba->iajb'`);
/// * `B = 2·(ia|jb) − hyb·(ib|ja)` with
///   `(ia|jb) = eri[i][nocc+a][j][nocc+b]` (`'iajb->iajb'`) and
///   `(ib|ja) = eri[i][nocc+b][j][nocc+a]` (`'ibja->iajb'`).
///
/// Triplet (`singlet = false`) drops both Coulomb terms (the
/// `gen_response(singlet=False)` convention); `hyb` is the exact-exchange
/// fraction (1.0 for Hartree-Fock).
pub fn build_ab(
    eri: &[f64],
    e_ia: &[f64],
    nocc: usize,
    nmo: usize,
    singlet: bool,
    hyb: f64,
) -> Result<(Vec<f64>, Vec<f64>), PbcTdscfError> {
    let nvir = nmo - nocc;
    let dim = nocc * nvir;
    if eri.len() != nocc * nmo * nmo * nmo || e_ia.len() != dim {
        return Err(PbcTdscfError::ShapeMismatch { expected: nocc * nmo * nmo * nmo, got: eri.len() });
    }
    let e = |i: usize, p: usize, q: usize, r: usize| -> f64 {
        eri[((i * nmo + p) * nmo + q) * nmo + r]
    };
    let mut a = vec![0.0f64; dim * dim];
    let mut b = vec![0.0f64; dim * dim];
    for i in 0..nocc {
        for aj in 0..nvir {
            let ia = i * nvir + aj;
            for j in 0..nocc {
                for bj in 0..nvir {
                    let jb = j * nvir + bj;
                    let ja = e(i, nocc + aj, nocc + bj, j);
                    let ka = e(i, j, nocc + bj, nocc + aj);
                    let jb2 = e(i, nocc + aj, j, nocc + bj);
                    let kb = e(i, nocc + bj, j, nocc + aj);
                    if singlet {
                        a[ia * dim + jb] = 2.0 * ja - hyb * ka;
                        b[ia * dim + jb] = 2.0 * jb2 - hyb * kb;
                    } else {
                        a[ia * dim + jb] = -hyb * ka;
                        b[ia * dim + jb] = -hyb * kb;
                    }
                }
            }
            a[ia * dim + ia] += e_ia[ia];
        }
    }
    Ok((a, b))
}

/// Gamma-point RHF-TDA driver: build `A`, solve densely, return `nroots`.
pub fn kernel_rhf_tda(
    eri: &[f64],
    e_ia: &[f64],
    nocc: usize,
    nmo: usize,
    hyb: f64,
    cfg: &TdaConfig,
) -> Result<TdaResult, PbcTdscfError> {
    let (a, _b) = build_ab(eri, e_ia, nocc, nmo, cfg.singlet, hyb)?;
    let mut out = tda_dense(&a, nocc * (nmo - nocc), cfg)?;
    out.kshift = 0;
    Ok(out)
}

/// Gamma-point RHF-TDHF driver via the symmetric Casida reduction.
///
/// `w² = eig(S)` with `S = sq·(A−B)·sq`, `sq = (A+B)^{1/2}` — similar to
/// `(A−B)(A+B)`, hence the same spectrum, but symmetric, so `eigh_gen`
/// applies. Requires both `(A−B)` and `(A+B)` positive definite; otherwise
/// the reference is an instability input ([`PbcTdscfError::UnstableReference`]),
/// routed to stability analysis rather than square-rooted into a complex.
pub fn kernel_rhf_tdhf(
    eri: &[f64],
    e_ia: &[f64],
    nocc: usize,
    nmo: usize,
    hyb: f64,
    cfg: &TdaConfig,
) -> Result<TdaResult, PbcTdscfError> {
    if !cfg.tda {
        // cfg.tda=false means "full TDHF requested" in the driver vocabulary;
        // this entry point IS the full-TDHF route, so proceed. The refusal in
        // davidson::tda_dense guards the TDA-only solver, not this path.
    }
    let (a, b) = build_ab(eri, e_ia, nocc, nmo, cfg.singlet, hyb)?;
    let dim = nocc * (nmo - nocc);
    if cfg.nroots > dim {
        return Err(PbcTdscfError::TooManyRoots { nroots: cfg.nroots, dim });
    }
    symm_tdhf(&a, &b, dim, cfg.nroots)
}

/// Shared symmetric-Casida TDHF core (gamma 19-06 and k-point 19-07).
///
/// `w² = eig(S)`, `S = sq·(A−B)·sq`, `sq = (A+B)^{1/2}` — similar to
/// `(A−B)(A+B)` hence the same spectrum, but symmetric, so `eigh_gen`
/// applies. Requires both `(A−B)` and `(A+B)` positive definite; otherwise
/// the reference is an instability input ([`PbcTdscfError::UnstableReference`]).
/// Returned roots carry `kshift = 0`; k-point callers re-stamp (19-07).
pub fn symm_tdhf(a: &[f64], b: &[f64], dim: usize, nroots: usize) -> Result<TdaResult, PbcTdscfError> {
    if a.len() != dim * dim || b.len() != dim * dim || nroots > dim || dim == 0 {
        return Err(PbcTdscfError::ShapeMismatch { expected: dim * dim, got: a.len().min(b.len()) });
    }
    // AmBp = A+B, AmBm = A−B.
    let mut sum = vec![0.0f64; dim * dim];
    let mut diff = vec![0.0f64; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            sum[i * dim + j] = a[i * dim + j] + b[i * dim + j];
            diff[i * dim + j] = a[i * dim + j] - b[i * dim + j];
        }
    }
    // diff must be positive definite: Cholesky-style check via eigh.
    let ident: Vec<f64> = {
        let mut s = vec![0.0f64; dim * dim];
        for i in 0..dim {
            s[i * dim + i] = 1.0;
        }
        s
    };
    let (d_evals, _) = eigh_gen(&diff, &ident, dim)?;
    if d_evals[0] <= 0.0 {
        return Err(PbcTdscfError::UnstableReference { lowest: d_evals[0] });
    }
    // Symmetric Casida route: M = (A−B)(A+B) is similar to the SYMMETRIC
    // S = sq·(A−B)·sq with sq = (A+B)^{1/2} (S and M share all eigenvalues;
    // M itself is genuinely non-symmetric — symmetrizing it element-wise
    // would be a same-shape wrong number, caught here during development).
    // Needs (A+B) SPD; if it is not, the reference is unstable territory.
    let (s_evals, s_vecs_f) = eigh_gen(&sum, &ident, dim)?;
    if s_evals[0] <= 0.0 {
        return Err(PbcTdscfError::UnstableReference { lowest: s_evals[0] });
    }
    // sq = U·diag(√λ)·Uᵀ (F-order columns: U[i][k] = s_vecs_f[k*dim+i]).
    let mut sq = vec![0.0f64; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let mut acc = 0.0f64;
            for k in 0..dim {
                acc += s_vecs_f[k * dim + i] * s_evals[k].sqrt() * s_vecs_f[k * dim + j];
            }
            sq[i * dim + j] = acc;
        }
    }
    // S = sq·diff·sq (ordered host products).
    let mut tmp = vec![0.0f64; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let mut terms = Vec::with_capacity(dim);
            for k in 0..dim {
                terms.push(sq[i * dim + k] * diff[k * dim + j]);
            }
            tmp[i * dim + j] = oracle_sum(&terms);
        }
    }
    let mut s = vec![0.0f64; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let mut terms = Vec::with_capacity(dim);
            for k in 0..dim {
                terms.push(tmp[i * dim + k] * sq[k * dim + j]);
            }
            s[i * dim + j] = oracle_sum(&terms);
        }
    }
    // S is symmetric by construction; assert the roundoff-only remainder.
    let mut asym = 0.0f64;
    for i in 0..dim {
        for j in 0..dim {
            asym = asym.max((s[i * dim + j] - s[j * dim + i]).abs());
        }
    }
    let s_scale: f64 = s.iter().map(|x| x.abs()).fold(0.0, f64::max);
    if asym > 1e-8 * s_scale.max(1.0) {
        return Err(PbcTdscfError::UnstableReference { lowest: -asym });
    }
    let (w2, _) = eigh_gen(&s, &ident, dim)?;
    if w2[0] <= 0.0 {
        return Err(PbcTdscfError::UnstableReference { lowest: w2[0] });
    }
    let n = nroots.min(dim);
    let energies: Vec<f64> = w2[..n].iter().map(|x| x.sqrt()).collect();
    Ok(TdaResult {
        energies,
        oscillator: vec![0.0f64; n],
        kshift: 0,
        converged: true,
    })
}
