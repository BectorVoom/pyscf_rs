//! CCSD wavefunction diagnostics — `get_t1_diagnostic` (Frobenius norm of t1),
//! `get_d1_diagnostic` / `get_d2_diagnostic` (eigh-based).
//!
//! Port of `pyscf/cc/ccsd.py:748-776`:
//!
//! ```python
//! def get_t1_diagnostic(t1):
//!     nelectron = 2 * t1.shape[0]
//!     return numpy.sqrt(numpy.linalg.norm(t1)**2 / nelectron)
//!
//! def get_d1_diagnostic(t1):  # Janssen, Chem. Phys. Lett. 290 (1998) 423
//!     f = lambda x: numpy.sqrt(numpy.sort(numpy.abs(x[0])))[-1]
//!     d1norm_ij = f(numpy.linalg.eigh(numpy.einsum('ia,ja->ij', t1, t1)))
//!     d1norm_ab = f(numpy.linalg.eigh(numpy.einsum('ia,ib->ab', t1, t1)))
//!     return max(d1norm_ij, d1norm_ab)
//!
//! def get_d2_diagnostic(t2):  # Nielsen, Chem. Phys. Lett. 310 (1999) 568
//!     f = lambda x: numpy.sqrt(numpy.sort(numpy.abs(x[0])))[-1]
//!     d2norm_ij = f(numpy.linalg.eigh(numpy.einsum('ikab,jkab->ij', t2, t2)))
//!     d2norm_ab = f(numpy.linalg.eigh(numpy.einsum('ijac,ijbc->ab', t2, t2)))
//!     return max(d2norm_ij, d2norm_ab)
//! ```
//!
//! **Reduction discipline (T-06-07-FP):** every Frobenius / einsum reduction
//! materializes per-element products into a `Vec` and reduces with
//! [`oracle_sum`] (thread-invariant, RAYON 1==8 bit-identical). The symmetric
//! `ij` / `ab` blocks are built as host loops (no `gemm`/`+=`) then diagonalized
//! with [`eigh_gen`] against an identity metric (a standard eigh of the
//! symmetric matrix — `S = I`). The diagnostics operate on the small active
//! `nocc`/`nvir` blocks, so no arena is needed (`<behavior>` / plan note).
//!
//! **Shape validation (T-06-07-SHAPE):** the `t1` / `t2` lengths are checked
//! against `nocc`/`nvir` BEFORE indexing — a mismatch returns
//! [`CcsdError::ShapeMismatch`], never an OOB panic
//! (`#![forbid(unsafe_code)]`).

use crate::error::CcsdError;
use pyscf_algebra::{eigh_gen, oracle_sum};

/// Build the row-major `n×n` identity used as the `eigh_gen` overlap metric
/// (`S = I` turns the generalized problem into a plain symmetric eigh).
fn identity(n: usize) -> Vec<f64> {
    let mut s = vec![0.0_f64; n * n];
    for i in 0..n {
        s[i * n + i] = 1.0;
    }
    s
}

/// `sqrt(max |eigenvalue|)` of a symmetric row-major `n×n` matrix `m` —
/// the upstream `f = lambda x: numpy.sqrt(numpy.sort(numpy.abs(x[0])))[-1]`.
///
/// `eigh_gen(m, I, n)` returns the eigenvalues (nondecreasing); we take the
/// largest `sqrt(|λ|)`. For a positive-semidefinite Gram matrix (`A·Aᵀ`) the
/// eigenvalues are already ≥ 0, but we apply `abs` for exact upstream parity.
fn max_sqrt_abs_eig(m: &[f64], n: usize) -> Result<f64, CcsdError> {
    if n == 0 {
        return Ok(0.0);
    }
    let s = identity(n);
    let (eigvals, _c) = eigh_gen(m, &s, n).map_err(CcsdError::from)?;
    let mut best = 0.0_f64;
    for &lam in eigvals.iter() {
        // eigh_gen pads dropped (linearly-dependent) directions with +∞; skip
        // any non-finite marker so a rank-deficient Gram block stays finite.
        if !lam.is_finite() {
            continue;
        }
        let v = lam.abs().sqrt();
        if v > best {
            best = v;
        }
    }
    Ok(best)
}

/// Lee–Taylor T1 diagnostic (`ccsd.py:748-751`): `||t1||_F / sqrt(nelec)`.
///
/// `t1` is a C-order `[nocc, nvir]` block (element `(i, a)` at `i*nvir + a`).
/// `nelec` is the number of correlated electrons (upstream
/// `nelectron = 2 * t1.shape[0] = 2 * nocc` for the closed-shell case; passed
/// explicitly so the caller controls the normalization). The Frobenius norm is
/// `sqrt(oracle_sum(t1[k]²))` — thread-invariant.
///
/// # Errors
/// [`CcsdError::ShapeMismatch`] if `t1.len() != nocc*nvir`, or if `nelec <= 0`.
pub fn get_t1_diagnostic(
    t1: &[f64],
    nocc: usize,
    nvir: usize,
    nelec: f64,
) -> Result<f64, CcsdError> {
    let expected = nocc * nvir;
    if t1.len() != expected {
        return Err(CcsdError::ShapeMismatch {
            expected,
            got: t1.len(),
        });
    }
    if nelec <= 0.0 {
        return Err(CcsdError::ShapeMismatch {
            expected: 1,
            got: 0,
        });
    }
    // ||t1||_F^2 = oracle_sum(t1[k]^2); the diagnostic is sqrt(norm^2 / nelec).
    let sq: Vec<f64> = t1.iter().map(|&x| x * x).collect();
    let norm_sq = oracle_sum(&sq);
    Ok((norm_sq / nelec).sqrt())
}

/// Janssen–Nielsen D1 diagnostic (`ccsd.py:753-762`):
/// `max(d1norm_ij, d1norm_ab)` where
/// `d1norm_ij = sqrt(max|eig|)` of `A_ij = einsum('ia,ja->ij', t1, t1) = t1·t1ᵀ`
/// (the `nocc×nocc` Gram) and `d1norm_ab = sqrt(max|eig|)` of
/// `A_ab = einsum('ia,ib->ab', t1, t1) = t1ᵀ·t1` (the `nvir×nvir` Gram).
///
/// The two Gram blocks share the same nonzero eigenvalues (`AAᵀ`/`AᵀA`), so the
/// values coincide; both are computed for exact upstream parity. Each Gram entry
/// is a host-loop materialize-then-[`oracle_sum`] reduction (no `gemm`/`+=`),
/// then diagonalized via [`eigh_gen`] against an identity metric.
///
/// # Errors
/// [`CcsdError::ShapeMismatch`] if `t1.len() != nocc*nvir`.
pub fn get_d1_diagnostic(t1: &[f64], nocc: usize, nvir: usize) -> Result<f64, CcsdError> {
    let expected = nocc * nvir;
    if t1.len() != expected {
        return Err(CcsdError::ShapeMismatch {
            expected,
            got: t1.len(),
        });
    }
    let t1e = |i: usize, a: usize| t1[i * nvir + a];

    // A_ij = sum_a t1[i,a]*t1[j,a]  (nocc×nocc, symmetric).
    let mut a_ij = vec![0.0_f64; nocc * nocc];
    let mut buf: Vec<f64> = Vec::with_capacity(nvir);
    for i in 0..nocc {
        for j in 0..nocc {
            buf.clear();
            for a in 0..nvir {
                buf.push(t1e(i, a) * t1e(j, a));
            }
            a_ij[i * nocc + j] = oracle_sum(&buf);
        }
    }
    let d1_ij = max_sqrt_abs_eig(&a_ij, nocc)?;

    // A_ab = sum_i t1[i,a]*t1[i,b]  (nvir×nvir, symmetric).
    let mut a_ab = vec![0.0_f64; nvir * nvir];
    for a in 0..nvir {
        for b in 0..nvir {
            buf.clear();
            for i in 0..nocc {
                buf.push(t1e(i, a) * t1e(i, b));
            }
            a_ab[a * nvir + b] = oracle_sum(&buf);
        }
    }
    let d1_ab = max_sqrt_abs_eig(&a_ab, nvir)?;

    Ok(d1_ij.max(d1_ab))
}

/// Nielsen D2 diagnostic (`ccsd.py:764-776`): `max(d2norm_ij, d2norm_ab)` where
/// `d2norm_ij = sqrt(max|eig|)` of `B_ij = einsum('ikab,jkab->ij', t2, t2)`
/// (the `nocc×nocc` block) and `d2norm_ab = sqrt(max|eig|)` of
/// `B_ab = einsum('ijac,ijbc->ab', t2, t2)` (the `nvir×nvir` block).
///
/// `t2` is a C-order `[nocc, nocc, nvir, nvir]` block (element `(i, j, a, b)` at
/// `((i*nocc + j)*nvir + a)*nvir + b`). Each Gram entry is a host-loop
/// materialize-then-[`oracle_sum`] reduction; the blocks are diagonalized via
/// [`eigh_gen`].
///
/// # Errors
/// [`CcsdError::ShapeMismatch`] if `t2.len() != nocc²·nvir²`.
pub fn get_d2_diagnostic(t2: &[f64], nocc: usize, nvir: usize) -> Result<f64, CcsdError> {
    let expected = nocc * nocc * nvir * nvir;
    if t2.len() != expected {
        return Err(CcsdError::ShapeMismatch {
            expected,
            got: t2.len(),
        });
    }
    let t2e = |i: usize, j: usize, a: usize, b: usize| t2[((i * nocc + j) * nvir + a) * nvir + b];

    // B_ij = sum_{k,a,b} t2[i,k,a,b]*t2[j,k,a,b]  (nocc×nocc, symmetric).
    let mut b_ij = vec![0.0_f64; nocc * nocc];
    let mut buf: Vec<f64> = Vec::with_capacity(nocc * nvir * nvir);
    for i in 0..nocc {
        for j in 0..nocc {
            buf.clear();
            for k in 0..nocc {
                for a in 0..nvir {
                    for b in 0..nvir {
                        buf.push(t2e(i, k, a, b) * t2e(j, k, a, b));
                    }
                }
            }
            b_ij[i * nocc + j] = oracle_sum(&buf);
        }
    }
    let d2_ij = max_sqrt_abs_eig(&b_ij, nocc)?;

    // B_ab = sum_{i,j,c} t2[i,j,a,c]*t2[i,j,b,c]  (nvir×nvir, symmetric).
    let mut b_ab = vec![0.0_f64; nvir * nvir];
    for a in 0..nvir {
        for b in 0..nvir {
            buf.clear();
            for i in 0..nocc {
                for j in 0..nocc {
                    for c in 0..nvir {
                        buf.push(t2e(i, j, a, c) * t2e(i, j, b, c));
                    }
                }
            }
            b_ab[a * nvir + b] = oracle_sum(&buf);
        }
    }
    let d2_ab = max_sqrt_abs_eig(&b_ab, nvir)?;

    Ok(d2_ij.max(d2_ab))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ||t1||_F / sqrt(nelec): t1 = [[3,0],[0,4]] → ||t1||_F = 5, nelec=4 → 2.5.
    #[test]
    fn t1_frobenius_norm_value() {
        let t1 = vec![3.0, 0.0, 0.0, 4.0]; // (0,0)=3, (1,1)=4
        let t1d = get_t1_diagnostic(&t1, 2, 2, 4.0).expect("t1");
        assert!((t1d - 2.5).abs() < 1e-12);
    }

    /// D1 of diag t1 = sqrt(16) = 4.
    #[test]
    fn d1_diag_value() {
        let t1 = vec![3.0, 0.0, 0.0, 4.0];
        let d1 = get_d1_diagnostic(&t1, 2, 2).expect("d1");
        assert!((d1 - 4.0).abs() < 1e-10);
    }

    /// D2 with a single t2[0,0,0,0]=2 → sqrt(4) = 2.
    #[test]
    fn d2_single_entry_value() {
        let mut t2 = vec![0.0_f64; 16];
        t2[0] = 2.0; // (0,0,0,0)
        let d2 = get_d2_diagnostic(&t2, 2, 2).expect("d2");
        assert!((d2 - 2.0).abs() < 1e-10);
    }

    /// Shape mismatch returns an error, never panics.
    #[test]
    fn shape_mismatch_is_error() {
        assert!(get_t1_diagnostic(&[1.0, 2.0, 3.0], 2, 2, 4.0).is_err());
        assert!(get_d1_diagnostic(&[1.0, 2.0, 3.0], 2, 2).is_err());
        assert!(get_d2_diagnostic(&[1.0; 5], 2, 2).is_err());
    }
}
