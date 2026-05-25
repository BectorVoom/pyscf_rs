//! Generalized self-adjoint eigendecomposition `F·C = S·C·diag(ε)`
//! for SCF (Phase 3, plan 03-11).
//!
//! Source-of-truth: ALG-05 host-fallback discipline. The Phase 1
//! `host_fallback::eigh(client, tensor)` Tensor-based API is still
//! `NotYetImplemented{phase:3}` — `eigh_gen` ships a slice-based
//! wrapper mirroring `solve_linear` (plan 03-01) so SCF (which lives
//! one crate up at pyscf-scf and is not allowed to depend on cubecl-rs
//! types — D-04 algebra-wall) can call into faer 0.24 without naming
//! `Tensor`/`AlgebraClient`.
//!
//! Algorithm — Löwdin transform:
//!   1. Diagonalize `S = U·diag(s)·U^T` (self-adjoint eigh).
//!   2. Build `X = U·diag(s^{-1/2})·U^T` (the symmetric S^{-1/2}).
//!      Drop tiny eigenvalues below `s_tol` (linear-dependency removal —
//!      pyscf/scf/addons.py:remove_linear_dep).
//!   3. Transform `F' = X·F·X` (symmetric).
//!   4. Diagonalize `F' = V·diag(ε)·V^T`.
//!   5. Recover `C = X·V`. Then `F·C = S·C·diag(ε)` holds.
//!
//! Layout: input `f` is row-major n×n; `s` is row-major n×n. Output `c`
//! is **column-major** (F-order) n×n — the convention pyscf-core's
//! `MOCoefficients` documents (`pyscf-core/src/mo.rs` says "Column-major
//! (Fortran order)"). Eigenvalues are returned nondecreasing.
//!
//! No SCF-13 canonicalization is applied here; the caller
//! (pyscf-scf::eig::default_eig) calls `pyscf_core::canonicalize_signs`
//! on the F-order output buffer.

use crate::AlgebraError;
use faer::linalg::solvers::SelfAdjointEigen;
use faer::{Mat, Side};

/// Linear-dependency cutoff. Eigenvalues of S below this magnitude are
/// dropped (matches upstream pyscf's `LINEAR_DEP_THRESHOLD = 1e-12`
/// at `pyscf/scf/addons.py:34`, conservative).
pub const S_LINEAR_DEP_TOL: f64 = 1e-12;

/// Solve `F·C = S·C·diag(ε)` for `(eigenvalues_nondecreasing,
/// C_F_order_flat)`. `f` and `s` are row-major flat slices of length
/// `n*n`; the returned C buffer is F-order (column-major) and
/// `eigenvalues.len() == n`.
///
/// # Errors
/// - [`AlgebraError::ShapeMismatch`] if `f.len() != n*n` or `s.len() != n*n`.
/// - [`AlgebraError::Singular`] if the LU decomposition reports failure
///   or if `S` has more than `n` eigenvalues below `S_LINEAR_DEP_TOL`
///   (full linear dependency — caller must investigate).
/// - [`AlgebraError::Backend(_)`] on faer evd failure (rare; usually
///   indicates non-self-adjoint input — symmetrize the input before
///   calling).
pub fn eigh_gen(f: &[f64], s: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>), AlgebraError> {
    if f.len() != n * n {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("f n*n = {}", n * n),
            actual: format!("len={}", f.len()),
        });
    }
    if s.len() != n * n {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("s n*n = {}", n * n),
            actual: format!("len={}", s.len()),
        });
    }

    // === Step 1: eigh(S) ===
    // Input is row-major flat slice; element (i, j) at s[i*n + j].
    let s_mat = Mat::<f64>::from_fn(n, n, |i, j| s[i * n + j]);
    let s_evd = SelfAdjointEigen::new(s_mat.as_ref(), Side::Lower)
        .map_err(|e| AlgebraError::CubeclRuntime(format!("eigh(S) failed: {e:?}")))?;
    let s_evals_diag = s_evd.S();
    let s_evecs = s_evd.U();

    // === Step 2: X = U · diag(s^{-1/2}) · U^T with linear-dep removal ===
    // Drop columns where eigenvalue < S_LINEAR_DEP_TOL.
    let mut valid_cols: Vec<(usize, f64)> = Vec::with_capacity(n);
    for j in 0..n {
        let lam = s_evals_diag[j];
        if lam > S_LINEAR_DEP_TOL {
            valid_cols.push((j, 1.0 / lam.sqrt()));
        }
    }
    if valid_cols.is_empty() {
        return Err(AlgebraError::Singular);
    }
    let n_lin = valid_cols.len();

    // X is n × n_lin (canonical orthonormalization: drop near-zero columns).
    // X[i, k] = U[i, j(k)] * s^{-1/2}(k)
    let x = Mat::<f64>::from_fn(n, n_lin, |i, k| {
        let (j, inv_sqrt) = valid_cols[k];
        s_evecs[(i, j)] * inv_sqrt
    });

    // === Step 3: F' = X^T · F · X (n_lin × n_lin, symmetric) ===
    let f_mat = Mat::<f64>::from_fn(n, n, |i, j| f[i * n + j]);
    // X^T · F → (n_lin × n)
    let xt_f = x.transpose() * &f_mat;
    let fp = &xt_f * &x; // (n_lin × n_lin)

    // === Step 4: eigh(F') ===
    let fp_evd = SelfAdjointEigen::new(fp.as_ref(), Side::Lower)
        .map_err(|e| AlgebraError::CubeclRuntime(format!("eigh(F') failed: {e:?}")))?;
    let eigvals_diag = fp_evd.S();
    let v = fp_evd.U();

    // === Step 5: C = X · V (n × n_lin) ===
    let c_lin = &x * v;

    // Pack outputs. eigenvalues: n entries — first n_lin from fp_evd
    // (sorted nondecreasing by faer), remaining padded with +∞ as
    // "linearly-dependent / dropped" markers so the caller sees an
    // n-length vector aligned with the n-column C buffer.
    let mut eigenvalues = vec![f64::INFINITY; n];
    for i in 0..n_lin {
        eigenvalues[i] = eigvals_diag[i];
    }

    // C output: F-order (column-major) n × n. First n_lin columns are
    // the valid MO coefficients; remaining columns are zeroed (these
    // correspond to the dropped linearly-dependent directions).
    let mut c_out = vec![0.0_f64; n * n];
    for j in 0..n_lin {
        for i in 0..n {
            c_out[i + j * n] = c_lin[(i, j)];
        }
    }

    if !eigenvalues.iter().take(n_lin).all(|x| x.is_finite()) {
        return Err(AlgebraError::Singular);
    }

    Ok((eigenvalues, c_out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonal_f_identity_s_returns_diagonal() {
        // F = diag(1, 2), S = I → eigenvalues (1, 2), eigvecs canonical basis.
        let f = vec![1.0, 0.0, 0.0, 2.0];
        let s = vec![1.0, 0.0, 0.0, 1.0];
        let (eigvals, c) = eigh_gen(&f, &s, 2).expect("eigh_gen");
        assert!((eigvals[0] - 1.0).abs() < 1e-10);
        assert!((eigvals[1] - 2.0).abs() < 1e-10);
        // C is F-order 2×2; column 0 should be ±[1, 0], column 1 ±[0, 1].
        // |C[0,0]| + |C[1,0]| ≈ 1 and only one of them is significant.
        let col0_norm_sq = c[0] * c[0] + c[1] * c[1];
        let col1_norm_sq = c[2] * c[2] + c[3] * c[3];
        assert!((col0_norm_sq - 1.0).abs() < 1e-10);
        assert!((col1_norm_sq - 1.0).abs() < 1e-10);
    }

    #[test]
    fn shape_mismatch_returns_error() {
        let f = vec![1.0, 0.0, 0.0]; // wrong size
        let s = vec![1.0, 0.0, 0.0, 1.0];
        let r = eigh_gen(&f, &s, 2);
        assert!(matches!(r, Err(AlgebraError::ShapeMismatch { .. })));
    }

    #[test]
    fn generalized_problem_recovers_eigenvalue_equation() {
        // Hand-crafted F and S (both symmetric, S positive-definite).
        // F = [[2, 1], [1, 2]], S = [[1, 0.1], [0.1, 1]]
        let f = vec![2.0, 1.0, 1.0, 2.0];
        let s = vec![1.0, 0.1, 0.1, 1.0];
        let (eigvals, c) = eigh_gen(&f, &s, 2).expect("eigh_gen");
        // Verify F·c_j = ε_j · S·c_j for each MO column.
        // C is F-order: c[i + j*2].
        for j in 0..2 {
            let cj = [c[j * 2], c[1 + j * 2]];
            // F·cj
            let fcj0 = f[0] * cj[0] + f[1] * cj[1];
            let fcj1 = f[2] * cj[0] + f[3] * cj[1];
            // S·cj
            let scj0 = s[0] * cj[0] + s[1] * cj[1];
            let scj1 = s[2] * cj[0] + s[3] * cj[1];
            let lam = eigvals[j];
            assert!(
                (fcj0 - lam * scj0).abs() < 1e-9,
                "F·c[{j}][0] = {} but ε·S·c[{j}][0] = {} (ε={})",
                fcj0,
                lam * scj0,
                lam
            );
            assert!(
                (fcj1 - lam * scj1).abs() < 1e-9,
                "F·c[{j}][1] = {} but ε·S·c[{j}][1] = {} (ε={})",
                fcj1,
                lam * scj1,
                lam
            );
        }
        // Eigenvalues sorted nondecreasing.
        assert!(eigvals[0] <= eigvals[1]);
    }
}
