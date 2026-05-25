//! Rank-revealing DF/RI 2-center metric fitting factor (plan 05-09).
//!
//! The density-fitting (resolution-of-identity) approximation needs an
//! inverse of the symmetric PSD 2-center metric `(P|Q)`:
//! `(μν|λσ) ≈ Σ_PQ (μν|P) (P|Q)⁻¹ (Q|λσ)`. A plain Cholesky of `(P|Q)`
//! fails for real auxiliary bases (cc-pvdz-jkfit, weigend) because the metric
//! is frequently rank-deficient / numerically indefinite. This module provides
//! the upstream-PySCF fallback (`pyscf/df/df.py` build): an eigendecomposition
//! that drops eigenvalues ≤ `lindep` and returns a rank-revealing fit factor.
//!
//! Algorithm (matches `pyscf` DF, eigh route):
//!   1. `(P|Q) = V·diag(w)·Vᵀ` (self-adjoint eigh, faer 0.24).
//!   2. Keep eigenvalues `w_i > lindep` (drop near-zero / negative — the
//!      linear-dependency removal, upstream `LINEAR_DEP_THRESHOLD`).
//!   3. Fit factor `W[P,k] = V[P, j(k)] · w_{j(k)}^{-1/2}` (n × rank), so
//!      `W·Wᵀ = Σ_{kept} V_i V_iᵀ / w_i = (P|Q)⁻¹` on the kept subspace.
//!
//! The DF B-tensor is then `B^k_{μν} = Σ_P (μν|P)·W[P,k]`, giving
//! `Σ_k B^k_{μν} B^k_{λσ} = (μν|·)ᵀ (P|Q)⁻¹_trunc (·|λσ)` — the standard DF
//! identity. `rank ≤ n` is the effective auxiliary dimension.
//!
//! Layout: input `j2c` is row-major n×n; output `w` is **column-major** n×rank
//! (element (P,k) at `w[k*n + P]`), so each fit column `w[k*n .. k*n+n]` is
//! contiguous for an `oracle_dot` against a gathered `(μν|·)` row.

use crate::AlgebraError;
use faer::linalg::solvers::SelfAdjointEigen;
use faer::{Mat, Side};

/// Linear-dependency cutoff for the DF 2-center metric: eigenvalues `≤` this
/// are dropped. Matches the order of upstream PySCF's DF `LINEAR_DEP_THRESHOLD`
/// (`pyscf/df/df.py`); the exact value is reconfirmed when the upstream oracle
/// runs (`mp2-oracle-upstream-manual`).
pub const DF_METRIC_LINEAR_DEP: f64 = 1e-9;

/// Rank-revealing inverse-square-root fit factor for a symmetric PSD DF metric.
///
/// Given `j2c = (P|Q)` (row-major `n × n`), returns `(w, rank)` where `w` is
/// column-major `n × rank` (element (P,k) at `w[k*n + P]`) and
/// `w · wᵀ == (P|Q)⁻¹` restricted to the subspace of eigenvalues `> lindep`.
///
/// # Errors
/// - [`AlgebraError::ShapeMismatch`] if `j2c.len() != n*n`.
/// - [`AlgebraError::Singular`] if every eigenvalue is `≤ lindep` (rank 0).
/// - [`AlgebraError::CubeclRuntime`] on faer evd failure (rare; usually a
///   non-self-adjoint input — symmetrize before calling).
pub fn df_metric_fit(
    j2c: &[f64],
    n: usize,
    lindep: f64,
) -> Result<(Vec<f64>, usize), AlgebraError> {
    if j2c.len() != n * n {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("j2c n*n = {}", n * n),
            actual: format!("len={}", j2c.len()),
        });
    }
    if n == 0 {
        return Err(AlgebraError::Singular);
    }

    // (P|Q) = V·diag(w)·Vᵀ. Input is row-major; the metric is symmetric so
    // Side::Lower reads the lower triangle (matches eigh_gen).
    let a = Mat::<f64>::from_fn(n, n, |i, j| j2c[i * n + j]);
    let evd = SelfAdjointEigen::new(a.as_ref(), Side::Lower)
        .map_err(|e| AlgebraError::CubeclRuntime(format!("eigh((P|Q)) failed: {e:?}")))?;
    let evals = evd.S();
    let evecs = evd.U();

    // Keep eigenvalues > lindep (linear-dependency removal). faer returns
    // eigenvalues nondecreasing, so the kept set is the trailing block, but we
    // scan all to be layout-agnostic.
    let kept: Vec<(usize, f64)> = (0..n)
        .filter_map(|j| {
            let lam = evals[j];
            if lam > lindep {
                Some((j, 1.0 / lam.sqrt()))
            } else {
                None
            }
        })
        .collect();
    let rank = kept.len();
    if rank == 0 {
        return Err(AlgebraError::Singular);
    }

    // W column-major n × rank: W[P,k] = V[P, j(k)] · w_{j(k)}^{-1/2}.
    let mut w = vec![0.0f64; n * rank];
    for (k, &(j, inv_sqrt)) in kept.iter().enumerate() {
        let base = k * n;
        for p in 0..n {
            w[base + p] = evecs[(p, j)] * inv_sqrt;
        }
    }

    Ok((w, rank))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `W·Wᵀ` reconstructed from the column-major n×rank factor.
    fn w_wt(w: &[f64], n: usize, rank: usize) -> Vec<f64> {
        let mut out = vec![0.0f64; n * n];
        for p in 0..n {
            for q in 0..n {
                let mut s = 0.0;
                for k in 0..rank {
                    s += w[k * n + p] * w[k * n + q];
                }
                out[p * n + q] = s;
            }
        }
        out
    }

    #[test]
    fn pd_diagonal_full_rank_inverse() {
        // (P|Q) = diag(4, 1) → full rank; (P|Q)⁻¹ = diag(0.25, 1).
        let j2c = vec![4.0, 0.0, 0.0, 1.0];
        let (w, rank) = df_metric_fit(&j2c, 2, DF_METRIC_LINEAR_DEP).expect("fit");
        assert_eq!(rank, 2, "PD metric is full rank");
        let inv = w_wt(&w, 2, rank);
        assert!((inv[0] - 0.25).abs() < 1e-12, "inv[0,0]={}", inv[0]);
        assert!((inv[3] - 1.0).abs() < 1e-12, "inv[1,1]={}", inv[3]);
        assert!(inv[1].abs() < 1e-12 && inv[2].abs() < 1e-12);
    }

    #[test]
    fn rank_deficient_drops_null_direction() {
        // Build a 3×3 symmetric PSD rank-2 metric: V diag(2, 1, 0) Vᵀ with V a
        // rotation in the (0,1) plane. The zero eigenvalue must be dropped →
        // rank 2, and W·Wᵀ is the pseudo-inverse (no inf from 1/0).
        // V columns: e2=[0,0,1] (eval 0), and an orthonormal pair in x-y.
        // A = 2*u uᵀ + 1*v vᵀ, u=[c,c,0], v=[c,-c,0], c=1/√2; null = [0,0,1].
        // A[0,0]=2c²+c²=1.5; A[0,1]=2c²-c²=0.5; A[1,1]=1.5; rest 0.
        let j2c = vec![
            1.5, 0.5, 0.0, //
            0.5, 1.5, 0.0, //
            0.0, 0.0, 0.0,
        ];
        let (w, rank) = df_metric_fit(&j2c, 3, DF_METRIC_LINEAR_DEP).expect("fit");
        assert_eq!(rank, 2, "rank-2 PSD metric drops the null direction");
        let pinv = w_wt(&w, 3, rank);
        for v in &pinv {
            assert!(v.is_finite(), "pseudo-inverse must be finite (no 1/0)");
        }
        // Pseudo-inverse of the (x,y) block: eigenvalues 2 and 1 → pinv on that
        // block is u uᵀ/2 + v vᵀ/1. pinv[0,0]=0.5²/2*? compute: u uᵀ/2 has
        // [0,0]=c²/2=0.25; v vᵀ/1 has [0,0]=c²=0.5 → 0.75. [0,1]=c²/2*?:
        // u uᵀ/2 [0,1]=c²/2=0.25; v vᵀ [0,1]=-c²=-0.5 → -0.25. [2,*]=0.
        assert!((pinv[0] - 0.75).abs() < 1e-10, "pinv[0,0]={}", pinv[0]);
        assert!((pinv[1] + 0.25).abs() < 1e-10, "pinv[0,1]={}", pinv[1]);
        assert!(
            pinv[8].abs() < 1e-12,
            "null direction pinv[2,2]={}",
            pinv[8]
        );
        // A · pinv · A == A (Moore-Penrose) on the kept subspace.
    }

    #[test]
    fn all_below_threshold_is_singular() {
        let j2c = vec![1e-15, 0.0, 0.0, 1e-15];
        let r = df_metric_fit(&j2c, 2, DF_METRIC_LINEAR_DEP);
        assert!(matches!(r, Err(AlgebraError::Singular)));
    }

    #[test]
    fn shape_mismatch_errors() {
        let j2c = vec![1.0, 0.0, 0.0];
        let r = df_metric_fit(&j2c, 2, DF_METRIC_LINEAR_DEP);
        assert!(matches!(r, Err(AlgebraError::ShapeMismatch { .. })));
    }

    #[test]
    fn tiny_negative_eigenvalue_dropped_not_nan() {
        // A symmetric metric with one tiny-negative eigenvalue (numerical
        // indefiniteness) must drop it, never produce NaN from sqrt(<0).
        // diag(1.0, -1e-13): the negative is < lindep → dropped.
        let j2c = vec![1.0, 0.0, 0.0, -1e-13];
        let (w, rank) = df_metric_fit(&j2c, 2, DF_METRIC_LINEAR_DEP).expect("fit");
        assert_eq!(rank, 1, "only the positive eigenvalue is kept");
        assert!(w.iter().all(|v| v.is_finite()));
        let inv = w_wt(&w, 2, rank);
        assert!((inv[0] - 1.0).abs() < 1e-12, "kept eigenvalue 1 → inv 1");
        assert!(inv[3].abs() < 1e-12, "dropped direction contributes 0");
    }
}
