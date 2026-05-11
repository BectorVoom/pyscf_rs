//! Host-faer LU solve for small square systems (Phase 3 — DIIS B-matrix).
//!
//! Source: ALG-05 host-fallback discipline. Phase 3 introduces this routine
//! for the DIIS Pulay system (RESEARCH Open Question 1 resolution): the
//! B-matrix has `-1` on the Lagrange row/col so the diagonal contains 0,
//! breaking naive Cholesky. faer 0.24 `linalg::solvers::FullPivLu` handles it.
//!
//! Input: row-major `n × n` matrix `a` (flat slice), `n`-element rhs `b`.
//! Output: `n`-element solution vector. Returns `AlgebraError::Singular` if
//! the LU produced a non-finite solution component (faer 0.24's
//! `FullPivLu::new` is panic-free but does not return `Result`; singularity
//! surfaces as inf/NaN in the solve, which we detect post-hoc).
//!
//! Mirrors the wrapper pattern in `host_fallback.rs`: copy flat slice into
//! `Mat`, run faer call, copy back out. No `unwrap` (see workspace clippy
//! lint `clippy::unwrap_used` warned at lib.rs).

use crate::error::AlgebraError;
use faer::{Mat, prelude::Solve, linalg::solvers::FullPivLu};

/// Solve `A·x = b` for `x`, where `A` is an `n×n` row-major flat slice and
/// `b` is an `n`-element rhs. Returns the solution as a `Vec<f64>` of length
/// `n`. Routes to faer 0.24 `FullPivLu` per ALG-05 host-fallback discipline.
///
/// # Errors
/// - [`AlgebraError::ShapeMismatch`] if `a.len() != n*n` or `b.len() != n`.
/// - [`AlgebraError::Singular`] if the LU solve produces non-finite values
///   (post-solve check; `FullPivLu::new` itself is panic-free per faer docs).
pub fn solve_linear(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>, AlgebraError> {
    if a.len() != n * n {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("{}x{}", n, n),
            actual:   format!("len={}", a.len()),
        });
    }
    if b.len() != n {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("rhs len={}", n),
            actual:   format!("len={}", b.len()),
        });
    }
    // Build faer `Mat` in row-major from the flat slice. `Mat::from_fn`
    // takes (rows, cols, |i, j| ...) — `a` is row-major so element (i, j)
    // lives at `a[i * n + j]`.
    let mat = Mat::<f64>::from_fn(n, n, |i, j| a[i * n + j]);
    let rhs = Mat::<f64>::from_fn(n, 1, |i, _| b[i]);
    // `FullPivLu::new` returns `Self` directly in faer 0.24 (no Result);
    // it does not panic on singular input but the solve produces non-finite
    // output. We detect that below.
    let lu = FullPivLu::new(mat.as_ref());
    let sol = lu.solve(rhs.as_ref());
    let out: Vec<f64> = (0..n).map(|i| sol[(i, 0)]).collect();
    if out.iter().any(|x| !x.is_finite()) {
        return Err(AlgebraError::Singular);
    }
    Ok(out)
}
