//! Host symmetric/Hermitian eigendecomposition via `lapack_rs`.
//!
//! This module is the single bridge between pyscf-algebra and the
//! OpenBLAS-track transcription in `lapack_rs` (bit-exact to scipy's
//! bundled OpenBLAS, not Reference-LAPACK). It replaces the faer
//! `SelfAdjointEigen` calls on every *standard* (ordinary, non-generalized)
//! eigendecomposition path:
//!
//! * [`crate::host_fallback::eigh`] (opaque-Tensor surface),
//! * [`crate::df_metric_fit`] (DF 2-center metric),
//! * the two inner standard decompositions (`S`, then `F'`) of the
//!   L\"owdin generalized solvers [`crate::eigh_gen`] and
//!   [`crate::zeigh::zeigh_gen_faer`].
//!
//! The L\"owdin glue itself (canonical orthonormalization, `F' = X\u1d40 F X`,
//! `C = X V`) stays here: it is the DSYGV/ZHEGV-equivalent algorithm, which
//! `lapack_rs` does not implement yet. Likewise Cholesky, LU solve, QR, SVD
//! and the non-symmetric eig stay on faer (see the coverage plan in
//! `lapack_rs/docs/pyscf-lapack-coverage-plan.md`).
//!
//! Layout convention: callers pass ROW-MAJOR flat slices (the pyscf-algebra
//! flat-slice convention); LAPACK wants COLUMN-MAJOR with `LDA = n`, so each
//! entry point transposes into a column-major work buffer, calls
//! `openblas::dsyevd` / `openblas::zheevd` (`Job::Vectors`, `Uplo::Lower` — matching
//! the previous `Side::Lower` reads), and returns eigenvalues ASCENDING plus
//! eigenvectors in COLUMN-MAJOR / F-order (`v[i + j*n] = V[(i,j)]`), the
//! convention [`crate::host_fallback::eigh`] documents.

use crate::AlgebraError;
use lapack_rs::{Job, Uplo};

/// Map a `lapack_rs` failure onto the algebra error surface.
///
/// The enum has no LAPACK-specific variant, so failures reuse
/// `CubeclRuntime`, consistent with the previous faer EVD failure mapping.
fn map_err(context: &'static str, e: lapack_rs::LapackError) -> AlgebraError {
    AlgebraError::CubeclRuntime(format!("{context}: {e:?}"))
}

/// Output of [`hermitian_eigh`]: ascending eigenvalues plus the planar
/// column-major parts of the eigenvector matrix.
pub type HermitianEigh = (Vec<f64>, Vec<f64>, Vec<f64>);

/// Standard symmetric eigendecomposition of a row-major `n×n` matrix.
///
/// Returns `(eigenvalues_ascending, eigenvectors_column_major)` where
/// `eigenvectors_column_major[i + j*n] = V[(i,j)]` and `A·V = V·diag(w)`.
/// Only the lower triangle is read (`Uplo::Lower`), matching the previous
/// faer `Side::Lower` behavior.
///
/// # Errors
/// - [`AlgebraError::ShapeMismatch`] if `row_major.len() != n*n`.
/// - [`AlgebraError::CubeclRuntime`] if the LAPACK driver fails to converge.
pub fn sym_eigh(row_major: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>), AlgebraError> {
    if row_major.len() != n * n {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("n*n = {}", n * n),
            actual: format!("len={}", row_major.len()),
        });
    }
    // Row-major element (i, j) at `i*n + j` becomes column-major (i, j) at
    // `i + j*n`: same matrix, transposed storage.
    let mut col = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            col[i + j * n] = row_major[i * n + j];
        }
    }
    let mut w = vec![0.0_f64; n];
    lapack_rs::openblas::dsyevd(Job::Vectors, Uplo::Lower, n, &mut col, &mut w)
        .map_err(|e| map_err("lapack_rs dsyevd failed", e))?;
    // On return `col` holds the orthonormal eigenvectors column-major and
    // `w` the eigenvalues ascending — exactly the F-order contract.
    Ok((w, col))
}

/// Standard Hermitian eigendecomposition of planar row-major `n×n` parts.
///
/// `re_row` / `im_row` hold element (i, j) at `i*n + j`. Returns
/// `(eigenvalues_ascending, re_column_major, im_column_major)` with
/// `A·V = V·diag(w)` and `Vᴴ·V = I`. Only the lower triangle is read
/// (`Uplo::Lower`), matching the previous faer `Side::Lower` behavior.
///
/// # Errors
/// - [`AlgebraError::ShapeMismatch`] if either part has length `!= n*n`.
/// - [`AlgebraError::CubeclRuntime`] if the LAPACK driver fails to converge.
pub fn hermitian_eigh(
    re_row: &[f64],
    im_row: &[f64],
    n: usize,
) -> Result<HermitianEigh, AlgebraError> {
    if re_row.len() != n * n {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("re n*n = {}", n * n),
            actual: format!("len={}", re_row.len()),
        });
    }
    if im_row.len() != n * n {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("im n*n = {}", n * n),
            actual: format!("len={}", im_row.len()),
        });
    }
    let mut re_col = vec![0.0_f64; n * n];
    let mut im_col = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            re_col[i + j * n] = re_row[i * n + j];
            im_col[i + j * n] = im_row[i * n + j];
        }
    }
    let mut w = vec![0.0_f64; n];
    lapack_rs::openblas::zheevd(Job::Vectors, Uplo::Lower, n, &mut re_col, &mut im_col, &mut w)
        .map_err(|e| map_err("lapack_rs zheevd failed", e))?;
    Ok((w, re_col, im_col))
}
