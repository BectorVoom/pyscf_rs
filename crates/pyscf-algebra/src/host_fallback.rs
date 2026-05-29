//! Host-fallback dense linear algebra (ALG-05). All four decompositions —
//! `eigh`/`cholesky`/`qr`/`svd` — route to faer 0.24 on host. On a GPU
//! `AlgebraClient`, the bodies copy the operand down with
//! `device_buffer::download::<f64>`, run faer on host, and `device_buffer::upload`
//! the factor(s) back (Vec<f64> round-trip per RESEARCH §9 + faer-ext Pitfall 3).
//! This module names ONLY `device_buffer::{download,upload}`, `faer::*`, and
//! `AlgebraError` — never a `cubecl::*` type (ALG-06 wall).
//!
//! quick-260529-oj6: refactored the four Phase-1 `NotYetImplemented` stubs into
//! real faer-backed bodies (mirroring `eigh_gen.rs` / `solve_linear.rs` wrapper
//! discipline: download → `Mat::from_fn` row-major → decompose → flat-Vec →
//! upload). The four public signatures are LOCKED (re-exported at `lib.rs:63`).
//!
//! Output layout conventions (DOCUMENTED and asserted by the oracle tests):
//!   * `eigh`: eigenvalues ASCENDING (as faer `SelfAdjointEigen` returns);
//!     eigenvectors in **column-major / F-order** (`evecs[i + j*n] = U[(i,j)]`),
//!     matching `eigh_gen.rs`'s `MOCoefficients` F-order convention.
//!   * `cholesky`: lower-triangular factor `L` in **row-major** order
//!     (`out[i*n + j] = L[(i,j)]`); the strict upper triangle is zero. Non-PD
//!     input is the documented failure mode → `AlgebraError::CubeclRuntime`.
//!   * `qr`: `Q` and `R` factors both in **row-major** order; `R` is
//!     upper-triangular, `Q` orthonormal.
//!   * `svd`: singular values DESCENDING and ≥ 0; `U` and `V` factors both in
//!     **row-major** order. `A ≈ U·diag(s)·Vᵀ`.
//!
//! All four currently require a **square** `n×n` Tensor (the Tensor surface
//! carries one `shape` only; rectangular qr/svd is a separate future change) —
//! a non-square or non-rank-2 shape yields `AlgebraError::ShapeMismatch`.

use crate::device_buffer;
use crate::{AlgebraClient, AlgebraError, Tensor};
use faer::linalg::solvers::SelfAdjointEigen;
use faer::{Mat, Side};

/// Validate that `matrix` is a rank-2 square `n×n` Tensor and return `n`.
/// Mirrors the flat-slice shape guards in `eigh_gen.rs` / `solve_linear.rs`.
fn square_n(matrix: &Tensor) -> Result<usize, AlgebraError> {
    if matrix.shape.len() != 2 || matrix.shape[0] != matrix.shape[1] {
        return Err(AlgebraError::ShapeMismatch {
            expected: "square n*n matrix".into(),
            actual: format!("shape={:?}", matrix.shape),
        });
    }
    Ok(matrix.shape[0])
}

/// Download `matrix` to a host `Vec<f64>` and belt-and-suspenders-check its
/// length against `n*n` (mirrors `eigh_gen.rs:53-64`).
fn download_square(
    client: &AlgebraClient,
    matrix: &Tensor,
    n: usize,
) -> Result<Vec<f64>, AlgebraError> {
    let data = device_buffer::download::<f64>(client, matrix)?;
    if data.len() != n * n {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("n*n = {}", n * n),
            actual: format!("len={}", data.len()),
        });
    }
    Ok(data)
}

/// Self-adjoint eigendecomposition. Returns `(eigenvalues, eigenvectors)`.
///
/// Eigenvalues are ASCENDING (faer `SelfAdjointEigen`); the eigenvector Tensor
/// is **column-major / F-order** (`evecs[i + j*n] = U[(i,j)]`), matching the
/// `eigh_gen.rs` MO-coefficient convention. Routes to faer on host via a
/// `device_buffer` Vec<f64> round-trip (ALG-05); requires a square `n×n` input.
///
/// # Errors
/// - [`AlgebraError::ShapeMismatch`] if `matrix` is not rank-2 square or the
///   downloaded length disagrees with `n*n`.
/// - [`AlgebraError::CubeclRuntime`] on faer evd failure (rare; usually a
///   non-self-adjoint input — only the lower triangle is read, `Side::Lower`).
pub fn eigh(client: &AlgebraClient, matrix: &Tensor) -> Result<(Vec<f64>, Tensor), AlgebraError> {
    let n = square_n(matrix)?;
    let data = download_square(client, matrix, n)?;

    // Row-major flat slice: element (i, j) at data[i*n + j].
    let m = Mat::<f64>::from_fn(n, n, |i, j| data[i * n + j]);
    let evd = SelfAdjointEigen::new(m.as_ref(), Side::Lower)
        .map_err(|e| AlgebraError::CubeclRuntime(format!("eigh failed: {e:?}")))?;

    // Eigenvalues ASCENDING (as faer returns).
    let eigenvalues: Vec<f64> = (0..n).map(|k| evd.S()[k]).collect();

    // Eigenvectors back to a flat Vec in column-major / F-order.
    let evecs_mat = evd.U();
    let mut evecs = vec![0.0_f64; n * n];
    for j in 0..n {
        for i in 0..n {
            evecs[i + j * n] = evecs_mat[(i, j)];
        }
    }
    let evec_tensor = device_buffer::upload::<f64>(client, &evecs, vec![n, n])?;

    Ok((eigenvalues, evec_tensor))
}

/// Cholesky LLT (positive-definite only). Returns the lower-triangular factor
/// `L` (`A = L·Lᵀ`) as a **row-major** Tensor (`out[i*n + j] = L[(i,j)]`); the
/// strict upper triangle is zero. Routes to faer on host via a `device_buffer`
/// Vec<f64> round-trip (ALG-05); requires a square `n×n` input.
///
/// # Errors
/// - [`AlgebraError::ShapeMismatch`] if `matrix` is not rank-2 square or the
///   downloaded length disagrees with `n*n`.
/// - [`AlgebraError::CubeclRuntime`] if the matrix is not positive-definite
///   (the documented failure mode; the enum has no dedicated NotPositiveDefinite
///   variant — reuse `CubeclRuntime`, consistent with `eigh_gen`).
pub fn cholesky(client: &AlgebraClient, matrix: &Tensor) -> Result<Tensor, AlgebraError> {
    let n = square_n(matrix)?;
    let data = download_square(client, matrix, n)?;

    let m = Mat::<f64>::from_fn(n, n, |i, j| data[i * n + j]);
    let llt = m.as_ref().llt(Side::Lower).map_err(|e| {
        AlgebraError::CubeclRuntime(format!(
            "cholesky failed (matrix not positive-definite?): {e:?}"
        ))
    })?;

    // faer `L` is a column-major Mat; read element (i, j) and store row-major.
    // Upper triangle of L is zero → `out` is lower-triangular.
    let l = llt.L();
    let mut out = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = l[(i, j)];
        }
    }

    device_buffer::upload::<f64>(client, &out, vec![n, n])
}

/// QR (no pivot). Phase 6 (CCSD intermediate canonicalization) wires.
pub fn qr(_client: &AlgebraClient, _matrix: &Tensor) -> Result<(Tensor, Tensor), AlgebraError> {
    Err(AlgebraError::NotYetImplemented {
        phase: 6,
        what: "qr — Phase 6 wires faer::Mat::qr",
    })
}

/// SVD (full). Phase 7 (gradient null-space projection) wires.
pub fn svd(
    _client: &AlgebraClient,
    _matrix: &Tensor,
) -> Result<(Tensor, Vec<f64>, Tensor), AlgebraError> {
    Err(AlgebraError::NotYetImplemented {
        phase: 7,
        what: "svd — Phase 7 wires faer::Mat::svd",
    })
}
