//! Small complex linear-algebra helpers shared by the periodic J/K builders.
//!
//! Two tiers, chosen by operand size:
//!
//! * [`zgemm`] routes to `pyscf_algebra::zgemm_dense` — the D-PBC-03 four-real-
//!   GEMM form on the active backend. Used for every contraction whose inner
//!   dimension is a grid axis.
//! * [`zmm_small`] is an in-order host fold. Used for the `nao x nao` algebra
//!   (density-matrix products, the `S D S` Ewald shift), where `nao` is 8 for
//!   the reference systems and a device launch costs more than the product.
//!   This is the same measured tradeoff plan 10-03 recorded for its Bloch
//!   contraction.

use pyscf_algebra::{AlgebraClient, CTensor, zgemm_dense};
use pyscf_core::{CoreError, PyscfRsError};

/// Device complex GEMM `C = A B`, `a` is `m x k`, `b` is `k x n`, row-major.
///
/// # Errors
/// [`CoreError::InvalidMolecule`] on a shape mismatch or a launch failure.
pub fn zgemm(
    client: &AlgebraClient,
    a: &CTensor,
    b: &CTensor,
    m: usize,
    k: usize,
    n: usize,
) -> Result<CTensor, PyscfRsError> {
    zgemm_dense(client, a, b, m, k, n).map_err(|e| {
        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "periodic JK: zgemm({m}x{k}x{n}) failed: {e}"
        )))
    })
}

/// Host complex GEMM `C = A B` for small operands, row-major.
///
/// The `k` loop runs in index order on a scalar accumulator, so the result is
/// reproducible across backends and thread counts (Pitfall 2).
pub fn zmm_small(a: &CTensor, b: &CTensor, m: usize, k: usize, n: usize) -> CTensor {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    let mut re = vec![0.0_f64; m * n];
    let mut im = vec![0.0_f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sr = 0.0_f64;
            let mut si = 0.0_f64;
            for t in 0..k {
                let (ar, ai) = (a.re[i * k + t], a.im[i * k + t]);
                let (br, bi) = (b.re[t * n + j], b.im[t * n + j]);
                sr += ar * br - ai * bi;
                si += ar * bi + ai * br;
            }
            re[i * n + j] = sr;
            im[i * n + j] = si;
        }
    }
    CTensor::from_planes(re, im)
}

/// Transpose of a row-major `rows x cols` complex matrix.
pub fn ztranspose(x: &CTensor, rows: usize, cols: usize) -> CTensor {
    debug_assert_eq!(x.len(), rows * cols);
    let mut re = vec![0.0_f64; x.len()];
    let mut im = vec![0.0_f64; x.len()];
    for i in 0..rows {
        for j in 0..cols {
            re[j * rows + i] = x.re[i * cols + j];
            im[j * rows + i] = x.im[i * cols + j];
        }
    }
    CTensor::from_planes(re, im)
}

/// `a += b`, element-wise.
pub fn zadd_assign(a: &mut CTensor, b: &CTensor) {
    debug_assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        a.re[i] += b.re[i];
        a.im[i] += b.im[i];
    }
}

/// `a += s * b` with a REAL scale.
pub fn zaxpy_real(a: &mut CTensor, s: f64, b: &CTensor) {
    debug_assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        a.re[i] += s * b.re[i];
        a.im[i] += s * b.im[i];
    }
}

/// `a *= s` with a REAL scale.
pub fn zscale_real(a: &mut CTensor, s: f64) {
    for v in a.re.iter_mut() {
        *v *= s;
    }
    for v in a.im.iter_mut() {
        *v *= s;
    }
}

/// `Tr(A B)` for row-major `n x n` operands — `einsum('ij,ji->', a, b)`.
pub fn ztrace_ab(a: &CTensor, b: &CTensor, n: usize) -> (f64, f64) {
    let mut sr = 0.0_f64;
    let mut si = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let (ar, ai) = (a.re[i * n + j], a.im[i * n + j]);
            let (br, bi) = (b.re[j * n + i], b.im[j * n + i]);
            sr += ar * br - ai * bi;
            si += ar * bi + ai * br;
        }
    }
    (sr, si)
}

/// Reinterpret a COLUMN-MAJOR (F-order) `ni x nj` complex matrix as ROW-MAJOR.
///
/// Every `pyscf-pbc-gto` Phase-10 product (`pbc_intor`, `get_ovlp`, `get_t`,
/// `get_pp_nl`) is F-order per component, while `pyscf_algebra::zeigh_gen` and
/// `zgemm_dense` are row-major. Phase 11 works ROW-MAJOR throughout and
/// converts once, here, on ingest. The conversion is a pure element move — no
/// arithmetic, so no rounding — and for the Hermitian matrices it is applied to
/// it is numerically the conjugate.
pub fn forder_to_c(m: &CTensor, ni: usize, nj: usize) -> CTensor {
    ztranspose(m, nj, ni)
}

/// The inverse of [`forder_to_c`] — ROW-MAJOR `ni x nj` back to F-order.
pub fn c_to_forder(m: &CTensor, ni: usize, nj: usize) -> CTensor {
    ztranspose(m, ni, nj)
}

/// Conjugate transpose of a row-major `n x n` matrix — the Hermitian
/// symmetriser's partner.
pub fn zconj_transpose_square(m: &CTensor, n: usize) -> CTensor {
    let t = ztranspose(m, n, n);
    CTensor::from_planes(t.re, t.im.into_iter().map(|v| -v).collect())
}

/// Replace `m` with `(m + mᴴ)/2`. Used where upstream relies on a `hermi=1`
/// flag to make a numerically-lopsided matrix exactly Hermitian.
pub fn hermitise(m: &mut CTensor, n: usize) {
    let h = zconj_transpose_square(m, n);
    for i in 0..m.len() {
        m.re[i] = 0.5 * (m.re[i] + h.re[i]);
        m.im[i] = 0.5 * (m.im[i] + h.im[i]);
    }
}
