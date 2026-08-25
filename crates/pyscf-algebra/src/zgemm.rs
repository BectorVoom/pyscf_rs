//! Complex dense GEMM expressed as FOUR real GEMMs — D-PBC-03.
//!
//! # Why four multiplications and not three
//!
//! The 3-multiplication Karatsuba form
//! (`t = (Ar+Ai)(Br+Bi)`, `re = ArBr − AiBi`, `im = t − ArBr − AiBi`) is
//! algebraically identical but NOT bit-identical in floating point: it folds
//! `Ar·Bi` and `Ai·Br` into a difference of three other products, so the
//! rounding of the imaginary plane depends on the magnitude of the real plane.
//! D-PBC-03 fixes the schoolbook 4-GEMM form so that:
//!
//!   * every periodic result is reproducible bit-for-bit across runs, thread
//!     counts and backends (each of the four products is a call into the SAME
//!     `gemm_dense` the molecular code path already uses and already validates);
//!   * a k-point result whose imaginary part cancels to zero (Γ-point, or a
//!     k/−k pair) cancels EXACTLY, because `im = Ar·Bi + Ai·Br` is a plain sum
//!     of the two products rather than a three-term cancellation;
//!   * a complex GEMM against a real operand degenerates to exactly the real
//!     GEMM result — `t2`/`t3` are then products with an all-zero plane.
//!
//! The evaluation ORDER is also load-bearing and must not be permuted, fused,
//! or short-circuited on an all-zero plane:
//!
//! ```text
//! t1 = gemm_dense(a.re, b.re)
//! t2 = gemm_dense(a.im, b.im)
//! t3 = gemm_dense(a.re, b.im)
//! t4 = gemm_dense(a.im, b.re)
//! re[i] = t1[i] - t2[i]
//! im[i] = t3[i] + t4[i]
//! ```

use crate::complex::CTensor;
use crate::{AlgebraClient, AlgebraError, gemm_dense, transpose_dense};

/// Complex dense matrix multiply `C = A · B`, row-major.
/// `a` is `m × k`, `b` is `k × n`; the result is `m × n`.
///
/// Implemented as the four real GEMMs of D-PBC-03, in the mandated order (see
/// the module docs). Do NOT reorder, fuse, or replace with the
/// 3-multiplication Karatsuba form: the bit-parity guarantee depends on this
/// exact expression.
pub fn zgemm_dense(
    client: &AlgebraClient,
    a: &CTensor,
    b: &CTensor,
    m: usize,
    k: usize,
    n: usize,
) -> Result<CTensor, AlgebraError> {
    if a.len() != m * k {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("a len {} (= {m}*{k})", m * k),
            actual: a.len().to_string(),
        });
    }
    if b.len() != k * n {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("b len {} (= {k}*{n})", k * n),
            actual: b.len().to_string(),
        });
    }

    // D-PBC-03 — the four real GEMMs, in this exact order.
    let t1 = gemm_dense::<f64>(client, &a.re, &b.re, m, k, n)?;
    let t2 = gemm_dense::<f64>(client, &a.im, &b.im, m, k, n)?;
    let t3 = gemm_dense::<f64>(client, &a.re, &b.im, m, k, n)?;
    let t4 = gemm_dense::<f64>(client, &a.im, &b.re, m, k, n)?;

    let mut re = vec![0.0_f64; m * n];
    let mut im = vec![0.0_f64; m * n];
    for i in 0..m * n {
        re[i] = t1[i] - t2[i];
        im[i] = t3[i] + t4[i];
    }
    Ok(CTensor::from_planes(re, im))
}

/// Conjugate-transposed complex dense matrix multiply `C = Aᴴ · B`, row-major.
///
/// `a` is the UN-transposed `k × m` operand, `b` is `k × n`; the result is
/// `m × n`. `Aᴴ` is materialised explicitly — [`transpose_dense`] on each plane
/// followed by negating the imaginary plane — and then handed to
/// [`zgemm_dense`]. Per D-PBC-03 this must NOT be fused into the four GEMMs
/// (e.g. by flipping signs on `t2`/`t3`): keeping `Aᴴ` explicit means the
/// resulting products are the same `gemm_dense` calls with the same operand
/// values as a caller who transposed by hand, so the two routes agree bit-for-bit.
pub fn zgemm_h_dense(
    client: &AlgebraClient,
    a: &CTensor,
    b: &CTensor,
    m: usize,
    k: usize,
    n: usize,
) -> Result<CTensor, AlgebraError> {
    if a.len() != k * m {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("a len {} (= {k}*{m}, the un-transposed operand)", k * m),
            actual: a.len().to_string(),
        });
    }

    // Materialise Aᴴ: transpose both planes (k×m -> m×k), then negate `im`.
    let ah_re = transpose_dense::<f64>(client, &a.re, k, m)?;
    let ah_im_t = transpose_dense::<f64>(client, &a.im, k, m)?;
    let ah_im: Vec<f64> = ah_im_t.iter().map(|v| -v).collect();
    let ah = CTensor::from_planes(ah_re, ah_im);

    zgemm_dense(client, &ah, b, m, k, n)
}
