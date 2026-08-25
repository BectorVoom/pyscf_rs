//! Complex BLAS-1/2 surface — PBC-MASTER-PLAN §5.2.
//!
//! Every routine here is expressed in terms of the EXISTING real cubecl
//! primitives (`axpy_dense`, `dot_dense`, `reduce_sum_dense`,
//! `transpose_dense`), so no new numeric type crosses the ALG-06 wall
//! (D-PBC-02 / RULE 8). The one exception is [`zhadamard_dense`]: there is no
//! real element-wise-multiply engine to build on, so it carries its own cubecl
//! kernel — see the note on that function.
//!
//! Like `zgemm`, the decompositions below use the schoolbook forms and a FIXED
//! evaluation order. Do not permute the calls or fold them into fewer
//! operations: bit-reproducibility across runs, thread counts and backends is
//! the acceptance criterion for the whole periodic milestone, and every
//! reassociation changes the low bits.

use crate::complex::CTensor;
use crate::scalar::DeviceScalar;
use crate::{
    AlgebraClient, AlgebraError, axpy_dense, dot_dense, reduce_sum_dense, transpose_dense,
};
use cubecl::Runtime;
use cubecl::bytes::Bytes;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;

/// Shared length guard for the two-operand routines.
fn same_len(op: &'static str, x: &CTensor, y: &CTensor) -> Result<usize, AlgebraError> {
    if x.len() != y.len() {
        return Err(AlgebraError::DimensionMismatch {
            op,
            lhs: vec![x.len()],
            rhs: vec![y.len()],
        });
    }
    Ok(x.len())
}

/// `y += alpha * x` for complex `alpha = (re, im)`.
///
/// Four real `axpy_dense` calls, in this order:
/// ```text
/// y.re += ar * x.re
/// y.re += (-ai) * x.im
/// y.im += ar * x.im
/// y.im += ai * x.re
/// ```
/// i.e. `y.re += ar*x.re − ai*x.im` and `y.im += ar*x.im + ai*x.re`. `x` is
/// never mutated, so the two planes of `y` may be updated independently.
pub fn zaxpy_dense(
    client: &AlgebraClient,
    alpha: (f64, f64),
    x: &CTensor,
    y: &mut CTensor,
) -> Result<(), AlgebraError> {
    let n = same_len("zaxpy", x, y)?;
    if n == 0 {
        return Ok(());
    }
    let (ar, ai) = alpha;
    axpy_dense::<f64>(client, ar, &x.re, &mut y.re)?;
    axpy_dense::<f64>(client, -ai, &x.im, &mut y.re)?;
    axpy_dense::<f64>(client, ar, &x.im, &mut y.im)?;
    axpy_dense::<f64>(client, ai, &x.re, &mut y.im)?;
    Ok(())
}

/// `x *= alpha` for complex `alpha = (re, im)`, in place.
///
/// Implemented "as [`zaxpy_dense`] with a temp" (§5.2): the original value is
/// snapshotted, `x` is zeroed, and the scaled snapshot is accumulated back.
/// Because the accumulator starts at exactly `0.0`, each `axpy` contributes
/// `0 + alpha*x`, which is the product with no extra rounding — so this agrees
/// bit-for-bit with a hand-written complex multiply.
pub fn zscal_dense(
    client: &AlgebraClient,
    alpha: (f64, f64),
    x: &mut CTensor,
) -> Result<(), AlgebraError> {
    if x.is_empty() {
        return Ok(());
    }
    let src = x.clone();
    x.re.iter_mut().for_each(|v| *v = 0.0);
    x.im.iter_mut().for_each(|v| *v = 0.0);
    zaxpy_dense(client, alpha, &src, x)
}

/// Conjugated inner product `xᴴ · y = Σ conj(x[i]) * y[i]`.
///
/// `re = dot(xr,yr) + dot(xi,yi)`; `im = dot(xr,yi) − dot(xi,yr)`.
///
/// NOTE: this is the DEVICE reduction. It is fast but its summation order
/// depends on the launch geometry. Numerical PBC code that must be
/// thread-count-invariant uses [`crate::zoracle::oracle_zdot`] instead
/// (D-PBC-17).
pub fn zdotc_dense(
    client: &AlgebraClient,
    x: &CTensor,
    y: &CTensor,
) -> Result<(f64, f64), AlgebraError> {
    let n = same_len("zdotc", x, y)?;
    if n == 0 {
        return Ok((0.0, 0.0));
    }
    let rr = dot_dense::<f64>(client, &x.re, &y.re)?;
    let ii = dot_dense::<f64>(client, &x.im, &y.im)?;
    let ri = dot_dense::<f64>(client, &x.re, &y.im)?;
    let ir = dot_dense::<f64>(client, &x.im, &y.re)?;
    Ok((rr + ii, ri - ir))
}

/// Un-conjugated inner product `xᵀ · y = Σ x[i] * y[i]`.
///
/// `re = dot(xr,yr) − dot(xi,yi)`; `im = dot(xr,yi) + dot(xi,yr)`.
/// Device reduction — see the ordering note on [`zdotc_dense`].
pub fn zdotu_dense(
    client: &AlgebraClient,
    x: &CTensor,
    y: &CTensor,
) -> Result<(f64, f64), AlgebraError> {
    let n = same_len("zdotu", x, y)?;
    if n == 0 {
        return Ok((0.0, 0.0));
    }
    let rr = dot_dense::<f64>(client, &x.re, &y.re)?;
    let ii = dot_dense::<f64>(client, &x.im, &y.im)?;
    let ri = dot_dense::<f64>(client, &x.re, &y.im)?;
    let ir = dot_dense::<f64>(client, &x.im, &y.re)?;
    Ok((rr - ii, ri + ir))
}

/// `Σ x[i]`, as two real `reduce_sum_dense` calls.
///
/// Device reduction — see the ordering note on [`zdotc_dense`]; the
/// thread-count-invariant sibling is [`crate::zoracle::oracle_zsum`].
pub fn zreduce_sum_dense(client: &AlgebraClient, x: &CTensor) -> Result<(f64, f64), AlgebraError> {
    let re = reduce_sum_dense::<f64>(client, &x.re)?;
    let im = reduce_sum_dense::<f64>(client, &x.im)?;
    Ok((re, im))
}

/// Plain (NON-conjugating) 2-D transpose of a row-major `rows × cols` complex
/// matrix, as two real `transpose_dense` calls. The result is `cols × rows`.
///
/// For the conjugate transpose, call [`CTensor::conj`] on the result (or use
/// [`crate::zgemm::zgemm_h_dense`], which fuses neither).
pub fn ztranspose_dense(
    client: &AlgebraClient,
    x: &CTensor,
    rows: usize,
    cols: usize,
) -> Result<CTensor, AlgebraError> {
    if x.len() != rows * cols {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("x len {} (= {rows}*{cols})", rows * cols),
            actual: x.len().to_string(),
        });
    }
    let re = transpose_dense::<f64>(client, &x.re, rows, cols)?;
    let im = transpose_dense::<f64>(client, &x.im, rows, cols)?;
    Ok(CTensor::from_planes(re, im))
}

// ---------------------------------------------------------------------------
// K-04 — element-wise complex multiply.
// ---------------------------------------------------------------------------

/// K-04 kernel, planar layout, one thread per element.
///
/// This is the IN-WALL mirror of `pyscf_kernels::pbc::zhadamard::zhadamard_kernel`.
/// The canonical K-04 kernel lives in `pyscf-kernels` per PBC-MASTER-PLAN §6 /
/// RULE 6, but `pyscf-kernels` DEPENDS ON this crate, so `zhadamard_dense`
/// cannot call it without creating a dependency cycle. Both copies are
/// byte-identical `#[cube(launch_unchecked)] fn …<F: Float>` bodies with the
/// same schoolbook 4-multiply arithmetic and the same `i < n` tail guard, so
/// they produce bit-identical results; keep them in lockstep if either changes.
#[cube(launch_unchecked)]
fn zhadamard_kernel<F: Float>(
    ar: &Array<F>,
    ai: &Array<F>,
    br: &Array<F>,
    bi: &Array<F>,
    cr: &mut Array<F>,
    ci: &mut Array<F>,
    n: usize,
) {
    let i = ABSOLUTE_POS;
    if i < n {
        cr[i] = ar[i] * br[i] - ai[i] * bi[i];
        ci[i] = ar[i] * bi[i] + ai[i] * br[i];
    }
}

/// Threads per cube for the zhadamard launch.
const ZHADAMARD_BLOCK: u32 = 256;

/// Host-slice launcher: upload the four input planes, allocate the two output
/// planes, run the kernel, read both back in ONE batched `client.read`.
fn launch_zhadamard<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    ar: &[F],
    ai: &[F],
    br: &[F],
    bi: &[F],
) -> (Vec<F>, Vec<F>) {
    let n = ar.len();
    let ar_h = client.create(Bytes::from_elems(ar.to_vec()));
    let ai_h = client.create(Bytes::from_elems(ai.to_vec()));
    let br_h = client.create(Bytes::from_elems(br.to_vec()));
    let bi_h = client.create(Bytes::from_elems(bi.to_vec()));
    let cr_h = client.empty(core::mem::size_of_val(ar));
    let ci_h = client.empty(core::mem::size_of_val(ar));
    let groups = (n as u32).div_ceil(ZHADAMARD_BLOCK);

    unsafe {
        zhadamard_kernel::launch_unchecked::<F, R>(
            client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(ZHADAMARD_BLOCK),
            // SAFETY: every handle is `n` elements of `F`; the kernel guards `i < n`.
            ArrayArg::from_raw_parts(ar_h, n),
            ArrayArg::from_raw_parts(ai_h, n),
            ArrayArg::from_raw_parts(br_h, n),
            ArrayArg::from_raw_parts(bi_h, n),
            ArrayArg::from_raw_parts(cr_h.clone(), n),
            ArrayArg::from_raw_parts(ci_h.clone(), n),
            // Scalar dimension arg is passed as a bare value (LaunchArg for usize).
            n,
        );
    }

    let bytes = client.read(vec![cr_h, ci_h]);
    (
        bytemuck::cast_slice::<u8, F>(&bytes[0]).to_vec(),
        bytemuck::cast_slice::<u8, F>(&bytes[1]).to_vec(),
    )
}

/// Element-wise complex multiply `c[i] = x[i] * y[i]` (K-04).
///
/// The only §5.2 routine with no real-primitive decomposition; it runs the
/// dedicated cubecl kernel above. Empty input returns an empty tensor without
/// launching.
pub fn zhadamard_dense(
    client: &AlgebraClient,
    x: &CTensor,
    y: &CTensor,
) -> Result<CTensor, AlgebraError> {
    let n = same_len("zhadamard", x, y)?;
    if n == 0 {
        return Ok(CTensor::zeros(0));
    }
    let (re, im) = dispatch_backend!(
        client,
        c,
        Rt,
        launch_zhadamard::<Rt, f64>(c, &x.re, &x.im, &y.re, &y.im)
    );
    Ok(CTensor::from_planes(re, im))
}
