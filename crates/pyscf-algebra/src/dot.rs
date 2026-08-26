//! DOT — generic-float cubecl reduction kernel + backend-dispatched host launcher.
//!
//! quick-260529-iji: refactored from the Phase-1 `NotYetImplemented` stub into
//! a real cubecl `#[cube(launch)]` kernel generic over the device float
//! `F: Float` (per docs/manual/Cubecl/Cubecl_generics.md), plus a host-slice
//! launcher dispatched off `AlgebraClient` so the cubecl `Runtime` generic
//! stays inside the ALG-06 wall.
//!
//! `dot(x, y) = sum(x[i] * y[i])` is the reduction sibling of GEMM (`gemm.rs`).
//! The device kernel computes the per-element products one-thread-per-element;
//! the final reduction sum runs on the host in `F`, keeping the kernel naive
//! and obviously-correct (no device atomics / tree-reduction).
//!
//! Two surfaces:
//!   * `dot()` — the opaque `Tensor`-based API. STILL a stub: `Tensor` carries
//!     a sentinel `BufferId` (no device allocator until Phase 2), so it cannot
//!     read device data yet.
//!   * `dot_dense()` — the working device path: takes host slices + an
//!     `AlgebraClient`, runs the generic kernel on the resolved backend (CPU,
//!     ROCm, CUDA, WGPU), and returns the scalar result. Exercised by the
//!     random oracle differential test (`tests/dot_oracle.rs`) on ROCm.

use crate::launch::{launch_1d, line_size_for, reduction_lanes, total_units, upload};
use crate::scalar::DeviceScalar;
use crate::{AlgebraClient, AlgebraError, Tensor};
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::server::Handle;

/// Vectorized grid-stride dot kernel: lane `tid` accumulates the products of the
/// `N`-wide vectors `x[tid + j*stride] * y[tid + j*stride]` and writes the
/// horizontal sum of its accumulator to `out[tid]`.
///
/// quick-260826-spd: the operands are now read as [`Vector<F, N>`], so each step
/// of the stride loop is one SIMD load per operand and one vector FMA rather
/// than a scalar pair. Adjacent lanes still read adjacent vectors, so the access
/// pattern stays fully coalesced.
///
/// The lane count is capped by [`reduction_lanes`] rather than derived from the
/// element count: one unit per element would emit one partial per element, which
/// is not a reduction at all. `stride` is the true total unit count of the grid,
/// passed in rather than recomputed from `CUBE_COUNT_X * CUBE_DIM_X` — that
/// product is only the whole grid when the cube count is one-dimensional, and
/// [`launch_1d`] is free to fold a large count onto the y axis.
///
/// The final horizontal fold is one `vector_sum` per lane, not per element, so
/// its cost is negligible even though float horizontal reduction lowers to a
/// straight-line chain (it is not associative, so no backend may re-tree it).
#[cube(launch_unchecked)]
fn dot_kernel<F: Float + CubeElement, N: Size>(
    x: &Array<Vector<F, N>>,
    y: &Array<Vector<F, N>>,
    out: &mut Array<F>,
    n_lines: usize,
    stride: usize,
) {
    let tid = ABSOLUTE_POS;
    if tid < stride {
        let mut acc = Vector::<F, N>::new(F::from_int(0));
        let mut i = tid;
        while i < n_lines {
            acc += x[i] * y[i];
            i += stride;
        }
        out[tid] = F::cast_from(acc.vector_sum());
    }
}

/// Core launch on resident device handles: compute the partial sums of the
/// element-wise product of `x`/`y` (length `n`) into a temporary device buffer,
/// read them back, and sum in `F`. No re-upload of `x`/`y` — both the host-slice
/// path ([`launch_dot`]) and the registry-backed `Tensor` path ([`dot`]) drive this.
fn launch_dot_on_handles<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    x: &Handle,
    y: &Handle,
    n: usize,
) -> F {
    if n == 0 {
        return F::from_int(0);
    }
    let line = line_size_for::<R, F>(client, n);
    let n_lines = n / line;
    let lanes = reduction_lanes(client, n_lines, line);
    let (count, dim) = launch_1d(client, lanes, (n_lines / lanes.max(1)) * line);
    // The grid may round up past `lanes`; every dispatched unit writes a partial,
    // so the buffer is sized from the geometry rather than from the request.
    let stride = total_units(&count, dim);
    let out_handle = client.empty(stride * core::mem::size_of::<F>());

    unsafe {
        dot_kernel::launch_unchecked::<F, R>(
            client,
            count,
            dim,
            line,
            // SAFETY: `n` matches the operand buffers and `stride` matches out_handle.
            ArrayArg::from_raw_parts(x.clone(), n),
            ArrayArg::from_raw_parts(y.clone(), n),
            ArrayArg::from_raw_parts(out_handle.clone(), stride),
            // Scalar kernel args are passed as bare values (LaunchArg for T = T).
            n_lines,
            stride,
        );
    }

    let bytes = client.read(vec![out_handle]);
    let partials: &[F] = bytemuck::cast_slice::<u8, F>(&bytes[0]);

    // Host-sum the partials in `F`. A plain `acc += p` add is NOT an FMA
    // (check-no-fma only flags FMA in pyscf_* symbols anyway).
    let mut acc = F::from_int(0);
    for &p in partials {
        acc += p;
    }
    acc
}

/// Host-slice launcher: upload `x`/`y`, then reduce on the device. Backs
/// `dot_dense`; the cubecl `Runtime` bound stays inside.
fn launch_dot<R: Runtime, F: DeviceScalar>(client: &ComputeClient<R>, x: &[F], y: &[F]) -> F {
    if x.is_empty() {
        return F::from_int(0);
    }
    let x_handle = upload(client, x);
    let y_handle = upload(client, y);
    launch_dot_on_handles::<R, F>(client, &x_handle, &y_handle, x.len())
}

/// Dense dot product on the device: `dot(x, y) = sum(x[i] * y[i])`.
///
/// Generic over `F: DeviceScalar` (f32 or f64). Validates that `x` and `y` have
/// equal length, then dispatches the generic cubecl kernel on whichever backend
/// `client` resolved to — the `Runtime` type is selected here and never appears
/// in the signature, honoring the ALG-06 wall.
pub fn dot_dense<F: DeviceScalar>(
    client: &AlgebraClient,
    x: &[F],
    y: &[F],
) -> Result<F, AlgebraError> {
    if x.len() != y.len() {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("y len {}", x.len()),
            actual: y.len().to_string(),
        });
    }

    let out = dispatch_backend!(client, c, Rt, launch_dot::<Rt, F>(c, x, y));
    Ok(out)
}

/// Dot reduction over the opaque `Tensor` surface: `dot(x, y) = sum(x[i]*y[i])`.
///
/// quick-260529-mtx-2: reduces directly on the operands' resident device handles
/// via the Phase-2 [`crate::device_buffer`] registry — `x`/`y` are not re-uploaded.
/// Both must be device-backed tensors built with [`crate::device_buffer::upload`];
/// a `Tensor::placeholder` (sentinel `BufferId`) yields
/// [`AlgebraError::UnallocatedBuffer`], and a buffer resident on another backend
/// yields [`AlgebraError::BackendMismatch`]. Pure reduction — no buffer is
/// modified. Empty operands return `0.0`.
pub fn dot(client: &AlgebraClient, x: &Tensor, y: &Tensor) -> Result<f64, AlgebraError> {
    let xb = crate::device_buffer::handle_of::<f64>(x.id.raw(), client, "dot")?;
    let yb = crate::device_buffer::handle_of::<f64>(y.id.raw(), client, "dot")?;
    if xb.len != yb.len {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("y len {}", xb.len),
            actual: yb.len.to_string(),
        });
    }
    if xb.len == 0 {
        return Ok(0.0);
    }
    let n = xb.len;
    let result = dispatch_backend!(
        client,
        c,
        Rt,
        launch_dot_on_handles::<Rt, f64>(c, &xb.handle, &yb.handle, n)
    );
    Ok(result)
}
