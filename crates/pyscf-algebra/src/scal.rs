//! SCAL — generic-float cubecl element-wise scale kernel + backend-dispatched host launcher.
//!
//! quick-260529-skl: refactored from the Phase-1 `NotYetImplemented` stub into
//! a real cubecl `#[cube(launch)]` kernel generic over the device float
//! `F: Float` (per docs/manual/Cubecl/Cubecl_generics.md), plus a host-slice
//! launcher dispatched off `AlgebraClient` so the cubecl `Runtime` generic
//! stays inside the ALG-06 wall. Mirrors the `dot.rs` (quick-260529-iji) and
//! `reduce.rs` (quick-260529-jcx) siblings exactly.
//!
//! `scal(alpha, x) = x[i] *= alpha` is the element-wise BLAS-1 scale: no
//! reduction, one thread per element, output the same shape as the input. The
//! device kernel scales in place (bounds-guarded against the launch tail); the
//! launcher reads the scaled buffer back. This keeps the kernel naive and
//! obviously-correct — a single multiply per element, no atomics.
//!
//! Two surfaces:
//!   * `scal()` — the opaque `Tensor`-based API. STILL a stub: `Tensor` carries
//!     a sentinel `BufferId` (no device allocator until Phase 2), so it cannot
//!     read device data yet.
//!   * `scal_dense()` — the working device path: takes a host slice + an
//!     `AlgebraClient`, runs the generic kernel on the resolved backend (CPU,
//!     ROCm, CUDA, WGPU), and scales the slice in place. Exercised by the random
//!     oracle differential test (`tests/scal_oracle.rs`) on ROCm.

use crate::launch::{launch_1d, line_size_for, upload};
use crate::scalar::DeviceScalar;
use crate::{AlgebraClient, AlgebraError, Tensor};
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::server::Handle;

/// Vectorized scale kernel: `x[i] *= alpha`, one unit per `N`-wide vector.
///
/// quick-260826-spd: the AXPY sibling's rewrite applies verbatim here — SIMD
/// vectors in place of a manual 4x scalar unroll, device-derived launch geometry
/// in place of a fixed 256-unit cube, and `alpha` as a real scalar launch
/// argument in place of a one-element device buffer allocated on every call.
/// See [`crate::axpy`] for the reasoning behind each.
#[cube(launch_unchecked)]
fn scal_kernel<F: Float + CubeElement, N: Size>(
    alpha: F,
    x: &mut Array<Vector<F, N>>,
    n_lines: usize,
) {
    if ABSOLUTE_POS < n_lines {
        x[ABSOLUTE_POS] *= Vector::<F, N>::new(alpha);
    }
}

/// Core launch on a resident device handle: `x[i] *= alpha`, in place on the `x`
/// handle, with NO host transfer. Both the host-slice path ([`launch_scal`]) and
/// the registry-backed `Tensor` path ([`scal`]) drive this. `R` stays generic so
/// a single body serves every backend; clones share the device binding, so the
/// in-place scale is visible through the registry's stored handle.
fn launch_scal_on_handle<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    alpha: F,
    x: &Handle,
    n: usize,
) {
    let line = line_size_for::<R, F>(client, n);
    let n_lines = n / line;
    // One multiply per element: the per-lane work is exactly the width.
    let (count, dim) = launch_1d(client, n_lines, line);

    unsafe {
        scal_kernel::launch_unchecked::<F, R>(
            client,
            count,
            dim,
            line,
            alpha,
            // SAFETY: `n` matches the buffer. `from_raw_parts` consumes the handle by
            // value, so clone the caller's handle (clones share the binding).
            ArrayArg::from_raw_parts(x.clone(), n),
            // Scalar dimension arg is passed as a bare value (LaunchArg for usize).
            n_lines,
        );
    }
}

/// Host-slice launcher: upload `x`, scale it in place on the device, read it
/// back. Backs `scal_dense`; the cubecl `Runtime` bound stays inside.
fn launch_scal<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    alpha: F,
    x: &[F],
) -> Vec<F> {
    let x_handle = upload(client, x);
    launch_scal_on_handle::<R, F>(client, alpha, &x_handle, x.len());
    let bytes = client.read(vec![x_handle]);
    bytemuck::cast_slice::<u8, F>(&bytes[0]).to_vec()
}

/// Dense element-wise scale on the device: `x[i] *= alpha`, in place.
///
/// Generic over `F: DeviceScalar` (f32 or f64). Empty input is a no-op (never
/// launches a zero-length grid). Dispatches the generic cubecl kernel on
/// whichever backend `client` resolved to — the `Runtime` type is selected here
/// and never appears in the signature, honoring the ALG-06 wall.
pub fn scal_dense<F: DeviceScalar>(
    client: &AlgebraClient,
    alpha: F,
    x: &mut [F],
) -> Result<(), AlgebraError> {
    // Empty input → no-op; never launch a zero-length grid.
    if x.is_empty() {
        return Ok(());
    }

    let scaled = dispatch_backend!(client, c, Rt, launch_scal::<Rt, F>(c, alpha, x));
    x.copy_from_slice(&scaled);
    Ok(())
}

/// Element-wise scale over the opaque `Tensor` surface: `x *= alpha`, updating
/// `x`'s resident device buffer in place.
///
/// quick-260529-mtx-2: launches directly on `x`'s resident device handle via the
/// Phase-2 [`crate::device_buffer`] registry — no host transfer. `x` must be a
/// device-backed tensor built with [`crate::device_buffer::upload`]; a
/// `Tensor::placeholder` (sentinel `BufferId`) yields
/// [`AlgebraError::UnallocatedBuffer`], and a buffer resident on another backend
/// yields [`AlgebraError::BackendMismatch`]. The kernel scales through a clone of
/// `x`'s handle, which shares the device binding with the registry's copy.
pub fn scal(client: &AlgebraClient, alpha: f64, x: &mut Tensor) -> Result<(), AlgebraError> {
    let xb = crate::device_buffer::handle_of::<f64>(x.id.raw(), client, "scal")?;
    if xb.len == 0 {
        return Ok(());
    }
    let n = xb.len;
    dispatch_backend!(
        client,
        c,
        Rt,
        launch_scal_on_handle::<Rt, f64>(c, alpha, &xb.handle, n)
    );
    Ok(())
}
