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

use crate::scalar::DeviceScalar;
use crate::{AlgebraClient, AlgebraError, Tensor};
use cubecl::Runtime;
use cubecl::bytes::Bytes;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::server::Handle;

/// Partial-sum dot kernel: thread `tid` multiplies and accumulates elements
/// `[tid*chunk, (tid+1)*chunk)` (clamped to `n`) into one partial in `F`,
/// writing it to `out[tid]`. Generic over the device float so the same kernel
/// monomorphizes for f32 (GPU speed path) and f64 (chemistry precision path) —
/// see `Cubecl_generics.md`. The final sum of the `groups` partials runs on the
/// host (see [`launch_dot`]).
/// Grid-stride coalesced dot product kernel: thread `tid` accumulates
/// elements `x[tid + k*stride] * y[tid + k*stride]` into `out[tid]`.
///
/// Within each warp/wavefront, adjacent threads access adjacent memory addresses
/// (unit stride), enabling 100% global memory coalescing on both inputs.
#[cube(launch_unchecked)]
fn dot_kernel<F: Float>(x: &Array<F>, y: &Array<F>, out: &mut Array<F>, n: usize) {
    let tid = ABSOLUTE_POS;
    let stride = CUBE_COUNT_X as usize * CUBE_DIM_X as usize;
    let mut acc = F::from_int(0);
    let mut i = tid;
    while i < n {
        acc += x[i] * y[i];
        i += stride;
    }
    out[tid] = acc;
}

/// Threads per block for the dot launch.
const BLOCK: u32 = 256;

/// Core launch on resident device handles: compute the per-element products and
/// partial sums of `x`/`y` (length `n`) into a temporary device buffer, read them
/// back, and sum in `F`. No re-upload of `x`/`y` — both the host-slice path
/// ([`launch_dot`]) and the registry-backed `Tensor` path ([`dot`]) drive this.
fn launch_dot_on_handles<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    x: &Handle,
    y: &Handle,
    n: usize,
) -> F {
    if n == 0 {
        return F::from_int(0);
    }
    // Launch an occupancy-tuned grid (clamped to hardware ceiling)
    let groups = ((n as u32).div_ceil(BLOCK * 16)).clamp(1, 64);
    let num_threads = (groups * BLOCK) as usize;
    let out_handle = client.empty(num_threads * core::mem::size_of::<F>());

    unsafe {
        dot_kernel::launch_unchecked::<F, R>(
            client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(BLOCK),
            // SAFETY: `n` matches the operand buffers and `num_threads` matches out_handle.
            ArrayArg::from_raw_parts(x.clone(), n),
            ArrayArg::from_raw_parts(y.clone(), n),
            ArrayArg::from_raw_parts(out_handle.clone(), num_threads),
            // Scalar kernel args are passed as bare values (LaunchArg for T = T).
            n,
        );
    }

    let bytes = client.read(vec![out_handle]);
    let products: Vec<F> = bytemuck::cast_slice::<u8, F>(&bytes[0]).to_vec();

    // Host-sum the partials in `F`.
    let mut acc = F::from_int(0);
    for &p in &products {
        acc += p;
    }
    acc
}

/// Host-slice launcher: upload `x`/`y`, then reduce on the device. Backs
/// `dot_dense`; the cubecl `Runtime` bound stays inside.
fn launch_dot<R: Runtime, F: DeviceScalar>(client: &ComputeClient<R>, x: &[F], y: &[F]) -> F {
    let x_handle = client.create(Bytes::from_elems(x.to_vec()));
    let y_handle = client.create(Bytes::from_elems(y.to_vec()));
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
