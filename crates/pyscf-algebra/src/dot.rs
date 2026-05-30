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

/// Naive one-thread-per-element products kernel: `out[i] = x[i] * y[i]`.
/// Generic over the device float so the same kernel monomorphizes for f32
/// (GPU speed path) and f64 (chemistry precision path) — see
/// `Cubecl_generics.md`. The final sum is performed on the host (see
/// [`launch_dot`]); this kernel does only the multiply.
#[cube(launch)]
fn dot_kernel<F: Float>(x: &Array<F>, y: &Array<F>, out: &mut Array<F>, n: usize) {
    // `ABSOLUTE_POS` and `Array` indices are `usize` in cubecl 0.10, so the
    // dimension scalar is `usize` too — no casts.
    let tid = ABSOLUTE_POS;
    // Bounds guard: the launch rounds the thread count up to a whole number of
    // blocks, so the tail threads must not write out of range.
    if tid < n {
        out[tid] = x[tid] * y[tid];
    }
}

/// Threads per block for the dot launch. One thread computes one product; the
/// grid is sized to cover `n` threads.
const BLOCK: u32 = 256;

/// Core launch on resident device handles: compute the per-element products of
/// `x`/`y` (length `n`) into a temporary device buffer, read them back, and sum
/// in `F`. No re-upload of `x`/`y` — both the host-slice path ([`launch_dot`])
/// and the registry-backed `Tensor` path ([`dot`]) drive this. The products
/// read-back is inherent to the host-sum reduction; the win is that the resident
/// operands are not copied to the device again per op.
fn launch_dot_on_handles<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    x: &Handle,
    y: &Handle,
    n: usize,
) -> F {
    let out_handle = client.empty(n * core::mem::size_of::<F>());

    let groups = n.div_ceil(BLOCK as usize) as u32;

    dot_kernel::launch::<F, R>(
        client,
        CubeCount::Static(groups, 1, 1),
        CubeDim::new_1d(BLOCK),
        // SAFETY: `n` matches the operand buffers and the just-allocated output.
        // `from_raw_parts` consumes the handle by value; clone the caller's
        // handles and the output handle (which must survive for the read-back).
        unsafe { ArrayArg::from_raw_parts(x.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(y.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n) },
        // Scalar kernel args are passed as bare values (LaunchArg for T = T).
        n,
    );

    let bytes = client.read(vec![out_handle]);
    let products: Vec<F> = bytemuck::cast_slice::<u8, F>(&bytes[0]).to_vec();

    // Host-sum in `F`. A plain `acc += p` add is NOT an FMA (check-no-fma only
    // flags FMA in pyscf_* symbols anyway) — the products kernel is the only
    // multiply.
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
