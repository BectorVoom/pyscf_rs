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

use crate::scalar::DeviceScalar;
use crate::{AlgebraClient, AlgebraError, Tensor};
use cubecl::Runtime;
use cubecl::bytes::Bytes;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::server::Handle;

/// Naive one-thread-per-element scale kernel: `x[i] *= alpha`. Generic over the
/// device float so the same kernel monomorphizes for f32 (GPU speed path) and
/// f64 (chemistry precision path) — see `Cubecl_generics.md`. Scales in place;
/// the launcher reads the buffer back (see [`launch_scal`]).
///
/// `alpha` is passed as a single-element `Array<F>` rather than a bare scalar:
/// a generic `F` used as a scalar launch arg would need extra `CubeElement` /
/// `ScalarArgSettings` bounds that `DeviceScalar` (sealed to f32/f64) does not
/// carry, so `F` only ever appears here as an `Array` element type — the same
/// shape that the `dot`/`reduce` siblings rely on.
#[cube(launch)]
fn scal_kernel<F: Float>(x: &mut Array<F>, alpha: &Array<F>, n: usize) {
    // `ABSOLUTE_POS` and `Array` indices are `usize` in cubecl 0.10, so the
    // dimension scalar is `usize` too — no casts.
    let tid = ABSOLUTE_POS;
    // Bounds guard: the launch rounds the thread count up to a whole number of
    // blocks, so the tail threads must not write out of range.
    if tid < n {
        x[tid] = x[tid] * alpha[0];
    }
}

/// Threads per block for the scal launch. One thread scales one element; the
/// grid is sized to cover `n` threads.
const BLOCK: u32 = 256;

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
    // `alpha` rides as a single-element device buffer (see `scal_kernel`).
    let alpha_handle = client.create(Bytes::from_elems(vec![alpha]));
    let groups = n.div_ceil(BLOCK as usize) as u32;

    scal_kernel::launch::<F, R>(
        client,
        CubeCount::Static(groups, 1, 1),
        CubeDim::new_1d(BLOCK),
        // SAFETY: `n` matches the buffer. `from_raw_parts` consumes the handle by
        // value, so clone the caller's handle (clones share the binding).
        unsafe { ArrayArg::from_raw_parts(x.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(alpha_handle, 1) },
        // Scalar dimension arg is passed as a bare value (LaunchArg for usize).
        n,
    );
}

/// Host-slice launcher: upload `x`, scale it in place on the device, read it
/// back. Backs `scal_dense`; the cubecl `Runtime` bound stays inside.
fn launch_scal<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    alpha: F,
    x: &[F],
) -> Vec<F> {
    let x_handle = client.create(Bytes::from_elems(x.to_vec()));
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

    let scaled = match client {
        AlgebraClient::Cpu(c) => launch_scal::<cubecl_cpu::CpuRuntime, F>(c, alpha, x),
        #[cfg(feature = "cuda")]
        AlgebraClient::Cuda(c) => launch_scal::<cubecl_cuda::CudaRuntime, F>(c, alpha, x),
        #[cfg(feature = "wgpu")]
        AlgebraClient::Wgpu(c) => launch_scal::<cubecl_wgpu::WgpuRuntime, F>(c, alpha, x),
        #[cfg(feature = "rocm")]
        AlgebraClient::Rocm(c) => launch_scal::<cubecl_hip::HipRuntime, F>(c, alpha, x),
    };
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
    match client {
        AlgebraClient::Cpu(c) => {
            launch_scal_on_handle::<cubecl_cpu::CpuRuntime, f64>(c, alpha, &xb.handle, n)
        }
        #[cfg(feature = "cuda")]
        AlgebraClient::Cuda(c) => {
            launch_scal_on_handle::<cubecl_cuda::CudaRuntime, f64>(c, alpha, &xb.handle, n)
        }
        #[cfg(feature = "wgpu")]
        AlgebraClient::Wgpu(c) => {
            launch_scal_on_handle::<cubecl_wgpu::WgpuRuntime, f64>(c, alpha, &xb.handle, n)
        }
        #[cfg(feature = "rocm")]
        AlgebraClient::Rocm(c) => {
            launch_scal_on_handle::<cubecl_hip::HipRuntime, f64>(c, alpha, &xb.handle, n)
        }
    }
    Ok(())
}
