//! AXPY — generic-float cubecl element-wise y += alpha*x kernel + backend-dispatched host launcher.
//!
//! quick-260529-mtx: refactored from the Phase-1 `NotYetImplemented` stub into
//! a real cubecl `#[cube(launch)]` kernel generic over the device float
//! `F: Float` (per docs/manual/Cubecl/Cubecl_generics.md), plus a host-slice
//! launcher dispatched off `AlgebraClient` so the cubecl `Runtime` generic
//! stays inside the ALG-06 wall. Mirrors the `dot.rs` (quick-260529-iji),
//! `reduce.rs` (quick-260529-jcx) and `scal.rs` (quick-260529-skl) siblings
//! exactly — AXPY is the two-array element-wise analog of SCAL.
//!
//! `axpy(alpha, x, y) = y[i] += alpha * x[i]` is the element-wise BLAS-1
//! update: no reduction, one thread per element, two arrays (`x` read-only,
//! `y` read-write in place), output the same shape as the inputs. The device
//! kernel updates `y` in place (bounds-guarded against the launch tail); the
//! launcher reads the updated `y` buffer back. This keeps the kernel naive and
//! obviously-correct — a single multiply-add per element, no atomics.
//!
//! Two surfaces:
//!   * `axpy()` — the opaque `Tensor`-based API. STILL a stub: `Tensor` carries
//!     a sentinel `BufferId` (no device allocator until Phase 2), so it cannot
//!     read device data yet.
//!   * `axpy_dense()` — the working device path: takes host slices + an
//!     `AlgebraClient`, runs the generic kernel on the resolved backend (CPU,
//!     ROCm, CUDA, WGPU), and updates `y` in place. Exercised by the random
//!     oracle differential test (`tests/axpy_oracle.rs`) on ROCm.

use crate::scalar::DeviceScalar;
use crate::{AlgebraClient, AlgebraError, Tensor};
use cubecl::Runtime;
use cubecl::bytes::Bytes;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::server::Handle;

/// Naive one-thread-per-element AXPY kernel: `y[i] += alpha * x[i]`. Generic
/// over the device float so the same kernel monomorphizes for f32 (GPU speed
/// path) and f64 (chemistry precision path) — see `Cubecl_generics.md`. `x` is
/// read-only, `y` is updated in place; the launcher reads the `y` buffer back
/// (see [`launch_axpy`]).
///
/// `alpha` is passed as a single-element `Array<F>` rather than a bare scalar:
/// a generic `F` used as a scalar launch arg would need extra `CubeElement` /
/// `ScalarArgSettings` bounds that `DeviceScalar` (sealed to f32/f64) does not
/// carry, so `F` only ever appears here as an `Array` element type — the same
/// shape that the `dot`/`reduce`/`scal` siblings rely on.
#[cube(launch)]
fn axpy_kernel<F: Float>(x: &Array<F>, y: &mut Array<F>, alpha: &Array<F>, n: usize) {
    // `ABSOLUTE_POS` and `Array` indices are `usize` in cubecl 0.10, so the
    // dimension scalar is `usize` too — no casts.
    let tid = ABSOLUTE_POS;
    // Bounds guard: the launch rounds the thread count up to a whole number of
    // blocks, so the tail threads must not write out of range.
    if tid < n {
        y[tid] = y[tid] + alpha[0] * x[tid];
    }
}

/// Threads per block for the axpy launch. One thread updates one element; the
/// grid is sized to cover `n` threads.
const BLOCK: u32 = 256;

/// Core launch on resident device handles: `y[i] += alpha * x[i]`, in place on
/// the `y` handle, with NO host transfer. Both the host-slice path
/// ([`launch_axpy`]) and the registry-backed `Tensor` path ([`axpy`]) drive this.
/// `R` is kept generic so a single body serves every backend.
///
/// cubecl handle clones share one device binding, so the in-place write to `y`
/// is visible through every clone — including the registry's stored handle.
fn launch_axpy_on_handles<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    alpha: F,
    x: &Handle,
    y: &Handle,
    n: usize,
) {
    // `alpha` rides as a single-element device buffer (see `axpy_kernel`).
    let alpha_handle = client.create(Bytes::from_elems(vec![alpha]));
    let groups = n.div_ceil(BLOCK as usize) as u32;

    axpy_kernel::launch::<F, R>(
        client,
        CubeCount::Static(groups, 1, 1),
        CubeDim::new_1d(BLOCK),
        // SAFETY: `n` matches both buffers. `from_raw_parts` consumes the handle
        // by value, so clone the caller's handles (clones share the binding).
        unsafe { ArrayArg::from_raw_parts(x.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(y.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(alpha_handle, 1) },
        // Scalar dimension arg is passed as a bare value (LaunchArg for usize).
        n,
    );
}

/// Host-slice launcher: upload `x`/`y`, run the kernel in place on `y`, read the
/// updated `y` back. Backs `axpy_dense`; the cubecl `Runtime` bound stays inside.
fn launch_axpy<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    alpha: F,
    x: &[F],
    y: &[F],
) -> Vec<F> {
    let x_handle = client.create(Bytes::from_elems(x.to_vec()));
    let y_handle = client.create(Bytes::from_elems(y.to_vec()));
    launch_axpy_on_handles::<R, F>(client, alpha, &x_handle, &y_handle, x.len());
    let bytes = client.read(vec![y_handle]);
    bytemuck::cast_slice::<u8, F>(&bytes[0]).to_vec()
}

/// Dense element-wise AXPY on the device: `y[i] += alpha * x[i]`, in place.
///
/// Generic over `F: DeviceScalar` (f32 or f64). On a length mismatch between
/// `x` and `y` it returns [`AlgebraError::DimensionMismatch`] (op `"axpy"`)
/// without launching or modifying `y`. Empty input is a no-op (never launches
/// a zero-length grid). Dispatches the generic cubecl kernel on whichever
/// backend `client` resolved to — the `Runtime` type is selected here and never
/// appears in the signature, honoring the ALG-06 wall.
pub fn axpy_dense<F: DeviceScalar>(
    client: &AlgebraClient,
    alpha: F,
    x: &[F],
    y: &mut [F],
) -> Result<(), AlgebraError> {
    // Length mismatch → clean error, never launch with mismatched buffers.
    if x.len() != y.len() {
        return Err(AlgebraError::DimensionMismatch {
            op: "axpy",
            lhs: vec![x.len()],
            rhs: vec![y.len()],
        });
    }

    // Empty input → no-op; never launch a zero-length grid. (Both are empty
    // here, given the length check above.)
    if x.is_empty() {
        return Ok(());
    }

    let result = match client {
        AlgebraClient::Cpu(c) => launch_axpy::<cubecl_cpu::CpuRuntime, F>(c, alpha, x, y),
        #[cfg(feature = "cuda")]
        AlgebraClient::Cuda(c) => launch_axpy::<cubecl_cuda::CudaRuntime, F>(c, alpha, x, y),
        #[cfg(feature = "wgpu")]
        AlgebraClient::Wgpu(c) => launch_axpy::<cubecl_wgpu::WgpuRuntime, F>(c, alpha, x, y),
        #[cfg(feature = "rocm")]
        AlgebraClient::Rocm(c) => launch_axpy::<cubecl_hip::HipRuntime, F>(c, alpha, x, y),
    };
    y.copy_from_slice(&result);
    Ok(())
}

/// Element-wise AXPY over the opaque `Tensor` surface: `y[i] += alpha * x[i]`,
/// updating `y`'s resident device buffer in place.
///
/// quick-260529-mtx-2: launches directly on the operands' resident device
/// handles via the Phase-2 [`crate::device_buffer`] registry — no host transfer.
/// Both `x` and `y` must be device-backed tensors built with
/// [`crate::device_buffer::upload`]; a `Tensor::placeholder` (sentinel
/// `BufferId`) yields [`AlgebraError::UnallocatedBuffer`], and a buffer resident
/// on another backend yields [`AlgebraError::BackendMismatch`]. The kernel
/// writes through a clone of `y`'s handle, which shares the device binding with
/// the registry's copy, so the update is visible on the next
/// [`crate::device_buffer::download`] — no read-back, no re-upload.
pub fn axpy(
    client: &AlgebraClient,
    alpha: f64,
    x: &Tensor,
    y: &mut Tensor,
) -> Result<(), AlgebraError> {
    let xb = crate::device_buffer::handle_of::<f64>(x.id.raw(), client, "axpy")?;
    let yb = crate::device_buffer::handle_of::<f64>(y.id.raw(), client, "axpy")?;
    if xb.len != yb.len {
        return Err(AlgebraError::DimensionMismatch {
            op: "axpy",
            lhs: vec![xb.len],
            rhs: vec![yb.len],
        });
    }
    if xb.len == 0 {
        return Ok(());
    }
    let n = xb.len;
    match client {
        AlgebraClient::Cpu(c) => launch_axpy_on_handles::<cubecl_cpu::CpuRuntime, f64>(
            c, alpha, &xb.handle, &yb.handle, n,
        ),
        #[cfg(feature = "cuda")]
        AlgebraClient::Cuda(c) => launch_axpy_on_handles::<cubecl_cuda::CudaRuntime, f64>(
            c, alpha, &xb.handle, &yb.handle, n,
        ),
        #[cfg(feature = "wgpu")]
        AlgebraClient::Wgpu(c) => launch_axpy_on_handles::<cubecl_wgpu::WgpuRuntime, f64>(
            c, alpha, &xb.handle, &yb.handle, n,
        ),
        #[cfg(feature = "rocm")]
        AlgebraClient::Rocm(c) => launch_axpy_on_handles::<cubecl_hip::HipRuntime, f64>(
            c, alpha, &xb.handle, &yb.handle, n,
        ),
    }
    Ok(())
}
