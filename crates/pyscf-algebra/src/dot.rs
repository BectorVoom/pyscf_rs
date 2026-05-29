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

/// Runtime-generic launcher: upload `x`/`y`, launch `dot_kernel`, read the
/// per-element products back to host, then sum them in `F`. `R` is kept generic
/// so a single body serves every backend; callers reach it only through
/// `dot_dense` (which picks `R` from the active `AlgebraClient`), so the cubecl
/// `Runtime` bound never escapes the wall.
fn launch_dot<R: Runtime, F: DeviceScalar>(client: &ComputeClient<R>, x: &[F], y: &[F]) -> F {
    let x_handle = client.create(Bytes::from_elems(x.to_vec()));
    let y_handle = client.create(Bytes::from_elems(y.to_vec()));
    let out_handle = client.empty(core::mem::size_of_val(x));

    let groups = x.len().div_ceil(BLOCK as usize) as u32;

    dot_kernel::launch::<F, R>(
        client,
        CubeCount::Static(groups, 1, 1),
        CubeDim::new_1d(BLOCK),
        // SAFETY: lengths match the buffers just allocated above. `from_raw_parts`
        // consumes the handle by value; clone the output handle so it survives
        // for the read-back below.
        unsafe { ArrayArg::from_raw_parts(x_handle, x.len()) },
        unsafe { ArrayArg::from_raw_parts(y_handle, y.len()) },
        unsafe { ArrayArg::from_raw_parts(out_handle.clone(), x.len()) },
        // Scalar kernel args are passed as bare values (LaunchArg for T = T).
        x.len(),
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

    let out = match client {
        AlgebraClient::Cpu(c) => launch_dot::<cubecl_cpu::CpuRuntime, F>(c, x, y),
        #[cfg(feature = "cuda")]
        AlgebraClient::Cuda(c) => launch_dot::<cubecl_cuda::CudaRuntime, F>(c, x, y),
        #[cfg(feature = "wgpu")]
        AlgebraClient::Wgpu(c) => launch_dot::<cubecl_wgpu::WgpuRuntime, F>(c, x, y),
        #[cfg(feature = "rocm")]
        AlgebraClient::Rocm(c) => launch_dot::<cubecl_hip::HipRuntime, F>(c, x, y),
    };
    Ok(out)
}

/// Dot reduction over the opaque `Tensor` surface: `dot(x, y) = sum(x[i]*y[i])`.
///
/// STILL a Phase-2 stub. `Tensor` carries a sentinel `BufferId` (the device
/// allocator lands in Phase 2), so there is no device buffer to read here yet.
/// The working device path is [`dot_dense`], which takes host slices directly.
pub fn dot(_client: &AlgebraClient, _x: &Tensor, _y: &Tensor) -> Result<f64, AlgebraError> {
    Err(AlgebraError::NotYetImplemented {
        phase: 2,
        what: "dot over Tensor (device allocator) — use dot_dense for the host-slice device path",
    })
}
