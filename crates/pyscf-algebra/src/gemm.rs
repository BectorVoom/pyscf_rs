//! GEMM — generic-float cubecl kernel + backend-dispatched host launcher.
//!
//! quick-260529-i2x: refactored from the Phase-1 `NotYetImplemented` stub
//! into a real cubecl `#[cube(launch)]` kernel generic over the device float
//! `F: Float` (per docs/manual/Cubecl/Cubecl_generics.md), plus a host-slice
//! launcher dispatched off `AlgebraClient` so the cubecl `Runtime` generic
//! stays inside the ALG-06 wall.
//!
//! Two surfaces:
//!   * `gemm()` — the opaque `Tensor`-based API. STILL a stub: `Tensor` carries
//!     a sentinel `BufferId` (no device allocator until Phase 2), so it cannot
//!     read device data yet. `backend_matrix.rs` pins this contract.
//!   * `gemm_dense()` — the working device path: takes host slices + an
//!     `AlgebraClient`, runs the generic kernel on the resolved backend (CPU,
//!     ROCm, CUDA, WGPU), and returns the result. Exercised by the random
//!     oracle differential test (`tests/gemm_oracle.rs`) on ROCm.

use crate::scalar::DeviceScalar;
use crate::{AlgebraClient, AlgebraError, Tensor};
use cubecl::Runtime;
use cubecl::bytes::Bytes;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;

/// Naive one-thread-per-output-element GEMM, `out = lhs @ rhs` (row-major).
/// `lhs` is `M×K`, `rhs` is `K×N`, `out` is `M×N`. Generic over the device
/// float so the same kernel monomorphizes for f32 (GPU speed path) and f64
/// (chemistry precision path) — see `Cubecl_generics.md`.
#[cube(launch)]
fn gemm_kernel<F: Float>(
    lhs: &Array<F>,
    rhs: &Array<F>,
    out: &mut Array<F>,
    m: usize,
    k: usize,
    n: usize,
) {
    // `ABSOLUTE_POS` and `Array` indices are `usize` in cubecl 0.10, so the
    // dimension scalars are `usize` too — no casts inside the hot loop.
    let tid = ABSOLUTE_POS;
    // Bounds guard: the launch rounds the thread count up to a whole number of
    // blocks, so the tail threads must not write out of range.
    if tid < m * n {
        let row = tid / n;
        let col = tid % n;
        let mut acc = F::from_int(0);
        // Runtime-bounded contraction over the shared K dimension.
        for j in 0..k {
            acc += lhs[row * k + j] * rhs[j * n + col];
        }
        out[tid] = acc;
    }
}

/// Threads per block for the GEMM launch. One thread computes one output
/// element; the grid is sized to cover `M*N` threads.
const BLOCK: u32 = 256;

/// Runtime-generic launcher: upload `lhs`/`rhs`, launch `gemm_kernel`, read the
/// `M×N` result back to host. `R` is kept generic so a single body serves every
/// backend; callers reach it only through `gemm_dense` (which picks `R` from the
/// active `AlgebraClient`), so the cubecl `Runtime` bound never escapes the wall.
fn launch_gemm<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    lhs: &[F],
    rhs: &[F],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<F> {
    let lhs_handle = client.create(Bytes::from_elems(lhs.to_vec()));
    let rhs_handle = client.create(Bytes::from_elems(rhs.to_vec()));
    let out_handle = client.empty(m * n * core::mem::size_of::<F>());

    let groups = (m * n).div_ceil(BLOCK as usize) as u32;

    gemm_kernel::launch::<F, R>(
        client,
        CubeCount::Static(groups, 1, 1),
        CubeDim::new_1d(BLOCK),
        // SAFETY: lengths match the buffers just allocated above. `from_raw_parts`
        // consumes the handle by value; clone the output handle so it survives
        // for the read-back below.
        unsafe { ArrayArg::from_raw_parts(lhs_handle, lhs.len()) },
        unsafe { ArrayArg::from_raw_parts(rhs_handle, rhs.len()) },
        unsafe { ArrayArg::from_raw_parts(out_handle.clone(), m * n) },
        // Scalar kernel args are passed as bare values (LaunchArg for T = T).
        m,
        k,
        n,
    );

    let bytes = client.read(vec![out_handle]);
    bytemuck::cast_slice::<u8, F>(&bytes[0]).to_vec()
}

/// Dense matrix multiply on the device: `out = lhs @ rhs`, row-major.
/// `lhs` is `M×K`, `rhs` is `K×N`; returns the `M×N` product.
///
/// Generic over `F: DeviceScalar` (f32 or f64). Validates the flat-slice
/// lengths against `(m, k, n)`, then dispatches the generic cubecl kernel on
/// whichever backend `client` resolved to — the `Runtime` type is selected
/// here and never appears in the signature, honoring the ALG-06 wall.
pub fn gemm_dense<F: DeviceScalar>(
    client: &AlgebraClient,
    lhs: &[F],
    rhs: &[F],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<F>, AlgebraError> {
    if lhs.len() != m * k {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("lhs len {} (= {m}*{k})", m * k),
            actual: lhs.len().to_string(),
        });
    }
    if rhs.len() != k * n {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("rhs len {} (= {k}*{n})", k * n),
            actual: rhs.len().to_string(),
        });
    }

    let out = match client {
        AlgebraClient::Cpu(c) => launch_gemm::<cubecl_cpu::CpuRuntime, F>(c, lhs, rhs, m, k, n),
        #[cfg(feature = "cuda")]
        AlgebraClient::Cuda(c) => launch_gemm::<cubecl_cuda::CudaRuntime, F>(c, lhs, rhs, m, k, n),
        #[cfg(feature = "wgpu")]
        AlgebraClient::Wgpu(c) => launch_gemm::<cubecl_wgpu::WgpuRuntime, F>(c, lhs, rhs, m, k, n),
        #[cfg(feature = "rocm")]
        AlgebraClient::Rocm(c) => launch_gemm::<cubecl_hip::HipRuntime, F>(c, lhs, rhs, m, k, n),
    };
    Ok(out)
}

/// Dense matrix multiply over the opaque `Tensor` surface: `out = lhs @ rhs`.
///
/// STILL a Phase-2 stub. `Tensor` carries a sentinel `BufferId` (the device
/// allocator lands in Phase 2), so there is no device buffer to read here yet.
/// The working device path is [`gemm_dense`], which takes host slices directly.
/// `backend_matrix.rs` asserts this returns `NotYetImplemented`.
pub fn gemm(
    _client: &AlgebraClient,
    _lhs: &Tensor,
    _rhs: &Tensor,
    _out: &mut Tensor,
) -> Result<(), AlgebraError> {
    Err(AlgebraError::NotYetImplemented {
        phase: 2,
        what: "gemm over Tensor (device allocator) — use gemm_dense for the host-slice device path",
    })
}
