//! TRANSPOSE — generic-float cubecl 2-D transpose kernel + backend-dispatched
//! host launcher + registry-backed `Tensor` surface.
//!
//! quick-260529-mtx (Phase-2): refactored from the Phase-1 `NotYetImplemented`
//! stub into a real cubecl `#[cube(launch)]` kernel generic over the device
//! float `F: Float` (per docs/manual/Cubecl/Cubecl_generics.md), plus a
//! host-slice launcher dispatched off `AlgebraClient` so the cubecl `Runtime`
//! generic stays inside the ALG-06 wall. Mirrors the `gemm.rs` sibling.
//!
//! `transpose(x) = out[c, r] = x[r, c]` for a row-major `M×N` input, producing a
//! row-major `N×M` output. It is a pure index permutation: one thread per input
//! element gathers `x[tid]` to its transposed output slot — no arithmetic, no
//! reduction, no atomics.
//!
//! Two surfaces:
//!   * `transpose()` — the opaque `Tensor`-based API, wired through the Phase-2
//!     [`crate::device_buffer`] registry: reads `x` from the registry, runs the
//!     kernel, and writes the `N×M` result into `out`'s buffer.
//!   * `transpose_dense()` — the working host-slice device path: takes a host
//!     slice + an `AlgebraClient`, runs the generic kernel on the resolved
//!     backend (CPU, ROCm, CUDA, WGPU), and returns the transposed `Vec`.
//!     Exercised by the random oracle differential test
//!     (`tests/transpose_oracle.rs`) on ROCm.

use crate::scalar::DeviceScalar;
use crate::{AlgebraClient, AlgebraError, Tensor};
use cubecl::Runtime;
use cubecl::bytes::Bytes;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;

/// Naive one-thread-per-element transpose: for the row-major `M×N` input,
/// thread `tid` reads `x[tid]` (logical position `row = tid/n`, `col = tid%n`)
/// and writes it to the `N×M` output slot `col*m + row`. Generic over the
/// device float so the same kernel monomorphizes for f32 (GPU speed path) and
/// f64 (chemistry precision path) — see `Cubecl_generics.md`.
#[cube(launch)]
fn transpose_kernel<F: Float>(x: &Array<F>, out: &mut Array<F>, m: usize, n: usize) {
    // `ABSOLUTE_POS` and `Array` indices are `usize` in cubecl 0.10, so the
    // dimension scalars are `usize` too — no casts.
    let tid = ABSOLUTE_POS;
    // Bounds guard: the launch rounds the thread count up to a whole number of
    // blocks, so the tail threads must not write out of range.
    if tid < m * n {
        let row = tid / n;
        let col = tid % n;
        out[col * m + row] = x[tid];
    }
}

/// Threads per block for the transpose launch. One thread moves one element;
/// the grid is sized to cover `M*N` threads.
const BLOCK: u32 = 256;

/// Runtime-generic launcher: upload `x`, launch `transpose_kernel`, read the
/// `N×M` result back to host. `R` is kept generic so a single body serves every
/// backend; callers reach it only through `transpose_dense` (which picks `R`
/// from the active `AlgebraClient`), so the cubecl `Runtime` bound never escapes
/// the wall.
fn launch_transpose<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    x: &[F],
    m: usize,
    n: usize,
) -> Vec<F> {
    let x_handle = client.create(Bytes::from_elems(x.to_vec()));
    let out_handle = client.empty(m * n * core::mem::size_of::<F>());

    let groups = (m * n).div_ceil(BLOCK as usize) as u32;

    transpose_kernel::launch::<F, R>(
        client,
        CubeCount::Static(groups, 1, 1),
        CubeDim::new_1d(BLOCK),
        // SAFETY: lengths match the buffers just allocated above. `from_raw_parts`
        // consumes the handle by value; clone the output handle so it survives
        // for the read-back below.
        unsafe { ArrayArg::from_raw_parts(x_handle, x.len()) },
        unsafe { ArrayArg::from_raw_parts(out_handle.clone(), m * n) },
        // Scalar dimension args are passed as bare values (LaunchArg for usize).
        m,
        n,
    );

    let bytes = client.read(vec![out_handle]);
    bytemuck::cast_slice::<u8, F>(&bytes[0]).to_vec()
}

/// Dense 2-D transpose on the device: a row-major `M×N` input → row-major `N×M`
/// output.
///
/// Generic over `F: DeviceScalar` (f32 or f64). Validates the flat-slice length
/// against `(m, n)`; empty input is a no-op (returns an empty `Vec`, never
/// launches a zero-length grid). Dispatches the generic cubecl kernel on
/// whichever backend `client` resolved to — the `Runtime` type is selected here
/// and never appears in the signature, honoring the ALG-06 wall.
pub fn transpose_dense<F: DeviceScalar>(
    client: &AlgebraClient,
    x: &[F],
    m: usize,
    n: usize,
) -> Result<Vec<F>, AlgebraError> {
    if x.len() != m * n {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("x len {} (= {m}*{n})", m * n),
            actual: x.len().to_string(),
        });
    }
    // Empty input → empty output; never launch a zero-length grid.
    if x.is_empty() {
        return Ok(Vec::new());
    }

    let out = match client {
        AlgebraClient::Cpu(c) => launch_transpose::<cubecl_cpu::CpuRuntime, F>(c, x, m, n),
        #[cfg(feature = "cuda")]
        AlgebraClient::Cuda(c) => launch_transpose::<cubecl_cuda::CudaRuntime, F>(c, x, m, n),
        #[cfg(feature = "wgpu")]
        AlgebraClient::Wgpu(c) => launch_transpose::<cubecl_wgpu::WgpuRuntime, F>(c, x, m, n),
        #[cfg(feature = "rocm")]
        AlgebraClient::Rocm(c) => launch_transpose::<cubecl_hip::HipRuntime, F>(c, x, m, n),
    };
    Ok(out)
}

/// 2-D transpose over the opaque `Tensor` surface: row-major `M×N` `x` →
/// row-major `N×M` `out`, written into `out`'s buffer in place.
///
/// quick-260529-mtx: wired through the Phase-2 [`crate::device_buffer`] registry.
/// `x` and `out` must be device-backed rank-2 tensors built with
/// [`crate::device_buffer::upload`]; a `Tensor::placeholder` (sentinel
/// `BufferId`) yields [`AlgebraError::UnallocatedBuffer`]. `x`'s `[M, N]` shape
/// drives the transpose; `out` must already be allocated with `M*N` elements
/// (an `[N, M]` tensor). Errors with `DimensionMismatch` if `x` is not rank-2.
pub fn transpose(
    client: &AlgebraClient,
    x: &Tensor,
    out: &mut Tensor,
) -> Result<(), AlgebraError> {
    if x.rank() != 2 {
        return Err(AlgebraError::DimensionMismatch {
            op: "transpose",
            lhs: x.shape.clone(),
            rhs: out.shape.clone(),
        });
    }
    let (m, n) = (x.shape[0], x.shape[1]);
    let x_host = crate::device_buffer::read_raw::<f64>(x.id.raw(), "transpose")?;
    let result = transpose_dense::<f64>(client, &x_host, m, n)?;
    crate::device_buffer::write_back::<f64>(out.id.raw(), &result, "transpose")
}
