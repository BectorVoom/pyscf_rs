//! GEMV — matrix-vector product `y = A @ x`, the `N=1` case of GEMM.
//!
//! quick-260529-mtx (Phase-2): refactored from the Phase-1 `NotYetImplemented`
//! stub. GEMV is the rank-1 column case of GEMM, so rather than carry a second
//! kernel it delegates to [`crate::gemm::gemm_dense`] with the output column
//! count fixed to 1 — `A` is `M×N`, `x` is the `N`-vector treated as `N×1`, and
//! `y = A @ x` is the `M`-vector (`M×1`). The cubecl `Runtime` generic stays
//! inside the wall because `gemm_dense` already dispatches off `AlgebraClient`.
//!
//! Two surfaces:
//!   * `gemv()` — the opaque `Tensor`-based API, wired through the Phase-2
//!     [`crate::device_buffer`] registry: reads `A` and `x` from the registry,
//!     computes `y`, and writes it into `y`'s buffer.
//!   * `gemv_dense()` — the working host-slice device path: takes host slices +
//!     an `AlgebraClient` and returns the `M`-vector `y`. Exercised by the
//!     random oracle differential test (`tests/gemv_oracle.rs`) on ROCm.

use crate::gemm::{gemm_dense, launch_gemm_on_handles};
use crate::scalar::DeviceScalar;
use crate::{AlgebraClient, AlgebraError, Tensor};

/// Dense matrix-vector product on the device: `y = A @ x`, where `A` is a
/// row-major `M×N` matrix and `x` is the `N`-vector; returns the `M`-vector `y`.
///
/// Generic over `F: DeviceScalar` (f32 or f64). Implemented as GEMM with
/// `N_cols = 1`, so `gemm_dense` performs the flat-slice length validation
/// (`A.len() == m*n`, `x.len() == n`) and the backend dispatch. The `Runtime`
/// type never appears in the signature, honoring the ALG-06 wall.
pub fn gemv_dense<F: DeviceScalar>(
    client: &AlgebraClient,
    a: &[F],
    x: &[F],
    m: usize,
    n: usize,
) -> Result<Vec<F>, AlgebraError> {
    // A (m×n) @ x (n×1) = y (m×1). One column → GEMM with N=1.
    gemm_dense::<F>(client, a, x, m, n, 1)
}

/// Matrix-vector product over the opaque `Tensor` surface: `y = A @ x`, written
/// into `y`'s resident buffer in place.
///
/// quick-260529-mtx-2: launches the GEMM kernel (with `N_cols = 1`) directly on
/// the operands' resident device handles via the Phase-2 [`crate::device_buffer`]
/// registry — no host transfer. `a`, `x`, and `y` must be device-backed tensors
/// built with [`crate::device_buffer::upload`]; a `Tensor::placeholder` (sentinel
/// `BufferId`) yields [`AlgebraError::UnallocatedBuffer`], and a buffer resident
/// on another backend yields [`AlgebraError::BackendMismatch`]. `a`'s `[M, N]`
/// shape drives the product; `x` must hold `N` elements and `y` must already be
/// allocated with `M`. Errors with `DimensionMismatch` if `a` is not rank-2 or
/// the operand element counts disagree with `(M, N)`.
pub fn gemv(
    client: &AlgebraClient,
    a: &Tensor,
    x: &Tensor,
    y: &mut Tensor,
) -> Result<(), AlgebraError> {
    if a.rank() != 2 {
        return Err(AlgebraError::DimensionMismatch {
            op: "gemv",
            lhs: a.shape.clone(),
            rhs: x.shape.clone(),
        });
    }
    let (m, n) = (a.shape[0], a.shape[1]);
    if x.numel() != n || y.numel() != m {
        return Err(AlgebraError::DimensionMismatch {
            op: "gemv",
            lhs: a.shape.clone(),
            rhs: vec![x.numel(), y.numel()],
        });
    }
    let ab = crate::device_buffer::handle_of::<f64>(a.id.raw(), client, "gemv")?;
    let xb = crate::device_buffer::handle_of::<f64>(x.id.raw(), client, "gemv")?;
    let yb = crate::device_buffer::handle_of::<f64>(y.id.raw(), client, "gemv")?;
    if m == 0 {
        return Ok(());
    }
    // A (m×n) @ x (n×1) = y (m×1): GEMM with N=1, in place on y's handle.
    match client {
        AlgebraClient::Cpu(c) => launch_gemm_on_handles::<cubecl_cpu::CpuRuntime, f64>(
            c, &ab.handle, &xb.handle, &yb.handle, m, n, 1,
        ),
        #[cfg(feature = "cuda")]
        AlgebraClient::Cuda(c) => launch_gemm_on_handles::<cubecl_cuda::CudaRuntime, f64>(
            c, &ab.handle, &xb.handle, &yb.handle, m, n, 1,
        ),
        #[cfg(feature = "wgpu")]
        AlgebraClient::Wgpu(c) => launch_gemm_on_handles::<cubecl_wgpu::WgpuRuntime, f64>(
            c, &ab.handle, &xb.handle, &yb.handle, m, n, 1,
        ),
        #[cfg(feature = "rocm")]
        AlgebraClient::Rocm(c) => launch_gemm_on_handles::<cubecl_hip::HipRuntime, f64>(
            c, &ab.handle, &xb.handle, &yb.handle, m, n, 1,
        ),
    }
    Ok(())
}
