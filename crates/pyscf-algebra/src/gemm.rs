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
use cubecl::server::Handle;

/// 2D Tiled GEMM kernel with shared memory staging and unrolled loop accumulation:
/// `out = lhs @ rhs` (row-major). `lhs` is `M×K`, `rhs` is `K×N`, `out` is `M×N`.
///
/// Threads within a 16×16 workgroup collaborate to load 16×16 tiles of `lhs` and
/// `rhs` into on-chip shared memory, synchronizing with `sync_cube()`, and accumulating
/// the matrix product with unrolled registers before writing back to global memory.
/// This reduces global memory traffic by 16× and ensures fully coalesced accesses.
const TILE_DIM: usize = 16;
const TILE_SIZE: usize = 256;

#[cube(launch_unchecked)]
fn gemm_kernel<F: Float>(
    lhs: &Array<F>,
    rhs: &Array<F>,
    out: &mut Array<F>,
    m: usize,
    k: usize,
    n: usize,
) {
    let mut lhs_shared = SharedMemory::<F>::new(TILE_SIZE);
    let mut rhs_shared = SharedMemory::<F>::new(TILE_SIZE);

    let tx = UNIT_POS_X as usize;
    let ty = UNIT_POS_Y as usize;
    let bx = CUBE_POS_X as usize;
    let by = CUBE_POS_Y as usize;

    let row = by * TILE_DIM + ty;
    let col = bx * TILE_DIM + tx;

    let mut acc = F::from_int(0);

    let num_tiles = k.div_ceil(TILE_DIM);

    for t in 0..num_tiles {
        let t_k = t * TILE_DIM;

        // Load LHS tile into shared memory
        let lhs_col = t_k + tx;
        if row < m && lhs_col < k {
            lhs_shared[ty * TILE_DIM + tx] = lhs[row * k + lhs_col];
        } else {
            lhs_shared[ty * TILE_DIM + tx] = F::from_int(0);
        }

        // Load RHS tile into shared memory
        let rhs_row = t_k + ty;
        if rhs_row < k && col < n {
            rhs_shared[ty * TILE_DIM + tx] = rhs[rhs_row * n + col];
        } else {
            rhs_shared[ty * TILE_DIM + tx] = F::from_int(0);
        }

        sync_cube();

        #[unroll]
        for p in 0..TILE_DIM {
            acc += lhs_shared[ty * TILE_DIM + p] * rhs_shared[p * TILE_DIM + tx];
        }

        sync_cube();
    }

    if row < m && col < n {
        out[row * n + col] = acc;
    }
}

/// Core launch on resident device handles: `out = lhs @ rhs` into the `out`
/// handle (`M×N`), with NO host transfer. Both the host-slice path
/// ([`launch_gemm`]) and the registry-backed `Tensor` path ([`gemm`], and `gemv`
/// via it) drive this; the caller owns `out` (a fresh temp for the dense path,
/// the resident result buffer for the `Tensor` path).
pub(crate) fn launch_gemm_on_handles<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    lhs: &Handle,
    rhs: &Handle,
    out: &Handle,
    m: usize,
    k: usize,
    n: usize,
) {
    let grid_x = (n as u32).div_ceil(TILE_DIM as u32);
    let grid_y = (m as u32).div_ceil(TILE_DIM as u32);

    unsafe {
        gemm_kernel::launch_unchecked::<F, R>(
            client,
            CubeCount::Static(grid_x, grid_y, 1),
            CubeDim {
                x: TILE_DIM as u32,
                y: TILE_DIM as u32,
                z: 1,
            },
            // SAFETY: lengths match the buffers (lhs m*k, rhs k*n, out m*n).
            // `from_raw_parts` consumes the handle by value, so clone the caller's
            // handles (clones share the binding).
            ArrayArg::from_raw_parts(lhs.clone(), m * k),
            ArrayArg::from_raw_parts(rhs.clone(), k * n),
            ArrayArg::from_raw_parts(out.clone(), m * n),
            // Scalar kernel args are passed as bare values (LaunchArg for T = T).
            m,
            k,
            n,
        );
    }
}

/// Host-slice launcher: upload `lhs`/`rhs`, allocate the result, run the kernel,
/// read it back. Backs `gemm_dense`; the cubecl `Runtime` bound stays inside.
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
    launch_gemm_on_handles::<R, F>(client, &lhs_handle, &rhs_handle, &out_handle, m, k, n);
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

    let out = dispatch_backend!(client, c, Rt, launch_gemm::<Rt, F>(c, lhs, rhs, m, k, n));
    Ok(out)
}

/// Dense matrix multiply over the opaque `Tensor` surface: `out = lhs @ rhs`,
/// written into `out`'s resident buffer in place.
///
/// quick-260529-mtx-2: launches directly on the operands' resident device
/// handles via the Phase-2 [`crate::device_buffer`] registry — no host transfer.
/// `lhs`, `rhs`, and `out` must be device-backed tensors built with
/// [`crate::device_buffer::upload`]; a `Tensor::placeholder` (sentinel
/// `BufferId`) yields [`AlgebraError::UnallocatedBuffer`], and a buffer resident
/// on another backend yields [`AlgebraError::BackendMismatch`]. Shapes drive the
/// product: `lhs` is `[M, K]`, `rhs` is `[K, N]`, and `out` must already be
/// allocated with `M*N` elements. Errors with `DimensionMismatch` if the
/// operands are not rank-2, the shared `K` disagrees, or `out` is the wrong size.
pub fn gemm(
    client: &AlgebraClient,
    lhs: &Tensor,
    rhs: &Tensor,
    out: &mut Tensor,
) -> Result<(), AlgebraError> {
    if lhs.rank() != 2 || rhs.rank() != 2 {
        return Err(AlgebraError::DimensionMismatch {
            op: "gemm",
            lhs: lhs.shape.clone(),
            rhs: rhs.shape.clone(),
        });
    }
    let (m, k) = (lhs.shape[0], lhs.shape[1]);
    let (k2, n) = (rhs.shape[0], rhs.shape[1]);
    if k != k2 || out.numel() != m * n {
        return Err(AlgebraError::DimensionMismatch {
            op: "gemm",
            lhs: lhs.shape.clone(),
            rhs: rhs.shape.clone(),
        });
    }
    let lb = crate::device_buffer::handle_of::<f64>(lhs.id.raw(), client, "gemm")?;
    let rb = crate::device_buffer::handle_of::<f64>(rhs.id.raw(), client, "gemm")?;
    let ob = crate::device_buffer::handle_of::<f64>(out.id.raw(), client, "gemm")?;
    if m * n == 0 {
        return Ok(());
    }
    dispatch_backend!(
        client,
        c,
        Rt,
        launch_gemm_on_handles::<Rt, f64>(c, &lb.handle, &rb.handle, &ob.handle, m, k, n)
    );
    Ok(())
}
