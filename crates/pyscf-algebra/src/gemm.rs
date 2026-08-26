//! GEMM — generic-float cubecl kernel + backend-dispatched host launcher.
//!
//! quick-260529-i2x: refactored from the Phase-1 `NotYetImplemented` stub
//! into a real cubecl `#[cube(launch)]` kernel generic over the device float
//! `F: Float` (per docs/manual/Cubecl/Cubecl_generics.md), plus a host-slice
//! launcher dispatched off `AlgebraClient` so the cubecl `Runtime` generic
//! stays inside the ALG-06 wall.
//!
//! Two surfaces:
//!   * `gemm()` — the opaque `Tensor`-based API, launching on the operands'
//!     resident device handles via the Phase-2 [`crate::device_buffer`] registry.
//!   * `gemm_dense()` — the host-slice path: takes host slices + an
//!     `AlgebraClient`, runs the generic kernel on the resolved backend (CPU,
//!     ROCm, CUDA, WGPU), and returns the result. Exercised by the random
//!     oracle differential test (`tests/gemm_oracle.rs`) on ROCm.
//!
//! # Two kernels, and why the barrier one is not the default
//!
//! This used to be a single shared-memory tiled kernel: stage a 16x16 tile of
//! each operand, `sync_cube`, accumulate, `sync_cube`. That is the textbook
//! shape for a GPU. It is also catastrophically wrong for CubeCL's CPU runtime —
//! the DEFAULT backend of this crate (`default = ["cpu"]`, ALG-03) — where a
//! cube barrier is not a hardware instruction and a 16x16 cube is 256 operating
//! system threads asked to synchronize twice per tile. A 64x64x64 product did
//! not finish in ten minutes on that path.
//!
//! So the kernels here never synchronize and never share:
//!
//!   * [`gemm_simple_kernel`] — one unit per output *vector*. A unit owns `N`
//!     adjacent columns of one output row: the `lhs` element is a scalar splat
//!     and the `rhs` row and accumulator are [`Vector`]s, so the inner `k` loop
//!     is one vector FMA per step instead of a scalar one.
//!   * [`gemm_row_tiled_kernel`] — the same, but one unit owns [`ROWS`] stacked
//!     output vectors. The `rhs` vector loaded on each `k` step is reused by
//!     every accumulator, which turns roughly one FLOP per byte into `2 * ROWS`
//!     FLOPs per byte. That is what a GPU needs to leave memory-bound territory;
//!     on a CPU it is merely not faster, never pathological.
//!
//! [`pick_plan`] chooses between them from the device's own properties. Both
//! compute the same product to within floating-point associativity.

use crate::launch::{has_planes, launch_1d, line_size_for, upload};
use crate::scalar::DeviceScalar;
use crate::{AlgebraClient, AlgebraError, Tensor};
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::server::Handle;

/// Output rows one unit accumulates in [`gemm_row_tiled_kernel`].
///
/// Every step of the `k` loop is one vector load from `rhs` reused across all
/// `ROWS` accumulators, so arithmetic per byte read grows with this number until
/// the register file runs out — and then falls off a cliff. Eight is the knee on
/// the RDNA3-class hardware this crate targets.
const ROWS: usize = 8;

/// Which no-barrier kernel a launch should use.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Plan {
    /// One unit per output vector. Portable and never pathological.
    Simple,
    /// One unit per [`ROWS`] stacked output vectors, accumulating in registers.
    RowTiled,
}

/// `PYSCF_GEMM_KERNEL`, read once: `simple` or `row_tiled` to override
/// [`pick_plan`]'s device-derived choice.
///
/// A backend tuning knob, not a semantic one — both kernels compute the same
/// product to within floating-point associativity. It exists because the plan is
/// otherwise chosen from hardware properties, which makes the row-tiled path
/// unreachable (and so untestable) on a machine whose only f64-capable backend
/// is the CPU runtime. `tests/gemm_kernels.rs` uses it to pin both paths against
/// the same reference.
fn plan_override() -> Option<Plan> {
    static OVERRIDE: std::sync::OnceLock<Option<Plan>> = std::sync::OnceLock::new();
    *OVERRIDE.get_or_init(|| match std::env::var("PYSCF_GEMM_KERNEL").as_deref() {
        Ok("simple") => Some(Plan::Simple),
        Ok("row_tiled") => Some(Plan::RowTiled),
        _ => None,
    })
}

/// Lanes the row-tiled kernel must still dispatch for its reuse to be worth the
/// eight-fold cut in parallelism it costs.
///
/// Row-tiling trades lanes for arithmetic intensity: `ROWS` output rows collapse
/// into one unit. That is the right trade only while there are still enough
/// units left to fill the device. The degenerate case is GEMV, which reaches
/// this module as GEMM with a single output column ([`crate::gemv`]): there
/// `n_lines` is 1, so row-tiling would run a 4096-row product on 512 lanes and
/// leave most of a GPU idle, with no `rhs` reuse to show for it — one column is
/// loaded once either way.
const ROW_TILED_LANES_MIN: usize = 4096;

/// Pick the kernel from the device's own properties and the operand shape.
///
/// Register-tiling only pays where a unit is a lane of a wide SIMD engine with a
/// large register file — i.e. where the device reports real hardware planes. On
/// CubeCL's CPU runtime a unit is a thread; `ROWS` accumulators per thread buys
/// nothing there, and the smaller lane count would leave cores idle on the
/// narrow matrices that dominate an SCF loop. `m < ROWS` cannot fill even one
/// row tile, and a lane count below [`ROW_TILED_LANES_MIN`] cannot fill a GPU,
/// so both take the simple path regardless.
///
/// The GPU arm of this choice is reasoned from the shape rather than measured:
/// the only f64-capable backend on the development machine is the CPU runtime,
/// whose `plane_size_max` is 1. `PYSCF_GEMM_KERNEL` overrides it — see
/// [`plan_override`] — and `tests/gemm_kernels.rs` pins both kernels' results.
fn pick_plan<R: Runtime>(client: &ComputeClient<R>, m: usize, n_lines: usize) -> Plan {
    if let Some(forced) = plan_override() {
        return forced;
    }
    let row_tiled_lanes = m.div_ceil(ROWS) * n_lines;
    if has_planes(client) && m >= ROWS && row_tiled_lanes >= ROW_TILED_LANES_MIN {
        Plan::RowTiled
    } else {
        Plan::Simple
    }
}

/// One unit per output vector: `out = lhs @ rhs` (row-major).
///
/// `n_lines` and the `rhs`/`out` indices are counted in vectors of `N` elements;
/// `lhs` stays scalar because a unit reads one `lhs` value per step and splats it
/// across the accumulator. No shared memory and no synchronization, so this is
/// safe and sensible on every backend.
#[cube(launch_unchecked)]
fn gemm_simple_kernel<F: Float + CubeElement, N: Size>(
    lhs: &Array<F>,
    rhs: &Array<Vector<F, N>>,
    out: &mut Array<Vector<F, N>>,
    m: usize,
    n_lines: usize,
    k: usize,
) {
    if ABSOLUTE_POS < m * n_lines {
        let row = ABSOLUTE_POS / n_lines;
        let col = ABSOLUTE_POS % n_lines;
        let lhs_base = row * k;
        let mut acc = Vector::<F, N>::new(F::from_int(0));
        for p in 0..k {
            acc += Vector::<F, N>::new(lhs[lhs_base + p]) * rhs[col + p * n_lines];
        }
        out[ABSOLUTE_POS] = acc;
    }
}

/// One unit per `ROWS` stacked output vectors.
///
/// The `rhs` vector loaded on each `k` step is reused by every accumulator, and
/// the `lhs` values a unit reads are the same for every unit in the plane — a
/// broadcast out of cache. Nothing is shared and nothing synchronizes.
///
/// The tail guard is hoisted out of the `k` loop: a row past the end reads row 0
/// instead of running off the buffer, and its result is simply never stored.
#[cube(launch_unchecked)]
fn gemm_row_tiled_kernel<F: Float + CubeElement, N: Size>(
    lhs: &Array<F>,
    rhs: &Array<Vector<F, N>>,
    out: &mut Array<Vector<F, N>>,
    m: usize,
    n_lines: usize,
    k: usize,
    lanes: usize,
    #[comptime] rows: usize,
) {
    if ABSOLUTE_POS < lanes {
        let row0 = (ABSOLUTE_POS / n_lines) * rows;
        let col = ABSOLUTE_POS % n_lines;

        let mut row_offset = Array::<usize>::new(rows);
        let mut acc = Array::<Vector<F, N>>::new(rows);
        #[unroll]
        for i in 0..rows {
            let row = select(row0 + i < m, row0 + i, 0usize);
            row_offset[i] = row * k;
            acc[i] = Vector::<F, N>::new(F::from_int(0));
        }

        for p in 0..k {
            let rv = rhs[col + p * n_lines];
            #[unroll]
            for i in 0..rows {
                acc[i] += Vector::<F, N>::new(lhs[row_offset[i] + p]) * rv;
            }
        }

        #[unroll]
        for i in 0..rows {
            if row0 + i < m {
                out[(row0 + i) * n_lines + col] = acc[i];
            }
        }
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
    // The width has to divide `n` exactly, so no unit's vector straddles two
    // rows of `rhs` or of `out`.
    let line = line_size_for::<R, F>(client, n);
    let n_lines = n / line;

    match pick_plan(client, m, n_lines) {
        Plan::Simple => {
            let lanes = m * n_lines;
            let (count, dim) = launch_1d(client, lanes, k * line);
            unsafe {
                gemm_simple_kernel::launch_unchecked::<F, R>(
                    client,
                    count,
                    dim,
                    line,
                    // SAFETY: lengths match the buffers (lhs m*k, rhs k*n, out m*n).
                    // `from_raw_parts` consumes the handle by value, so clone the
                    // caller's handles (clones share the binding).
                    ArrayArg::from_raw_parts(lhs.clone(), m * k),
                    ArrayArg::from_raw_parts(rhs.clone(), k * n),
                    ArrayArg::from_raw_parts(out.clone(), m * n),
                    // Scalar kernel args are passed as bare values (LaunchArg for T = T).
                    m,
                    n_lines,
                    k,
                );
            }
        }
        Plan::RowTiled => {
            let lanes = m.div_ceil(ROWS) * n_lines;
            let (count, dim) = launch_1d(client, lanes, k * line * ROWS);
            unsafe {
                gemm_row_tiled_kernel::launch_unchecked::<F, R>(
                    client,
                    count,
                    dim,
                    line,
                    // SAFETY: as above.
                    ArrayArg::from_raw_parts(lhs.clone(), m * k),
                    ArrayArg::from_raw_parts(rhs.clone(), k * n),
                    ArrayArg::from_raw_parts(out.clone(), m * n),
                    m,
                    n_lines,
                    k,
                    lanes,
                    ROWS,
                );
            }
        }
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
    let lhs_handle = upload(client, lhs);
    let rhs_handle = upload(client, rhs);
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
    // Never launch a zero-length grid; an empty product is an empty result.
    if m * n == 0 {
        return Ok(Vec::new());
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
