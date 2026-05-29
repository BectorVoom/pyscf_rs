//! REDUCE — generic-float cubecl reduction kernel + backend-dispatched host launcher.
//!
//! quick-260529-jcx: refactored from the Phase-1 `NotYetImplemented` stub into
//! a real cubecl `#[cube(launch)]` kernel generic over the device float
//! `F: Float` (per docs/manual/Cubecl/Cubecl_generics.md), plus a host-slice
//! launcher dispatched off `AlgebraClient` so the cubecl `Runtime` generic
//! stays inside the ALG-06 wall. Mirrors the `dot.rs` sibling
//! (quick-260529-iji) exactly.
//!
//! `reduce_sum(x) = sum(x[i])` is the axis-free reduction. The device kernel
//! computes `groups` PARTIAL sums — each thread sequentially accumulates a
//! contiguous `CHUNK` slice of the input into one partial in `F` (bounds-guarded
//! against the tail) — and the host sums those `groups` partials in `F`. This
//! keeps the kernel naive and obviously-correct (no device atomics /
//! tree-reduction) while doing the bulk of the work on-device.
//!
//! Two surfaces:
//!   * `reduce_sum()` — the opaque `Tensor`-based API. STILL a stub: `Tensor`
//!     carries a sentinel `BufferId` (no device allocator until Phase 2), so it
//!     cannot read device data yet.
//!   * `reduce_sum_dense()` — the working device path: takes a host slice + an
//!     `AlgebraClient`, runs the generic kernel on the resolved backend (CPU,
//!     ROCm, CUDA, WGPU), and returns the scalar sum. Exercised by the random
//!     oracle differential test (`tests/reduce_oracle.rs`) on ROCm.

use crate::scalar::DeviceScalar;
use crate::{AlgebraClient, AlgebraError, Tensor};
use cubecl::Runtime;
use cubecl::bytes::Bytes;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;

/// Partial-sum kernel: thread `tid` sequentially sums input indices
/// `[tid*chunk, (tid+1)*chunk)` (clamped to `n`) into one partial in `F`,
/// writing it to `out[tid]`. Generic over the device float so the same kernel
/// monomorphizes for f32 (GPU speed path) and f64 (chemistry precision path) —
/// see `Cubecl_generics.md`. The final sum of the `groups` partials runs on the
/// host (see [`launch_reduce_sum`]).
#[cube(launch)]
fn reduce_kernel<F: Float>(x: &Array<F>, out: &mut Array<F>, n: usize, chunk: usize) {
    // `ABSOLUTE_POS` and `Array` indices are `usize` in cubecl 0.10, so the
    // dimension scalars are `usize` too — no casts.
    let tid = ABSOLUTE_POS;
    let start = tid * chunk;
    // Bounds guard: the launch rounds the thread count up to a whole number of
    // blocks, so tail threads (start >= n) write the identity and never index
    // out of range.
    let mut acc = F::from_int(0);
    if start < n {
        let mut end = start + chunk;
        if end > n {
            end = n;
        }
        // Sequential accumulation over this thread's contiguous slice.
        let mut i = start;
        while i < end {
            acc += x[i];
            i += 1;
        }
    }
    out[tid] = acc;
}

/// Threads per block for the reduce launch. One thread produces one partial; the
/// grid is sized to cover all partials.
const BLOCK: u32 = 256;

/// Elements summed sequentially per thread (per partial). `n.div_ceil(CHUNK)`
/// gives the number of partials; the grid covers them. Kept naive and
/// obviously-correct.
const CHUNK: usize = 256;

/// Runtime-generic launcher: upload `x`, allocate the partials buffer, launch
/// `reduce_kernel`, read the partials back to host, then sum them in `F`. `R` is
/// kept generic so a single body serves every backend; callers reach it only
/// through `reduce_sum_dense` (which picks `R` from the active `AlgebraClient`),
/// so the cubecl `Runtime` bound never escapes the wall.
fn launch_reduce_sum<R: Runtime, F: DeviceScalar>(client: &ComputeClient<R>, x: &[F]) -> F {
    // Empty input → identity sum; never launch a zero-length grid.
    if x.is_empty() {
        return F::from_int(0);
    }

    let n = x.len();
    let partials = n.div_ceil(CHUNK);

    let x_handle = client.create(Bytes::from_elems(x.to_vec()));
    let out_handle = client.empty(partials * core::mem::size_of::<F>());

    let groups = (partials as u32).div_ceil(BLOCK);

    reduce_kernel::launch::<F, R>(
        client,
        CubeCount::Static(groups, 1, 1),
        CubeDim::new_1d(BLOCK),
        // SAFETY: lengths match the buffers just allocated above. `from_raw_parts`
        // consumes the handle by value; clone the output handle so it survives
        // for the read-back below.
        unsafe { ArrayArg::from_raw_parts(x_handle, n) },
        unsafe { ArrayArg::from_raw_parts(out_handle.clone(), partials) },
        // Scalar kernel args are passed as bare values (LaunchArg for T = T).
        n,
        CHUNK,
    );

    let bytes = client.read(vec![out_handle]);
    let parts: Vec<F> = bytemuck::cast_slice::<u8, F>(&bytes[0]).to_vec();

    // Host-sum the partials in `F`. A plain `acc += p` add is NOT an FMA
    // (check-no-fma only flags FMA in pyscf_* symbols anyway).
    let mut acc = F::from_int(0);
    for &p in &parts {
        acc += p;
    }
    acc
}

/// Dense axis-free sum on the device: `reduce_sum(x) = sum(x[i])`.
///
/// Generic over `F: DeviceScalar` (f32 or f64). Empty input returns the identity
/// `0`. Dispatches the generic cubecl kernel on whichever backend `client`
/// resolved to — the `Runtime` type is selected here and never appears in the
/// signature, honoring the ALG-06 wall. Matches `oracle_sum`'s signature shape.
pub fn reduce_sum_dense<F: DeviceScalar>(
    client: &AlgebraClient,
    x: &[F],
) -> Result<F, AlgebraError> {
    let out = match client {
        AlgebraClient::Cpu(c) => launch_reduce_sum::<cubecl_cpu::CpuRuntime, F>(c, x),
        #[cfg(feature = "cuda")]
        AlgebraClient::Cuda(c) => launch_reduce_sum::<cubecl_cuda::CudaRuntime, F>(c, x),
        #[cfg(feature = "wgpu")]
        AlgebraClient::Wgpu(c) => launch_reduce_sum::<cubecl_wgpu::WgpuRuntime, F>(c, x),
        #[cfg(feature = "rocm")]
        AlgebraClient::Rocm(c) => launch_reduce_sum::<cubecl_hip::HipRuntime, F>(c, x),
    };
    Ok(out)
}

/// Total reduction over the opaque `Tensor` surface: `out[0] = sum(x[i])`,
/// writing the scalar into `out`'s buffer in place.
///
/// quick-260529-mtx: wired through the Phase-2 [`crate::device_buffer`] registry.
/// Both `x` and `out` must be device-backed tensors built with
/// [`crate::device_buffer::upload`]; a `Tensor::placeholder` (sentinel
/// `BufferId`) yields [`AlgebraError::UnallocatedBuffer`]. `x` is read from the
/// registry and summed by the oracle-tested [`reduce_sum_dense`] launcher on
/// `client`'s backend; the scalar result is written back to `out`.
///
/// Scope: the underlying device kernel is axis-free (full reduction), so only
/// the total-reduction case is wired — `out` must be a single-element buffer.
/// A non-scalar `out` (per-axis reduction) returns `NotYetImplemented`; `axis`
/// is accepted for API compatibility but the full reduction collapses every
/// axis. (Per-axis reduction needs a strided kernel beyond `reduce_sum_dense`.)
pub fn reduce_sum(
    client: &AlgebraClient,
    x: &Tensor,
    _axis: usize,
    out: &mut Tensor,
) -> Result<(), AlgebraError> {
    if out.numel() != 1 {
        return Err(AlgebraError::NotYetImplemented {
            phase: 2,
            what: "per-axis reduce_sum over Tensor — only full reduction to a scalar `out` is wired; \
                   use reduce_sum_dense for the host-slice total sum",
        });
    }
    let x_host = crate::device_buffer::read_raw::<f64>(x.id.raw(), "reduce_sum")?;
    let total = reduce_sum_dense::<f64>(client, &x_host)?;
    crate::device_buffer::write_back::<f64>(out.id.raw(), &[total], "reduce_sum")
}
