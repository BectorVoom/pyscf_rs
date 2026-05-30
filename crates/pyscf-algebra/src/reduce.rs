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
//! Surfaces:
//!   * `reduce_sum_dense()` — the working device path for the axis-free TOTAL
//!     sum: takes a host slice + an `AlgebraClient`, runs the partial-sum kernel
//!     on the resolved backend (CPU, ROCm, CUDA, WGPU), and returns the scalar
//!     sum. Exercised by `tests/reduce_oracle.rs` on ROCm.
//!   * `reduce_sum_axis_dense()` — the PER-AXIS device path (quick-260529-mtx):
//!     a strided kernel sums one axis of a row-major tensor, returning the
//!     reduced slice. Exercised by `tests/reduce_axis_oracle.rs` on ROCm.
//!   * `reduce_sum()` — the opaque `Tensor`-based API, wired through the Phase-2
//!     `crate::device_buffer` registry onto the strided per-axis kernel.

use crate::scalar::DeviceScalar;
use crate::{AlgebraClient, AlgebraError, Tensor};
use cubecl::Runtime;
use cubecl::bytes::Bytes;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::server::Handle;

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

/// Strided per-axis reduction kernel: thread `tid` owns one output element
/// `(o, i)` and sums the `axis_len` inputs that collapse into it, walking the
/// reduced axis with stride `inner`. For a row-major tensor flattened as
/// `((o * axis_len) + a) * inner + i`, the inputs summed for output `(o, i)` are
/// `x[o*axis_len*inner + i + a*inner]` for `a in 0..axis_len`. Generic over the
/// device float so the same kernel monomorphizes for f32 (GPU speed path) and
/// f64 (chemistry precision path) — see `Cubecl_generics.md`.
#[cube(launch)]
fn reduce_axis_kernel<F: Float>(
    x: &Array<F>,
    out: &mut Array<F>,
    outer: usize,
    axis_len: usize,
    inner: usize,
) {
    // `ABSOLUTE_POS` and `Array` indices are `usize` in cubecl 0.10 — no casts.
    let tid = ABSOLUTE_POS;
    let n_out = outer * inner;
    // Bounds guard: the launch rounds the thread count up to a whole number of
    // blocks, so tail threads must not write out of range.
    if tid < n_out {
        let o = tid / inner;
        let i = tid % inner;
        let base = o * axis_len * inner + i;
        // Sequential strided accumulation along the reduced axis, in `F`.
        let mut acc = F::from_int(0);
        let mut a = 0;
        while a < axis_len {
            acc += x[base + a * inner];
            a += 1;
        }
        out[tid] = acc;
    }
}

/// Core launch on resident device handles: strided per-axis reduction of the `x`
/// handle into the `out` handle (`outer*inner` elements), with NO host transfer.
/// Both the host-slice path ([`reduce_sum_axis_dense`]) and the registry-backed
/// `Tensor` path ([`reduce_sum`]) drive this; the caller owns `out` (a fresh temp
/// for the dense path, the resident result buffer for the `Tensor` path).
fn launch_reduce_axis_on_handles<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    x: &Handle,
    out: &Handle,
    x_len: usize,
    outer: usize,
    axis_len: usize,
    inner: usize,
) {
    let n_out = outer * inner;
    let groups = (n_out as u32).div_ceil(BLOCK);

    reduce_axis_kernel::launch::<F, R>(
        client,
        CubeCount::Static(groups, 1, 1),
        CubeDim::new_1d(BLOCK),
        // SAFETY: `x_len == outer*axis_len*inner` and `out` holds `n_out`.
        // `from_raw_parts` consumes the handle by value, so clone the caller's
        // handles (clones share the binding).
        unsafe { ArrayArg::from_raw_parts(x.clone(), x_len) },
        unsafe { ArrayArg::from_raw_parts(out.clone(), n_out) },
        // Scalar dimension args are passed as bare values (LaunchArg for usize).
        outer,
        axis_len,
        inner,
    );
}

/// Host-slice launcher: upload `x`, allocate the `outer*inner` output, run the
/// strided kernel, read it back. Backs `reduce_sum_axis_dense`; the cubecl
/// `Runtime` bound stays inside.
fn launch_reduce_axis<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    x: &[F],
    outer: usize,
    axis_len: usize,
    inner: usize,
) -> Vec<F> {
    let n_out = outer * inner;
    let x_handle = client.create(Bytes::from_elems(x.to_vec()));
    let out_handle = client.empty(n_out * core::mem::size_of::<F>());
    launch_reduce_axis_on_handles::<R, F>(
        client,
        &x_handle,
        &out_handle,
        x.len(),
        outer,
        axis_len,
        inner,
    );
    let bytes = client.read(vec![out_handle]);
    bytemuck::cast_slice::<u8, F>(&bytes[0]).to_vec()
}

/// `(outer, axis_len, inner)` decomposition of `shape` about `axis`: the product
/// of the dims before `axis`, the reduced dim itself, and the product of the
/// dims after it. The flat layout is row-major, so this triple is all the
/// strided kernel needs. Returns `DimensionMismatch` if `axis` is out of range.
fn axis_decomposition(shape: &[usize], axis: usize) -> Result<(usize, usize, usize), AlgebraError> {
    if axis >= shape.len() {
        return Err(AlgebraError::DimensionMismatch {
            op: "reduce_sum_axis",
            lhs: shape.to_vec(),
            rhs: vec![axis],
        });
    }
    let outer: usize = shape[..axis].iter().product();
    let axis_len = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    Ok((outer, axis_len, inner))
}

/// The output shape of a per-axis reduction: `shape` with `axis` removed.
pub fn reduced_shape(shape: &[usize], axis: usize) -> Vec<usize> {
    shape
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| if i == axis { None } else { Some(d) })
        .collect()
}

/// Dense per-axis sum on the device: reduce `x` (row-major, logical shape
/// `shape`) along `axis`, returning the `outer*inner` result in row-major order
/// over the surviving axes (i.e. over [`reduced_shape`]`(shape, axis)`).
///
/// Generic over `F: DeviceScalar` (f32 or f64). Validates `axis` is in range and
/// `x.len() == product(shape)`. A zero-size output (some surviving dim is 0)
/// returns an empty `Vec`; a zero-length reduced axis returns all-zero sums (the
/// empty sum) without launching a grid. Dispatches the strided kernel on
/// whichever backend `client` resolved to — the `Runtime` type never appears in
/// the signature (ALG-06 wall). A 1-D `shape` reduced along axis 0 is the total
/// sum (`outer == inner == 1`).
pub fn reduce_sum_axis_dense<F: DeviceScalar>(
    client: &AlgebraClient,
    x: &[F],
    shape: &[usize],
    axis: usize,
) -> Result<Vec<F>, AlgebraError> {
    let (outer, axis_len, inner) = axis_decomposition(shape, axis)?;
    let expected = outer * axis_len * inner;
    if x.len() != expected {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("x len {expected} (= product{shape:?})"),
            actual: x.len().to_string(),
        });
    }

    let n_out = outer * inner;
    // A surviving dim is 0 → empty output; never launch a zero-length grid.
    if n_out == 0 {
        return Ok(Vec::new());
    }
    // Reducing a zero-length axis → every output is the empty sum (0). The input
    // is empty here, so fill directly rather than launch on an empty buffer.
    if axis_len == 0 {
        return Ok(vec![F::from_int(0); n_out]);
    }

    let out = match client {
        AlgebraClient::Cpu(c) => {
            launch_reduce_axis::<cubecl_cpu::CpuRuntime, F>(c, x, outer, axis_len, inner)
        }
        #[cfg(feature = "cuda")]
        AlgebraClient::Cuda(c) => {
            launch_reduce_axis::<cubecl_cuda::CudaRuntime, F>(c, x, outer, axis_len, inner)
        }
        #[cfg(feature = "wgpu")]
        AlgebraClient::Wgpu(c) => {
            launch_reduce_axis::<cubecl_wgpu::WgpuRuntime, F>(c, x, outer, axis_len, inner)
        }
        #[cfg(feature = "rocm")]
        AlgebraClient::Rocm(c) => {
            launch_reduce_axis::<cubecl_hip::HipRuntime, F>(c, x, outer, axis_len, inner)
        }
    };
    Ok(out)
}

/// Per-axis reduction over the opaque `Tensor` surface: `out = sum(x, axis)`,
/// written into `out`'s resident buffer in place. On success `out`'s shape is set
/// to the reduced shape (`x.shape` with `axis` removed).
///
/// quick-260529-mtx-2: launches the strided kernel directly on the resident
/// device handles via the Phase-2 [`crate::device_buffer`] registry — no host
/// transfer. Both `x` and `out` must be device-backed tensors built with
/// [`crate::device_buffer::upload`]; a `Tensor::placeholder` (sentinel
/// `BufferId`) yields [`AlgebraError::UnallocatedBuffer`], and a buffer resident
/// on another backend yields [`AlgebraError::BackendMismatch`]. `x`'s shape
/// drives the reduction; `out` must already be allocated with
/// `product(reduced_shape(x.shape, axis))` elements. For a 1-D `x`, reducing
/// axis 0 collapses to the scalar total sum.
///
/// Errors with `DimensionMismatch` if `axis` is out of range or `out`'s element
/// count does not match the reduced size.
pub fn reduce_sum(
    client: &AlgebraClient,
    x: &Tensor,
    axis: usize,
    out: &mut Tensor,
) -> Result<(), AlgebraError> {
    // Validate `axis` against x's shape (metadata-only) before touching the
    // registry, then fetch operands operand-first (missing x → UnallocatedBuffer).
    let (outer, axis_len, inner) = axis_decomposition(&x.shape, axis)?;
    let xb = crate::device_buffer::handle_of::<f64>(x.id.raw(), client, "reduce_sum")?;

    let out_shape = reduced_shape(&x.shape, axis);
    let n_out = outer * inner;
    let ob = crate::device_buffer::handle_of::<f64>(out.id.raw(), client, "reduce_sum")?;
    if ob.len != n_out {
        return Err(AlgebraError::DimensionMismatch {
            op: "reduce_sum",
            lhs: out.shape.clone(),
            rhs: out_shape,
        });
    }

    if n_out != 0 {
        let x_len = xb.len;
        match client {
            AlgebraClient::Cpu(c) => launch_reduce_axis_on_handles::<cubecl_cpu::CpuRuntime, f64>(
                c, &xb.handle, &ob.handle, x_len, outer, axis_len, inner,
            ),
            #[cfg(feature = "cuda")]
            AlgebraClient::Cuda(c) => {
                launch_reduce_axis_on_handles::<cubecl_cuda::CudaRuntime, f64>(
                    c, &xb.handle, &ob.handle, x_len, outer, axis_len, inner,
                )
            }
            #[cfg(feature = "wgpu")]
            AlgebraClient::Wgpu(c) => {
                launch_reduce_axis_on_handles::<cubecl_wgpu::WgpuRuntime, f64>(
                    c, &xb.handle, &ob.handle, x_len, outer, axis_len, inner,
                )
            }
            #[cfg(feature = "rocm")]
            AlgebraClient::Rocm(c) => launch_reduce_axis_on_handles::<cubecl_hip::HipRuntime, f64>(
                c, &xb.handle, &ob.handle, x_len, outer, axis_len, inner,
            ),
        }
    }
    out.shape = out_shape;
    Ok(())
}
