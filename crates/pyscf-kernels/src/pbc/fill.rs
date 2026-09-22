//! B-01 — the single device fill kernel: write a literal `0.0` over a buffer.
//!
//! Moved out of `local_vmat.rs` (where it was `zero_imag_kernel`, serving only
//! the gamma-point imaginary plane) so that [`crate::pbc::AoKAccumulator::zeros`]
//! can allocate its two planes with `client.empty` (T4 — a DIRTY recycled
//! buffer) and fill them on the device instead of uploading a host
//! `vec![0.0f64; nkpts · n]` twice. A buffer of literal `0.0` is a buffer of
//! literal `0.0` however it was produced, so this is bit-exact by construction.
//!
//! Generic over the device float (`F: Float`, AGENTS.md §3 / RULE 5).

use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::server::Handle;
use pyscf_algebra::launch::launch_1d;

use crate::scalar::DeviceScalar;

/// Write a literal `0.0` over `n` elements starting at `base`, spaced `stride`
/// apart, in place.
///
/// `base`/`stride` serve both accumulator layouts: `(0, 1)` fills a whole
/// contiguous buffer ([`fill_zero`]), while `(k · stride_k, 1)` zeroes one
/// k-point's gamma plane and `(k, nkpts)` its point-major strided form.
#[cube(launch_unchecked)]
pub(crate) fn fill_zero_kernel<F: Float>(buf: &mut Array<F>, base: usize, stride: usize, n: usize) {
    let i = ABSOLUTE_POS;
    if i < n {
        buf[base + i * stride] = F::from_int(0);
    }
}

/// Zero `n` elements of `handle`'s buffer starting at `base`, spaced `stride`
/// apart. `ao_len` is the buffer's full element count (the bounds the kernel
/// is checked against); `n` is how many lanes to launch.
pub(crate) fn fill_range<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    handle: &Handle,
    ao_len: usize,
    base: usize,
    stride: usize,
    n: usize,
) {
    if n == 0 {
        return;
    }
    let (count, dim) = launch_1d(client, n, 1);
    unsafe {
        fill_zero_kernel::launch_unchecked::<F, R>(
            client,
            count,
            dim,
            // SAFETY: the caller pairs `ao_len` with `(base, stride, n)` that
            // keep every `base + i · stride` in range, and the kernel guards
            // `i < n`.
            ArrayArg::from_raw_parts(handle.clone(), ao_len),
            base,
            stride,
            n,
        );
    }
}

/// Zero a whole `len`-element buffer. `base = 0`, `stride = 1`, `n = len`.
pub(crate) fn fill_zero<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    handle: &Handle,
    len: usize,
) {
    fill_range::<R, F>(client, handle, len, 0, 1, len);
}
