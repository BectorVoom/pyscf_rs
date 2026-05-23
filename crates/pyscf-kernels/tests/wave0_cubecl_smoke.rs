//! W0-T2 smoke test: cubecl-cpu kernel launch from inside pyscf-kernels.
//!
//! Proves before plan 02-06 invests in the eval_gto port that:
//!   1. cubecl 0.10.0 + cubecl-cpu compile + link from this crate
//!   2. The `#[cube(launch_unchecked)]` macro expands cleanly
//!   3. `CpuRuntime::client(&device).create / .empty / .read_one` round-trips
//!   4. `client.create_from_slice(&[u8])` upload, kernel launch, and host
//!      readback agree on numerics for a trivial element-wise op (vector_add)
//!
//! Pattern lifted from cubecl 0.10.0 README + cubecl-core runtime_tests.
//! Per Phase 1 D-12 the workspace is pinned at `cubecl = "=0.10.0"`.
//!
//! Algebra wall: pyscf-kernels is on the ALG-06 carve-out allowlist alongside
//! pyscf-algebra and pyscf-runtime; this `cubecl::*` import is therefore
//! permitted here but forbidden in pyscf-gto, pyscf-scf, etc. Plan 02-01
//! Task 2 also adds pyscf-kernels to xtask/check_dependency_wall.rs.

#![cfg(feature = "cpu")]

use cubecl::prelude::*;
use cubecl_cpu::{CpuDevice, CpuRuntime};

/// Trivial element-wise add kernel — the cubecl-equivalent of "Hello, World".
#[cube(launch_unchecked)]
fn vector_add(lhs: &Array<f32>, rhs: &Array<f32>, out: &mut Array<f32>) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        out[i] = lhs[i] + rhs[i];
    }
}

#[test]
fn cubecl_cpu_vector_add_round_trip() {
    let device = CpuDevice;
    let client = CpuRuntime::client(&device);

    const N: usize = 256;
    let lhs_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let rhs_host: Vec<f32> = (0..N).map(|i| (2 * i) as f32).collect();

    let lhs = client.create_from_slice(bytemuck::cast_slice(&lhs_host));
    let rhs = client.create_from_slice(bytemuck::cast_slice(&rhs_host));
    let out = client.empty(N * std::mem::size_of::<f32>());

    // Launch grid: 32 threads per block, ceil(N/32) blocks. With N=256 that's
    // exactly 8 blocks of 32 threads = 256 threads, one per element.
    const BLOCK: u32 = 32;
    let blocks = (N as u32).div_ceil(BLOCK);

    // SAFETY: Buffer lengths match the slice lengths we passed at create time;
    // the kernel's bounds-check (`if i < out.len()`) guards against the small
    // overrun if N were not a multiple of BLOCK. Aliasing rule respected:
    // lhs/rhs are read-only Arrays, out is &mut. Handles are cloned before
    // ArrayArg::from_raw_parts because that fn consumes the Handle (cubecl
    // 0.10.0 signature: `(Handle, usize) -> ArrayArg<R>`); we keep the
    // originals for the post-launch read_one(out) below.
    unsafe {
        vector_add::launch_unchecked::<CpuRuntime>(
            &client,
            CubeCount::Static(blocks, 1, 1),
            CubeDim::new_1d(BLOCK),
            ArrayArg::from_raw_parts(lhs.clone(), N),
            ArrayArg::from_raw_parts(rhs.clone(), N),
            ArrayArg::from_raw_parts(out.clone(), N),
        );
    }

    let bytes = client.read_one(out).expect("read_one out");
    let result: &[f32] = bytemuck::cast_slice(&bytes);

    assert_eq!(
        result.len(),
        N,
        "result length must equal N (got {})",
        result.len()
    );
    for (i, &got) in result.iter().enumerate() {
        let expected = (3 * i) as f32; // lhs[i] + rhs[i] = i + 2i = 3i
        approx::assert_abs_diff_eq!(got, expected, epsilon = 1e-6);
    }
}
