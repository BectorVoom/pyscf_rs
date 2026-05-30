//! quick-260529-mtx: randomized **oracle differential** test for the cubecl
//! generic-float strided per-axis reduction kernel (`reduce_sum_axis_dense`).
//!
//! Strategy (per docs/rust_crate_test_guideline.md — differential testing
//! against a trusted reference, randomized inputs, reproducible seed): the host
//! `out[o,i] = sum_a x[(o*axis_len + a)*inner + i]` computed with the SAME
//! ascending-`a` accumulation order as the device kernel is the ground truth.
//! Because the summation order matches, agreement is tight (a single device add
//! vs a single host add per term). We run the kernel on random tensors over a
//! spread of shapes and reduced axes and assert every output element matches.
//!
//! - `reduce_axis_kernel_matches_oracle_on_cpu` always runs (default `cpu` feature).
//! - `reduce_axis_kernel_matches_oracle_on_rocm` (`#[cfg(feature = "rocm")]`)
//!   runs the SAME differential check on real AMD hardware (gfx1152) via
//!   `cubecl_hip::HipRuntime`.
//!
//! Clients are constructed directly (not via `select_backend`) so the CPU and
//! ROCm tests never race on the process-global `PYSCF_BACKEND` env var.
//!
//! Verified in scope: device per-axis sum == host reference within 1e-12 over
//!   1-D/2-D/3-D shapes, every axis, prime dims, single-element axes, and
//!   `outer*inner` straddling the BLOCK=256 launch boundary (bounds-guard tail).
//!   Not verified: f32 precision path.

use cubecl::Runtime; // brings `::client` into scope for the concrete runtimes
use pyscf_algebra::{AlgebraClient, reduce_sum_axis_dense};

/// Deterministic LCG (Knuth/MMIX constants) → reproducible "random" tensors
/// without pulling in the `rand` crate. Maps the high bits to `[-1.0, 1.0)`.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (self.0 >> 11) as f64 / (1u64 << 53) as f64; // [0, 1)
        u * 2.0 - 1.0 // [-1, 1)
    }
}

fn random_tensor(rng: &mut Lcg, len: usize) -> Vec<f64> {
    (0..len).map(|_| rng.next_f64()).collect()
}

/// Host per-axis reduction in the same ascending-`a` order as the kernel.
fn host_reduce_axis(x: &[f64], shape: &[usize], axis: usize) -> Vec<f64> {
    let outer: usize = shape[..axis].iter().product();
    let axis_len = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let mut out = vec![0.0_f64; outer * inner];
    for o in 0..outer {
        for i in 0..inner {
            let base = o * axis_len * inner + i;
            let mut acc = 0.0_f64;
            for a in 0..axis_len {
                acc += x[base + a * inner];
            }
            out[o * inner + i] = acc;
        }
    }
    out
}

/// Run one random per-axis reduction on `client` and return the max abs
/// elementwise diff against the host reference.
fn check_case(client: &AlgebraClient, shape: &[usize], axis: usize, seed: u64) -> f64 {
    let len: usize = shape.iter().product();
    let mut rng = Lcg::new(seed);
    let x = random_tensor(&mut rng, len);
    let reference = host_reduce_axis(&x, shape, axis);

    let device = reduce_sum_axis_dense::<f64>(client, &x, shape, axis)
        .expect("reduce_sum_axis_dense should succeed");

    assert_eq!(
        device.len(),
        reference.len(),
        "reduced length must match host"
    );
    device
        .iter()
        .zip(reference.iter())
        .map(|(d, r)| (d - r).abs())
        .fold(0.0_f64, f64::max)
}

/// Sequential same-order adds → agreement is effectively exact; this tolerance
/// just absorbs any backend rounding-mode quirk while catching a wrong stride.
const TOL: f64 = 1e-12;

/// `(shape, axis)` cases: 1-D total, 2-D both axes (square/tall/wide/prime),
/// 3-D every axis, single-element axis, and `outer*inner` straddling BLOCK=256.
const CASES: &[(&[usize], usize)] = &[
    (&[1], 0),
    (&[5], 0),
    (&[2, 3], 0),
    (&[2, 3], 1),
    (&[3, 2], 0),
    (&[3, 2], 1),
    (&[16, 16], 0),
    (&[13, 17], 0),
    (&[13, 17], 1),
    (&[4, 5, 6], 0),
    (&[4, 5, 6], 1),
    (&[4, 5, 6], 2),
    (&[257, 1], 0),
    (&[1, 257], 1),
    (&[300, 2], 1),
];

/// Drive every case through `check_case` and assert the bound.
fn run_all(client: &AlgebraClient, base_seed: u64, label: &str) {
    for (idx, &(shape, axis)) in CASES.iter().enumerate() {
        let diff = check_case(client, shape, axis, base_seed.wrapping_add(idx as u64));
        assert!(
            diff < TOL,
            "{label} reduce_axis shape {shape:?} axis {axis}: max abs diff {diff:e} >= tol {TOL:e}"
        );
    }
}

#[test]
fn reduce_axis_kernel_matches_oracle_on_cpu() {
    let client = AlgebraClient::Cpu(cubecl_cpu::CpuRuntime::client(&cubecl_cpu::CpuDevice));
    run_all(&client, 0x5EDA_C100_u64, "CPU");
}

#[cfg(feature = "rocm")]
#[test]
fn reduce_axis_kernel_matches_oracle_on_rocm() {
    // Construct the HIP client directly on the default AMD device (gfx1152).
    let client = AlgebraClient::Rocm(cubecl_hip::HipRuntime::client(
        &cubecl_hip::AmdDevice::default(),
    ));
    assert!(
        matches!(client, AlgebraClient::Rocm(_)),
        "test must run on the ROCm backend, not a fallback"
    );
    run_all(&client, 0x5EDA_C10C_u64, "ROCm");
}
