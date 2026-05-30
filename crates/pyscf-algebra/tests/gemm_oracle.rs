//! quick-260529-i2x: randomized **oracle differential** test for the cubecl
//! generic-float GEMM kernel (`gemm_dense`).
//!
//! Strategy (per docs/rust_crate_test_guideline.md — differential testing
//! against a trusted reference, randomized inputs, reproducible seed): the
//! CPU `oracle_einsum("ij,jk->ik", ...)` is the bit-deterministic ground truth.
//! We run the device kernel on random matrices and assert the device result
//! matches the oracle within a tight tolerance.
//!
//! - `gemm_kernel_matches_oracle_on_cpu` always runs (default `cpu` feature).
//! - `gemm_kernel_matches_oracle_on_rocm` (`#[cfg(feature = "rocm")]`) runs the
//!   SAME differential check on real AMD hardware (gfx1152) via
//!   `cubecl_hip::HipRuntime` — the backend the task explicitly requested.
//!
//! Clients are constructed directly (not via `select_backend`) so the CPU and
//! ROCm tests never race on the process-global `PYSCF_BACKEND` env var.
//!
//! Verified in scope: device GEMM == CPU oracle within 1e-9 over a spread of
//!   random shapes (square, tall, wide, prime dims, single element).
//! Not verified here: f32 precision path (oracle is f64-only — see f32_smoke.rs
//!   for the f32 metadata path); shapes large enough to need tiling.

use cubecl::Runtime; // brings `::client` into scope for the concrete runtimes
use pyscf_algebra::{AlgebraClient, gemm_dense, oracle_einsum};

/// Deterministic LCG (Knuth/MMIX constants) → reproducible "random" matrices
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

fn random_matrix(rng: &mut Lcg, len: usize) -> Vec<f64> {
    (0..len).map(|_| rng.next_f64()).collect()
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// Run one random GEMM on `client` and compare against the oracle.
/// Returns the max elementwise absolute difference.
fn check_case(client: &AlgebraClient, m: usize, k: usize, n: usize, seed: u64) -> f64 {
    let mut rng = Lcg::new(seed);
    let lhs = random_matrix(&mut rng, m * k);
    let rhs = random_matrix(&mut rng, k * n);

    let device = gemm_dense::<f64>(client, &lhs, &rhs, m, k, n)
        .expect("gemm_dense should succeed for valid shapes");
    let reference = oracle_einsum("ij,jk->ik", &lhs, (m, k), &rhs, (k, n))
        .expect("oracle_einsum must accept the ij,jk->ik contraction");

    assert_eq!(device.len(), m * n, "device output length");
    assert_eq!(reference.len(), m * n, "oracle output length");
    max_abs_diff(&device, &reference)
}

/// Naive device accumulation vs pairwise oracle, f64, small K → differences are
/// far below this bound (~1e-15 in practice). Generous enough to absorb any
/// backend reassociation while still catching a wrong kernel.
const TOL: f64 = 1e-9;

/// Square, tall, wide, prime, and degenerate (1×1×1) shapes — exercises the
/// row/col index math and the bounds-guard tail in the kernel.
const SHAPES: &[(usize, usize, usize)] = &[
    (1, 1, 1),
    (2, 3, 2),
    (4, 4, 4),
    (8, 5, 3),
    (16, 16, 16),
    (7, 13, 11),
    (32, 8, 24),
];

#[test]
fn gemm_kernel_matches_oracle_on_cpu() {
    let client = AlgebraClient::Cpu(cubecl_cpu::CpuRuntime::client(&cubecl_cpu::CpuDevice));
    for (i, &(m, k, n)) in SHAPES.iter().enumerate() {
        let diff = check_case(&client, m, k, n, 0x51ED_C0DE_u64 + i as u64);
        assert!(
            diff < TOL,
            "CPU GEMM {m}x{k}x{n}: max abs diff {diff:e} >= tol {TOL:e}"
        );
    }
}

#[cfg(feature = "rocm")]
#[test]
fn gemm_kernel_matches_oracle_on_rocm() {
    // Construct the HIP client directly on the default AMD device (gfx1152).
    let client = AlgebraClient::Rocm(cubecl_hip::HipRuntime::client(
        &cubecl_hip::AmdDevice::default(),
    ));
    assert!(
        matches!(client, AlgebraClient::Rocm(_)),
        "test must run on the ROCm backend, not a fallback"
    );
    for (i, &(m, k, n)) in SHAPES.iter().enumerate() {
        let diff = check_case(&client, m, k, n, 0x0CA0_1A7E_u64 + i as u64);
        assert!(
            diff < TOL,
            "ROCm GEMM {m}x{k}x{n}: max abs diff {diff:e} >= tol {TOL:e}"
        );
    }
}
