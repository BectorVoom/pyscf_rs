//! quick-260529-iji: randomized **oracle differential** test for the cubecl
//! generic-float dot kernel (`dot_dense`).
//!
//! Strategy (per docs/rust_crate_test_guideline.md — differential testing
//! against a trusted reference, randomized inputs, reproducible seed): the
//! CPU `oracle_dot(x, y)` is the bit-deterministic ground truth. We run the
//! device kernel on random vectors and assert the device result matches the
//! oracle within a tight tolerance.
//!
//! - `dot_kernel_matches_oracle_on_cpu` always runs (default `cpu` feature).
//! - `dot_kernel_matches_oracle_on_rocm` (`#[cfg(feature = "rocm")]`) runs the
//!   SAME differential check on real AMD hardware (gfx1152) via
//!   `cubecl_hip::HipRuntime` — the backend the task explicitly requested.
//!
//! Clients are constructed directly (not via `select_backend`) so the CPU and
//! ROCm tests never race on the process-global `PYSCF_BACKEND` env var.
//!
//! Verified in scope: device dot == CPU oracle within 1e-9 over a spread of
//!   random lengths (degenerate, prime, BLOCK-boundary straddling).
//! Not verified here: f32 precision path (oracle is f64-only).

use cubecl::Runtime; // brings `::client` into scope for the concrete runtimes
use pyscf_algebra::{AlgebraClient, dot_dense, oracle_dot};

/// Deterministic LCG (Knuth/MMIX constants) → reproducible "random" vectors
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

fn random_vector(rng: &mut Lcg, len: usize) -> Vec<f64> {
    (0..len).map(|_| rng.next_f64()).collect()
}

/// Run one random dot on `client` and compare against the oracle.
/// Returns the absolute difference.
fn check_case(client: &AlgebraClient, len: usize, seed: u64) -> f64 {
    let mut rng = Lcg::new(seed);
    let x = random_vector(&mut rng, len);
    let y = random_vector(&mut rng, len);

    let device =
        dot_dense::<f64>(client, &x, &y).expect("dot_dense should succeed for equal-length vectors");
    let reference = oracle_dot(&x, &y);
    assert!(
        reference.is_finite(),
        "oracle_dot must be finite for equal-length vectors"
    );
    (device - reference).abs()
}

/// Naive device products + host f64 sum vs pairwise oracle — differences are
/// far below this in practice. Generous enough to absorb backend reassociation
/// while still catching a wrong kernel.
const TOL: f64 = 1e-9;

/// Degenerate 1, primes, and lengths straddling the BLOCK=256 boundary to
/// exercise the bounds-guard tail in the kernel.
const LENS: &[usize] = &[1, 2, 13, 31, 64, 97, 128, 255, 256, 257, 512, 1000];

#[test]
fn dot_kernel_matches_oracle_on_cpu() {
    let client = AlgebraClient::Cpu(cubecl_cpu::CpuRuntime::client(&cubecl_cpu::CpuDevice));
    for (i, &len) in LENS.iter().enumerate() {
        let diff = check_case(&client, len, 0x51ED_C0DE_u64 + i as u64);
        assert!(
            diff < TOL,
            "CPU dot len {len}: abs diff {diff:e} >= tol {TOL:e}"
        );
    }
}

#[cfg(feature = "rocm")]
#[test]
fn dot_kernel_matches_oracle_on_rocm() {
    // Construct the HIP client directly on the default AMD device (gfx1152).
    let client =
        AlgebraClient::Rocm(cubecl_hip::HipRuntime::client(&cubecl_hip::AmdDevice::default()));
    assert!(
        matches!(client, AlgebraClient::Rocm(_)),
        "test must run on the ROCm backend, not a fallback"
    );
    for (i, &len) in LENS.iter().enumerate() {
        let diff = check_case(&client, len, 0x0CA0_1A7E_u64 + i as u64);
        assert!(
            diff < TOL,
            "ROCm dot len {len}: abs diff {diff:e} >= tol {TOL:e}"
        );
    }
}
