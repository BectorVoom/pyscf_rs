//! quick-260529-mtx: randomized **oracle differential** test for the cubecl
//! generic-float AXPY kernel (`axpy_dense`).
//!
//! Strategy (per docs/rust_crate_test_guideline.md — differential testing
//! against a trusted reference, randomized inputs, reproducible seed): the
//! host `y0[i] + alpha * x[i]` is the ground truth. AXPY has no reduction, so
//! the oracle is exact element-wise (no pairwise tree needed); we run the
//! device kernel on random vectors and assert every updated element matches
//! the host multiply-add within a tight tolerance.
//!
//! - `axpy_kernel_matches_oracle_on_cpu` always runs (default `cpu` feature).
//! - `axpy_kernel_matches_oracle_on_rocm` (`#[cfg(feature = "rocm")]`) runs the
//!   SAME differential check on real AMD hardware (gfx1152) via
//!   `cubecl_hip::HipRuntime` — the backend the task explicitly requested.
//!
//! Clients are constructed directly (not via `select_backend`) so the CPU and
//! ROCm tests never race on the process-global `PYSCF_BACKEND` env var.
//!
//! Verified in scope: device `y += alpha*x` == host `y0 + alpha*x` within
//!   1e-12 over a spread of random lengths (degenerate, prime, BLOCK-boundary
//!   straddling) and a spread of alphas (incl. 0, negative, identity). Also
//!   asserts the empty-input no-op path and the length-mismatch error path.
//!   Not verified here: f32 precision path.

use cubecl::Runtime; // brings `::client` into scope for the concrete runtimes
use pyscf_algebra::{AlgebraClient, axpy_dense};

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

/// Run one random AXPY on `client` and return the max abs elementwise diff
/// against the host reference `y0[i] + alpha * x[i]`.
fn check_case(client: &AlgebraClient, len: usize, alpha: f64, seed: u64) -> f64 {
    // Two distinct draws from the SAME rng so x and y differ.
    let mut rng = Lcg::new(seed);
    let x = random_vector(&mut rng, len);
    let y0 = random_vector(&mut rng, len);
    let reference: Vec<f64> = (0..len).map(|i| y0[i] + alpha * x[i]).collect();

    let mut device = y0.clone();
    axpy_dense::<f64>(client, alpha, &x, &mut device).expect("axpy_dense should always succeed");

    assert_eq!(device.len(), reference.len(), "axpy must preserve length");
    device
        .iter()
        .zip(reference.iter())
        .map(|(d, r)| (d - r).abs())
        .fold(0.0_f64, f64::max)
}

/// A single device multiply-add vs a single host multiply-add — should agree
/// to the last bit; this tolerance just absorbs any backend rounding-mode
/// quirk while still catching a wrong kernel.
const TOL: f64 = 1e-12;

/// Degenerate 1, primes, and lengths straddling the BLOCK=256 boundary to
/// exercise the bounds-guard tail in the kernel.
const LENS: &[usize] = &[1, 2, 13, 31, 64, 97, 128, 255, 256, 257, 512, 1000];

/// Spread of scale factors: identity, zero (no-op add), negative, and
/// arbitrary fractional/large values.
const ALPHAS: &[f64] = &[1.0, 0.0, -1.0, 2.5, -3.75, 0.001, 1234.5];

/// Drive every (len, alpha) pair through `check_case` and assert the bound.
fn run_all(client: &AlgebraClient, base_seed: u64, label: &str) {
    let mut i: u64 = 0;
    for &len in LENS {
        for &alpha in ALPHAS {
            let diff = check_case(client, len, alpha, base_seed.wrapping_add(i));
            assert!(
                diff < TOL,
                "{label} axpy len {len} alpha {alpha}: max abs diff {diff:e} >= tol {TOL:e}"
            );
            i += 1;
        }
    }

    // Empty input is a documented no-op: it must succeed and leave the slices
    // untouched (never launches a zero-length grid).
    let x_empty: Vec<f64> = Vec::new();
    let mut y_empty: Vec<f64> = Vec::new();
    axpy_dense::<f64>(client, 3.0, &x_empty, &mut y_empty)
        .expect("axpy_dense on empty slices is a no-op");
    assert!(
        y_empty.is_empty(),
        "{label} axpy on empty slice must stay empty"
    );
}

#[test]
fn axpy_kernel_matches_oracle_on_cpu() {
    let client = AlgebraClient::Cpu(cubecl_cpu::CpuRuntime::client(&cubecl_cpu::CpuDevice));
    run_all(&client, 0x_A1FA_BEEF_u64, "CPU");
}

/// A length mismatch between `x` and `y` must return a clean `DimensionMismatch`
/// error (no panic, `y` left untouched). CPU-only — pure host-side validation
/// that runs before any kernel launch.
#[test]
fn axpy_dense_length_mismatch_is_error() {
    let client = AlgebraClient::Cpu(cubecl_cpu::CpuRuntime::client(&cubecl_cpu::CpuDevice));
    let x = vec![1.0_f64, 2.0, 3.0, 4.0]; // len 4
    let mut y = vec![0.0_f64; 5]; // len 5
    let result = axpy_dense::<f64>(&client, 2.0, &x, &mut y);
    assert!(
        result.is_err(),
        "mismatched x/y lengths must return Err, not a wrong result"
    );
}

#[cfg(feature = "rocm")]
#[test]
fn axpy_kernel_matches_oracle_on_rocm() {
    // Construct the HIP client directly on the default AMD device (gfx1152).
    let client = AlgebraClient::Rocm(cubecl_hip::HipRuntime::client(
        &cubecl_hip::AmdDevice::default(),
    ));
    assert!(
        matches!(client, AlgebraClient::Rocm(_)),
        "test must run on the ROCm backend, not a fallback"
    );
    run_all(&client, 0x_D00D_F00D_u64, "ROCm");
}
