//! quick-260529-mtx: randomized **oracle differential** test for the cubecl
//! generic-float 2-D transpose kernel (`transpose_dense`).
//!
//! Strategy (per docs/rust_crate_test_guideline.md — differential testing
//! against a trusted reference, randomized inputs, reproducible seed): transpose
//! is a pure index permutation, so the host `out[c*m + r] = x[r*n + c]` is the
//! exact ground truth (no rounding — the values are merely moved). We run the
//! device kernel on random `M×N` matrices and assert every element lands in its
//! transposed slot bit-for-bit.
//!
//! - `transpose_kernel_matches_oracle_on_cpu` always runs (default `cpu` feature).
//! - `transpose_kernel_matches_oracle_on_rocm` (`#[cfg(feature = "rocm")]`) runs
//!   the SAME differential check on real AMD hardware (gfx1152) via
//!   `cubecl_hip::HipRuntime`.
//!
//! Clients are constructed directly (not via `select_backend`) so the CPU and
//! ROCm tests never race on the process-global `PYSCF_BACKEND` env var.
//!
//! Verified in scope: device `N×M` transpose == host permutation EXACTLY over a
//!   spread of shapes (square, tall, wide, prime dims, single row/col, lengths
//!   straddling the BLOCK=256 launch boundary to exercise the bounds-guard tail).
//!   Also the empty-input no-op. Not verified: f32 precision path.

use cubecl::Runtime; // brings `::client` into scope for the concrete runtimes
use pyscf_algebra::{AlgebraClient, transpose_dense};

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

/// Host transpose: row-major `M×N` → row-major `N×M`. The trusted reference.
fn host_transpose(x: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; m * n];
    for r in 0..m {
        for c in 0..n {
            out[c * m + r] = x[r * n + c];
        }
    }
    out
}

/// Run one random transpose on `client` and return the max abs elementwise diff
/// against the host permutation.
fn check_case(client: &AlgebraClient, m: usize, n: usize, seed: u64) -> f64 {
    let mut rng = Lcg::new(seed);
    let x = random_matrix(&mut rng, m * n);
    let reference = host_transpose(&x, m, n);

    let device = transpose_dense::<f64>(client, &x, m, n).expect("transpose_dense should succeed");

    assert_eq!(
        device.len(),
        reference.len(),
        "transpose must preserve size"
    );
    device
        .iter()
        .zip(reference.iter())
        .map(|(d, r)| (d - r).abs())
        .fold(0.0_f64, f64::max)
}

/// A pure permutation moves bytes unchanged — agreement is exact. This zero
/// tolerance catches any wrong index math or a dropped tail element.
const TOL: f64 = 0.0;

/// `(M, N)` shapes: square, tall, wide, prime dims, single row/col, and shapes
/// whose `M*N` straddles the BLOCK=256 boundary to exercise the bounds guard.
const SHAPES: &[(usize, usize)] = &[
    (1, 1),
    (1, 7),
    (7, 1),
    (2, 3),
    (3, 2),
    (16, 16),
    (13, 17),
    (255, 1),
    (1, 256),
    (16, 17),
    (17, 16),
    (32, 9),
];

/// Drive every shape through `check_case` and assert exact equality.
fn run_all(client: &AlgebraClient, base_seed: u64, label: &str) {
    for (i, &(m, n)) in SHAPES.iter().enumerate() {
        let diff = check_case(client, m, n, base_seed.wrapping_add(i as u64));
        assert!(
            diff <= TOL,
            "{label} transpose {m}x{n}: max abs diff {diff:e} > tol {TOL:e}"
        );
    }

    // Empty input is a documented no-op: must succeed and return empty.
    let empty = transpose_dense::<f64>(client, &[], 0, 0).expect("transpose_dense empty no-op");
    assert!(
        empty.is_empty(),
        "{label} transpose of empty must stay empty"
    );
}

#[test]
fn transpose_kernel_matches_oracle_on_cpu() {
    let client = AlgebraClient::Cpu(cubecl_cpu::CpuRuntime::client(&cubecl_cpu::CpuDevice));
    run_all(&client, 0x7A11_5E00_u64, "CPU");
}

#[cfg(feature = "rocm")]
#[test]
fn transpose_kernel_matches_oracle_on_rocm() {
    // Construct the HIP client directly on the default AMD device (gfx1152).
    let client = AlgebraClient::Rocm(cubecl_hip::HipRuntime::client(
        &cubecl_hip::AmdDevice::default(),
    ));
    assert!(
        matches!(client, AlgebraClient::Rocm(_)),
        "test must run on the ROCm backend, not a fallback"
    );
    run_all(&client, 0x7A11_5E0C_u64, "ROCm");
}
