//! quick-260529-oj6: randomized **oracle differential** tests for the four
//! `host_fallback` dense decompositions (`eigh`/`cholesky`/`qr`/`svd`), routed
//! to faer 0.24 on host via the `device_buffer` Vec<f64> round-trip (ALG-05).
//!
//! Strategy (per docs/rust_crate_test_guideline.md — differential testing
//! against a trusted reference, randomized inputs, reproducible seed): unlike
//! the exact-permutation transpose oracle, these are FLOATING decompositions, so
//! we assert each one's DEFINING PROPERTY (the algebraic identity that uniquely
//! characterizes the factorization) within a numeric tolerance — NOT bitwise
//! equality. The reference math (matmul/transpose/identity) is plain row-major
//! `Vec<f64>` with NO faer call, so the assertion is independent of the
//! implementation's faer backend (a true differential oracle).
//!
//!   * eigh:     `A ≈ U·diag(λ)·Uᵀ`, eigenvalues ascending.
//!   * cholesky: `L·Lᵀ ≈ A`, `L` lower-triangular.
//!   * qr:       `Q·R ≈ A`, `Qᵀ·Q ≈ I`.
//!   * svd:      `U·diag(s)·Vᵀ ≈ A`, singular values descending and ≥ 0.
//!
//! Each decomposition has a `*_matches_oracle_on_cpu` test that always runs
//! (default `cpu` feature) and a `#[cfg(feature = "rocm")]` `*_on_rocm` test that
//! runs the SAME differential check on real AMD hardware (gfx1152) via
//! `cubecl_hip::HipRuntime` — mirroring `tests/transpose_oracle.rs`.
//!
//! Clients are constructed directly (not via `select_backend`) so the CPU and
//! ROCm tests never race on the process-global `PYSCF_BACKEND` env var.
//!
//! Verified scope: the algebraic identity for each decomposition over a spread
//! of square sizes (1,2,3,5,8) with SPD inputs (eigh/cholesky) and general
//! inputs (qr/svd). Not verified: rectangular inputs (host_fallback is
//! square-only by the locked Tensor surface), f32 precision, non-PD cholesky
//! rejection (covered by the body's error contract, not asserted here).

use cubecl::Runtime; // brings `::client` into scope for the concrete runtimes
use pyscf_algebra::{AlgebraClient, cholesky, download, eigh, qr, svd, upload};

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

/// Floating decompositions agree with the reference only to round-off — this
/// tolerance is intentionally NON-ZERO, unlike the exact transpose permutation
/// in transpose_oracle.rs:79.
const TOL: f64 = 1e-9;

/// Square sizes to drive (square only — host_fallback requires square inputs).
const SIZES: &[usize] = &[1, 2, 3, 5, 8];

// === Host reference math (row-major n×n, plain Vec — no faer) ===

/// Row-major `n×n` · `n×n` matrix product.
fn matmul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0_f64;
            for k in 0..n {
                acc += a[i * n + k] * b[k * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

/// Row-major `n×n` transpose.
fn transpose(a: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            out[j * n + i] = a[i * n + j];
        }
    }
    out
}

/// Max elementwise absolute difference.
fn max_abs_diff(x: &[f64], y: &[f64]) -> f64 {
    x.iter()
        .zip(y.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

/// `n×n` identity, row-major.
fn identity(n: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; n * n];
    for i in 0..n {
        out[i * n + i] = 1.0;
    }
    out
}

/// Build an SPD symmetric matrix `A = MᵀM + n·I` (diagonally dominant → safely
/// positive-definite for eigh and cholesky).
fn make_spd(rng: &mut Lcg, n: usize) -> Vec<f64> {
    let m = random_matrix(rng, n * n);
    let mut a = matmul(&transpose(&m, n), &m, n);
    for i in 0..n {
        a[i * n + i] += n as f64;
    }
    a
}

/// General (non-symmetric) input for qr and svd.
fn make_general(rng: &mut Lcg, n: usize) -> Vec<f64> {
    random_matrix(rng, n * n)
}

// === Per-decomposition drivers (shared by CPU and ROCm tests) ===

fn run_eigh(client: &AlgebraClient, base_seed: u64, label: &str) {
    for (i, &n) in SIZES.iter().enumerate() {
        let mut rng = Lcg::new(base_seed.wrapping_add(i as u64));
        let input = make_spd(&mut rng, n);
        let t = upload::<f64>(client, &input, vec![n, n]).expect("upload");

        let (evals, u_t) = eigh(client, &t).expect("eigh");
        let u_fortran = download::<f64>(client, &u_t).expect("download U");

        // U is F-order (column-major): element (i,j) at u[i + j*n]. Convert to
        // row-major so the row-major reference matmul applies.
        let mut u_rm = vec![0.0_f64; n * n];
        for r in 0..n {
            for c in 0..n {
                u_rm[r * n + c] = u_fortran[r + c * n];
            }
        }

        // A_rec = U · diag(λ) · Uᵀ.
        let mut diag = vec![0.0_f64; n * n];
        for k in 0..n {
            diag[k * n + k] = evals[k];
        }
        let ud = matmul(&u_rm, &diag, n);
        let a_rec = matmul(&ud, &transpose(&u_rm, n), n);

        let diff = max_abs_diff(&a_rec, &input);
        assert!(
            diff <= TOL,
            "{label} eigh n={n}: ‖U·diag(λ)·Uᵀ - A‖_max = {diff:e} > tol {TOL:e}"
        );

        // Eigenvalues ascending.
        for k in 1..n {
            assert!(
                evals[k - 1] <= evals[k] + TOL,
                "{label} eigh n={n}: eigenvalues not ascending at k={k}: {} > {}",
                evals[k - 1],
                evals[k]
            );
        }
    }
}

fn run_cholesky(client: &AlgebraClient, base_seed: u64, label: &str) {
    for (i, &n) in SIZES.iter().enumerate() {
        let mut rng = Lcg::new(base_seed.wrapping_add(i as u64));
        let input = make_spd(&mut rng, n);
        let t = upload::<f64>(client, &input, vec![n, n]).expect("upload");

        let l_t = cholesky(client, &t).expect("cholesky");
        let l = download::<f64>(client, &l_t).expect("download L"); // row-major

        // L is lower-triangular: strict upper triangle ≈ 0.
        for r in 0..n {
            for c in (r + 1)..n {
                assert!(
                    l[r * n + c].abs() <= TOL,
                    "{label} cholesky n={n}: L[{r},{c}] = {} not ≈ 0 (not lower-tri)",
                    l[r * n + c]
                );
            }
        }

        // L · Lᵀ ≈ A.
        let llt = matmul(&l, &transpose(&l, n), n);
        let diff = max_abs_diff(&llt, &input);
        assert!(
            diff <= TOL,
            "{label} cholesky n={n}: ‖L·Lᵀ - A‖_max = {diff:e} > tol {TOL:e}"
        );
    }
}

fn run_qr(client: &AlgebraClient, base_seed: u64, label: &str) {
    for (i, &n) in SIZES.iter().enumerate() {
        let mut rng = Lcg::new(base_seed.wrapping_add(i as u64));
        let input = make_general(&mut rng, n);
        let t = upload::<f64>(client, &input, vec![n, n]).expect("upload");

        let (q_t, r_t) = qr(client, &t).expect("qr");
        let q = download::<f64>(client, &q_t).expect("download Q"); // row-major
        let r = download::<f64>(client, &r_t).expect("download R"); // row-major

        // Q · R ≈ A.
        let qr_prod = matmul(&q, &r, n);
        let diff = max_abs_diff(&qr_prod, &input);
        assert!(
            diff <= TOL,
            "{label} qr n={n}: ‖Q·R - A‖_max = {diff:e} > tol {TOL:e}"
        );

        // Qᵀ · Q ≈ I.
        let qtq = matmul(&transpose(&q, n), &q, n);
        let diff_i = max_abs_diff(&qtq, &identity(n));
        assert!(
            diff_i <= TOL,
            "{label} qr n={n}: ‖Qᵀ·Q - I‖_max = {diff_i:e} > tol {TOL:e}"
        );
    }
}

fn run_svd(client: &AlgebraClient, base_seed: u64, label: &str) {
    for (i, &n) in SIZES.iter().enumerate() {
        let mut rng = Lcg::new(base_seed.wrapping_add(i as u64));
        let input = make_general(&mut rng, n);
        let t = upload::<f64>(client, &input, vec![n, n]).expect("upload");

        let (u_t, s, v_t) = svd(client, &t).expect("svd");
        let u = download::<f64>(client, &u_t).expect("download U"); // row-major
        let v = download::<f64>(client, &v_t).expect("download V"); // row-major

        // U · diag(s) · Vᵀ ≈ A.
        let mut diag = vec![0.0_f64; n * n];
        for k in 0..n {
            diag[k * n + k] = s[k];
        }
        let ud = matmul(&u, &diag, n);
        let a_rec = matmul(&ud, &transpose(&v, n), n);
        let diff = max_abs_diff(&a_rec, &input);
        assert!(
            diff <= TOL,
            "{label} svd n={n}: ‖U·diag(s)·Vᵀ - A‖_max = {diff:e} > tol {TOL:e}"
        );

        // Singular values descending and non-negative.
        for (k, &sk) in s.iter().enumerate() {
            assert!(
                sk >= -TOL,
                "{label} svd n={n}: singular value s[{k}] = {sk} < 0"
            );
        }
        for k in 1..n {
            assert!(
                s[k - 1] >= s[k] - TOL,
                "{label} svd n={n}: singular values not descending at k={k}: {} < {}",
                s[k - 1],
                s[k]
            );
        }
    }
}

// === CPU tests (always run) ===

fn cpu_client() -> AlgebraClient {
    AlgebraClient::Cpu(cubecl_cpu::CpuRuntime::client(&cubecl_cpu::CpuDevice))
}

#[test]
fn eigh_matches_oracle_on_cpu() {
    run_eigh(&cpu_client(), 0xE160_0000_u64, "CPU");
}

#[test]
fn cholesky_matches_oracle_on_cpu() {
    run_cholesky(&cpu_client(), 0xC01E_0000_u64, "CPU");
}

#[test]
fn qr_matches_oracle_on_cpu() {
    run_qr(&cpu_client(), 0x9C00_0000_u64, "CPU");
}

#[test]
fn svd_matches_oracle_on_cpu() {
    run_svd(&cpu_client(), 0x5DD0_0000_u64, "CPU");
}

// === ROCm tests (cfg-gated, run on real AMD gfx1152) ===

#[cfg(feature = "rocm")]
fn rocm_client() -> AlgebraClient {
    AlgebraClient::Rocm(cubecl_hip::HipRuntime::client(
        &cubecl_hip::AmdDevice::default(),
    ))
}

#[cfg(feature = "rocm")]
#[test]
fn eigh_matches_oracle_on_rocm() {
    let client = rocm_client();
    assert!(
        matches!(client, AlgebraClient::Rocm(_)),
        "test must run on the ROCm backend, not a fallback"
    );
    run_eigh(&client, 0xE160_0C00_u64, "ROCm");
}

#[cfg(feature = "rocm")]
#[test]
fn cholesky_matches_oracle_on_rocm() {
    let client = rocm_client();
    assert!(
        matches!(client, AlgebraClient::Rocm(_)),
        "test must run on the ROCm backend, not a fallback"
    );
    run_cholesky(&client, 0xC01E_0C00_u64, "ROCm");
}

#[cfg(feature = "rocm")]
#[test]
fn qr_matches_oracle_on_rocm() {
    let client = rocm_client();
    assert!(
        matches!(client, AlgebraClient::Rocm(_)),
        "test must run on the ROCm backend, not a fallback"
    );
    run_qr(&client, 0x9C00_0C00_u64, "ROCm");
}

#[cfg(feature = "rocm")]
#[test]
fn svd_matches_oracle_on_rocm() {
    let client = rocm_client();
    assert!(
        matches!(client, AlgebraClient::Rocm(_)),
        "test must run on the ROCm backend, not a fallback"
    );
    run_svd(&client, 0x5DD0_0C00_u64, "ROCm");
}
