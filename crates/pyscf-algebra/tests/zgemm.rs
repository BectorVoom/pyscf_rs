//! Plan 09-02 Task 8 — `zgemm_dense` / `zgemm_h_dense` (D-PBC-03).
//!
//! Verified here: a 64x64 random complex product matches a NAIVE host triple
//! loop (independent of the four-real-GEMM decomposition) to 1e-12;
//! `zgemm_h_dense(a, a)` is Hermitian to 1e-12; the four-GEMM form degenerates
//! exactly to the real GEMM when the imaginary planes are zero; shape errors
//! are rejected before any launch.
//!
//! Not verified here: GPU backends (the CPU runtime is constructed directly so
//! the test never races on the process-global `PYSCF_BACKEND`); f32.

use cubecl::Runtime;
use pyscf_algebra::{AlgebraClient, CTensor, gemm_dense, zgemm_dense, zgemm_h_dense};

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
        let u = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
        u * 2.0 - 1.0
    }
}

fn random_ctensor(rng: &mut Lcg, len: usize) -> CTensor {
    let re: Vec<f64> = (0..len).map(|_| rng.next_f64()).collect();
    let im: Vec<f64> = (0..len).map(|_| rng.next_f64()).collect();
    CTensor::from_interleaved(
        &re.iter()
            .zip(im.iter())
            .flat_map(|(r, i)| [*r, *i])
            .collect::<Vec<f64>>(),
    )
}

/// Naive host reference: `c = a * b`, row-major, plain complex triple loop.
/// Deliberately written WITHOUT the four-real-GEMM decomposition so it is an
/// independent oracle for D-PBC-03.
fn reference_zgemm(a: &CTensor, b: &CTensor, m: usize, k: usize, n: usize) -> CTensor {
    let mut re = vec![0.0_f64; m * n];
    let mut im = vec![0.0_f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let (mut acc_r, mut acc_i) = (0.0_f64, 0.0_f64);
            for p in 0..k {
                let (ar, ai) = (a.re[i * k + p], a.im[i * k + p]);
                let (br, bi) = (b.re[p * n + j], b.im[p * n + j]);
                acc_r += ar * br - ai * bi;
                acc_i += ar * bi + ai * br;
            }
            re[i * n + j] = acc_r;
            im[i * n + j] = acc_i;
        }
    }
    CTensor::from_interleaved(
        &re.iter()
            .zip(im.iter())
            .flat_map(|(r, i)| [*r, *i])
            .collect::<Vec<f64>>(),
    )
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn cpu_client() -> AlgebraClient {
    AlgebraClient::Cpu(cubecl_cpu::CpuRuntime::client(&cubecl_cpu::CpuDevice))
}

const TOL: f64 = 1e-12;

#[test]
fn zgemm_64x64_matches_naive_host_reference() {
    let client = cpu_client();
    let (m, k, n) = (64, 64, 64);
    let mut rng = Lcg::new(0x0902_0002);
    let a = random_ctensor(&mut rng, m * k);
    let b = random_ctensor(&mut rng, k * n);

    let got = zgemm_dense(&client, &a, &b, m, k, n).expect("zgemm_dense");
    let want = reference_zgemm(&a, &b, m, k, n);

    assert_eq!(got.len(), m * n);
    let dre = max_abs_diff(&got.re, &want.re);
    let dim = max_abs_diff(&got.im, &want.im);
    assert!(dre < TOL, "real plane max|diff| = {dre:e} (tol {TOL:e})");
    assert!(dim < TOL, "imag plane max|diff| = {dim:e} (tol {TOL:e})");
}

#[test]
fn zgemm_on_non_square_shapes_matches_reference() {
    let client = cpu_client();
    let mut rng = Lcg::new(0x0902_0003);
    for &(m, k, n) in &[(1, 1, 1), (3, 7, 5), (17, 2, 31), (8, 8, 1)] {
        let a = random_ctensor(&mut rng, m * k);
        let b = random_ctensor(&mut rng, k * n);
        let got = zgemm_dense(&client, &a, &b, m, k, n).expect("zgemm_dense");
        let want = reference_zgemm(&a, &b, m, k, n);
        let d = max_abs_diff(&got.re, &want.re).max(max_abs_diff(&got.im, &want.im));
        assert!(d < TOL, "({m},{k},{n}) max|diff| = {d:e}");
    }
}

/// D-PBC-03 consequence: with zero imaginary planes the four-GEMM form must
/// reproduce the REAL `gemm_dense` result BIT-for-bit — `t2`/`t3` are exactly
/// zero, so `t1 - 0` and `0 + 0` introduce no rounding.
#[test]
fn zgemm_with_zero_imaginary_planes_is_bit_identical_to_real_gemm() {
    let client = cpu_client();
    let (m, k, n) = (16, 16, 16);
    let mut rng = Lcg::new(0x0902_0004);
    let ar: Vec<f64> = (0..m * k).map(|_| rng.next_f64()).collect();
    let br: Vec<f64> = (0..k * n).map(|_| rng.next_f64()).collect();
    let a = CTensor::from_real(&ar);
    let b = CTensor::from_real(&br);

    let got = zgemm_dense(&client, &a, &b, m, k, n).expect("zgemm_dense");
    let want = gemm_dense::<f64>(&client, &ar, &br, m, k, n).expect("gemm_dense");
    for (i, (g, w)) in got.re.iter().zip(want.iter()).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "real plane at {i}");
        assert_eq!(got.im[i], 0.0, "imag plane must cancel exactly at {i}");
    }
}

#[test]
fn zgemm_h_of_a_with_itself_is_hermitian() {
    let client = cpu_client();
    let n = 64;
    let mut rng = Lcg::new(0x0902_0005);
    // `a` is the UN-transposed k x m operand; square here so Aᴴ·A is n x n.
    let a = random_ctensor(&mut rng, n * n);

    let g = zgemm_h_dense(&client, &a, &a, n, n, n).expect("zgemm_h_dense");
    assert_eq!(g.len(), n * n);

    let mut worst = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let (r1, i1) = (g.re[i * n + j], g.im[i * n + j]);
            let (r2, i2) = (g.re[j * n + i], g.im[j * n + i]);
            worst = worst.max((r1 - r2).abs()).max((i1 + i2).abs());
        }
        // The diagonal of Aᴴ·A is real and non-negative.
        assert!(
            g.im[i * n + i].abs() < TOL,
            "diagonal {i} imaginary part {} is not zero",
            g.im[i * n + i]
        );
        assert!(g.re[i * n + i] >= 0.0, "diagonal {i} is negative");
    }
    assert!(worst < TOL, "Hermitian violation max = {worst:e}");
}

#[test]
fn zgemm_h_matches_an_explicitly_conjugate_transposed_reference() {
    let client = cpu_client();
    let (m, k, n) = (5, 7, 3);
    let mut rng = Lcg::new(0x0902_0006);
    let a = random_ctensor(&mut rng, k * m); // k x m
    let b = random_ctensor(&mut rng, k * n);

    // Host-side Aᴴ (m x k).
    let mut ah = CTensor::zeros(m * k);
    for i in 0..k {
        for j in 0..m {
            ah.re[j * k + i] = a.re[i * m + j];
            ah.im[j * k + i] = -a.im[i * m + j];
        }
    }
    let want = reference_zgemm(&ah, &b, m, k, n);
    let got = zgemm_h_dense(&client, &a, &b, m, k, n).expect("zgemm_h_dense");
    let d = max_abs_diff(&got.re, &want.re).max(max_abs_diff(&got.im, &want.im));
    assert!(d < TOL, "max|diff| = {d:e}");
}

#[test]
fn zgemm_rejects_bad_shapes_without_launching() {
    let client = cpu_client();
    let a = CTensor::zeros(6); // claims 2x3
    let b = CTensor::zeros(6); // claims 3x2
    assert!(zgemm_dense(&client, &a, &b, 2, 3, 2).is_ok());
    assert!(zgemm_dense(&client, &a, &b, 3, 3, 2).is_err());
    assert!(zgemm_dense(&client, &a, &b, 2, 3, 3).is_err());
    assert!(zgemm_h_dense(&client, &a, &b, 3, 3, 2).is_err());
}
