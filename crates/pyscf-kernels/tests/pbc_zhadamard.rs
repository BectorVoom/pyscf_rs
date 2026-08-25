//! Plan 09-02 Task 8 — K-04 `zhadamard` (PBC-MASTER-PLAN §6).
//!
//! Verified here: the device kernel matches a naive host complex-multiply
//! reference to 1e-14 over a spread of lengths (including ones that are not a
//! multiple of the cube dimension, exercising the `i < n` tail guard); the
//! `pyscf-algebra` in-wall mirror (`zhadamard_dense`) agrees with it BIT-for-bit;
//! plane-length mismatches are rejected without launching.
//!
//! Clients are constructed directly (not via `select_backend`) so this never
//! races on the process-global `PYSCF_BACKEND`.

#![cfg(feature = "cpu")]

use cubecl::Runtime;
use pyscf_algebra::{AlgebraClient, CTensor, zhadamard_dense};
use pyscf_kernels::zhadamard;

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

fn cpu_client() -> AlgebraClient {
    AlgebraClient::Cpu(cubecl_cpu::CpuRuntime::client(&cubecl_cpu::CpuDevice))
}

/// Naive host reference: `c[i] = a[i] * b[i]`, schoolbook 4-multiply form.
fn reference(ar: &[f64], ai: &[f64], br: &[f64], bi: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = ar.len();
    let mut cr = vec![0.0; n];
    let mut ci = vec![0.0; n];
    for i in 0..n {
        cr[i] = ar[i] * br[i] - ai[i] * bi[i];
        ci[i] = ar[i] * bi[i] + ai[i] * br[i];
    }
    (cr, ci)
}

const TOL: f64 = 1e-14;

#[test]
fn zhadamard_matches_host_reference() {
    let client = cpu_client();
    let mut rng = Lcg::new(0x0902_0401);
    // 255/257/1000 straddle the 256-thread cube dimension — the tail guard.
    for &n in &[1usize, 255, 256, 257, 1000, 4096] {
        let ar: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
        let ai: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
        let br: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
        let bi: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();

        let (gr, gi) = zhadamard(&client, &ar, &ai, &br, &bi).expect("zhadamard");
        let (wr, wi) = reference(&ar, &ai, &br, &bi);
        assert_eq!(gr.len(), n);
        assert_eq!(gi.len(), n);

        let d = (0..n)
            .map(|i| (gr[i] - wr[i]).abs().max((gi[i] - wi[i]).abs()))
            .fold(0.0_f64, f64::max);
        assert!(d < TOL, "n = {n}: max|diff| = {d:e} (tol {TOL:e})");
    }
}

/// The `pyscf-algebra` mirror of this kernel exists because `pyscf-kernels`
/// depends on `pyscf-algebra` and the call cannot go upward. The two copies are
/// byte-identical kernel bodies, so they must agree BIT-for-bit — this test is
/// the guard that keeps them in lockstep.
#[test]
fn kernels_and_algebra_copies_agree_bit_for_bit() {
    let client = cpu_client();
    let mut rng = Lcg::new(0x0902_0402);
    let n = 1024;
    let x = CTensor::from_planes(
        (0..n).map(|_| rng.next_f64()).collect(),
        (0..n).map(|_| rng.next_f64()).collect(),
    );
    let y = CTensor::from_planes(
        (0..n).map(|_| rng.next_f64()).collect(),
        (0..n).map(|_| rng.next_f64()).collect(),
    );

    let (kr, ki) = zhadamard(&client, &x.re, &x.im, &y.re, &y.im).expect("zhadamard");
    let a = zhadamard_dense(&client, &x, &y).expect("zhadamard_dense");
    for i in 0..n {
        assert_eq!(kr[i].to_bits(), a.re[i].to_bits(), "re plane at {i}");
        assert_eq!(ki[i].to_bits(), a.im[i].to_bits(), "im plane at {i}");
    }
}

#[test]
fn zhadamard_handles_empty_and_rejects_mismatched_planes() {
    let client = cpu_client();
    let (r, i) = zhadamard(&client, &[], &[], &[], &[]).expect("empty is a no-op");
    assert!(r.is_empty() && i.is_empty());

    assert!(zhadamard(&client, &[1.0], &[1.0, 2.0], &[1.0], &[1.0]).is_err());
    assert!(zhadamard(&client, &[1.0], &[1.0], &[], &[1.0]).is_err());
}

/// A purely real operand must leave the imaginary plane EXACTLY zero — the
/// D-PBC-03 cancellation property the schoolbook form guarantees.
#[test]
fn real_times_real_has_exactly_zero_imaginary_plane() {
    let client = cpu_client();
    let mut rng = Lcg::new(0x0902_0403);
    let n = 512;
    let ar: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
    let br: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
    let zeros = vec![0.0_f64; n];

    let (cr, ci) = zhadamard(&client, &ar, &zeros, &br, &zeros).expect("zhadamard");
    for i in 0..n {
        assert_eq!(cr[i].to_bits(), (ar[i] * br[i]).to_bits(), "re at {i}");
        assert_eq!(ci[i], 0.0, "im at {i} must be exactly zero");
    }
}
