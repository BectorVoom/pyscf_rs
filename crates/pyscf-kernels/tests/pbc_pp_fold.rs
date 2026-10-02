//! K-PP3 (`B += S · A`, `pyscf/pbc/df/fft.py:114-176`): device fold against a
//! host double loop, and bit-identity across G-block splits.

#![cfg(feature = "cpu")]

use cubecl::Runtime;
use pyscf_algebra::AlgebraClient;
use pyscf_kernels::pp_fold;

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

fn random_planes(n: usize, seed: &mut Lcg) -> (Vec<f64>, Vec<f64>) {
    (
        (0..n).map(|_| seed.next_f64()).collect(),
        (0..n).map(|_| seed.next_f64()).collect(),
    )
}

/// Host reference: the same double loop in plain Rust, `g` ascending.
fn fold_reference(
    s_re: &[f64],
    s_im: &[f64],
    a_re: &[f64],
    a_im: &[f64],
    b_re: &[f64],
    b_im: &[f64],
    nrow: usize,
    nao: usize,
    nbs: usize,
    nb: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut out_re = b_re.to_vec();
    let mut out_im = b_im.to_vec();
    for row in 0..nrow {
        for q in 0..nao {
            for g in 0..nb {
                let (xr, xi) = (s_re[row * nbs + g], s_im[row * nbs + g]);
                let (yr, yi) = (a_re[q * nbs + g], a_im[q * nbs + g]);
                out_re[row * nao + q] += xr * yr - xi * yi;
                out_im[row * nao + q] += xr * yi + xi * yr;
            }
        }
    }
    (out_re, out_im)
}

#[test]
fn fold_matches_the_host_sum() {
    let (nrow, nao, nbs, nb) = (5, 7, 33, 33);
    let mut rng = Lcg::new(11);
    let (s_re, s_im) = random_planes(nrow * nbs, &mut rng);
    let (a_re, a_im) = random_planes(nao * nbs, &mut rng);
    let (b_re, b_im) = random_planes(nrow * nao, &mut rng);
    let client = cpu_client();
    let (got_re, got_im) = pp_fold(
        &client, &s_re, &s_im, &a_re, &a_im, &b_re, &b_im, nrow, nao, nbs, nb,
    )
    .expect("pp_fold");
    let (want_re, want_im) =
        fold_reference(&s_re, &s_im, &a_re, &a_im, &b_re, &b_im, nrow, nao, nbs, nb);
    for i in 0..nrow * nao {
        assert!((got_re[i] - want_re[i]).abs() < 1e-12, "re[{i}]");
        assert!((got_im[i] - want_im[i]).abs() < 1e-12, "im[{i}]");
    }
}

#[test]
fn fold_is_bit_identical_across_blocks() {
    let (nrow, nao, full) = (5, 7, 33);
    let mut rng = Lcg::new(29);
    let (s_full_re, s_full_im) = random_planes(nrow * full, &mut rng);
    let (a_full_re, a_full_im) = random_planes(nao * full, &mut rng);
    let (b0_re, b0_im) = random_planes(nrow * nao, &mut rng);
    let client = cpu_client();
    // (a) one call with all 33 points.
    let (a_re, a_im) = pp_fold(
        &client, &s_full_re, &s_full_im, &a_full_re, &a_full_im, &b0_re, &b0_im, nrow, nao, full,
        full,
    )
    .expect("one block");
    // (b) three calls with nbs = 13: 0..13, 13..26, 26..33.
    let nbs = 13;
    let mut b_re = b0_re.clone();
    let mut b_im = b0_im.clone();
    for (g0, nb) in [(0, 13), (13, 13), (26, 7)] {
        let mut s_re = vec![0.0; nrow * nbs];
        let mut s_im = vec![0.0; nrow * nbs];
        let mut a_re = vec![0.0; nao * nbs];
        let mut a_im = vec![0.0; nao * nbs];
        for row in 0..nrow {
            s_re[row * nbs..row * nbs + nb]
                .copy_from_slice(&s_full_re[row * full + g0..row * full + g0 + nb]);
            s_im[row * nbs..row * nbs + nb]
                .copy_from_slice(&s_full_im[row * full + g0..row * full + g0 + nb]);
        }
        for q in 0..nao {
            a_re[q * nbs..q * nbs + nb]
                .copy_from_slice(&a_full_re[q * full + g0..q * full + g0 + nb]);
            a_im[q * nbs..q * nbs + nb]
                .copy_from_slice(&a_full_im[q * full + g0..q * full + g0 + nb]);
        }
        (b_re, b_im) = pp_fold(
            &client, &s_re, &s_im, &a_re, &a_im, &b_re, &b_im, nrow, nao, nbs, nb,
        )
        .expect("block");
    }
    for i in 0..nrow * nao {
        assert_eq!(a_re[i].to_bits(), b_re[i].to_bits(), "re[{i}]");
        assert_eq!(a_im[i].to_bits(), b_im[i].to_bits(), "im[{i}]");
    }
}
