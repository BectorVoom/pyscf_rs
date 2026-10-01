//! BAND-08 — `band_vmat`, the on-device band contraction with per-variable
//! grid weights, against a host reference on random planes.
//!
//! The device lane sums `g` serially; the reference here does the same, so
//! the two agree to the last bit on the CPU runtime. (The production host
//! route reduces pairwise, so THAT comparison is a tolerance gate — see
//! `pyscf-pbc-dft/tests/band_vmat_device.rs`.)
#![cfg(feature = "cpu")]

use cubecl::Runtime;
use pyscf_algebra::AlgebraClient;
use pyscf_kernels::pbc::{AoPlanes, band_vmat, local_vmat};

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

/// `v[k][p, q] = Σ_g conj(ao⁰[k, p, g]) · Σ_n wv[n, g] · aoⁿ[k, q, g]`, the
/// kernel's own order (serial in `g`, `n` ascending inside).
#[allow(clippy::too_many_arguments)]
fn reference(
    re: &[f64],
    im: &[f64],
    wv: &[f64],
    nvar: usize,
    comp: usize,
    nkpts: usize,
    nao: usize,
    ngrids: usize,
    gamma: &[bool],
) -> Vec<(Vec<f64>, Vec<f64>)> {
    let n = comp * nao * ngrids;
    (0..nkpts)
        .map(|k| {
            let ar = &re[k * n..(k + 1) * n];
            let ai_raw = &im[k * n..(k + 1) * n];
            let zeros = vec![0.0; n];
            let ai: &[f64] = if gamma[k] { &zeros } else { ai_raw };
            let mut out_re = vec![0.0_f64; nao * nao];
            let mut out_im = vec![0.0_f64; nao * nao];
            for p in 0..nao {
                for q in 0..nao {
                    let (mut sr, mut si) = (0.0_f64, 0.0_f64);
                    for g in 0..ngrids {
                        let pr = ar[p * ngrids + g];
                        let pi = -ai[p * ngrids + g];
                        let (mut wr, mut wi) = (0.0_f64, 0.0_f64);
                        for c in 0..nvar {
                            let w = wv[c * ngrids + g];
                            wr += w * ar[c * ngrids * nao + q * ngrids + g];
                            wi += w * ai[c * ngrids * nao + q * ngrids + g];
                        }
                        sr += pr * wr - pi * wi;
                        si += pr * wi + pi * wr;
                    }
                    out_re[p * nao + q] = sr;
                    out_im[p * nao + q] = si;
                }
            }
            (out_re, out_im)
        })
        .collect()
}

fn planes(seed: u64, len: usize) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Lcg::new(seed);
    let re: Vec<f64> = (0..len).map(|_| rng.next_f64()).collect();
    let im: Vec<f64> = (0..len).map(|_| rng.next_f64()).collect();
    (re, im)
}

fn assert_bitwise(got: &[(Vec<f64>, Vec<f64>)], want: &[(Vec<f64>, Vec<f64>)]) {
    assert_eq!(got.len(), want.len());
    for (k, ((gr, gi), (wr, wi))) in got.iter().zip(want).enumerate() {
        assert_eq!(gr.len(), wr.len());
        for i in 0..gr.len() {
            assert_eq!(
                gr[i].to_bits(),
                wr[i].to_bits(),
                "re k={k} i={i}: {} vs {}",
                gr[i],
                wr[i]
            );
            assert_eq!(
                gi[i].to_bits(),
                wi[i].to_bits(),
                "im k={k} i={i}: {} vs {}",
                gi[i],
                wi[i]
            );
        }
    }
}

#[test]
fn band_vmat_deriv1_matches_serial_reference_bitwise() {
    let (nkpts, nao, ngrids, comp, nvar) = (3, 5, 301, 4, 4);
    let (re, im) = planes(7, nkpts * comp * nao * ngrids);
    let mut rng = Lcg::new(99);
    let wv: Vec<f64> = (0..nvar * ngrids).map(|_| rng.next_f64()).collect();
    let gamma = [true, false, false];
    let client = cpu_client();
    let got = band_vmat(
        &client,
        &AoPlanes { re: &re, im: &im },
        &wv,
        nvar,
        comp,
        nkpts,
        nao,
        ngrids,
        &gamma,
    )
    .expect("band_vmat");
    let want = reference(&re, &im, &wv, nvar, comp, nkpts, nao, ngrids, &gamma);
    assert_bitwise(&got, &want);
}

#[test]
fn band_vmat_with_one_variable_reduces_to_local_vmat() {
    // `nvar = 1` on a `comp = 1` table is K-14f's contraction. The two kernels
    // round differently per point (`(pr·qr − pi·qi)·w` there, `pr·(w·qr) −
    // pi·(w·qi)` here), so this is a tolerance gate, not a bitwise one.
    let (nkpts, nao, ngrids) = (2, 6, 257);
    let (re, im) = planes(11, nkpts * nao * ngrids);
    let mut rng = Lcg::new(5);
    let vr: Vec<f64> = (0..ngrids).map(|_| rng.next_f64()).collect();
    let gamma = [false, true];
    let client = cpu_client();
    let ao = AoPlanes { re: &re, im: &im };
    let got = band_vmat(&client, &ao, &vr, 1, 1, nkpts, nao, ngrids, &gamma).expect("band_vmat");
    let want = local_vmat(&client, &ao, &vr, nkpts, nao, ngrids, &gamma).expect("local_vmat");
    for (k, ((gr, gi), (wr, wi))) in got.iter().zip(&want).enumerate() {
        for i in 0..gr.len() {
            let scale = 1.0 + wr[i].abs().max(wi[i].abs());
            assert!(
                (gr[i] - wr[i]).abs() <= 1e-12 * scale,
                "re k={k} i={i}: {} vs {}",
                gr[i],
                wr[i]
            );
            assert!(
                (gi[i] - wi[i]).abs() <= 1e-12 * scale,
                "im k={k} i={i}: {} vs {}",
                gi[i],
                wi[i]
            );
        }
    }
}

#[test]
fn band_vmat_nvar_below_comp_uses_only_the_leading_components() {
    // `nvar = 2` on a `comp = 4` table: the two unused components must not
    // leak into the answer.
    let (nkpts, nao, ngrids, comp, nvar) = (2, 4, 130, 4, 2);
    let (re, im) = planes(3, nkpts * comp * nao * ngrids);
    let mut rng = Lcg::new(17);
    let wv: Vec<f64> = (0..nvar * ngrids).map(|_| rng.next_f64()).collect();
    let gamma = [false, false];
    let client = cpu_client();
    let got = band_vmat(
        &client,
        &AoPlanes { re: &re, im: &im },
        &wv,
        nvar,
        comp,
        nkpts,
        nao,
        ngrids,
        &gamma,
    )
    .expect("band_vmat");
    let want = reference(&re, &im, &wv, nvar, comp, nkpts, nao, ngrids, &gamma);
    assert_bitwise(&got, &want);
}

#[test]
fn band_vmat_rejects_wrong_shapes() {
    let client = cpu_client();
    let (re, im) = planes(1, 2 * 4 * 3 * 10);
    let ao = AoPlanes { re: &re, im: &im };
    let wv = vec![0.0; 4 * 10];
    // nvar > comp
    assert!(band_vmat(&client, &ao, &wv, 5, 4, 2, 3, 10, &[false, false]).is_err());
    // wv too short
    assert!(band_vmat(&client, &ao, &wv[..10], 4, 4, 2, 3, 10, &[false, false]).is_err());
    // gamma count
    assert!(band_vmat(&client, &ao, &wv, 4, 4, 2, 3, 10, &[false]).is_err());
    // plane length
    assert!(band_vmat(&client, &ao, &wv, 4, 4, 2, 3, 11, &[false, false]).is_err());
}
