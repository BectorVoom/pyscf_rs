//! K-14f — `local_vmat`, the on-device local-potential contraction.
//!
//! Verified here:
//!
//! * the kernel reproduces the host loop it replaces
//!   (`pyscf_pbc_df::fftdf::Fftdf::contract_local_potential`) BIT for BIT, at
//!   several `(nkpts, nao, ngrids)` shapes including ones whose lane count is
//!   not a multiple of the cube dimension (the `local < lanes` tail guard);
//! * a gamma k-point's imaginary AO plane is treated as a literal `0.0`, so the
//!   result matches a host reference fed a zeroed plane down to the SIGN of
//!   every zero — the thing a `* 0.0` scaling would get wrong;
//! * the result is exactly Hermitian across `(p, q)`, which the summation
//!   order guarantees and a reordered reduction would break;
//! * shape disagreements are rejected without launching.
//!
//! The client is constructed directly (not via `select_backend`) so this never
//! races on the process-global `PYSCF_BACKEND`.

#![cfg(feature = "cpu")]

use cubecl::Runtime;
use pyscf_algebra::AlgebraClient;
use pyscf_kernels::pbc::{AoPlanes, local_vmat};

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

/// The host loop this kernel replaces, transcribed from
/// `pyscf-pbc-df/src/fftdf.rs`'s `contract_local_potential` — same operand
/// order, same associativity, same `-a.im[..]` conjugation.
fn reference(
    re: &[f64],
    im: &[f64],
    vr: &[f64],
    nkpts: usize,
    nao: usize,
    ngrids: usize,
) -> Vec<(Vec<f64>, Vec<f64>)> {
    let n = nao * ngrids;
    (0..nkpts)
        .map(|k| {
            let (ar, ai) = (&re[k * n..(k + 1) * n], &im[k * n..(k + 1) * n]);
            let mut out_re = vec![0.0_f64; nao * nao];
            let mut out_im = vec![0.0_f64; nao * nao];
            for p in 0..nao {
                for q in 0..nao {
                    let mut sr = 0.0_f64;
                    let mut si = 0.0_f64;
                    let (pb, qb) = (p * ngrids, q * ngrids);
                    for g in 0..ngrids {
                        let (pr, pi) = (ar[pb + g], -ai[pb + g]);
                        let (qr, qi) = (ar[qb + g], ai[qb + g]);
                        let w = vr[g];
                        sr += (pr * qr - pi * qi) * w;
                        si += (pr * qi + pi * qr) * w;
                    }
                    out_re[p * nao + q] = sr;
                    out_im[p * nao + q] = si;
                }
            }
            (out_re, out_im)
        })
        .collect()
}

fn random_planes(
    seed: u64,
    nkpts: usize,
    nao: usize,
    ngrids: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = Lcg::new(seed);
    let n = nkpts * nao * ngrids;
    let re: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
    let im: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
    let vr: Vec<f64> = (0..ngrids).map(|_| rng.next_f64()).collect();
    (re, im, vr)
}

fn assert_bitwise(got: &[(Vec<f64>, Vec<f64>)], want: &[(Vec<f64>, Vec<f64>)], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: k-point count");
    for (k, ((gr, gi), (wr, wi))) in got.iter().zip(want).enumerate() {
        for (i, (g, w)) in gr.iter().zip(wr).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "{what}: re[{k}][{i}] = {g:e} vs {w:e}"
            );
        }
        for (i, (g, w)) in gi.iter().zip(wi).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "{what}: im[{k}][{i}] = {g:e} vs {w:e}"
            );
        }
    }
}

#[test]
fn matches_the_host_contraction_bitwise() {
    let client = cpu_client();
    // Shapes chosen so `nkpts * nao^2` is variously a multiple and a
    // non-multiple of any plausible cube dimension.
    for (seed, nkpts, nao, ngrids) in [
        (1u64, 1usize, 1usize, 1usize),
        (2, 1, 4, 37),
        (3, 3, 7, 61),
        (4, 2, 13, 128),
        (5, 5, 9, 257),
    ] {
        let (re, im, vr) = random_planes(seed, nkpts, nao, ngrids);
        let gamma = vec![false; nkpts];
        let got = local_vmat(
            &client,
            &AoPlanes { re: &re, im: &im },
            &vr,
            nkpts,
            nao,
            ngrids,
            &gamma,
        )
        .expect("local_vmat");
        let want = reference(&re, &im, &vr, nkpts, nao, ngrids);
        assert_bitwise(&got, &want, &format!("({nkpts}, {nao}, {ngrids})"));
    }
}

#[test]
fn gamma_drops_the_imaginary_plane_exactly() {
    let client = cpu_client();
    let (nkpts, nao, ngrids) = (3usize, 6usize, 53usize);
    let (re, im, vr) = random_planes(11, nkpts, nao, ngrids);
    // k = 0 and k = 2 are gamma: the kernel must read `+0.0` there, not the
    // residue it was handed.
    let gamma = vec![true, false, true];
    let got = local_vmat(
        &client,
        &AoPlanes { re: &re, im: &im },
        &vr,
        nkpts,
        nao,
        ngrids,
        &gamma,
    )
    .expect("local_vmat");

    // The reference is fed exactly what `eval_ao_kpts` hands the host route at
    // gamma — a freshly zeroed imaginary plane, `vec![0.0; n]`.
    let n = nao * ngrids;
    let mut im_zeroed = im.clone();
    for (k, &is_gamma) in gamma.iter().enumerate() {
        if is_gamma {
            im_zeroed[k * n..(k + 1) * n].fill(0.0);
        }
    }
    let want = reference(&re, &im_zeroed, &vr, nkpts, nao, ngrids);
    assert_bitwise(&got, &want, "gamma");

    // And the gamma blocks' imaginary parts are zero, with the same sign the
    // host route produces (`+0.0`, from `pr * 0.0 + (-0.0) * qr`).
    for (k, &is_gamma) in gamma.iter().enumerate() {
        if is_gamma {
            for (i, v) in got[k].1.iter().enumerate() {
                assert_eq!(v.to_bits(), 0.0_f64.to_bits(), "gamma im[{k}][{i}] = {v:e}");
            }
        }
    }
}

#[test]
fn the_result_is_exactly_hermitian() {
    let client = cpu_client();
    let (nkpts, nao, ngrids) = (2usize, 8usize, 71usize);
    let (re, im, vr) = random_planes(23, nkpts, nao, ngrids);
    let got = local_vmat(
        &client,
        &AoPlanes { re: &re, im: &im },
        &vr,
        nkpts,
        nao,
        ngrids,
        &vec![false; nkpts],
    )
    .expect("local_vmat");
    for (k, (r, i)) in got.iter().enumerate() {
        for p in 0..nao {
            for q in 0..nao {
                assert_eq!(
                    r[p * nao + q].to_bits(),
                    r[q * nao + p].to_bits(),
                    "re[{k}][{p},{q}] is not symmetric"
                );
                assert_eq!(
                    i[p * nao + q],
                    -i[q * nao + p],
                    "im[{k}][{p},{q}] is not antisymmetric"
                );
            }
        }
    }
}

#[test]
fn shape_disagreements_are_rejected() {
    let client = cpu_client();
    let (nkpts, nao, ngrids) = (2usize, 3usize, 5usize);
    let (re, im, vr) = random_planes(31, nkpts, nao, ngrids);
    let gamma = vec![false; nkpts];

    // A short potential.
    assert!(
        local_vmat(
            &client,
            &AoPlanes { re: &re, im: &im },
            &vr[..ngrids - 1],
            nkpts,
            nao,
            ngrids,
            &gamma,
        )
        .is_err(),
        "a vr shorter than the grid must be refused"
    );
    // A short AO plane.
    assert!(
        local_vmat(
            &client,
            &AoPlanes {
                re: &re[..re.len() - 1],
                im: &im,
            },
            &vr,
            nkpts,
            nao,
            ngrids,
            &gamma,
        )
        .is_err(),
        "a truncated AO plane must be refused"
    );
    // One gamma flag too few.
    assert!(
        local_vmat(
            &client,
            &AoPlanes { re: &re, im: &im },
            &vr,
            nkpts,
            nao,
            ngrids,
            &gamma[..1],
        )
        .is_err(),
        "a gamma list shorter than nkpts must be refused"
    );
}

#[test]
fn a_degenerate_shape_returns_empty_planes_without_launching() {
    let client = cpu_client();
    let out = local_vmat(
        &client,
        &AoPlanes { re: &[], im: &[] },
        &[],
        3,
        4,
        0,
        &[false, false, false],
    )
    .expect("degenerate local_vmat");
    assert_eq!(out.len(), 3);
    assert!(out.iter().all(|(r, i)| r.is_empty() && i.is_empty()));
}
