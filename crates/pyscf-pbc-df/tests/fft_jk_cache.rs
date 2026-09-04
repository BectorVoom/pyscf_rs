//! W-01 (`.planning/pbc/KRKS-OPTIMISATION-PLAN.md`) — `Fftdf::coulg_and_expmikr`
//! hoists `get_coulG(dk)`/`expmikr(dk)` out of `get_k_kpts`'s `Nk^2` pair loop
//! into a cache keyed on `(dk, omega, exxdiv)`. This must be EXACT: same
//! values, computed once instead of `Nk^2` times, nothing reordered.
//!
//! Two independent checks:
//!
//! 1. The cached accessor returns exactly what a fresh, uncached computation
//!    of `get_coulG` + the `expmikr` phase table would (the pre-W-01 formula,
//!    reproduced here verbatim from the `fft_jk.rs` history rather than
//!    re-derived, so this test cannot share a bug with the cache).
//! 2. `get_k_kpts` gives BIT-IDENTICAL output whether the cache is cold (first
//!    call on a fresh `Fftdf`) or warm (second call on the same `Fftdf`), over
//!    `{gamma, 2x2x2, 1x1x3} x {omega: None, Some(0.11), Some(-0.11)}`.

mod common;

use pyscf_algebra::CTensor;
use pyscf_pbc_df::df_jk::KMats;
use pyscf_pbc_df::{Fftdf, get_k_kpts};
use pyscf_pbc_gto::{CoulGArgs, ExxDiv, get_coulg, get_gv, is_zero, make_kpts_default};

const MESH_FAST: [usize; 3] = [11, 11, 11];

fn model_dm(nao: usize, nkpts: usize) -> KMats {
    (0..nkpts)
        .map(|k| {
            let mut m = CTensor::zeros(nao * nao);
            for p in 0..nao {
                for q in 0..nao {
                    let v =
                        0.3 / (1.0 + (p as f64 - q as f64).abs()) + if p == q { 1.0 } else { 0.0 };
                    m.re[p * nao + q] = v * (1.0 + 0.1 * k as f64);
                }
            }
            m
        })
        .collect()
}

/// The pre-W-01 formula (`fft_jk.rs`'s `get_k_kpts`, before the cache),
/// reproduced verbatim so this test is an independent computation.
fn uncached_coulg_and_expmikr(
    df: &Fftdf,
    dk: [f64; 3],
    exxdiv: Option<ExxDiv>,
    omega: Option<f64>,
) -> (Vec<f64>, Option<CTensor>) {
    let gv = get_gv(&df.cell, Some(df.mesh)).expect("gv");
    let coulg = get_coulg(
        &df.cell,
        CoulGArgs {
            k: dk,
            exxdiv,
            kpts: Some(&df.kpts),
            mesh: Some(df.mesh),
            gv: Some(&gv),
            wrap_around: true,
            omega,
        },
    )
    .expect("get_coulg");
    let expmikr = if is_zero(&dk) {
        None
    } else {
        let ngrids = df.grids.coords.len();
        let mut re = vec![0.0_f64; ngrids];
        let mut im = vec![0.0_f64; ngrids];
        for (g, r) in df.grids.coords.iter().enumerate() {
            let ph = -(r[0] * dk[0] + r[1] * dk[1] + r[2] * dk[2]);
            re[g] = ph.cos();
            im[g] = ph.sin();
        }
        Some(CTensor::from_planes(re, im))
    };
    (coulg, expmikr)
}

#[test]
fn cached_accessor_matches_the_uncached_formula() {
    let cell = common::diamond();
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("kpts");
    let df = Fftdf::with_mesh(cell, &kpts, MESH_FAST).expect("FFTDF");
    let gv = get_gv(&df.cell, Some(df.mesh)).expect("gv");

    for &omega in &[None, Some(0.11), Some(-0.11)] {
        for k1 in 0..kpts.len() {
            for k2 in 0..kpts.len() {
                let dk = [
                    kpts[k2][0] - kpts[k1][0],
                    kpts[k2][1] - kpts[k1][1],
                    kpts[k2][2] - kpts[k1][2],
                ];
                let (want_coulg, want_expmikr) = uncached_coulg_and_expmikr(&df, dk, None, omega);
                let got = df
                    .coulg_and_expmikr(dk, omega, None, &kpts, &gv)
                    .expect("coulg_and_expmikr");

                assert_eq!(
                    got.0, want_coulg,
                    "coulG mismatch at dk={dk:?} omega={omega:?}"
                );
                match (&got.1, &want_expmikr) {
                    (None, None) => {}
                    (Some(g), Some(w)) => {
                        assert_eq!(g.re, w.re, "expmikr.re mismatch at dk={dk:?}");
                        assert_eq!(g.im, w.im, "expmikr.im mismatch at dk={dk:?}");
                    }
                    _ => panic!("expmikr presence mismatch at dk={dk:?}"),
                }
            }
        }
    }
}

#[test]
fn get_k_kpts_is_bit_identical_cold_vs_warm_cache() {
    let cell = common::diamond();
    let nao = cell.mol.nao_nr;

    for nk in [[1usize, 1, 1], [2, 2, 2], [1, 1, 3]] {
        let kpts = make_kpts_default(&cell.clone(), nk).expect("kpts");
        let dms = vec![model_dm(nao, kpts.len())];

        for &omega in &[None, Some(0.11), Some(-0.11)] {
            let df = Fftdf::with_mesh(cell.clone(), &kpts, MESH_FAST).expect("FFTDF");
            let cold = get_k_kpts(&df, &dms, 1, &kpts, None, Some(ExxDiv::Ewald), omega)
                .expect("cold get_k_kpts");
            let warm = get_k_kpts(&df, &dms, 1, &kpts, None, Some(ExxDiv::Ewald), omega)
                .expect("warm get_k_kpts");

            for (c, w) in cold.iter().zip(warm.iter()) {
                for (cm, wm) in c.iter().zip(w.iter()) {
                    assert_eq!(
                        cm.re, wm.re,
                        "get_k_kpts .re differs cold vs warm cache: nk={nk:?} omega={omega:?}"
                    );
                    assert_eq!(
                        cm.im, wm.im,
                        "get_k_kpts .im differs cold vs warm cache: nk={nk:?} omega={omega:?}"
                    );
                }
            }
        }
    }
}
