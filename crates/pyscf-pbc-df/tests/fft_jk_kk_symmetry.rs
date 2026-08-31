//! W-08 (`.planning/pbc/KRKS-OPTIMISATION-PLAN.md`) — the opt-in k-pair
//! symmetry in `get_k_kpts`.
//!
//! # What is being tested, and why it is derived rather than ported
//!
//! W-08 says to port upstream's `kk_adapted_iter` (`pbc/df/aft_jk.py`). It does
//! not carry over — see the doc comment on `get_k_kpts_opts` — so the identity
//! actually exploited here is
//!
//! ```text
//! vR^{(k2,k1)}[(i,j),g] = conj( vR^{(k1,k2)}[(j,i),g] )
//! ```
//!
//! which halves the number of 3-D transforms. Because it is derived and not
//! ported, it needs a test that could actually catch it being wrong, and that
//! is what this file is: the symmetric route is compared against the FULL
//! `Nk^2` loop — the code that has been carrying Gate A since Phase 11 — over
//! several k-meshes and both range-separation settings.
//!
//! The agreement asserted is `1e-13` RELATIVE, which is W-08's own stated
//! tolerance. It is not bit-identity: the same terms reach `vk[k]` in a
//! different order, which is exactly why the flag is opt-in and why a gate run
//! with it on has to be re-baselined.
//!
//! The precondition errors are asserted too. Every one of them is a
//! CORRECTNESS precondition of the identity — an even mesh axis really does
//! break `G_(-n) = -G_n` — so silently falling back to the plain loop would be
//! worse than failing: the caller would believe the flag was honoured.

mod common;

use pyscf_algebra::CTensor;
use pyscf_pbc_df::df_jk::KMats;
use pyscf_pbc_df::{Fftdf, get_k_kpts, get_k_kpts_opts};
use pyscf_pbc_gto::{ExxDiv, make_kpts_default};

/// ODD on every axis — the identity's precondition.
const MESH_ODD: [usize; 3] = [11, 11, 11];
/// EVEN on every axis — must be REJECTED, not silently downgraded.
const MESH_EVEN: [usize; 3] = [12, 12, 12];

/// W-08's own stated tolerance.
const TOL: f64 = 1e-13;

fn model_dm(nao: usize, nkpts: usize) -> KMats {
    (0..nkpts)
        .map(|k| {
            let mut m = CTensor::zeros(nao * nao);
            for p in 0..nao {
                for q in 0..nao {
                    let v =
                        0.3 / (1.0 + (p as f64 - q as f64).abs()) + if p == q { 1.0 } else { 0.0 };
                    m.re[p * nao + q] = v * (1.0 + 0.1 * k as f64);
                    m.im[p * nao + q] = 0.05 * (p as f64 - q as f64) * (1.0 + 0.03 * k as f64);
                }
            }
            // Hermitian — the identity assumes it, and `hermi = 1` asserts it.
            for p in 0..nao {
                for q in 0..p {
                    m.re[q * nao + p] = m.re[p * nao + q];
                    m.im[q * nao + p] = -m.im[p * nao + q];
                }
                m.im[p * nao + p] = 0.0;
            }
            m
        })
        .collect()
}

#[test]
fn symmetric_route_matches_the_full_pair_loop() {
    let cell = common::diamond();
    for nk in [[1, 1, 1], [2, 2, 2], [1, 1, 3], [1, 2, 2]] {
        let kpts = make_kpts_default(&cell, nk).expect("kpts");
        let df = Fftdf::with_mesh(cell.clone(), &kpts, MESH_ODD).expect("fftdf");
        let dms = vec![model_dm(cell.mol.nao_nr, kpts.len())];

        for omega in [None, Some(0.11), Some(-0.11)] {
            for exxdiv in [None, Some(ExxDiv::Ewald)] {
                let full = get_k_kpts(&df, &dms, 1, &kpts, None, exxdiv, omega)
                    .expect("full get_k_kpts");
                let sym = get_k_kpts_opts(&df, &dms, 1, &kpts, None, exxdiv, omega, true)
                    .expect("symmetric get_k_kpts");

                let mut worst = 0.0_f64;
                for (iset, (sa, sb)) in full.iter().zip(&sym).enumerate() {
                    for (k, (a, b)) in sa.iter().zip(sb).enumerate() {
                        for i in 0..a.len() {
                            let scale = a.re[i].abs().max(a.im[i].abs()).max(1e-8);
                            let d = ((a.re[i] - b.re[i]).abs())
                                .max((a.im[i] - b.im[i]).abs())
                                / scale;
                            if d > worst {
                                worst = d;
                            }
                            assert!(
                                d < TOL,
                                "nk={nk:?} omega={omega:?} exxdiv={exxdiv:?} set {iset} \
                                 band {k} element {i}: full ({}, {}) vs symmetric \
                                 ({}, {}) — relative {d:.3e} exceeds {TOL:.0e}",
                                a.re[i],
                                a.im[i],
                                b.re[i],
                                b.im[i]
                            );
                        }
                    }
                }
                println!(
                    "nk={nk:?} omega={omega:?} exxdiv={exxdiv:?}: worst relative \
                     difference {worst:.3e}"
                );
            }
        }
    }
}

#[test]
fn an_even_mesh_axis_is_rejected_not_silently_downgraded() {
    let cell = common::diamond();
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("kpts");
    let df = Fftdf::with_mesh(cell.clone(), &kpts, MESH_EVEN).expect("fftdf");
    let dms = vec![model_dm(cell.mol.nao_nr, kpts.len())];
    let err = get_k_kpts_opts(&df, &dms, 1, &kpts, None, None, None, true)
        .expect_err("an even mesh must be rejected");
    let msg = format!("{err}");
    assert!(
        msg.contains("EVEN"),
        "the error must name the even axis as the reason, got: {msg}"
    );
}

#[test]
fn a_non_hermitian_request_is_rejected() {
    let cell = common::diamond();
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("kpts");
    let df = Fftdf::with_mesh(cell.clone(), &kpts, MESH_ODD).expect("fftdf");
    let dms = vec![model_dm(cell.mol.nao_nr, kpts.len())];
    let err = get_k_kpts_opts(&df, &dms, 0, &kpts, None, None, None, true)
        .expect_err("hermi != 1 must be rejected");
    assert!(format!("{err}").contains("hermi"), "{err}");
}

#[test]
fn band_kpoints_are_rejected() {
    let cell = common::diamond();
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("kpts");
    let df = Fftdf::with_mesh(cell.clone(), &kpts, MESH_ODD).expect("fftdf");
    let dms = vec![model_dm(cell.mol.nao_nr, kpts.len())];
    let band = [[0.05, 0.0, 0.0]];
    let err = get_k_kpts_opts(&df, &dms, 1, &kpts, Some(&band), None, None, true)
        .expect_err("band k-points must be rejected");
    assert!(format!("{err}").contains("band"), "{err}");
}
