//! `pyscf_pbc_df::gdf::jk` — J and K from `cderi` (plan 14-04).
//!
//! **The point of the phase, measured**: FFTDF and AFTDF sweep a plane-wave
//! grid every SCF iteration; GDF sweeps the fitted 3-index tensor instead, with
//! `L` running over `naux = 108` where `G` runs over `47³`. Upstream's own
//! wall-clock on diamond `gth-szv` 2×2×2 is 6.4 s for a GDF-driven `KRHF`
//! against 30.0 s (FFTDF) and 450.6 s (AFTDF).

mod common;

use pyscf_algebra::CTensor;
use pyscf_pbc_df::df_jk::KMats;
use pyscf_pbc_df::gdf::Gdf;
use pyscf_pbc_df::gdf::jk::{get_j_kpts, get_k_kpts};
use pyscf_pbc_df::traits::{JkOpts, PeriodicDf};

fn kpts(cell: &pyscf_pbc_gto::Cell, km: [usize; 3]) -> Vec<[f64; 3]> {
    pyscf_pbc_gto::kpts_mesh::make_kpts(cell, km, false, true, None).expect("kpts")
}

/// A Hermitian, positive, deterministic density — enough structure that a
/// transposed contraction shows up.
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

fn built(cell: pyscf_pbc_gto::Cell, km: [usize; 3]) -> (Gdf, Vec<[f64; 3]>) {
    let k = kpts(&cell, km);
    let mut d = Gdf::new(cell, &k);
    d.build().expect("GDF::build");
    (d, k)
}

#[test]
fn vj_and_vk_are_hermitian() {
    let (d, k) = built(common::he_all_electron(), [2, 2, 2]);
    let nao = d.cell.mol.nao_nr;
    let dms = vec![model_dm(nao, k.len())];

    for (label, mats) in [
        ("vj", get_j_kpts(&d, &dms, &k).expect("get_j_kpts")),
        ("vk", get_k_kpts(&d, &dms, &k, None).expect("get_k_kpts")),
    ] {
        for (ik, m) in mats[0].iter().enumerate() {
            let mut worst = 0.0_f64;
            for p in 0..nao {
                for q in 0..nao {
                    let (a, b) = (p * nao + q, q * nao + p);
                    worst = worst.max((m.re[a] - m.re[b]).abs());
                    worst = worst.max((m.im[a] + m.im[b]).abs());
                }
            }
            assert!(worst < 1e-11, "{label} k={ik}: asymmetry {worst:e}");
        }
    }
}

/// **The fitting identity.** `vj` built from `cderi` must equal `vj` built by
/// contracting the fitted ERI with the same density — same numbers, two routes.
/// This tests the CONTRACTION, not the fit, so it is exact.
#[test]
fn vj_matches_an_explicit_eri_contraction() {
    let (d, k) = built(common::he_all_electron(), [1, 1, 1]);
    let nao = d.cell.mol.nao_nr;
    let n2 = nao * nao;
    let dms = vec![model_dm(nao, 1)];
    let vj = get_j_kpts(&d, &dms, &k).expect("get_j_kpts");

    // eri[mu nu, rho sigma] = SUM_L C[L, mu nu] · conj(C[L, rho sigma])
    let blk = &d.sr_loop(0, 0, false).expect("sr_loop")[0];
    let mut want = CTensor::zeros(n2);
    for a in 0..n2 {
        for b in 0..n2 {
            let (mut er, mut ei) = (0.0_f64, 0.0_f64);
            for l in 0..blk.naux {
                let (ar, ai) = (blk.re[l * n2 + a], blk.im[l * n2 + a]);
                let (br, bi) = (blk.re[l * n2 + b], blk.im[l * n2 + b]);
                er += ar * br + ai * bi;
                ei += ai * br - ar * bi;
            }
            // vj[a] += eri[a, b] * dm[b transposed]
            let (mu, nu) = (b / nao, b % nao);
            let (dr, di) = (dms[0][0].re[nu * nao + mu], dms[0][0].im[nu * nao + mu]);
            want.re[a] += er * dr - ei * di;
            want.im[a] += er * di + ei * dr;
        }
    }
    let mut worst = 0.0_f64;
    for p in 0..n2 {
        worst = worst.max((vj[0][0].re[p] - want.re[p]).abs());
    }
    assert!(
        worst < 1e-12,
        "vj from cderi != vj from the explicit ERI contraction: {worst:e}"
    );
}

/// The `exxdiv = Ewald` correction is applied to the ASSEMBLED `vk`, not folded
/// into the kernel — the structural difference between GDF and AFTDF. Phase 13
/// measured (risk R-15) that this term is ~96 % of the MATRIX difference
/// between builders while barely moving the ENERGY, so the two need different
/// tolerances and this test gates the matrix.
#[test]
fn exxdiv_shifts_vk_by_the_madelung_term() {
    let (d, k) = built(common::he_all_electron(), [2, 2, 2]);
    let nao = d.cell.mol.nao_nr;
    let dms = vec![model_dm(nao, k.len())];
    let plain = get_k_kpts(&d, &dms, &k, None).expect("no exxdiv");
    let ewald = get_k_kpts(&d, &dms, &k, Some(pyscf_pbc_gto::ExxDiv::Ewald)).expect("ewald");

    // The difference must be exactly what `ewald_exxdiv_for_g0` applies.
    let mut reference = plain.clone();
    pyscf_pbc_df::df_jk::ewald_exxdiv_for_g0(&d.cell, &k, &dms, &mut reference, None)
        .expect("ewald_exxdiv_for_g0");
    let mut worst = 0.0_f64;
    for (a, b) in ewald[0].iter().zip(reference[0].iter()) {
        for p in 0..nao * nao {
            worst = worst.max((a.re[p] - b.re[p]).abs());
            worst = worst.max((a.im[p] - b.im[p]).abs());
        }
    }
    assert!(worst < 1e-14, "exxdiv term mismatch: {worst:e}");
    // …and it must actually MOVE the matrix, or the test proves nothing.
    let moved = plain[0]
        .iter()
        .zip(ewald[0].iter())
        .flat_map(|(a, b)| a.re.iter().zip(b.re.iter()))
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
    assert!(moved > 1e-6, "exxdiv did not change vk at all ({moved:e})");
}

/// An `exxdiv` GDF does not support must be REFUSED, exactly as upstream
/// raises (`df_jk.py:288-292`).
#[test]
fn unsupported_exxdiv_is_refused() {
    let (d, k) = built(common::he_all_electron(), [1, 1, 1]);
    let nao = d.cell.mol.nao_nr;
    let dms = vec![model_dm(nao, 1)];
    let e = get_k_kpts(&d, &dms, &k, Some(pyscf_pbc_gto::ExxDiv::VcutSph))
        .expect_err("GDF supports only Ewald or None");
    assert!(format!("{e}").contains("exxdiv"), "got: {e}");
}

/// `get_jk` drives through the trait object, which is what the eight k-point
/// SCF drivers actually call (D-PBC-22).
#[test]
fn get_jk_through_the_trait_object() {
    let (d, k) = built(common::he_all_electron(), [2, 2, 2]);
    let nao = d.cell.mol.nao_nr;
    let dms = vec![model_dm(nao, k.len())];
    let df: Box<dyn PeriodicDf> = Box::new(d);
    let out = df
        .get_jk(&dms, &k, JkOpts::hermitian())
        .expect("get_jk through the trait object");
    assert!(out.vj.is_some() && out.vk.is_some());
    assert_eq!(out.vj.as_ref().expect("vj")[0].len(), k.len());

    // `with_k = false` must not compute K.
    let only_j = df
        .get_jk(
            &dms,
            &k,
            JkOpts {
                with_k: false,
                ..JkOpts::hermitian()
            },
        )
        .expect("J only");
    assert!(only_j.vj.is_some() && only_j.vk.is_none());
}

/// `omega` (the range-separated kernel) needs `GDF.range_coulomb`, which is
/// plan 14-07. It must be refused, not silently ignored — an ignored `omega`
/// gives a plausible full-range answer to an RSH functional.
#[test]
fn omega_is_refused() {
    let (d, k) = built(common::he_all_electron(), [1, 1, 1]);
    let nao = d.cell.mol.nao_nr;
    let dms = vec![model_dm(nao, 1)];
    let e = d
        .get_jk(
            &dms,
            &k,
            JkOpts {
                omega: Some(0.2),
                ..JkOpts::hermitian()
            },
        )
        .expect_err("omega is 14-07");
    assert!(
        format!("{e}").contains("omega") || format!("{e}").contains("14-07"),
        "got: {e}"
    );
}
