//! SCF-03 — `get_j_kpts` over grid blocks on the DEVICE route
//! (`PYSCF_PBC_FFTJK_DEVICE=1`, forced on the CPU runtime): each block's AO
//! table stays on the device and is contracted there by the tiled kernels.
//!
//! * tiled against the one-lane-per-output kernels (`PYSCF_PBC_GRID_TILED=0`):
//!   the same operations in the same order, so bitwise;
//! * against the blocked HOST route and the one-table route: a different
//!   summation order over the grid, so to rounding.
//!
//! One test per binary: the switches are environment variables.

mod common;

use pyscf_algebra::CTensor;
use pyscf_pbc_df::df_jk::KMats;
use pyscf_pbc_df::{Fftdf, get_j_kpts};
use pyscf_pbc_gto::make_kpts_default;

fn model_dm(nao: usize, nkpts: usize) -> KMats {
    (0..nkpts)
        .map(|k| {
            let mut m = CTensor::zeros(nao * nao);
            for p in 0..nao {
                for q in 0..nao {
                    let v =
                        0.3 / (1.0 + (p as f64 - q as f64).abs()) + if p == q { 1.0 } else { 0.0 };
                    m.re[p * nao + q] = v * (1.0 + 0.1 * k as f64);
                    // Hermitian, with a k-dependent imaginary part.
                    m.im[p * nao + q] = 0.05 * k as f64 * (p as f64 - q as f64);
                }
            }
            m
        })
        .collect()
}

fn worst_and_scale(a: &[Vec<KMats>], b: &[Vec<KMats>]) -> (f64, f64, bool) {
    let (mut worst, mut scale, mut bitwise) = (0.0_f64, 0.0_f64, true);
    for (sa, sb) in a.iter().zip(b) {
        assert_eq!(sa.len(), sb.len());
        for (ka, kb) in sa.iter().zip(sb) {
            assert_eq!(ka.len(), kb.len());
            for (ma, mb) in ka.iter().zip(kb) {
                for t in 0..ma.re.len() {
                    worst = worst
                        .max((ma.re[t] - mb.re[t]).abs())
                        .max((ma.im[t] - mb.im[t]).abs());
                    scale = scale.max(ma.re[t].abs()).max(ma.im[t].abs());
                    bitwise &= ma.re[t].to_bits() == mb.re[t].to_bits()
                        && ma.im[t].to_bits() == mb.im[t].to_bits();
                }
            }
        }
    }
    (worst, scale, bitwise)
}

#[test]
fn blocked_device_vj_keeps_the_table_on_the_device() {
    let cell = common::diamond();
    let nao = cell.mol.nao_nr;
    let kpts = make_kpts_default(&cell, [2, 1, 1]).expect("kpts");
    let dms = vec![model_dm(nao, kpts.len())];
    let band = make_kpts_default(&cell, [1, 1, 3]).expect("band kpts");

    // Sampling list, then a band list, per route.
    let run = |device: &str, tiled: &str, budget_mb: Option<f64>| {
        // SAFETY: single-threaded here — the only test in this binary.
        unsafe {
            std::env::set_var("PYSCF_PBC_FFTJK_DEVICE", device);
            std::env::set_var("PYSCF_PBC_GRID_TILED", tiled);
            match budget_mb {
                Some(mb) => std::env::set_var("PYSCF_PBC_FFTJK_AO_BUDGET_MB", format!("{mb}")),
                None => std::env::remove_var("PYSCF_PBC_FFTJK_AO_BUDGET_MB"),
            }
        }
        let df = Fftdf::new(cell.clone(), &kpts).expect("fftdf");
        assert_eq!(
            df.j_grid_blocks(kpts.len()).is_some(),
            budget_mb.is_some(),
            "the budget decides the route"
        );
        vec![
            get_j_kpts(&df, &dms, 1, &kpts, None, None).expect("vj"),
            get_j_kpts(&df, &dms, 1, &kpts, Some(&band), None).expect("vj band"),
        ]
    };

    // A budget of a few grid planes forces many blocks, the last one ragged.
    let per_point_mb = 16.0 * (nao * band.len().max(kpts.len())) as f64 / (1024.0 * 1024.0);
    let budget = Some(per_point_mb * 997.0);

    let whole_host = run("0", "1", None);
    let blocked_host = run("0", "1", budget);
    let blocked_reference = run("1", "0", budget);
    let blocked_tiled = run("1", "1", budget);
    unsafe {
        std::env::remove_var("PYSCF_PBC_FFTJK_DEVICE");
        std::env::remove_var("PYSCF_PBC_GRID_TILED");
        std::env::remove_var("PYSCF_PBC_FFTJK_AO_BUDGET_MB");
    }

    let (_, scale, bitwise) = worst_and_scale(&blocked_tiled, &blocked_reference);
    assert!(scale > 1e-3, "vj is vacuous: max |vj| = {scale:e}");
    assert!(bitwise, "tiled and per-output device vj must agree bitwise");

    for (name, other) in [
        ("blocked host", &blocked_host),
        ("one-table host", &whole_host),
    ] {
        let (worst, scale, _) = worst_and_scale(&blocked_tiled, other);
        assert!(
            worst <= 1e-12 * scale.max(1.0),
            "blocked device vj differs from the {name} route by {worst:e} (scale {scale:e})"
        );
        eprintln!("SCF-03 blocked device vj vs {name}: {worst:e} (scale {scale:e})");
    }

    // A fingerprint of the per-output device route, for a same-inputs A/B
    // across revisions of the route itself.
    let mut h = 0xcbf29ce484222325_u64;
    for set in &blocked_reference {
        for ks in set {
            for m in ks {
                for v in m.re.iter().chain(&m.im) {
                    h = (h ^ v.to_bits()).wrapping_mul(0x100000001b3);
                }
            }
        }
    }
    eprintln!("SCF-03 blocked device vj (per-output kernels) fingerprint {h:016x}");
}
