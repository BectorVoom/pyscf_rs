//! `get_j_kpts` over grid blocks (`PYSCF_PBC_FFTJK_AO_BUDGET_MB`): a cell
//! whose whole `(nkpts, nao, ngrids)` AO table exceeds the budget walks the
//! grid in blocks instead of materialising it. The density at a grid point
//! does not depend on the partition; only the order of the `Σ_g` in the
//! `vj` contraction changes, so blocked and one-table `vj` agree to rounding.
//!
//! One test per binary: the budget is an environment variable.

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
                }
            }
            m
        })
        .collect()
}

#[test]
fn blocked_vj_matches_one_table_vj() {
    let cell = common::diamond();
    let nao = cell.mol.nao_nr;
    let kpts = make_kpts_default(&cell, [2, 1, 1]).expect("kpts");
    let dms = vec![model_dm(nao, kpts.len())];
    let band = make_kpts_default(&cell, [1, 1, 3]).expect("band kpts");

    // SAFETY: single-threaded here — the only test in this binary.
    unsafe { std::env::remove_var("PYSCF_PBC_FFTJK_AO_BUDGET_MB") };
    let df = Fftdf::new(cell.clone(), &kpts).expect("fftdf");
    assert!(df.j_grid_blocks(kpts.len()).is_none(), "small cell must keep the one-table route");
    let whole = get_j_kpts(&df, &dms, 1, &kpts, None, None).expect("vj");
    let whole_band = get_j_kpts(&df, &dms, 1, &kpts, Some(&band), None).expect("vj band");

    // A budget of a few grid planes forces many blocks, the last one ragged.
    let per_point_mb = 16.0 * (nao * band.len().max(kpts.len())) as f64 / (1024.0 * 1024.0);
    let budget_mb = per_point_mb * 997.0;
    unsafe { std::env::set_var("PYSCF_PBC_FFTJK_AO_BUDGET_MB", format!("{budget_mb}")) };
    let df = Fftdf::new(cell, &kpts).expect("fftdf");
    let blocks = df.j_grid_blocks(kpts.len()).expect("budget must force blocking");
    assert!(blocks.len() > 2, "want several blocks, got {}", blocks.len());
    assert_eq!(blocks.last().unwrap().1, df.ngrids(), "blocks must cover the grid");
    let blocked = get_j_kpts(&df, &dms, 1, &kpts, None, None).expect("vj blocked");
    let blocked_band = get_j_kpts(&df, &dms, 1, &kpts, Some(&band), None).expect("vj band blocked");

    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for (a, b) in [(&whole, &blocked), (&whole_band, &blocked_band)] {
        assert_eq!(a.len(), b.len());
        for (ka, kb) in a.iter().zip(b.iter()) {
            assert_eq!(ka.len(), kb.len());
            for (ma, mb) in ka.iter().zip(kb.iter()) {
                for t in 0..ma.re.len() {
                    worst = worst.max((ma.re[t] - mb.re[t]).abs()).max((ma.im[t] - mb.im[t]).abs());
                    scale = scale.max(ma.re[t].abs()).max(ma.im[t].abs());
                }
            }
        }
    }
    assert!(scale > 1e-3, "vj is vacuous: max |vj| = {scale:e}");
    assert!(worst <= 1e-12 * scale.max(1.0), "blocked vj differs by {worst:e} (scale {scale:e})");
}
