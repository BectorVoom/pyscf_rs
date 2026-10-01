//! `KNumInt::block_ranges_ao` — the grid partition capped by the AO table's
//! size (`PYSCF_PBC_NUMINT_AO_BUDGET_MB`, default 4096 MB). A table that fits
//! keeps `block_ranges`' partition exactly (every existing energy keeps its
//! bits); one that does not — nao 910, 9 k-points, 253k points, deriv 1 is
//! 133 GB — is re-split into blocks that each fit.

use pyscf_pbc_dft::numint::{BLKSIZE, KNumInt};
use pyscf_pbc_dft::xc::XcType;

const BUDGET: usize = 4096 * 1024 * 1024;

fn covers(ranges: &[(usize, usize)], ngrids: usize) {
    assert_eq!(ranges.first().map(|r| r.0), Some(0));
    assert_eq!(ranges.last().map(|r| r.1), Some(ngrids));
    assert!(ranges.windows(2).all(|w| w[0].1 == w[1].0), "ranges must tile the grid");
}

#[test]
fn small_table_keeps_the_memory_partition() {
    let ni = KNumInt::new(&[[0.0; 3]; 4]);
    for (ngrids, nao) in [(3375, 26), (136_000, 312)] {
        let ty = XcType::Gga;
        if 16 * ty.ncomp() * 4 * nao * ngrids > BUDGET {
            continue;
        }
        assert_eq!(ni.block_ranges_ao(ngrids, ty, 4, nao), ni.block_ranges(ngrids, ty, 4));
    }
}

#[test]
fn large_table_is_split_under_the_budget() {
    let ni = KNumInt::new(&[[0.0; 3]; 9]);
    let (ngrids, nao, nk) = (43 * 43 * 137, 910, 9);
    for ty in [XcType::Lda, XcType::Gga] {
        let whole = ni.block_ranges(ngrids, ty, nk);
        let per_point = 16 * ty.ncomp() * nk * nao;
        assert!(whole.iter().any(|&(a, b)| (b - a) * per_point > BUDGET), "fixture must overflow");
        let r = ni.block_ranges_ao(ngrids, ty, nk, nao);
        covers(&r, ngrids);
        assert!(r.len() > 1);
        for &(a, b) in &r {
            assert!((b - a) * per_point <= BUDGET, "block of {} points is over budget", b - a);
        }
        assert!(r[..r.len() - 1].iter().all(|&(a, b)| (b - a) % BLKSIZE == 0));
    }
}
