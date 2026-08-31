//! W-07 (`.planning/pbc/KRKS-OPTIMISATION-PLAN.md`) — the grid-block partition.
//!
//! # What changed
//!
//! `nr_rks` used to accumulate `nelec` and `excsum` with a running `+=` over
//! the grid blocks: a naive sequential sum on the two quantities that land
//! directly in the total energy, which is precisely the shape D-PBC-17 exists
//! to forbid. It now collects one partial per block and reduces THOSE through
//! `oracle_sum`.
//!
//! # What this test asserts, and the plan deviation it records
//!
//! W-07's stated DONE criterion is "`nr_rks` output must be **bit-identical**
//! across `PYSCF_PBC_NUMINT_BLKSIZE` ∈ {128, 1024, 8192, whole-grid} once the
//! block-independent accumulation of the previous paragraph is in".
//!
//! **That criterion is not achievable and the plan's own reasoning for it does
//! not hold.** `oracle_sum` is a pairwise tree over a fixed chunk of 128, so
//! its shape is a function of the input LENGTH. Reducing per-block partials
//! gives `oracle_sum([oracle_sum(b0), oracle_sum(b1), …])`, which is a
//! different tree from `oracle_sum(b0 ++ b1 ++ …)` whenever there is more than
//! one block — and floating-point addition is not associative, so the two
//! differ in the last bits by construction. The only partition-independent
//! formulation is to concatenate every block and reduce once, which defeats
//! the entire purpose of blocking (it materialises the whole grid).
//!
//! So the honest contract is the one asserted here:
//!
//! 1. **Bit-identical for the DEFAULT partition** — one block covering the
//!    whole grid — which is what every shipped configuration uses and what
//!    Gate A's residuals were measured on. This is the property that made the
//!    change safe to land.
//! 2. **Agreeing to 1e-13 relative across partitions**, i.e. the residual from
//!    re-partitioning is three orders of magnitude inside the 1e-11 KRKS gate.
//! 3. The partition is genuinely varied — the test would be vacuous if
//!    `max_memory` did not actually produce different block counts, so that is
//!    asserted directly.

mod common;

use pyscf_algebra::CTensor;
use pyscf_pbc_dft::gen_grid::PeriodicGrids;
use pyscf_pbc_dft::numint::{BLKSIZE, KNumInt};
use pyscf_pbc_dft::xc::XcType;
use pyscf_pbc_gto::make_kpts_default;

const MESH: [usize; 3] = [15, 15, 15];

fn model_dms(nao: usize, nkpts: usize) -> Vec<Vec<CTensor>> {
    vec![(0..nkpts)
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
        .collect()]
}

/// `max_memory` in MB that yields a block of about `want` grid points for this
/// shape. `block_ranges` computes `((max_memory * 1e6) / (comp*2*nkpts*16*128))
/// * 128`, so invert that.
fn max_memory_for_block(want: usize, comp: usize, nkpts: usize) -> f64 {
    let denom = (comp * 2 * nkpts * 16 * BLKSIZE) as f64;
    (want / BLKSIZE).max(1) as f64 * denom / 1e6
}

#[test]
fn block_partition_is_actually_varied() {
    let cell = common::diamond();
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("kpts");
    let ngrids = MESH[0] * MESH[1] * MESH[2];
    let mut counts = Vec::new();
    for want in [BLKSIZE, 1024, 8192] {
        let mut ni = KNumInt::new(&kpts);
        ni.max_memory = max_memory_for_block(want, XcType::Gga.ncomp(), kpts.len());
        counts.push(ni.block_ranges(ngrids, XcType::Gga, kpts.len()).len());
    }
    assert!(
        counts[0] > 1 && counts.windows(2).all(|w| w[0] >= w[1]),
        "the partitions did not actually differ: block counts {counts:?} — this test \
         would be vacuous"
    );
    // The shipped default (4000 MB) must be the single whole-grid block that
    // Gate A's residuals were measured on.
    let default_blocks = KNumInt::new(&kpts)
        .block_ranges(ngrids, XcType::Gga, kpts.len())
        .len();
    assert_eq!(
        default_blocks, 1,
        "the DEFAULT max_memory must still give exactly one block covering the whole grid"
    );
}

#[test]
fn nr_rks_is_bit_identical_for_the_default_whole_grid_partition() {
    let cell = common::diamond();
    let grids = PeriodicGrids::uniform(&cell, Some(MESH)).expect("grids");
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("kpts");
    let dms = model_dms(cell.mol.nao_nr, kpts.len());

    let run = || {
        let ni = KNumInt::new(&kpts);
        ni.nr_rks(&cell, &grids, "PBE", &dms, 1, None).expect("nr_rks")
    };
    let a = run();
    let b = run();
    assert_eq!(a.nelec[0].to_bits(), b.nelec[0].to_bits());
    assert_eq!(a.excsum[0].to_bits(), b.excsum[0].to_bits());
    for (ma, mb) in a.vmat[0].iter().zip(&b.vmat[0]) {
        for i in 0..ma.len() {
            assert_eq!(ma.re[i].to_bits(), mb.re[i].to_bits());
            assert_eq!(ma.im[i].to_bits(), mb.im[i].to_bits());
        }
    }
}

#[test]
fn nr_rks_agrees_across_block_partitions_far_inside_the_gate() {
    let cell = common::diamond();
    let grids = PeriodicGrids::uniform(&cell, Some(MESH)).expect("grids");
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("kpts");
    let dms = model_dms(cell.mol.nao_nr, kpts.len());
    let comp = XcType::Gga.ncomp();

    let run = |want: Option<usize>| {
        let mut ni = KNumInt::new(&kpts);
        if let Some(w) = want {
            ni.max_memory = max_memory_for_block(w, comp, kpts.len());
        }
        ni.nr_rks(&cell, &grids, "PBE", &dms, 1, None).expect("nr_rks")
    };
    // The DEFAULT budget — one block covering the whole grid — is the reference.
    let reference = run(None);
    // Three orders of magnitude inside the 1e-11 KRKS gate — see the module
    // docs for why bit-identity across partitions is not achievable.
    const TOL: f64 = 1e-13;
    for want in [BLKSIZE, 1024, 8192] {
        let got = run(Some(want));
        let rel = |a: f64, b: f64| (a - b).abs() / a.abs().max(1.0);
        assert!(
            rel(reference.nelec[0], got.nelec[0]) < TOL,
            "nelec moved too far at block {want}: {} vs {}",
            reference.nelec[0],
            got.nelec[0]
        );
        assert!(
            rel(reference.excsum[0], got.excsum[0]) < TOL,
            "excsum moved too far at block {want}: {} vs {}",
            reference.excsum[0],
            got.excsum[0]
        );
        for (k, (a, b)) in reference.vmat[0].iter().zip(&got.vmat[0]).enumerate() {
            for i in 0..a.len() {
                assert!(
                    rel(a.re[i], b.re[i]) < TOL && rel(a.im[i], b.im[i]) < TOL,
                    "vmat[{k}][{i}] moved too far at block {want}: \
                     ({}, {}) vs ({}, {})",
                    a.re[i],
                    a.im[i],
                    b.re[i],
                    b.im[i]
                );
            }
        }
    }
}
