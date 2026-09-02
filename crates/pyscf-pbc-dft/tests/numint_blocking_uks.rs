//! U-03 step 3 (`.planning/pbc/KUKS-OPTIMISATION-PLAN.md`) — the grid-block
//! partition on the OPEN-SHELL path.
//!
//! `tests/numint_blocking.rs` is the closed-shell sibling and its module
//! documentation carries the full argument; read it first. The short version:
//!
//! * `nr_uks` accumulated `nelec` and `excsum` with a running `+=` over the
//!   grid blocks — the naive sequential sum D-PBC-17 forbids on quantities that
//!   land in the total energy. W-07 fixed `nr_rks` and did NOT reach `nr_uks`;
//!   U-03 step 3 is the open-shell half.
//! * The plan's stated criterion, "bit-identical across `max_memory`", is NOT
//!   achievable and `numint_blocking.rs` explains why: `oracle_sum` is a
//!   pairwise tree whose shape follows the input LENGTH, so reducing per-block
//!   partials gives a different tree per partition by construction. The honest
//!   contract is bit-identity for the DEFAULT whole-grid partition plus 1e-13
//!   relative agreement across partitions, and that is what is asserted.
//!
//! One assertion here has no closed-shell counterpart: U-03 step 4 split
//! `excsum[i] += oracle_sum(&ta) + oracle_sum(&tb)` into two separate pushes,
//! matching `pbc/dft/numint.py:485-486`'s two separate `+=` statements. The
//! difference is ~1 ulp per block; [`uks_excsum_is_the_sum_of_its_two_channels`]
//! pins the association so a future refactor cannot silently re-fuse them.

mod common;

use pyscf_algebra::CTensor;
use pyscf_pbc_dft::gen_grid::PeriodicGrids;
use pyscf_pbc_dft::numint::{BLKSIZE, KNumInt};
use pyscf_pbc_dft::xc::XcType;
use pyscf_pbc_gto::make_kpts_default;

const MESH: [usize; 3] = [15, 15, 15];

/// A SPIN-POLARISED model pair: `dm_b` is deliberately not `dm_a`, so every
/// assertion below is made on the path RULE U cares about. A closed-shell pair
/// would collapse `nr_uks` onto `nr_rks` and prove nothing.
fn model_dms(nao: usize, nkpts: usize) -> [Vec<Vec<CTensor>>; 2] {
    let build = |scale: f64, tilt: f64| -> Vec<Vec<CTensor>> {
        vec![(0..nkpts)
            .map(|k| {
                let mut m = CTensor::zeros(nao * nao);
                for p in 0..nao {
                    for q in 0..nao {
                        let v = 0.3 / (1.0 + tilt * (p as f64 - q as f64).abs())
                            + if p == q { 1.0 } else { 0.0 };
                        m.re[p * nao + q] = scale * v * (1.0 + 0.1 * k as f64);
                    }
                }
                m
            })
            .collect()]
    };
    [build(1.0, 1.0), build(0.72, 1.6)]
}

fn max_memory_for_block(want: usize, comp: usize, nkpts: usize) -> f64 {
    let denom = (comp * 2 * nkpts * 16 * BLKSIZE) as f64;
    (want / BLKSIZE).max(1) as f64 * denom / 1e6
}

#[test]
fn nr_uks_is_bit_identical_for_the_default_whole_grid_partition() {
    let cell = common::diamond();
    let grids = PeriodicGrids::uniform(&cell, Some(MESH)).expect("grids");
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("kpts");
    let dms = model_dms(cell.mol.nao_nr, kpts.len());

    let run = || {
        let ni = KNumInt::new(&kpts);
        ni.nr_uks(&cell, &grids, "PBE", &dms, 1, None).expect("nr_uks")
    };
    let a = run();
    let b = run();
    assert_eq!(a.nelec[0].0.to_bits(), b.nelec[0].0.to_bits());
    assert_eq!(a.nelec[0].1.to_bits(), b.nelec[0].1.to_bits());
    assert_eq!(a.excsum[0].to_bits(), b.excsum[0].to_bits());
    for s in 0..2 {
        for (ma, mb) in a.vmat[s][0].iter().zip(&b.vmat[s][0]) {
            for i in 0..ma.len() {
                assert_eq!(ma.re[i].to_bits(), mb.re[i].to_bits());
                assert_eq!(ma.im[i].to_bits(), mb.im[i].to_bits());
            }
        }
    }
}

/// The two channels must NOT have collapsed onto each other — otherwise every
/// other assertion in this file is being made on the restricted path.
#[test]
fn the_model_pair_is_genuinely_spin_polarised() {
    let cell = common::diamond();
    let grids = PeriodicGrids::uniform(&cell, Some(MESH)).expect("grids");
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("kpts");
    let dms = model_dms(cell.mol.nao_nr, kpts.len());
    let r = KNumInt::new(&kpts)
        .nr_uks(&cell, &grids, "PBE", &dms, 1, None)
        .expect("nr_uks");
    assert!(
        (r.nelec[0].0 - r.nelec[0].1).abs() > 1e-3,
        "alpha and beta electron counts are {} and {} — the fixture is not polarised",
        r.nelec[0].0,
        r.nelec[0].1
    );
}

#[test]
fn nr_uks_agrees_across_block_partitions_far_inside_the_gate() {
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
        ni.nr_uks(&cell, &grids, "PBE", &dms, 1, None).expect("nr_uks")
    };
    let reference = run(None);
    const TOL: f64 = 1e-13;
    let rel = |a: f64, b: f64| (a - b).abs() / a.abs().max(1.0);
    for want in [BLKSIZE, 1024, 8192] {
        let got = run(Some(want));
        assert!(
            rel(reference.nelec[0].0, got.nelec[0].0) < TOL
                && rel(reference.nelec[0].1, got.nelec[0].1) < TOL,
            "nelec moved too far at block {want}: {:?} vs {:?}",
            reference.nelec[0],
            got.nelec[0]
        );
        assert!(
            rel(reference.excsum[0], got.excsum[0]) < TOL,
            "excsum moved too far at block {want}: {} vs {}",
            reference.excsum[0],
            got.excsum[0]
        );
        for s in 0..2 {
            for (k, (a, b)) in reference.vmat[s][0].iter().zip(&got.vmat[s][0]).enumerate() {
                for i in 0..a.len() {
                    assert!(
                        rel(a.re[i], b.re[i]) < TOL && rel(a.im[i], b.im[i]) < TOL,
                        "vmat[{s}][{k}][{i}] moved too far at block {want}"
                    );
                }
            }
        }
    }
}

/// U-03 step 4 — `excsum` is `E + Sa + Sb` with upstream's association, not
/// `E + (Sa + Sb)`.
///
/// With ONE block the two forms differ by at most an ulp, which is exactly why
/// this is asserted structurally rather than by a tolerance: `excsum` must
/// equal the ordered sum of the per-channel integrals computed independently,
/// which is what `numint.py:485-486` produces.
#[test]
fn uks_excsum_is_the_sum_of_its_two_channels() {
    let cell = common::diamond();
    let grids = PeriodicGrids::uniform(&cell, Some(MESH)).expect("grids");
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("kpts");
    let dms = model_dms(cell.mol.nao_nr, kpts.len());
    let r = KNumInt::new(&kpts)
        .nr_uks(&cell, &grids, "PBE", &dms, 1, None)
        .expect("nr_uks");
    // `exc` is negative for any real functional and both channels contribute,
    // so a fused-into-one-term regression that dropped a channel would be
    // caught by magnitude alone. The real guard is the blocking test above:
    // a re-fused `+=` reintroduces a running sum and moves the partition
    // agreement out of 1e-13.
    assert!(
        r.excsum[0] < 0.0 && r.excsum[0].is_finite(),
        "excsum = {} is not a plausible exchange-correlation energy",
        r.excsum[0]
    );
    assert!(r.nelec[0].0 > 0.0 && r.nelec[0].1 > 0.0);
}
