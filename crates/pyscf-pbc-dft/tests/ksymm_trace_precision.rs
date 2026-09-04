//! D-PBC-17 for the **`weights_ibz`** traces — plan item P-02 of
//! `.planning/pbc/KUKS-KSYMM-MULTIGRID-OPTIMISATION-PLAN.md`.
//!
//! # Why these were a gap
//!
//! U-03 put `veff::trace_ab` / `trace_dm_v` and `krdm::trace_ab` on the
//! FOUND-06 ordered primitives, and `veff_trace_precision.rs` gates them. The
//! k-symmetric drivers were written afterwards (17-08) and carry their OWN
//! contraction, because they weight each irreducible k-point by its star
//! multiplicity rather than by `1/nkpts`: `KsymAdaptedKrks::weighted_trace`
//! and `KsymAdaptedKuks::weighted_trace_uks`. Both were hand-rolled nests with
//! a naive `nao^2` inner sum and a naive `nkpts_ibz` outer fold, and both feed
//! `ecoul`, the hybrid `exc` correction AND `energy_elec`'s `e1` for every
//! k-symmetric KS driver — so `e1`, a term of every ksymm total energy, was in
//! exactly the state U-03 found `krdm::trace_ab` in.
//!
//! Worse, `weighted_trace`'s doc comment asserted "the accumulation is ordered
//! (D-PBC-17)". It was serial, which is not the same claim: serial buys
//! thread-independence and nothing else, while D-PBC-17 asks for the ordered
//! primitive whose error bound is `O(log2 n · eps)` instead of `O(n · eps)`.
//! P-02 corrected the code and the comment together.
//!
//! # What is asserted
//!
//! 1. **Bit-identity at the reference cell size**, against a reproduction of
//!    the pre-P-02 nests kept in this file (RULE 4: reference implementations
//!    live in the test, never in the source). `oracle_sum`'s base case below
//!    `PAIRWISE_CHUNK` (128) is a strict left-to-right fold from `0.0`, which
//!    is what the nests did — so at `nao <= 11` and
//!    `nset * nkpts_ibz <= 128` **nothing may move**. Every cell this
//!    repository gates on has `nao = 8`.
//! 2. **The shared-stack forms are bit-identical to the cloned-stack forms**
//!    they replaced — the `vec![jtot.clone(), jtot.clone()]` and
//!    `vec![h1e.to_vec(), h1e.to_vec()]` allocations P-02 deleted.
//! 3. **A strictly better error bound where the tree engages** (`nao >= 12`),
//!    measured over an ensemble rather than asserted from theory — the
//!    `veff_trace_precision.rs` discipline, and for its reason: on a single
//!    draw the naive fold can happen to land closer.

use pyscf_algebra::CTensor;
use pyscf_pbc_dft::veff::{weighted_trace_dm_v, weighted_trace_dm_v_shared};

/// The PRE-P-02 `KsymAdaptedKrks::weighted_trace` / `weighted_trace_uks`,
/// reproduced verbatim. The two differed only in whether they iterated the
/// channel axis, so one function with a channel loop covers both: the
/// restricted form is this with `dms.len() == 1`.
fn naive_weighted_trace(dms: &[Vec<CTensor>], v: &[Vec<CTensor>], w: &[f64], nao: usize) -> f64 {
    let mut acc = 0.0;
    for (spin, dset) in dms.iter().enumerate() {
        for (k, d) in dset.iter().enumerate() {
            let vk = &v[spin][k];
            let mut t = 0.0;
            for i in 0..nao {
                for j in 0..nao {
                    let dij = i * nao + j;
                    let vji = j * nao + i;
                    t += d.re[dij] * vk.re[vji] - d.im[dij] * vk.im[vji];
                }
            }
            acc += w[k] * t;
        }
    }
    acc
}

/// Neumaier-compensated reference — accurate to within one rounding of the
/// exact sum, far tighter than either route under test.
fn compensated_weighted_trace(
    dms: &[Vec<CTensor>],
    v: &[Vec<CTensor>],
    w: &[f64],
    nao: usize,
) -> f64 {
    let mut acc = 0.0_f64;
    let mut comp = 0.0_f64;
    let mut add = |x: f64, acc: &mut f64, comp: &mut f64| {
        let s = *acc + x;
        *comp += if acc.abs() >= x.abs() {
            (*acc - s) + x
        } else {
            (x - s) + *acc
        };
        *acc = s;
    };
    for (spin, dset) in dms.iter().enumerate() {
        for (k, d) in dset.iter().enumerate() {
            let vk = &v[spin][k];
            for i in 0..nao {
                for j in 0..nao {
                    let dij = i * nao + j;
                    let vji = j * nao + i;
                    let x = w[k] * (d.re[dij] * vk.re[vji] - d.im[dij] * vk.im[vji]);
                    add(x, &mut acc, &mut comp);
                }
            }
        }
    }
    acc + comp
}

/// Deterministic pseudo-random `n x n` complex matrix. `spread` alternates
/// large and small magnitudes — the classic ill-conditioned sum, where a
/// naive fold loses the small terms into the running total.
fn matrix(n: usize, seed: u64, spread: bool) -> CTensor {
    let mut s = seed | 1;
    let mut next = move || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let mut re = vec![0.0_f64; n * n];
    let mut im = vec![0.0_f64; n * n];
    for k in 0..n * n {
        let (mut r, mut i) = (next(), next());
        if spread {
            let scale = if k % 2 == 0 { 1e8 } else { 1e-8 };
            r *= scale;
            i *= scale;
        }
        re[k] = r;
        im[k] = i;
    }
    CTensor::from_planes(re, im)
}

fn stack(nset: usize, nk: usize, nao: usize, seed: u64, spread: bool) -> Vec<Vec<CTensor>> {
    (0..nset)
        .map(|s| {
            (0..nk)
                .map(|k| matrix(nao, seed ^ ((s * 64 + k) as u64) << 8, spread))
                .collect()
        })
        .collect()
}

/// `si [2,2,2]`'s measured star sizes are `[1, 3, 4]` of 8
/// (`krks_ksymm.rs::si_222_stars_have_unequal_sizes_so_the_weighting_is_observable`),
/// so a UNEQUAL weight vector is the honest fixture: a uniform one would let
/// a dropped weight pass unnoticed, which is the trap that test exists for.
fn unequal_weights(nk: usize) -> Vec<f64> {
    let raw: Vec<f64> = (0..nk).map(|k| (k + 1) as f64).collect();
    let total: f64 = raw.iter().sum();
    raw.into_iter().map(|x| x / total).collect()
}

#[test]
fn weighted_trace_is_bit_identical_at_the_reference_cell_size() {
    // nao = 8 is every cell this repository gates on; 11 is the largest nao
    // whose n^2 still fits PAIRWISE_CHUNK = 128, i.e. the last size at which
    // the ordered route is REQUIRED to reproduce the nest exactly.
    for nao in [8_usize, 11] {
        // nkpts_ibz = 3 is `si [2,2,2]`; 10 is a larger fold. nset covers
        // KRKS (1) and KUKS (2).
        for (nset, nk) in [(1_usize, 3_usize), (1, 10), (2, 3), (2, 10)] {
            let w = unequal_weights(nk);
            for spread in [false, true] {
                let dms = stack(nset, nk, nao, 0x51ED_C0DE ^ nao as u64, spread);
                let v = stack(nset, nk, nao, 0xB105_F00D ^ nao as u64, spread);
                let got = weighted_trace_dm_v(&dms, &v, &w, nao);
                let want = naive_weighted_trace(&dms, &v, &w, nao);
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "nao={nao} nset={nset} nkpts_ibz={nk} spread={spread}: \
                     the ksymm energy trace moved ({got} vs {want}) — below \
                     PAIRWISE_CHUNK oracle_sum is a strict left-to-right fold, \
                     so nothing may move here"
                );
            }
        }
    }
}

#[test]
fn the_shared_stack_form_is_bit_identical_to_the_cloned_form() {
    // This is the assertion that licenses P-02's deletion of
    // `vec![jtot.clone(), jtot.clone()]` and `vec![h1e.to_vec(), h1e.to_vec()]`
    // from `KsymAdaptedKuks`: same operands, same partial order, same reducer.
    for nao in [8_usize, 26] {
        for nk in [3_usize, 10] {
            let w = unequal_weights(nk);
            let dms = stack(2, nk, nao, 0xFEED_BEEF ^ nao as u64, false);
            let shared: Vec<CTensor> = (0..nk)
                .map(|k| matrix(nao, 0xC0DE_0042 ^ (k as u64) << 8, false))
                .collect();
            let cloned = vec![shared.clone(), shared.clone()];

            let got = weighted_trace_dm_v_shared(&dms, &shared, &w, nao);
            let want = weighted_trace_dm_v(&dms, &cloned, &w, nao);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "nao={nao} nkpts_ibz={nk}: the shared-stack trace diverged from \
                 the cloned-stack trace it replaced ({got} vs {want})"
            );
        }
    }
}

#[test]
fn the_ordered_route_is_more_accurate_where_the_tree_engages() {
    // A pairwise tree improves the error BOUND, not every individual draw, so
    // the claim is asserted over an ensemble — `veff_trace_precision.rs`'s
    // reasoning, which was arrived at after a single-draw assertion failed on
    // a seed where the naive fold happened to land closer.
    const TRIALS: usize = 200;

    // nao >= 12 puts n^2 past PAIRWISE_CHUNK. 26 is `gth-dzvp` on the Si
    // reference cell; 64 is a large-cell stand-in.
    for nao in [26_usize, 64] {
        let nk = 10;
        let w = unequal_weights(nk);
        let mut sum_ordered = 0.0_f64;
        let mut sum_naive = 0.0_f64;
        let mut ordered_wins = 0usize;

        for t in 0..TRIALS {
            let seed = (t as u64) << 20;
            let dms = stack(2, nk, nao, 0xC0FF_EE00 ^ nao as u64 ^ seed, true);
            let v = stack(2, nk, nao, 0x0BAD_BEEF ^ nao as u64 ^ seed, true);

            let reference = compensated_weighted_trace(&dms, &v, &w, nao);
            let got = weighted_trace_dm_v(&dms, &v, &w, nao);
            let naive = naive_weighted_trace(&dms, &v, &w, nao);

            let scale = reference.abs().max(1e-30);
            let err_ordered = (got - reference).abs() / scale;
            let err_naive = (naive - reference).abs() / scale;
            sum_ordered += err_ordered;
            sum_naive += err_naive;
            if err_ordered < err_naive {
                ordered_wins += 1;
            }
        }

        let mean_ordered = sum_ordered / TRIALS as f64;
        let mean_naive = sum_naive / TRIALS as f64;
        println!(
            "nao={nao} nkpts_ibz={nk}: mean relative error  ordered {mean_ordered:.3e}  \
             naive {mean_naive:.3e}  (ordered closer on {ordered_wins}/{TRIALS} draws)"
        );
        assert!(
            mean_ordered < mean_naive,
            "nao={nao}: the ordered route must have the smaller MEAN error \
             (ordered {mean_ordered:e}, naive {mean_naive:e}) — that is the \
             whole reason for the pairwise tree"
        );
    }
}
