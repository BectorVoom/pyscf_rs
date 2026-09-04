//! D-PBC-17 for `veff::trace_ab` / `veff::trace_dm_v` — the `Tr(dm·v)`
//! reductions that produce `ecoul` and the exchange half of `exc`.
//!
//! # Why these two were a gap
//!
//! W-05 put `fft_jk`'s grid contractions on the FOUND-06 ordered primitives and
//! W-07 did the same for `nr_rks`'s block sums, but both items' file lists
//! stopped short of `veff.rs`. `trace_dm_v` is nevertheless squarely on the
//! energy path: `krks.rs:193` and `kuks.rs:248` take its `.0` as `ecoul`, and
//! `krks.rs:203` / `kuks.rs:258` subtract its `.0` as the exchange term. It was
//! an `n^2`-term inner sum and an `(nset*nkpts)`-term outer sum, both naive
//! running `+=`.
//!
//! # What is asserted
//!
//! 1. **Bit-identity at the reference cell size.** `oracle_sum`'s base case for
//!    `len <= PAIRWISE_CHUNK` (128) is a strict left-to-right fold from `0.0`,
//!    which is exactly what the replaced loops did. So for `nao <= 11` — every
//!    cell the KRKS gate runs on, all of which have `nao = 8` — the change must
//!    move NOTHING. This is asserted against a reproduction of the pre-change
//!    loops kept here (RULE 4: reference implementations live in the test file,
//!    never in the source).
//! 2. **A strictly better error bound where the tree engages.** For `nao >= 12`
//!    the reduction becomes a pairwise tree, and on an ill-conditioned operand
//!    pair it must land closer to a compensated reference than the naive fold
//!    does. This is the actual precision claim, and it is measured rather than
//!    asserted from theory.

use pyscf_algebra::CTensor;
use pyscf_pbc_dft::veff::{trace_ab, trace_dm_v};

/// The PRE-CHANGE `trace_ab`, reproduced verbatim so this test is an
/// independent computation rather than a re-run of the code under test.
fn naive_trace_ab(a: &CTensor, b: &CTensor, n: usize) -> (f64, f64) {
    let mut sr = 0.0_f64;
    let mut si = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let (ar, ai) = (a.re[i * n + j], a.im[i * n + j]);
            let (br, bi) = (b.re[j * n + i], b.im[j * n + i]);
            sr += ar * br - ai * bi;
            si += ar * bi + ai * br;
        }
    }
    (sr, si)
}

/// The PRE-CHANGE `trace_dm_v`.
fn naive_trace_dm_v(dms: &[Vec<CTensor>], v: &[Vec<CTensor>], nao: usize) -> (f64, f64) {
    let mut re = 0.0_f64;
    let mut im = 0.0_f64;
    for (s, set) in dms.iter().enumerate() {
        for (k, d) in set.iter().enumerate() {
            let (r, i) = naive_trace_ab(d, &v[s][k], nao);
            re += r;
            im += i;
        }
    }
    (re, im)
}

/// Neumaier-compensated `Tr(A B)` — the reference both routes are scored
/// against. Carrying the lost low-order bits in a separate accumulator makes
/// this accurate to within one rounding of the exact sum, which is far tighter
/// than either route under test.
fn compensated_trace_ab(a: &CTensor, b: &CTensor, n: usize) -> (f64, f64) {
    let mut acc = [0.0_f64; 2];
    let mut comp = [0.0_f64; 2];
    for i in 0..n {
        for j in 0..n {
            let (ar, ai) = (a.re[i * n + j], a.im[i * n + j]);
            let (br, bi) = (b.re[j * n + i], b.im[j * n + i]);
            for (slot, x) in [ar * br - ai * bi, ar * bi + ai * br]
                .into_iter()
                .enumerate()
            {
                let s = acc[slot] + x;
                comp[slot] += if acc[slot].abs() >= x.abs() {
                    (acc[slot] - s) + x
                } else {
                    (x - s) + acc[slot]
                };
                acc[slot] = s;
            }
        }
    }
    (acc[0] + comp[0], acc[1] + comp[1])
}

/// Deterministic pseudo-random `n x n` complex matrix.
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
            // Alternating large/small magnitudes: the classic ill-conditioned
            // sum, where a naive fold loses the small terms into the large
            // running total and a pairwise tree does not.
            let scale = if k % 2 == 0 { 1e8 } else { 1e-8 };
            r *= scale;
            i *= scale;
        }
        re[k] = r;
        im[k] = i;
    }
    CTensor::from_planes(re, im)
}

#[test]
fn trace_ab_is_bit_identical_below_the_pairwise_chunk() {
    // nao = 8 is every cell the KRKS gate runs on; 11 is the largest nao whose
    // n^2 still fits PAIRWISE_CHUNK = 128, i.e. the last size at which the
    // ordered route is required to reproduce the naive one exactly.
    for nao in [1_usize, 2, 8, 11] {
        for spread in [false, true] {
            let a = matrix(nao, 0x51ED_C0DE ^ nao as u64, spread);
            let b = matrix(nao, 0xB105_F00D ^ nao as u64, spread);
            let (gr, gi) = trace_ab(&a, &b, nao);
            let (nr, ni) = naive_trace_ab(&a, &b, nao);
            assert_eq!(
                gr.to_bits(),
                nr.to_bits(),
                "nao={nao} spread={spread}: real part moved ({gr} vs {nr}) — \
                 oracle_sum's base case is a strict left-to-right fold below \
                 PAIRWISE_CHUNK, so nothing may move here"
            );
            assert_eq!(
                gi.to_bits(),
                ni.to_bits(),
                "nao={nao} spread={spread}: imag moved"
            );
        }
    }
}

#[test]
fn trace_dm_v_is_bit_identical_at_the_reference_cell_size() {
    let nao = 8;
    for (nset, nkpts) in [(1_usize, 1_usize), (1, 8), (2, 8)] {
        let dms: Vec<Vec<CTensor>> = (0..nset)
            .map(|s| {
                (0..nkpts)
                    .map(|k| matrix(nao, 0x1234 + (s * 64 + k) as u64, false))
                    .collect()
            })
            .collect();
        let v: Vec<Vec<CTensor>> = (0..nset)
            .map(|s| {
                (0..nkpts)
                    .map(|k| matrix(nao, 0x9876 + (s * 64 + k) as u64, false))
                    .collect()
            })
            .collect();
        let (gr, gi) = trace_dm_v(&dms, &v, nao);
        let (nr, ni) = naive_trace_dm_v(&dms, &v, nao);
        assert_eq!(
            gr.to_bits(),
            nr.to_bits(),
            "nset={nset} nkpts={nkpts}: ecoul would move ({gr} vs {nr})"
        );
        assert_eq!(
            gi.to_bits(),
            ni.to_bits(),
            "nset={nset} nkpts={nkpts}: imag moved"
        );
    }
}

#[test]
fn the_ordered_route_is_more_accurate_where_the_tree_engages() {
    // A pairwise tree improves the ERROR BOUND, not every individual draw: on
    // any single random operand pair the naive fold can happen to land closer,
    // and it does (measured: at nao=12 one particular seed gives naive
    // 1.2e-15 against ordered 2.0e-15). The claim worth asserting is therefore
    // the one the bound actually makes — that the ordered route is better ON
    // AVERAGE over an ensemble — so this averages over many trials rather than
    // asserting on one.
    const TRIALS: usize = 400;

    // nao >= 12 puts n^2 past PAIRWISE_CHUNK; 26 is `gth-dzvp` on the Si
    // reference cell and 64 is a large-cell stand-in.
    for nao in [12_usize, 26, 64] {
        let mut sum_ordered = 0.0_f64;
        let mut sum_naive = 0.0_f64;
        let mut ordered_wins = 0usize;

        for t in 0..TRIALS {
            let seed = (t as u64) << 20;
            let a = matrix(nao, 0xC0FF_EE00 ^ nao as u64 ^ seed, true);
            let b = matrix(nao, 0x0BAD_BEEF ^ nao as u64 ^ seed, true);

            let (rr, _) = compensated_trace_ab(&a, &b, nao);
            let (gr, _) = trace_ab(&a, &b, nao);
            let (nr, _) = naive_trace_ab(&a, &b, nao);

            let scale = rr.abs().max(1e-30);
            let err_ordered = (gr - rr).abs() / scale;
            let err_naive = (nr - rr).abs() / scale;
            sum_ordered += err_ordered;
            sum_naive += err_naive;
            if err_ordered < err_naive {
                ordered_wins += 1;
            }
        }

        let mean_ordered = sum_ordered / TRIALS as f64;
        let mean_naive = sum_naive / TRIALS as f64;
        println!(
            "nao={nao} (n^2={}): mean relative error over {TRIALS} trials — \
             ordered {mean_ordered:.3e}, naive {mean_naive:.3e} \
             ({:.2}x better, ordered closer in {ordered_wins}/{TRIALS})",
            nao * nao,
            mean_naive / mean_ordered.max(f64::MIN_POSITIVE),
        );
        assert!(
            mean_ordered < mean_naive,
            "nao={nao}: the ordered reduction is not better ON AVERAGE \
             (ordered {mean_ordered:.3e}, naive {mean_naive:.3e} over {TRIALS} \
             trials) — past PAIRWISE_CHUNK the tree is supposed to shrink the \
             error bound from O(n^2*eps) to O(log2(n^2)*eps)"
        );
    }
}
