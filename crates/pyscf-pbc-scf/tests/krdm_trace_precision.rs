//! U-03 step 1 — D-PBC-17 for `krdm::trace_ab`, the SECOND copy.
//!
//! # Why this file exists
//!
//! `Tr(A B)` over a row-major complex pair exists TWICE in this workspace, as
//! two textually identical functions in two crates:
//!
//! * `pyscf-pbc-dft::veff::trace_ab` — `ecoul` and the exchange half of `exc`.
//!   Ordered by the D-PBC-17 pass; covered by
//!   `pyscf-pbc-dft/tests/veff_trace_precision.rs`.
//! * `pyscf-pbc-scf::krdm::trace_ab` — **`e1`**, the one-electron energy, via
//!   [`energy_elec`], plus `coulomb_imag` and the initial guess's electron
//!   count. That pass did NOT reach it, and `krdm.rs` imported only `CTensor`.
//!
//! Fixing one left `e1` on the naive path, and `e1` is a term of every
//! `Krks`/`Kuks`/`Kroks`/`Krkspu` total energy. Both are now ordered; this file
//! is the second copy's evidence, and it asserts the same two properties the
//! first copy's does.
//!
//! The two remain separate functions only because the ALG-06 crate split puts
//! them on opposite sides of it. If that ever stops being true, delete one.

use pyscf_algebra::CTensor;
use pyscf_pbc_scf::krdm::{energy_elec, trace_ab};

/// The PRE-CHANGE `krdm::trace_ab`, reproduced verbatim so this test is an
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

/// Neumaier-compensated `Tr(A B)` — the reference both routes are scored
/// against.
fn compensated_trace_ab(a: &CTensor, b: &CTensor, n: usize) -> f64 {
    let mut acc = 0.0_f64;
    let mut comp = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let (ar, ai) = (a.re[i * n + j], a.im[i * n + j]);
            let (br, bi) = (b.re[j * n + i], b.im[j * n + i]);
            let x = ar * br - ai * bi;
            let s = acc + x;
            comp += if acc.abs() >= x.abs() {
                (acc - s) + x
            } else {
                (x - s) + acc
            };
            acc = s;
        }
    }
    acc + comp
}

/// Deterministic pseudo-random `n x n` complex matrix. `spread` alternates
/// magnitudes by 16 orders — the classic ill-conditioned sum, where a naive
/// fold loses the small terms into the large running total.
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

/// Below `PAIRWISE_CHUNK` the ordered route must move NOTHING: `oracle_sum`'s
/// base case is a strict left-to-right fold from `0.0`, which is what the
/// replaced loop did. `nao = 8` is every cell the periodic gates run on.
#[test]
fn krdm_trace_ab_is_bit_identical_below_the_pairwise_chunk() {
    for nao in [1_usize, 2, 8, 11] {
        for spread in [false, true] {
            let a = matrix(nao, 0x51ED_C0DE ^ nao as u64, spread);
            let b = matrix(nao, 0xB105_F00D ^ nao as u64, spread);
            let (gr, gi) = trace_ab(&a, &b, nao);
            let (nr, ni) = naive_trace_ab(&a, &b, nao);
            assert_eq!(
                gr.to_bits(),
                nr.to_bits(),
                "nao={nao} spread={spread}: e1 would move ({gr} vs {nr})"
            );
            assert_eq!(
                gi.to_bits(),
                ni.to_bits(),
                "nao={nao} spread={spread}: imag moved"
            );
        }
    }
}

/// `energy_elec`'s `(nset * nkpts)`-long OUTER chain is ordered too — U-03
/// step 2. At the reference cell size the inner traces are bit-identical to
/// the old ones, so the whole result must be, including for `nset = 2`.
#[test]
fn energy_elec_is_bit_identical_at_the_reference_cell_size() {
    let nao = 8;
    for (nset, nkpts) in [(1_usize, 1_usize), (1, 8), (2, 8)] {
        let dms: Vec<Vec<CTensor>> = (0..nset)
            .map(|s| {
                (0..nkpts)
                    .map(|k| matrix(nao, 0x1234 + (s * 64 + k) as u64, false))
                    .collect()
            })
            .collect();
        let h1e: Vec<CTensor> = (0..nkpts)
            .map(|k| matrix(nao, 0x4321 + k as u64, false))
            .collect();
        let vhf: Vec<Vec<CTensor>> = (0..nset)
            .map(|s| {
                (0..nkpts)
                    .map(|k| matrix(nao, 0x9876 + (s * 64 + k) as u64, false))
                    .collect()
            })
            .collect();

        // The PRE-CHANGE `energy_elec`: naive inner traces folded by a naive
        // outer loop.
        let inv = 1.0 / nkpts as f64;
        let mut e1 = 0.0_f64;
        let mut ec = 0.0_f64;
        for (set, dmset) in dms.iter().enumerate() {
            for k in 0..nkpts {
                e1 += naive_trace_ab(&dmset[k], &h1e[k], nao).0;
                ec += naive_trace_ab(&dmset[k], &vhf[set][k], nao).0;
            }
        }
        let want_coul = inv * ec * 0.5;
        let want = (inv * e1 + want_coul, want_coul);

        let got = energy_elec(&dms, &h1e, &vhf, nao);
        assert_eq!(
            got.0.to_bits(),
            want.0.to_bits(),
            "nset={nset} nkpts={nkpts}: e_elec moved ({} vs {})",
            got.0,
            want.0
        );
        assert_eq!(
            got.1.to_bits(),
            want.1.to_bits(),
            "nset={nset} nkpts={nkpts}: e_coul moved"
        );
    }
}

/// The actual precision claim, measured rather than asserted from theory: past
/// `PAIRWISE_CHUNK` the tree engages and the ordered route is closer to the
/// compensated reference ON AVERAGE. (Not on every draw — a pairwise tree
/// improves the error BOUND, and on any single operand pair the naive fold can
/// land closer. That is why this averages over an ensemble.)
#[test]
fn the_ordered_route_is_more_accurate_where_the_tree_engages() {
    const TRIALS: usize = 400;
    for nao in [12_usize, 26, 64] {
        let mut sum_ordered = 0.0_f64;
        let mut sum_naive = 0.0_f64;
        let mut ordered_wins = 0usize;
        for t in 0..TRIALS {
            let seed = (t as u64) << 20;
            let a = matrix(nao, 0xC0FF_EE00 ^ nao as u64 ^ seed, true);
            let b = matrix(nao, 0x0BAD_BEEF ^ nao as u64 ^ seed, true);
            let reference = compensated_trace_ab(&a, &b, nao);
            let scale = reference.abs().max(1e-30);
            let err_ordered = (trace_ab(&a, &b, nao).0 - reference).abs() / scale;
            let err_naive = (naive_trace_ab(&a, &b, nao).0 - reference).abs() / scale;
            sum_ordered += err_ordered;
            sum_naive += err_naive;
            if err_ordered < err_naive {
                ordered_wins += 1;
            }
        }
        let mean_ordered = sum_ordered / TRIALS as f64;
        let mean_naive = sum_naive / TRIALS as f64;
        println!(
            "krdm::trace_ab nao={nao:<3} mean relative error  ordered {mean_ordered:.3e}  \
             naive {mean_naive:.3e}  ordered wins {ordered_wins}/{TRIALS}"
        );
        assert!(
            mean_ordered < mean_naive,
            "nao={nao}: the ordered route ({mean_ordered:.3e}) is not better on average \
             than the naive fold ({mean_naive:.3e})"
        );
    }
}
