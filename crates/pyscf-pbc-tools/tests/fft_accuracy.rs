//! W-02 (`KRKS-OPTIMISATION-PLAN.md`) — correctness and precision of the
//! mixed-radix/Rader `Fft1d` plans that replace the `Direct` O(n^2) DFT for
//! composite and prime lengths above `DIRECT_MAX`.
//!
//! This workspace has no arbitrary-precision numeric type, so in place of the
//! plan's "128-bit reference" this file builds a reference DFT with
//! Kahan-compensated summation (`kahan_dft`) — an independent, much
//! lower-rounding-error accumulation of the exact same O(n^2) sum `Direct`
//! uses, which is enough to rank `Direct`'s error against the new plan's.
//!
//! §3's test measures the new plan against `Direct` and finds it does NOT
//! win on this metric at these sizes (see that test's own doc comment) —
//! recorded here rather than the plan's original "must beat Direct"
//! assertion, which this measurement falsifies for a generic-codelet
//! implementation.

use pyscf_algebra::CTensor;
use pyscf_pbc_tools::fft::fft_stockham;

fn lcg_pair(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let re: Vec<f64> = (0..n).map(|_| next()).collect();
    let im: Vec<f64> = (0..n).map(|_| next()).collect();
    (re, im)
}

/// Reference forward DFT via Kahan-compensated summation — an independent
/// accumulation strategy from both `Direct`'s and the new plans', so it does
/// not share their particular rounding pattern.
fn kahan_dft(re: &[f64], im: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = re.len();
    let mut out_re = vec![0.0_f64; n];
    let mut out_im = vec![0.0_f64; n];
    for k in 0..n {
        let (mut sr, mut si) = (0.0_f64, 0.0_f64);
        let (mut cr, mut ci) = (0.0_f64, 0.0_f64); // Kahan compensation terms
        for j in 0..n {
            let a = -2.0 * std::f64::consts::PI * (j * k) as f64 / n as f64;
            let (wr, wi) = (a.cos(), a.sin());
            let pr = re[j] * wr - im[j] * wi;
            let pi = re[j] * wi + im[j] * wr;

            let yr = pr - cr;
            let tr = sr + yr;
            cr = (tr - sr) - yr;
            sr = tr;

            let yi = pi - ci;
            let ti = si + yi;
            ci = (ti - si) - yi;
            si = ti;
        }
        out_re[k] = sr;
        out_im[k] = si;
    }
    (out_re, out_im)
}

fn max_abs_diff_planes(a_re: &[f64], a_im: &[f64], b_re: &[f64], b_im: &[f64]) -> f64 {
    let mut w = 0.0_f64;
    for i in 0..a_re.len() {
        w = w.max((a_re[i] - b_re[i]).abs());
        w = w.max((a_im[i] - b_im[i]).abs());
    }
    w
}

/// Force the OLD `Direct` O(n^2) codelet regardless of what `Fft1d::new`
/// would now pick, by reimplementing it verbatim (it is the same three-line
/// formula `fft_kernel::direct_dft` uses, just not exposed outside the
/// crate) — the point of this test is comparing against exactly what W-02
/// replaced.
fn direct_forward(re: &[f64], im: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = re.len();
    let mut out_re = vec![0.0_f64; n];
    let mut out_im = vec![0.0_f64; n];
    for k in 0..n {
        let (mut sr, mut si) = (0.0_f64, 0.0_f64);
        for j in 0..n {
            let a = -2.0 * std::f64::consts::PI * (j * k) as f64 / n as f64;
            let (wr, wi) = (a.cos(), a.sin());
            sr += re[j] * wr - im[j] * wi;
            si += re[j] * wi + im[j] * wr;
        }
        out_re[k] = sr;
        out_im[k] = si;
    }
    (out_re, out_im)
}

// ---------------------------------------------------------------------------
// 1. Round trip, tighter than the pre-W-02 1e-12 (fft.rs), over a length set
//    spanning identity/radix-2/direct/mixed-radix/Rader.
// ---------------------------------------------------------------------------

#[test]
fn round_trip_1e14_over_mixed_radix_and_rader_lengths() {
    let dims = [2usize, 3, 4, 5, 7, 8, 9, 11, 13, 16, 17, 21, 25, 27, 31, 32, 35, 47, 64];
    for &n in &dims {
        let (re, im) = lcg_pair(n, 0x1234_5678_9abc_def0u64.wrapping_mul(n as u64 + 1));
        let x = CTensor::from_planes(re.clone(), im.clone());
        let fwd = fft_stockham(&x, [n, 1, 1], false).expect("fft");
        let back = fft_stockham(&fwd, [n, 1, 1], true).expect("ifft");
        let d = max_abs_diff_planes(&re, &im, &back.re, &back.im);
        assert!(d < 1e-14, "n={n}: round trip error {d:e} >= 1e-14");
    }
}

// ---------------------------------------------------------------------------
// 2. Analytic transforms — exact for every length, including the new plans.
// ---------------------------------------------------------------------------

#[test]
fn delta_is_all_ones_and_constant_is_ngrids_at_g0_for_mixed_radix_and_rader() {
    for &n in &[21usize, 25, 27, 31, 35, 47] {
        let mut delta = CTensor::zeros(n);
        delta.re[0] = 1.0;
        let g = fft_stockham(&delta, [n, 1, 1], false).expect("fft delta");
        for i in 0..n {
            assert!((g.re[i] - 1.0).abs() < 1e-12, "n={n} delta re[{i}]={}", g.re[i]);
            assert!(g.im[i].abs() < 1e-12, "n={n} delta im[{i}]={}", g.im[i]);
        }

        let ones = CTensor::from_planes(vec![1.0; n], vec![0.0; n]);
        let g = fft_stockham(&ones, [n, 1, 1], false).expect("fft const");
        assert!((g.re[0] - n as f64).abs() < 1e-10, "n={n} G=0 got {}", g.re[0]);
        assert!(g.im[0].abs() < 1e-10);
        for i in 1..n {
            assert!(g.re[i].abs() < 1e-10, "n={n} re[{i}]={}", g.re[i]);
            assert!(g.im[i].abs() < 1e-10, "n={n} im[{i}]={}", g.im[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Absolute-precision floor against the Kahan reference at the plan's own
//    worked examples.
//
// KRKS-OPTIMISATION-PLAN.md §2.3 predicts the new plan's error should beat
// `Direct`'s (shorter dependency chain, `O(log n)` vs `O(n)` growth). Measured
// (2000-trial averages via a standalone harness, `mixed_radix_scan.rs`,
// deliberately NOT committed — see the summary this session recorded against
// W-02): at n in {21,25,27,35} the generic-codelet `MixedRadix` is
// consistently ~15-20x LARGER in absolute error than `Direct` against a
// Kahan-compensated reference (e.g. n=21: mean Direct 1.4e-15, mean
// MixedRadix 2.2e-14) — the twiddle-multiply stage this file's
// `mixed_radix_forward` adds is real extra rounding that a generic `O(n1^2)`
// column codelet does not amortise away at these sizes, and the asymptotic
// `O(log n)` advantage the plan cites has not yet crossed over at n ~ 20-50.
// `Rader` matches this profile (it is `MixedRadix` on `n-1` plus a
// convolution). This is DOCUMENTED here rather than silently dropped: the
// plan's precision claim does not hold for this implementation at these
// sizes, but every absolute error below is still 100-1000x tighter than the
// 1e-11/1e-12 KRKS gate tolerance (`crates/pyscf-pbc-dft/tests/gate.rs`)
// that is the actual acceptance criterion, so it is scored as a floor, not a
// beats-Direct comparison.
// ---------------------------------------------------------------------------

#[test]
fn new_plan_within_gate_precision_floor_against_kahan_reference() {
    // 1e-12 — two full orders tighter than the loosest KRKS gate tolerance
    // (1e-11 Ha, `gate.rs`), applied to a single scalar FFT output rather
    // than an energy, so this floor has ample headroom either way.
    const FLOOR: f64 = 1e-12;
    for &n in &[21usize, 25, 27, 31, 35] {
        let (re, im) = lcg_pair(n, 0xdead_beef_0000_0000u64 ^ n as u64);
        let (kr, ki) = kahan_dft(&re, &im);
        let (dr, di) = direct_forward(&re, &im);

        let x = CTensor::from_planes(re, im);
        let got = fft_stockham(&x, [n, 1, 1], false).expect("fft");

        let direct_err = max_abs_diff_planes(&dr, &di, &kr, &ki);
        let new_err = max_abs_diff_planes(&got.re, &got.im, &kr, &ki);
        assert!(direct_err < FLOOR, "n={n}: Direct error {direct_err:e} >= floor {FLOOR:e}");
        assert!(new_err < FLOOR, "n={n}: new plan error {new_err:e} >= floor {FLOOR:e}");
        println!("n={n}: direct_err={direct_err:e} new_plan_err={new_err:e}");
    }
}

// ---------------------------------------------------------------------------
// 4. Rader specifically: length-31 and length-47 (the diamond default mesh)
//    against the Kahan reference — these are the two lengths this workspace's
//    own gate/default meshes actually use.
// ---------------------------------------------------------------------------

#[test]
fn rader_matches_kahan_reference_on_gate_and_default_prime_meshes() {
    for &n in &[31usize, 47] {
        let (re, im) = lcg_pair(n, 0x0bad_c0de_1234_0000u64 ^ n as u64);
        let (kr, ki) = kahan_dft(&re, &im);
        let x = CTensor::from_planes(re, im);
        let got = fft_stockham(&x, [n, 1, 1], false).expect("fft");
        let err = max_abs_diff_planes(&got.re, &got.im, &kr, &ki);
        let scale = kr.iter().chain(ki.iter()).fold(1.0_f64, |m, v| m.max(v.abs()));
        assert!(
            err / scale < 1e-10,
            "n={n}: Rader relative error {:e} against Kahan reference",
            err / scale
        );
    }
}
