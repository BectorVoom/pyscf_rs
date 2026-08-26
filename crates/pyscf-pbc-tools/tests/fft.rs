//! Plan 11-01 / 11-03 — the complex 3-D FFT.
//!
//! Four layers, none of which needs Python at run time:
//!
//! 1. **Round trip** — `ifft(fft(x)) == x` to 1e-12 over meshes with odd and
//!    prime axes (3, 5, 7, 11, 13, 17), which is where a mishandled negative
//!    frequency fold shows up first.
//! 2. **Analytic** — a delta transforms to all-ones; a constant transforms to
//!    `ngrids * delta_{G,0}`.
//! 3. **Engine parity (D-PBC-06)** — `fft_blas` vs `fft_stockham` to 1e-13 over
//!    200 random `(mesh, n_batch)` combinations. This is the condition that
//!    licenses `stockham` as the default engine.
//! 4. **Upstream** — both engines against `tools.fft` arrays captured from live
//!    PySCF 2.12.1 in `fixtures/fft_reference.rs` (tier-2 hard-coded numbers,
//!    D-PBC-19). Regenerate with:
//!
//!    ```python
//!    import numpy as np
//!    from pyscf.pbc import tools
//!    # the same LCG as `lcg()` below, then `tools.fft(f, mesh)`
//!    ```

#[path = "fixtures/fft_reference.rs"]
mod fft_reference;

use pyscf_algebra::CTensor;
use pyscf_pbc_tools::fft::{fft_blas, fft_stockham, ifft_blas};
use pyscf_pbc_tools::{fftk, ifftk};

/// The same 64-bit LCG the fixture generator used, mapped into `[-1, 1)`.
fn lcg(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

fn random_ctensor(n: usize, seed: u64) -> CTensor {
    let v = lcg(2 * n, seed);
    CTensor::from_planes(
        v.iter().step_by(2).copied().collect(),
        v.iter().skip(1).step_by(2).copied().collect(),
    )
}

fn max_abs_diff(a: &CTensor, b: &CTensor) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut w = 0.0_f64;
    for i in 0..a.len() {
        w = w.max((a.re[i] - b.re[i]).abs());
        w = w.max((a.im[i] - b.im[i]).abs());
    }
    w
}

// ---------------------------------------------------------------------------
// 1. Round trip
// ---------------------------------------------------------------------------

#[test]
fn round_trip_over_odd_and_prime_meshes() {
    let dims = [1usize, 2, 3, 4, 5, 7, 8, 11, 13, 17];
    let mut seed = 7u64;
    let mut cases = 0usize;
    for &mx in &dims {
        for &my in &[3usize, 5, 8] {
            for &mz in &[1usize, 7, 13] {
                for &nb in &[1usize, 3] {
                    let mesh = [mx, my, mz];
                    let ng = mx * my * mz;
                    seed = seed.wrapping_mul(31).wrapping_add(17);
                    let x = random_ctensor(nb * ng, seed);
                    let g = fft_stockham(&x, mesh, false).expect("fft");
                    let back = fft_stockham(&g, mesh, true).expect("ifft");
                    assert!(
                        max_abs_diff(&x, &back) < 1e-12,
                        "stockham round trip failed on mesh {mesh:?} nb {nb}: {}",
                        max_abs_diff(&x, &back)
                    );
                    let gb = fft_blas(&x, mesh).expect("fft_blas");
                    let bb = ifft_blas(&gb, mesh).expect("ifft_blas");
                    assert!(
                        max_abs_diff(&x, &bb) < 1e-12,
                        "blas round trip failed on mesh {mesh:?} nb {nb}"
                    );
                    cases += 1;
                }
            }
        }
    }
    assert!(cases >= 90, "expected a broad sweep, ran {cases}");
}

// ---------------------------------------------------------------------------
// 2. Analytic transforms
// ---------------------------------------------------------------------------

#[test]
fn delta_transforms_to_all_ones() {
    let mesh = [5usize, 4, 3];
    let ng = 60;
    let mut x = CTensor::zeros(ng);
    x.re[0] = 1.0;
    for g in [
        fft_stockham(&x, mesh, false).expect("fft"),
        fft_blas(&x, mesh).expect("fft_blas"),
    ] {
        for i in 0..ng {
            assert!((g.re[i] - 1.0).abs() < 1e-13, "re[{i}] = {}", g.re[i]);
            assert!(g.im[i].abs() < 1e-13, "im[{i}] = {}", g.im[i]);
        }
    }
}

#[test]
fn constant_transforms_to_ngrids_at_g0() {
    let mesh = [7usize, 5, 3];
    let ng = 105;
    let x = CTensor::from_planes(vec![1.0; ng], vec![0.0; ng]);
    for g in [
        fft_stockham(&x, mesh, false).expect("fft"),
        fft_blas(&x, mesh).expect("fft_blas"),
    ] {
        assert!((g.re[0] - ng as f64).abs() < 1e-10, "G=0 got {}", g.re[0]);
        assert!(g.im[0].abs() < 1e-10);
        for i in 1..ng {
            assert!(g.re[i].abs() < 1e-10, "re[{i}] = {}", g.re[i]);
            assert!(g.im[i].abs() < 1e-10, "im[{i}] = {}", g.im[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Engine parity — the D-PBC-06 gate for the `stockham` default
// ---------------------------------------------------------------------------

#[test]
fn stockham_matches_blas_on_200_random_cases() {
    let mut seed = 20_260_826u64;
    let mut worst = 0.0_f64;
    for case in 0..200 {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let pick = |shift: u32, lo: usize, hi: usize| -> usize {
            lo + ((seed >> shift) as usize) % (hi - lo + 1)
        };
        let mesh = [pick(5, 1, 12), pick(17, 1, 12), pick(29, 1, 12)];
        let nb = pick(41, 1, 3);
        let ng = mesh[0] * mesh[1] * mesh[2];
        let x = random_ctensor(nb * ng, seed ^ 0x5555);

        let a = fft_stockham(&x, mesh, false).expect("fft_stockham");
        let b = fft_blas(&x, mesh).expect("fft_blas");
        let scale = b.re.iter().chain(b.im.iter()).fold(1.0_f64, |m, v| m.max(v.abs()));
        let d = max_abs_diff(&a, &b) / scale;
        assert!(
            d < 1e-13,
            "case {case}: forward engines disagree by {d} on mesh {mesh:?} nb {nb}"
        );
        worst = worst.max(d);

        let ia = fft_stockham(&x, mesh, true).expect("ifft_stockham");
        let ib = ifft_blas(&x, mesh).expect("ifft_blas");
        let iscale = ib.re.iter().chain(ib.im.iter()).fold(1.0_f64, |m, v| m.max(v.abs()));
        let di = max_abs_diff(&ia, &ib) / iscale;
        assert!(
            di < 1e-13,
            "case {case}: inverse engines disagree by {di} on mesh {mesh:?} nb {nb}"
        );
        worst = worst.max(di);
    }
    println!("engine parity: worst relative deviation over 200 cases = {worst:e}");
}

// ---------------------------------------------------------------------------
// 4. Upstream reference
// ---------------------------------------------------------------------------

#[test]
fn matches_upstream_fft_on_3x3x3() {
    check_upstream([3, 3, 3], &fft_reference::M333_RE, &fft_reference::M333_IM);
}

#[test]
fn matches_upstream_fft_on_4x5x7() {
    check_upstream([4, 5, 7], &fft_reference::M457_RE, &fft_reference::M457_IM);
}

fn check_upstream(mesh: [usize; 3], want_re: &[f64], want_im: &[f64]) {
    let ng = mesh[0] * mesh[1] * mesh[2];
    assert_eq!(want_re.len(), ng);
    let x = random_ctensor(ng, 12345);
    let want = CTensor::from_planes(want_re.to_vec(), want_im.to_vec());
    for (name, got) in [
        ("stockham", fft_stockham(&x, mesh, false).expect("fft")),
        ("blas", fft_blas(&x, mesh).expect("fft_blas")),
    ] {
        let d = max_abs_diff(&got, &want);
        assert!(d < 1e-12, "{name} deviates from upstream by {d} on {mesh:?}");
    }
}

// ---------------------------------------------------------------------------
// fftk / ifftk
// ---------------------------------------------------------------------------

/// `ifftk(fftk(f, expmikr), conj(expmikr)) == f`: the two phase factors are
/// exact inverses, so the round trip is the plain FFT round trip.
#[test]
fn fftk_ifftk_round_trip() {
    let mesh = [5usize, 3, 4];
    let ng = 60;
    let x = random_ctensor(2 * ng, 99);
    let ph = random_ctensor(ng, 4242);
    // Normalise the phase to unit modulus so it is a genuine e^{-ikr}.
    let mut pr = vec![0.0; ng];
    let mut pi = vec![0.0; ng];
    for i in 0..ng {
        let m = (ph.re[i] * ph.re[i] + ph.im[i] * ph.im[i]).sqrt().max(1e-12);
        pr[i] = ph.re[i] / m;
        pi[i] = ph.im[i] / m;
    }
    let expmikr = CTensor::from_planes(pr.clone(), pi.clone());
    let expikr = CTensor::from_planes(pr, pi.iter().map(|v| -v).collect());

    let g = fftk(&x, mesh, &expmikr).expect("fftk");
    let back = ifftk(&g, mesh, &expikr).expect("ifftk");
    assert!(
        max_abs_diff(&x, &back) < 1e-12,
        "fftk/ifftk round trip: {}",
        max_abs_diff(&x, &back)
    );
}
