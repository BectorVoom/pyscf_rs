//! The pocketfft port (`pyscf_pbc_tools::pocketfft`): mathematics against a
//! naive DFT, and the long-double constants against exact rational
//! arithmetic. The bit-for-bit match with `scipy.fft` is pinned by
//! `pyscf-pbc-df/tests/get_nuc_bitexact.rs` (it needs the upstream venv).

use pyscf_pbc_tools::pocketfft::{
    Cmplx, Plan1d, c2c_3d, ld_quotient_to_f64, norm_fct_inv, uses_bluestein,
};

fn naive_dft(x: &[Cmplx], fwd: bool) -> Vec<Cmplx> {
    let n = x.len();
    let s = if fwd { -1.0 } else { 1.0 };
    (0..n)
        .map(|k| {
            let (mut r, mut i) = (0.0, 0.0);
            for (j, v) in x.iter().enumerate() {
                let t = s * 2.0 * std::f64::consts::PI * ((j * k) % n) as f64 / n as f64;
                r += v.r * t.cos() - v.i * t.sin();
                i += v.r * t.sin() + v.i * t.cos();
            }
            Cmplx { r, i }
        })
        .collect()
}

/// Every 1-D length through [`Plan1d`] — `cfftp` and Bluestein alike —
/// against the textbook DFT, both directions, including twiddled (`ido > 1`)
/// stages.
#[test]
fn cfftp_matches_a_naive_dft() {
    for n in 1..=200usize {
        let x: Vec<Cmplx> = (0..n)
            .map(|j| Cmplx {
                r: ((j * 37) % 11) as f64 - 5.0,
                i: ((j * 17) % 7) as f64 - 3.0,
            })
            .collect();
        let plan = Plan1d::new(n);
        for fwd in [true, false] {
            let mut y = x.clone();
            plan.exec(&mut y, 1.0, fwd);
            let want = naive_dft(&x, fwd);
            let scale = want
                .iter()
                .fold(1.0_f64, |m, v| m.max(v.r.abs()).max(v.i.abs()));
            for (a, b) in y.iter().zip(&want) {
                assert!(
                    (a.r - b.r).abs() < 1e-12 * scale && (a.i - b.i).abs() < 1e-12 * scale,
                    "n = {n}, fwd = {fwd}: {a:?} vs {b:?}"
                );
            }
        }
    }
}

/// `pocketfft_c`'s plan choice: below 50, or with `lpf^2 <= n`, always
/// `cfftp`; otherwise Bluestein only when its (fudged) cost guess wins. The
/// expected list is `pocketfft_c`'s logic re-evaluated independently in Python
/// for `n < 180` — the first Bluestein length is 89, not the first prime >= 50.
#[test]
fn bluestein_plan_choice_matches_pocketfft() {
    let want = [
        89, 101, 103, 107, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179,
    ];
    let got: Vec<usize> = (1..180).filter(|&n| uses_bluestein(n)).collect();
    assert_eq!(got, want);
}

/// An inverse 3-D transform of a forward one returns the input, with the
/// `1/N` applied once (on the first axis).
#[test]
fn c2c_3d_round_trips() {
    let mesh = [5, 6, 11];
    let n: usize = mesh.iter().product();
    let re: Vec<f64> = (0..n).map(|j| ((j * 31) % 13) as f64 - 6.0).collect();
    let im: Vec<f64> = (0..n).map(|j| ((j * 7) % 5) as f64 - 2.0).collect();
    let (fr, fi) = c2c_3d(&re, &im, mesh, true, 1.0);
    let (br, bi) = c2c_3d(&fr, &fi, mesh, false, norm_fct_inv(n as u64));
    for j in 0..n {
        assert!(
            (br[j] - re[j]).abs() < 1e-12 && (bi[j] - im[j]).abs() < 1e-12,
            "j = {j}"
        );
    }
}

/// `double(0.25L*pi/n)` and `double(1/ldbl(N))` against values computed with
/// exact rationals (round to 64 bits, then to 53). `n = 3, 6, 13` are cases
/// where the double-rounded angle differs from `0.25*PI/n` evaluated in f64.
#[test]
fn long_double_quotients_round_twice() {
    const PI_LD: u64 = 0xC90F_DAA2_2168_C235;
    let ang = |n: u64| ld_quotient_to_f64(PI_LD, -64, n);
    assert_eq!(ang(3).to_bits(), 0x3fd0_c152_382d_7366);
    assert_eq!(ang(6).to_bits(), 0x3fc0_c152_382d_7366);
    assert_eq!(ang(13).to_bits(), 0x3fae_eebf_2ca2_ada8);
    assert_eq!(ang(1331), 0.25 * std::f64::consts::PI / 1331.0);
    assert_ne!(ang(3), 0.25 * std::f64::consts::PI / 3.0);
    for n in [1u64, 3, 729, 1000, 1331, 1728] {
        assert_eq!(norm_fct_inv(n), 1.0 / n as f64, "1/{n}");
    }
    // The first N where the double-rounded reciprocal is NOT 1.0/N.
    assert_ne!(norm_fct_inv(2731), 1.0 / 2731.0);
}
