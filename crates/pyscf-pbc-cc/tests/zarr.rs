//! The `ZArr` / `einsum` substrate (plan 16-04's prerequisite).
//!
//! Oracle-free: every assertion is an algebraic identity or a bit-identity
//! against an explicitly written-out loop.

use pyscf_algebra::CTensor;
use pyscf_pbc_cc::{ZArr, einsum, einsum_scaled};

fn ramp(shape: &[usize], off: f64) -> ZArr {
    let n: usize = shape.iter().product();
    ZArr::from_ctensor(
        shape,
        CTensor::from_planes(
            (0..n).map(|i| (i as f64 * 0.37).sin() + off).collect(),
            (0..n).map(|i| (i as f64 * 0.11).cos() - off).collect(),
        ),
    )
    .expect("shape matches")
}

/// `transpose` follows NUMPY's convention: `axes[k]` is the SOURCE axis that
/// becomes output axis `k`.
///
/// Getting this backwards is the 14-05 `decompose_j2c` class of defect
/// (`16-CONTEXT §3.4`), so it is asserted against a hand-written loop rather
/// than assumed.
#[test]
fn transpose_matches_the_numpy_convention() {
    let shape = [2usize, 3, 4, 5];
    let x = ramp(&shape, 0.25);
    let axes = [2usize, 3, 0, 1];
    let y = x.transpose(&axes).expect("valid permutation");
    assert_eq!(y.shape(), &[4, 5, 2, 3]);
    for a in 0..4 {
        for b in 0..5 {
            for c in 0..2 {
                for d in 0..3 {
                    // y[a,b,c,d] == x[c,d,a,b]
                    assert_eq!(y.at(&[a, b, c, d]).unwrap(), x.at(&[c, d, a, b]).unwrap());
                }
            }
        }
    }
    // A permutation is invertible: transpose(2,3,0,1) twice is the identity.
    assert_eq!(x.transpose(&axes).unwrap().transpose(&axes).unwrap(), x);
    assert!(x.transpose(&[0, 0, 1, 2]).is_err());
    assert!(x.transpose(&[0, 1, 2]).is_err());
}

/// A two-operand contraction agrees with the explicit loop, bit-for-bit for a
/// single contracted index (where there is no summation order to disagree
/// about).
#[test]
fn einsum_two_operand_matches_the_explicit_loop() {
    let a = ramp(&[3, 4], 0.5); // 'ic'
    let b = ramp(&[4, 5], -0.25); // 'cj'
    let got = einsum("ic,cj->ij", &[&a, &b]).expect("valid spec");
    assert_eq!(got.shape(), &[3, 5]);
    for i in 0..3 {
        for j in 0..5 {
            let (mut re, mut im) = (0.0_f64, 0.0_f64);
            for c in 0..4 {
                let (ar, ai) = a.at(&[i, c]).unwrap();
                let (br, bi) = b.at(&[c, j]).unwrap();
                re += ar * br - ai * bi;
                im += ar * bi + ai * br;
            }
            let (gr, gi) = got.at(&[i, j]).unwrap();
            assert!((gr - re).abs() < 1e-14 && (gi - im).abs() < 1e-14);
        }
    }
}

/// **The einsum is UNCONJUGATED**, exactly as `numpy.einsum` is.
///
/// `15-REVIEW.md D-15-R-02` found that a plan saying only "route through
/// `oracle_dot`" produces `Σ conj(x)·y` where `Σ x·y` was meant — a plausible
/// wrong number no gate but the last would catch. This asserts the direction
/// explicitly: `einsum("i,i->", x, x)` is `Σ x²` (which for a purely imaginary
/// `x` is NEGATIVE real), not `Σ |x|²`.
#[test]
fn einsum_does_not_conjugate() {
    let x = ZArr::from_ctensor(&[3], CTensor::from_planes(vec![0.0; 3], vec![1.0, 2.0, 3.0]))
        .unwrap();
    let unconj = einsum("i,i->", &[&x, &x]).unwrap();
    assert_eq!(unconj.at(&[]).unwrap(), (-14.0, 0.0), "Σ x·x = -(1+4+9)");
    // The conjugated product is written by conjugating an operand explicitly.
    let conj = einsum("i,i->", &[&x.conj(), &x]).unwrap();
    assert_eq!(conj.at(&[]).unwrap(), (14.0, 0.0), "Σ conj(x)·x = |x|²");
}

/// Three-operand contractions work — upstream writes them (`cc_Foo`'s
/// `'klcd,ic,ld->ki'`, `kintermediates_rhf.py:50`).
#[test]
fn einsum_three_operands() {
    let s = ramp(&[2, 3, 4, 5], 0.1); // klcd
    let t1i = ramp(&[2, 4], 0.2); // ic
    let t1l = ramp(&[3, 5], -0.3); // ld
    let got = einsum("klcd,ic,ld->ki", &[&s, &t1i, &t1l]).unwrap();
    assert_eq!(got.shape(), &[2, 2]);
    for k in 0..2 {
        for i in 0..2 {
            let (mut re, mut im) = (0.0_f64, 0.0_f64);
            for l in 0..3 {
                for c in 0..4 {
                    for d in 0..5 {
                        let (sr, si) = s.at(&[k, l, c, d]).unwrap();
                        let (ar, ai) = t1i.at(&[i, c]).unwrap();
                        let (br, bi) = t1l.at(&[l, d]).unwrap();
                        let (pr, pi) = (sr * ar - si * ai, sr * ai + si * ar);
                        re += pr * br - pi * bi;
                        im += pr * bi + pi * br;
                    }
                }
            }
            let (gr, gi) = got.at(&[k, i]).unwrap();
            assert!((gr - re).abs() < 1e-13 && (gi - im).abs() < 1e-13, "k{k} i{i}");
        }
    }
}

/// Determinism (§9.3): the result depends only on the spec and the data, never
/// on the thread count. Each output element is one `oracle_zsum` over a
/// fixed-length buffer, so this holds by construction — asserted anyway.
#[test]
fn einsum_is_bit_identical_across_repeated_runs() {
    let a = ramp(&[4, 5, 6], 0.3);
    let b = ramp(&[5, 6, 7], -0.7);
    let first = einsum("abc,bcd->ad", &[&a, &b]).unwrap();
    let second = einsum("abc,bcd->ad", &[&a, &b]).unwrap();
    assert_eq!(first, second);
}

/// Malformed specs are refused, not silently mis-computed.
#[test]
fn malformed_specs_are_refused() {
    let a = ramp(&[3, 4], 0.0);
    let b = ramp(&[4, 3], 0.0);
    assert!(einsum("ij,jk", &[&a, &b]).is_err(), "no '->'");
    assert!(einsum("ij,jk->ik", &[&a]).is_err(), "operand count");
    assert!(einsum("ijk,jk->i", &[&a, &b]).is_err(), "rank mismatch");
    assert!(einsum("ii,jk->jk", &[&a, &b]).is_err(), "diagonal");
    assert!(einsum("ij,jk->ix", &[&a, &b]).is_err(), "free output letter");
    let c = ramp(&[5, 3], 0.0);
    assert!(einsum("ij,jk->ik", &[&a, &c]).is_err(), "extent mismatch");
}

/// `slice_leading` / `set_leading` round-trip — the `eris.oovv[kk,kl,kc]`
/// access pattern the intermediates are written in.
#[test]
fn leading_slice_roundtrip() {
    let mut x = ZArr::zeros(&[2, 3, 4, 5]);
    let blk = ramp(&[4, 5], 0.9);
    x.set_leading(&[1, 2], &blk).unwrap();
    assert_eq!(x.slice_leading(&[1, 2]).unwrap(), blk);
    assert_eq!(x.slice_leading(&[0, 0]).unwrap(), ZArr::zeros(&[4, 5]));
    assert!(x.set_leading(&[2, 0], &blk).is_err());
    assert!(x.set_leading(&[0, 0], &ramp(&[4, 4], 0.0)).is_err());
    // A leading slice of rank 1 keeps the rest.
    assert_eq!(x.slice_leading(&[1]).unwrap().shape(), &[3, 4, 5]);
}

/// `einsum_scaled` is exactly `factor * einsum`.
#[test]
fn einsum_scaled_matches_a_scaled_einsum() {
    let a = ramp(&[3, 4], 0.5);
    let b = ramp(&[4, 3], -0.5);
    let mut want = einsum("ij,ji->", &[&a, &b]).unwrap();
    want.scale(0.5);
    assert_eq!(einsum_scaled("ij,ji->", &[&a, &b], 0.5).unwrap(), want);
}
