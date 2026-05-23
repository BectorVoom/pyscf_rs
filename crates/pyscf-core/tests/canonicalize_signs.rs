//! Phase 3 plan 03-01 — SCF-13 unit tests for `canonicalize_signs`.
//!
//! Source: RESEARCH §"Pattern 9: Sign Canonicalization" + §"Anti-Patterns"
//! line 1028 ("must use strict-greater-than comparison so ties break to
//! lowest index"). The reference algorithm is at upstream
//! `pyscf/scf/hf.py:1349-1357` (inline `def eig`).
//!
//! The 6 tests cover: idempotency, sign-flip on negative leader, no-flip
//! on positive leader, tie-break-to-lowest-index (the regression that
//! `>=` would silently break — Pitfall 12 anchor), cross-vendor sign
//! reproducibility, and F-order indexing correctness.

use pyscf_core::canonicalize_signs;

/// F-order 3×nmo matrix builder helper: rows = nao, cols = nmo.
fn f_order(rows: &[[f64; 2]]) -> Vec<f64> {
    let nao = rows.len();
    let nmo = 2;
    let mut out = vec![0.0; nao * nmo];
    for (i, row) in rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            out[i + j * nao] = v;
        }
    }
    out
}

#[test]
fn idempotent() {
    let mut c = f_order(&[[-0.7, 0.3], [0.5, -0.9], [0.2, 0.1]]);
    canonicalize_signs(&mut c, 3, 2);
    let snapshot = c.clone();
    canonicalize_signs(&mut c, 3, 2);
    assert_eq!(c, snapshot, "second call must be a no-op");
}

#[test]
fn flips_when_max_is_negative() {
    // col 0: largest-|.| at row 0 = 0.7, but c[0,0] = -0.7 < 0 → flip whole col
    // col 1: largest-|.| at row 1 = 0.9, c[1,1] = -0.9 < 0 → flip whole col
    let mut c = f_order(&[[-0.7, 0.3], [0.5, -0.9], [0.2, 0.1]]);
    canonicalize_signs(&mut c, 3, 2);
    // After: col 0 = [0.7, -0.5, -0.2], col 1 = [-0.3, 0.9, -0.1]
    assert!((c[0] - 0.7).abs() < 1e-15, "c[0,0]={}", c[0]);
    assert!((c[1] - -0.5).abs() < 1e-15, "c[1,0]={}", c[1]);
    assert!((c[2] - -0.2).abs() < 1e-15, "c[2,0]={}", c[2]);
    assert!((c[3] - -0.3).abs() < 1e-15, "c[0,1]={}", c[3]);
    assert!((c[4] - 0.9).abs() < 1e-15, "c[1,1]={}", c[4]);
    assert!((c[5] - -0.1).abs() < 1e-15, "c[2,1]={}", c[5]);
}

#[test]
fn no_flip_when_max_is_positive() {
    let mut c = f_order(&[[0.7, 0.3], [-0.5, 0.9], [0.2, 0.1]]);
    let snap = c.clone();
    canonicalize_signs(&mut c, 3, 2);
    assert_eq!(c, snap, "positive-leading cols should not flip");
}

#[test]
fn tie_break_to_lowest_index() {
    // Both row 1 and row 3 have |c|=0.8 in col 0; STRICT > picks row 1.
    // c[1,0] = -0.8 < 0 → flip whole col. Conversely if we wrongly used >=,
    // we'd pick row 3 (c[3,0] = +0.8 > 0) → no flip. We assert the FLIP.
    let mut c = vec![0.0; 4]; // nao=4, nmo=1
    c[0] = 0.1;
    c[1] = -0.8;
    c[2] = 0.2;
    c[3] = 0.8;
    canonicalize_signs(&mut c, 4, 1);
    // After tie-break-to-lowest: col flipped (because c[1] < 0).
    assert!((c[0] - -0.1).abs() < 1e-15);
    assert!((c[1] - 0.8).abs() < 1e-15);
    assert!((c[2] - -0.2).abs() < 1e-15);
    assert!((c[3] - -0.8).abs() < 1e-15);
}

#[test]
fn cross_vendor_reproducibility() {
    // Simulate MKL vs Accelerate producing eigenvectors with opposite signs.
    let mut mkl_like = f_order(&[[-0.7, 0.3], [0.5, -0.9], [0.2, 0.1]]);
    let mut acc_like = mkl_like.iter().map(|v| -v).collect::<Vec<_>>();
    canonicalize_signs(&mut mkl_like, 3, 2);
    canonicalize_signs(&mut acc_like, 3, 2);
    assert_eq!(mkl_like, acc_like, "vendor sign flip must wash out");
}

#[test]
fn f_order_indexing_not_c_order() {
    // 3×2 matrix in F-order vs same data in C-order should yield DIFFERENT
    // results — confirming the function reads F-order strides.
    // F-order: c[0]=(0,0), c[1]=(1,0), c[2]=(2,0), c[3]=(0,1), c[4]=(1,1), c[5]=(2,1)
    let mut c_fo = vec![-0.9, 0.1, 0.2, 0.3, -0.5, 0.4];
    canonicalize_signs(&mut c_fo, 3, 2);
    // Col 0 leader: |−0.9| at row 0 → flip whole col 0.
    // After: c_fo[0..3] = [0.9, -0.1, -0.2]
    assert!((c_fo[0] - 0.9).abs() < 1e-15);
    assert!((c_fo[1] - -0.1).abs() < 1e-15);
    assert!((c_fo[2] - -0.2).abs() < 1e-15);
}
