//! CCSD-09: T1/D1/D2 wavefunction diagnostics on synthetic amplitudes with
//! hand-computed Frobenius/eigh references (port of `pyscf/cc/ccsd.py:748-776`).
//!
//! These are pure value checks against analytically-known references — no live
//! PySCF needed (byte-identity vs upstream is the 06-08 workflow_dispatch arm).
//! Each diagnostic's reference is derived by hand in the test comments so the
//! ported definition is pinned independent of any kernel run.

use pyscf_ccsd::diagnostics::{get_d1_diagnostic, get_d2_diagnostic, get_t1_diagnostic};

/// C-order flat index for a t1 block `[nocc, nvir]`: element (i, a) at i*nvir + a.
fn t1_idx(nvir: usize, i: usize, a: usize) -> usize {
    i * nvir + a
}

/// C-order flat index for a t2 block `[nocc, nocc, nvir, nvir]`:
/// element (i, j, a, b) at ((i*nocc + j)*nvir + a)*nvir + b.
fn t2_idx(nocc: usize, nvir: usize, i: usize, j: usize, a: usize, b: usize) -> usize {
    ((i * nocc + j) * nvir + a) * nvir + b
}

/// T1 = ||t1||_F / sqrt(nelec). Synthetic t1 (nocc=2, nvir=2) = [[3,0],[0,4]]:
/// ||t1||_F^2 = 9 + 16 = 25, ||t1||_F = 5. With nelec = 2*nocc = 4 → T1 = 5/2 = 2.5.
#[test]
fn t1_diagnostic_matches_hand_computed_frobenius() {
    let nocc = 2;
    let nvir = 2;
    let mut t1 = vec![0.0_f64; nocc * nvir];
    t1[t1_idx(nvir, 0, 0)] = 3.0;
    t1[t1_idx(nvir, 1, 1)] = 4.0;
    let nelec = 2.0 * nocc as f64; // upstream nelectron = 2 * t1.shape[0]
    let t1d = get_t1_diagnostic(&t1, nocc, nvir, nelec).expect("t1 diagnostic");
    assert!(
        (t1d - 2.5).abs() < 1e-12,
        "T1 diagnostic = {t1d}, expected 2.5 (||t1||_F=5 / sqrt(nelec=4))"
    );
}

/// D1 = max over (ij, ab) blocks of sqrt(max|eigenvalue|).
/// For t1 = [[3,0],[0,4]]: A_ij = t1.t1^T = diag(9,16), A_ab = t1^T.t1 = diag(9,16).
/// Largest sqrt(|eig|) of each = sqrt(16) = 4 → D1 = 4.
#[test]
fn d1_diagnostic_matches_hand_computed_eigh() {
    let nocc = 2;
    let nvir = 2;
    let mut t1 = vec![0.0_f64; nocc * nvir];
    t1[t1_idx(nvir, 0, 0)] = 3.0;
    t1[t1_idx(nvir, 1, 1)] = 4.0;
    let d1 = get_d1_diagnostic(&t1, nocc, nvir).expect("d1 diagnostic");
    assert!(
        (d1 - 4.0).abs() < 1e-10,
        "D1 diagnostic = {d1}, expected 4.0 (sqrt of max eigenvalue 16 of t1.t1^T)"
    );
}

/// D1 on a non-diagonal t1 — the ij and ab blocks share the same nonzero
/// eigenvalues (AA^T and A^TA), so both yield the same D1; this checks the
/// off-diagonal eigh path. t1 = [[1, 2]] (nocc=1, nvir=2):
/// A_ij = t1.t1^T = [[5]] (1*1 + 2*2 = 5), eig {5}, sqrt(5).
/// A_ab = t1^T.t1 = [[1,2],[2,4]], eigenvalues {0, 5}, sqrt(5).
/// D1 = max(sqrt(5), sqrt(5)) = sqrt(5).
#[test]
fn d1_diagnostic_offdiagonal_shares_eigenvalues() {
    let nocc = 1;
    let nvir = 2;
    let t1 = vec![1.0, 2.0]; // (0,0)=1, (0,1)=2
    let d1 = get_d1_diagnostic(&t1, nocc, nvir).expect("d1 diagnostic");
    let expected = 5.0_f64.sqrt();
    assert!(
        (d1 - expected).abs() < 1e-10,
        "D1 diagnostic = {d1}, expected sqrt(5) = {expected}"
    );
}

/// D2 = max over (ij, ab) blocks of sqrt(max|eigenvalue|).
/// Synthetic t2 (nocc=2, nvir=2) with only t2[0,0,0,0] = 2 nonzero:
/// B_ij = einsum('ikab,jkab->ij') → B[0,0] = 4 (2*2), else 0; eig {4,0}, sqrt(4)=2.
/// B_ab = einsum('ijac,ijbc->ab') → B[0,0] = 4, else 0; sqrt(4)=2. D2 = 2.
#[test]
fn d2_diagnostic_matches_hand_computed_eigh() {
    let nocc = 2;
    let nvir = 2;
    let mut t2 = vec![0.0_f64; nocc * nocc * nvir * nvir];
    t2[t2_idx(nocc, nvir, 0, 0, 0, 0)] = 2.0;
    let d2 = get_d2_diagnostic(&t2, nocc, nvir).expect("d2 diagnostic");
    assert!(
        (d2 - 2.0).abs() < 1e-10,
        "D2 diagnostic = {d2}, expected 2.0 (sqrt of max eigenvalue 4)"
    );
}

/// Shape validation: a t1 whose length disagrees with nocc*nvir is a
/// ShapeMismatch, never an OOB panic (#![forbid(unsafe_code)] discipline).
#[test]
fn diagnostics_shape_mismatch_errors_not_panics() {
    let nocc = 2;
    let nvir = 2;
    let bad_t1 = vec![1.0, 2.0, 3.0]; // length 3 != 4
    assert!(get_t1_diagnostic(&bad_t1, nocc, nvir, 4.0).is_err());
    assert!(get_d1_diagnostic(&bad_t1, nocc, nvir).is_err());
    let bad_t2 = vec![1.0; 5]; // length 5 != 16
    assert!(get_d2_diagnostic(&bad_t2, nocc, nvir).is_err());
}

/// The T1 diagnostic is thread-invariant (oracle_sum reduction): the value is
/// identical regardless of RAYON_NUM_THREADS (RAYON 1==8). We only assert the
/// determinism property here (the harness fixes the thread count); the value
/// match is covered above.
#[test]
fn t1_diagnostic_is_deterministic() {
    let nocc = 4;
    let nvir = 3;
    // A reproducible spread of values.
    let t1: Vec<f64> = (0..nocc * nvir).map(|k| (k as f64 - 5.0) * 0.1).collect();
    let nelec = 2.0 * nocc as f64;
    let a = get_t1_diagnostic(&t1, nocc, nvir, nelec).expect("t1");
    let b = get_t1_diagnostic(&t1, nocc, nvir, nelec).expect("t1");
    assert_eq!(
        a.to_bits(),
        b.to_bits(),
        "T1 diagnostic must be deterministic"
    );
}
