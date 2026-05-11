//! Phase 3 plan 03-01 — solve_linear unit tests.
//!
//! Source: RESEARCH §"Open Question 1" — DIIS B-matrix solve. The Pulay
//! system has -1 on the Lagrange row/col so the diagonal contains 0; faer
//! 0.24 `FullPivLu` handles it correctly (singular-detection via the
//! post-solve `is_finite` check — see `solve_linear.rs` for rationale).

use pyscf_algebra::{solve_linear, AlgebraError};

#[test]
fn identity_returns_rhs() {
    let a = [1.0, 0.0, 0.0, 1.0];
    let b = [2.0, 3.0];
    let x = solve_linear(&a, &b, 2).expect("identity");
    assert!((x[0] - 2.0).abs() < 1e-14);
    assert!((x[1] - 3.0).abs() < 1e-14);
}

#[test]
fn diis_b_matrix_shape_with_lagrange_row() {
    // 3×3 DIIS B-matrix with Lagrange-row pattern (zero on bottom-right
    // diagonal element). Construct: B[i,j] = (i+1)*(j+1) for i,j<2, -1 on
    // last row/col, 0 in corner. RHS = [0,0,-1].
    let a = [
        1.0,  2.0, -1.0,
        2.0,  4.5, -1.0,
       -1.0, -1.0,  0.0,
    ];
    let b = [0.0, 0.0, -1.0];
    // Manual solve: this system has a unique solution per Pulay's method.
    // We only assert the solve succeeds and the residual is small.
    let x = solve_linear(&a, &b, 3).expect("diis-shape solve");
    // Residual r = A·x - b should be near zero element-wise.
    for i in 0..3 {
        let r: f64 = (0..3).map(|j| a[i * 3 + j] * x[j]).sum::<f64>() - b[i];
        assert!(r.abs() < 1e-10, "residual too large at row {}: {}", i, r);
    }
}

#[test]
fn singular_matrix_returns_err() {
    // All-zeros row → singular.
    let a = [1.0, 2.0, 2.0, 4.0]; // det=0
    let b = [1.0, 2.0];
    match solve_linear(&a, &b, 2) {
        Err(AlgebraError::Singular)         => (),
        Err(AlgebraError::ShapeMismatch{..}) => panic!("expected Singular, got ShapeMismatch"),
        Err(e) => panic!("unexpected error variant: {:?}", e),
        Ok(_) => panic!("expected singular error, got Ok"),
    }
}

#[test]
fn shape_mismatch_returns_err() {
    let a = [1.0, 0.0, 0.0]; // 3 elements but n=2 → mismatch
    let b = [1.0, 2.0];
    assert!(matches!(solve_linear(&a, &b, 2), Err(AlgebraError::ShapeMismatch{..})));
}
