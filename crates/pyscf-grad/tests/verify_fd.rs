//! Structural tests for the D-01 / GRAD-09 always-on FD numeric gate.
//!
//! These prove the central-difference harness ([`pyscf_grad::verify_fd`]) is
//! correct + always-on BEFORE any method body lands (07-03+). The reference is
//! a trivial closed-form energy whose analytical gradient is exact, so the
//! harness MUST pass at ≤1e-6 Ha/Bohr; the RHF wave then wires its real
//! `as_scanner` energy + analytical gradient into the same harness.
//!
//! Run scoped + single-threaded (the user-memory + CI discipline):
//!   cargo test -p pyscf-grad --locked -- --test-threads=1 verify_fd

use pyscf_grad::verify_fd::{DEFAULT_DISP, FD_TOL};
use pyscf_grad::{GradError, verify_fd};

/// A separable quadratic energy `E(R) = sum_ia sum_c k_iac * (R_iac - r0_iac)^2`,
/// whose exact gradient is `dE/dR_iac = 2 * k_iac * (R_iac - r0_iac)`. A central
/// difference of a quadratic is EXACT (no truncation error), so the FD harness
/// agrees with the analytical gradient to machine precision.
fn quadratic_energy(k: &[[f64; 3]], r0: &[[f64; 3]], coords: &[[f64; 3]]) -> f64 {
    let mut terms = Vec::new();
    for ia in 0..coords.len() {
        for c in 0..3 {
            let d = coords[ia][c] - r0[ia][c];
            terms.push(k[ia][c] * d * d);
        }
    }
    pyscf_algebra::oracle_sum(&terms)
}

fn quadratic_grad(k: &[[f64; 3]], r0: &[[f64; 3]], coords: &[[f64; 3]]) -> Vec<[f64; 3]> {
    (0..coords.len())
        .map(|ia| {
            let mut row = [0.0f64; 3];
            for c in 0..3 {
                row[c] = 2.0 * k[ia][c] * (coords[ia][c] - r0[ia][c]);
            }
            row
        })
        .collect()
}

#[test]
fn fd_matches_analytical_on_quadratic_reference() {
    // A 2-atom system displaced off its quadratic minimum.
    let coords = [[0.10, -0.20, 0.05], [1.40, 0.30, -0.15]];
    let k = [[1.5, 0.8, 2.1], [0.6, 1.1, 0.9]];
    let r0 = [[0.00, 0.00, 0.00], [1.40, 0.00, 0.00]];

    let analytical = quadratic_grad(&k, &r0, &coords);

    let report = verify_fd(
        &coords,
        &analytical,
        |c| Ok(quadratic_energy(&k, &r0, c)),
        DEFAULT_DISP,
        FD_TOL,
    )
    .expect("verify_fd should run on a valid quadratic reference");

    assert!(
        report.passed,
        "FD must agree with the exact analytical gradient within {FD_TOL} Ha/Bohr; \
         got max|fd - analytical| = {:e}",
        report.max_abs_diff
    );
    // A quadratic's central difference is exact — agreement is at machine
    // precision, far tighter than the 1e-6 gate.
    assert!(report.max_abs_diff < 1e-9, "got {:e}", report.max_abs_diff);
    assert_eq!(report.fd_grad.len(), coords.len());
}

#[test]
fn fd_flags_a_wrong_analytical_gradient() {
    // Same reference, but feed a deliberately wrong analytical gradient: the
    // gate MUST NOT pass (proves it is a real check, not a no-op).
    let coords = [[0.10, -0.20, 0.05], [1.40, 0.30, -0.15]];
    let k = [[1.5, 0.8, 2.1], [0.6, 1.1, 0.9]];
    let r0 = [[0.00, 0.00, 0.00], [1.40, 0.00, 0.00]];

    let mut wrong = quadratic_grad(&k, &r0, &coords);
    wrong[0][0] += 0.01; // perturb one component well above the 1e-6 tol

    let report = verify_fd(
        &coords,
        &wrong,
        |c| Ok(quadratic_energy(&k, &r0, c)),
        DEFAULT_DISP,
        FD_TOL,
    )
    .expect("verify_fd should still run");

    assert!(!report.passed, "a wrong analytical gradient must fail the gate");
    assert!(report.max_abs_diff >= 1e-3);
}

#[test]
fn fd_rejects_nonpositive_disp() {
    // T-07-05: a zero / negative / non-finite disp is rejected BEFORE any
    // energy eval, with a clear GradError (no panic, no OOB).
    let coords = [[0.0, 0.0, 0.0]];
    let analytical = [[0.0, 0.0, 0.0]];

    for bad in [0.0, -1e-4, f64::NAN, f64::INFINITY] {
        let err = verify_fd(&coords, &analytical, |_| Ok(0.0), bad, FD_TOL)
            .expect_err("non-positive / non-finite disp must be rejected");
        // The error bridges through Core(InvalidMolecule(..)); confirm the
        // GradError::InvalidDisplacement message is carried.
        let msg = format!("{err}");
        assert!(
            msg.contains("invalid finite-difference displacement"),
            "expected InvalidDisplacement message, got: {msg}"
        );
    }

    // Sanity: GradError::InvalidDisplacement Display is well-formed.
    let direct = GradError::InvalidDisplacement { disp: 0.0 };
    assert!(format!("{direct}").contains("must be finite and > 0"));
}

#[test]
fn fd_rejects_shape_mismatch() {
    // analytical row count must equal coords row count.
    let coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let analytical = [[0.0, 0.0, 0.0]]; // one row short
    let err = verify_fd(&coords, &analytical, |_| Ok(0.0), DEFAULT_DISP, FD_TOL)
        .expect_err("row-count mismatch must be rejected");
    assert!(format!("{err}").contains("shape mismatch"));
}
