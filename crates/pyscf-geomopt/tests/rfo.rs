//! Integration test: the RFO step on model Hessians — step direction +
//! trust-radius cap + negative-eigenvalue shift (GEOMOPT-06, Task 2 of plan
//! 07-04). Always-on, no SCF/cintx dependency (the optimizer STRUCTURE is
//! provable on analytic model Hessians).

use pyscf_geomopt::ConvParams;
use pyscf_geomopt::rfo::{BfgsHessian, EPSILON_NEG_EIG, rfo_step};

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[test]
fn rfo_step_descends_on_1d_quadratic() {
    // E(x) = x² + 4x → H = [2], g = [4], minimum at x = −2.
    let hess = BfgsHessian {
        h: vec![2.0],
        nint: 1,
        n_updates: 0,
    };
    let g = [4.0];
    let (dq, _pred, _trust) = rfo_step(&hess, &g, 10.0, &ConvParams::gau(), None, None);
    // The RFO step is DOWNHILL and DAMPED relative to the Newton step (−2):
    // the augmented-Hessian eigenvalue scaling shortens the step. For
    // [[2,4],[4,0]] the lowest mode gives dq = −4/(2−λ) ≈ −0.78. The defining
    // RFO property is monotone descent, not Newton-equality.
    assert!(dq[0] < 0.0, "downhill step expected, got {}", dq[0]);
    assert!(
        dq[0] > -2.0 && dq[0] < -0.1,
        "RFO step is a damped descent in (−2, 0), got {}",
        dq[0]
    );
}

#[test]
fn rfo_step_descends_on_2d_quadratic() {
    // H = diag(2, 4), g = [2, 4] → Newton step = [−1, −1].
    let hess = BfgsHessian {
        h: vec![2.0, 0.0, 0.0, 4.0],
        nint: 2,
        n_updates: 0,
    };
    let g = [2.0, 4.0];
    let (dq, _pred, _trust) = rfo_step(&hess, &g, 10.0, &ConvParams::gau(), None, None);
    assert!(
        dq[0] < 0.0 && dq[1] < 0.0,
        "both components downhill: {dq:?}"
    );
}

#[test]
fn negative_eigenvalue_is_shifted_for_minimization() {
    // H = [−1] (negative curvature). Without the v0 shift a naive Newton step
    // −g/H = +2 would go UPHILL; the neg-eig shift makes the step downhill.
    let hess = BfgsHessian {
        h: vec![-1.0],
        nint: 1,
        n_updates: 0,
    };
    assert!(hess_min_eig_below_epsilon(&hess));
    let g = [2.0];
    let (dq, _pred, _trust) = rfo_step(&hess, &g, 10.0, &ConvParams::gau(), None, None);
    assert!(
        dq[0] < 0.0,
        "neg-eig shift must produce a downhill step, got {}",
        dq[0]
    );
}

/// The Hessian's smallest eigenvalue is below the neg-eig shift threshold.
fn hess_min_eig_below_epsilon(hess: &BfgsHessian) -> bool {
    // 1×1 case: the eigenvalue is the single element.
    hess.h[0] < EPSILON_NEG_EIG
}

#[test]
fn trust_radius_caps_a_large_step() {
    // A near-flat Hessian → a huge Newton step that must be capped at trust.
    let hess = BfgsHessian {
        h: vec![0.01],
        nint: 1,
        n_updates: 0,
    };
    let g = [1.0];
    let trust = 0.1;
    let (dq, _pred, _trust) = rfo_step(&hess, &g, trust, &ConvParams::gau(), None, None);
    assert!(
        norm(&dq) <= trust + 1e-9,
        "RFO step norm {} must be ≤ trust {trust}",
        norm(&dq)
    );
}

#[test]
fn bfgs_update_builds_curvature() {
    // Identity Hessian + a positive-curvature (dq, dg) pair → updated Hessian
    // stays symmetric and the update counter advances.
    let mut hess = BfgsHessian::identity(2);
    hess.bfgs_update(&[0.1, 0.05], &[0.2, 0.1]);
    assert_eq!(hess.n_updates, 1);
    assert!(
        (hess.h[1] - hess.h[2]).abs() < 1e-12,
        "Hessian stays symmetric"
    );
}
