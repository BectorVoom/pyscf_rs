//! `eph_fd` tests (19-18): central-difference exactness, convergence rate,
//! refusal discipline, and determinism.
//!
//! `eph_fd` is a finite difference — its floor is the cancellation floor
//! recorded in `src/eph_fd.rs` (~1e-8 absolute at the default step), never
//! X2C's 1e-8-on-the-energy gate. No assertion here applies X2C's tolerance.
//!
//! Run scoped: `cargo test -p pyscf-pbc-eph --test eph_fd`

use pyscf_pbc_eph::eph_fd::{EPH_FD_DEFAULT_DISP, central_difference, eph_fd_coupling};

/// Central differences are exact on linear functions (to roundoff).
#[test]
fn central_difference_exact_on_linear() {
    // q(R) = 3R + 1 sampled at ±d/2: derivative is exactly 3.
    let d = EPH_FD_DEFAULT_DISP;
    let qp = vec![3.0 * (d / 2.0) + 1.0, -2.0 * (d / 2.0)];
    let qm = vec![3.0 * (-d / 2.0) + 1.0, -2.0 * (-d / 2.0)];
    let dq = central_difference(&qp, &qm, d).expect("must run");
    assert!((dq[0] - 3.0).abs() < 1e-12);
    assert!((dq[1] + 2.0).abs() < 1e-12);
}

/// Quadratic convergence under step halving (truncation is O(disp²)).
#[test]
fn central_difference_converges_quadratically() {
    // q(R) = R³ + R, q'(0) = 1; central difference at ±d/2 errs by d²/4·q'''/6.
    let f = |r: f64| r * r * r + r;
    let err = |d: f64| (central_difference(&[f(d / 2.0)], &[f(-d / 2.0)], d).unwrap()[0] - 1.0).abs();
    let (e1, e2) = (err(1e-2), err(5e-3));
    let ratio = e1 / e2;
    assert!((ratio - 4.0).abs() < 0.05, "expected ~4x, got {ratio}");
}

/// Refusals: length mismatch, zero/negative/nonfinite step.
#[test]
fn central_difference_refuses_bad_input() {
    assert!(central_difference(&[1.0], &[1.0, 2.0], 1e-4).is_err());
    assert!(central_difference(&[1.0], &[2.0], 0.0).is_err());
    assert!(central_difference(&[1.0], &[2.0], -1e-4).is_err());
    assert!(central_difference(&[1.0], &[2.0], f64::NAN).is_err());
    assert!(central_difference(&[1.0], &[2.0], f64::INFINITY).is_err());
}

/// The coupling entry point agrees with the difference on one mode.
#[test]
fn eph_fd_coupling_matches_central_difference() {
    let ep = vec![-0.5, 0.3, 0.8];
    let em = vec![-0.4, 0.25, 0.7];
    let a = eph_fd_coupling(&ep, &em, EPH_FD_DEFAULT_DISP).unwrap();
    let b = central_difference(&ep, &em, EPH_FD_DEFAULT_DISP).unwrap();
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.to_bits(), y.to_bits());
    }
}

/// Determinism at 1 vs 8 rayon workers inside one process.
#[test]
fn eph_fd_deterministic_across_thread_counts() {
    let run = || {
        let ep: Vec<f64> = (0..64).map(|i| (i as f64).sin()).collect();
        let em: Vec<f64> = (0..64).map(|i| (i as f64).cos()).collect();
        eph_fd_coupling(&ep, &em, EPH_FD_DEFAULT_DISP).expect("must run")
    };
    let pool1 = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let pool8 = rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap();
    let (a, b) = (pool1.install(run), pool8.install(run));
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.to_bits(), y.to_bits());
    }
}
