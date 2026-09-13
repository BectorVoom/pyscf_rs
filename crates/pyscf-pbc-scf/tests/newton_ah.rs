//! Second-order SCF tests (19-04): AH step, Newton-vs-Roothaan energy gate.
//!
//! The analytic 2-level closed-shell model (Hcore 2×2 + on-site U, one pair)
//! is a genuine SCF problem — Fock depends on the density — minimized by two
//! different routes: Roothaan diagonalization loops (first order) and the
//! augmented-Hessian Newton driver (second order). The gate is the ENERGY
//! (19-01 scale: 1e-8 Ha, upstream `test_newton.py` asserts 8dp); nothing
//! asserts a cycle count.
//!
//! The live-cell arm (diamond/H2 integrals through the k-point mean field)
//! is `#[ignore]`d: it needs `--release` + the DF grid and is the documented
//! human-verify run. Its upstream number is committed below.
//!
//! Run scoped: `cargo test -p pyscf-pbc-scf --test newton_ah`

use pyscf_pbc_scf::newton_ah::{NewtonConfig, NewtonModel, PbscfNewtonError, ah_step, kernel_newton};

/// Analytic 2-level closed shell: Hcore + on-site U on function 0, one pair.
/// Orbital angle θ: |o> = cosθ|0> + sinθ|1>, D = 2|o><o|.
struct TwoLevel {
    h11: f64,
    h22: f64,
    h12: f64,
    u: f64,
    theta: f64,
}

impl TwoLevel {
    fn energy_of(&self, th: f64) -> f64 {
        let (c, s) = (th.cos(), th.sin());
        2.0 * (self.h11 * c * c + 2.0 * self.h12 * c * s + self.h22 * s * s)
            + 2.0 * self.u * c.powi(4)
    }
    fn grad_of(&self, th: f64) -> f64 {
        let (c, s) = (th.cos(), th.sin());
        let sin2 = 2.0 * c * s;
        let cos2 = c * c - s * s;
        2.0 * (-self.h11 * sin2 + 2.0 * self.h12 * cos2 + self.h22 * sin2)
            - 8.0 * self.u * c.powi(3) * s
    }
    fn hess_of(&self, th: f64) -> f64 {
        let (c, s) = (th.cos(), th.sin());
        let sin2 = 2.0 * c * s;
        let cos2 = c * c - s * s;
        4.0 * ((self.h22 - self.h11) * cos2 - 2.0 * self.h12 * sin2)
            - 8.0 * self.u * (c.powi(4) - 3.0 * c * c * s * s)
    }
    /// Roothaan first-order loop on the same physics (damped diagonalization).
    fn roothaan(&self, th0: f64) -> f64 {
        let mut th = th0;
        let mut e_prev = self.energy_of(th);
        for _ in 0..500 {
            // Fock in the current-MO basis: F = H + U·D00·|0><0|, rotated.
            let (c, s) = (th.cos(), th.sin());
            let d00 = 2.0 * c * c;
            // 2x2 Fock in AO basis.
            let (f11, f22, f12) = (self.h11 + self.u * d00, self.h22, self.h12);
            // Lowest eigenvector angle of [[f11,f12],[f12,f22]].
            let phi = 0.5 * (2.0 * f12).atan2(f11 - f22);
            // Ground state: pick the branch continuously connected (lowest E).
            let e_a = self.energy_of(phi);
            let e_b = self.energy_of(phi + std::f64::consts::FRAC_PI_2);
            let th_new = if e_a < e_b { phi } else { phi + std::f64::consts::FRAC_PI_2 };
            // Damped orbital update toward the diagonalizer.
            th += 0.5 * angle_diff(th_new, th);
            let e = self.energy_of(th);
            if (e - e_prev).abs() < 1e-12 {
                return e;
            }
            e_prev = e;
        }
        e_prev
    }
}

fn angle_diff(a: f64, b: f64) -> f64 {
    let mut d = (a - b) % std::f64::consts::PI;
    if d > std::f64::consts::FRAC_PI_2 {
        d -= std::f64::consts::PI;
    }
    if d < -std::f64::consts::FRAC_PI_2 {
        d += std::f64::consts::PI;
    }
    d
}

impl NewtonModel for TwoLevel {
    fn dim(&self) -> usize {
        1
    }
    fn gradient(&self) -> Vec<f64> {
        vec![self.grad_of(self.theta)]
    }
    fn hop(&self, x: &[f64]) -> Result<Vec<f64>, pyscf_core::PyscfRsError> {
        Ok(vec![self.hess_of(self.theta) * x[0]])
    }
    fn apply_step(&mut self, dx: &[f64]) {
        self.theta += dx[0];
    }
    fn energy(&self) -> f64 {
        self.energy_of(self.theta)
    }
}

fn model() -> TwoLevel {
    TwoLevel { h11: -1.0, h22: -0.2, h12: -0.3, u: 0.5, theta: 0.9 }
}

/// AH micro-cycles converge to the Newton point on a quadratic.
///
/// One AH step is damped (the homogeneous constraint), not the full Newton
/// step: [[0,g],[g,H]]'s lowest eigenvector gives dx = λmin/g-fit — CIAH
/// iterates the micro-cycles. Assert the iterated point, plus that one step
/// already reduces |g|.
#[test]
fn ah_step_converges_on_quadratic() {
    // E(x) = 2 + 3x + 4x², Newton point x* = −3/8, g(x) = 3 + 8x, H = 8.
    let hop = |x: &[f64]| -> Result<Vec<f64>, pyscf_core::PyscfRsError> { Ok(vec![8.0 * x[0]]) };
    let mut x = 0.5f64;
    let g0 = (3.0 + 8.0 * x).abs();
    let (dx0, _) = ah_step(&[3.0 + 8.0 * x], &hop, 1, 0.0, 100.0).expect("AH must solve");
    x += dx0[0];
    assert!((3.0 + 8.0 * x).abs() < g0, "one AH step must reduce |g|");
    for _ in 0..50 {
        let g = 3.0 + 8.0 * x;
        if g.abs() < 1e-13 {
            break;
        }
        let (dx, _) = ah_step(&[g], &hop, 1, 0.0, 100.0).expect("AH must solve");
        x += dx[0];
    }
    assert!((x + 3.0 / 8.0).abs() < 1e-9, "x = {x}");
}

/// Asymmetric hop is refused (caller bug, never silently symmetrized).
#[test]
fn ah_step_refuses_asymmetric_hop() {
    let g = vec![0.1, 0.2];
    let hop = |x: &[f64]| -> Result<Vec<f64>, pyscf_core::PyscfRsError> {
        // Antisymmetric: H[0][1] = +1, H[1][0] = −1.
        Ok(vec![x[1], -x[0]])
    };
    assert!(matches!(
        ah_step(&g, &hop, 2, 0.0, 100.0),
        Err(PbscfNewtonError::AsymmetricHop { .. })
    ));
}

/// The gate: Newton and Roothaan reach the same minimum energy.
#[test]
fn newton_and_first_order_agree_on_energy() {
    let m = model();
    let e_first = m.roothaan(0.9);
    let mut n = model();
    let cfg = NewtonConfig { conv_tol_grad: 1e-10, max_macro: 50, max_micro: 20, max_step: 10.0 };
    let (e_second, _cycles) = kernel_newton(&mut n, &cfg).expect("Newton must converge");
    assert!(
        (e_second - e_first).abs() < 1e-8,
        "Newton {e_second:.12} vs first-order {e_first:.12}"
    );
}

/// Newton lowers the energy monotonically on the model.
#[test]
fn newton_descends() {
    let mut n = model();
    let e0 = n.energy();
    let cfg = NewtonConfig::default();
    let (e1, _) = kernel_newton(&mut n, &cfg).expect("Newton must converge");
    assert!(e1 < e0, "no descent: {e0} -> {e1}");
}

/// Trust region caps over-long steps (scale reported, not silent).
#[test]
fn trust_region_caps_step() {
    let g = vec![10.0];
    let hop = |x: &[f64]| -> Result<Vec<f64>, pyscf_core::PyscfRsError> { Ok(vec![x[0]]) };
    let (dx, scale) = ah_step(&g, &hop, 1, 0.0, 0.5).expect("AH must solve");
    assert!(scale < 1.0);
    assert!(dx[0].abs() <= 0.5 + 1e-15);
}

/// Determinism at 1 vs 8 rayon workers inside one process.
#[test]
fn newton_deterministic_across_thread_counts() {
    let run = || {
        let mut n = model();
        let cfg = NewtonConfig { conv_tol_grad: 1e-10, max_macro: 50, max_micro: 20, max_step: 10.0 };
        kernel_newton(&mut n, &cfg).expect("Newton must converge").0
    };
    let pool1 = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let pool8 = rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap();
    let (a, b) = (pool1.install(run), pool8.install(run));
    assert_eq!(a.to_bits(), b.to_bits());
}

/// Live-cell arm (human-verify): second-order SCF on a real cell through the
/// k-point mean field. Needs `--release` + DF grids; upstream's number
/// (`test_newton.py::test_nr_rhf`, 8dp) is committed for the manual run.
#[test]
#[ignore = "live-cell human-verify: needs --release + DF grids (19-04 Task 3)"]
fn live_newton_matches_upstream_energy() {
    // Upstream: H2-like cell, RHF.newton(), e_tot = -10.137043711032916 (8dp).
    // This arm converges `kernel_newton` over the live KRHF Fock build and
    // asserts 1e-8 Ha. Outstanding: the live Fock wiring (production-cell
    // `NewtonModel` impl over `Krhf`).
    panic!("live arm not yet wired to the KRHF Fock build");
}
