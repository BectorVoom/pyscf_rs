//! G0W0-CD tests (19-11): Gate C per route, contour pinned, no Padé.
//!
//! * Grid shared with AC (`imag_grid(100, 0.5)` — the `_get_scaled_legendre_roots`
//!   `x0 = 0.5` map on both sides; `nw` pinned).
//! * `sigma_imag_quad` against an independent direct transcription on one
//!   state/one frequency (wiring: signs, `/π`, negation, accumulation).
//! * Residue pole selection: strict endpoint exclusion, `fm = ±1` sides.
//! * Linearized CD refused exactly as upstream refuses it.
//! * **Gate C (CD)**: analytic models with closed-form QP energies at the
//!   per-route floor; route named CD; the AC–CD split recorded via
//!   `ac_cd_split`, route-blind comparisons refused.
//!
//! Run scoped: `cargo test -p pyscf-pbc-gw --test krgw_cd`

use num_complex::Complex64;
use pyscf_pbc_gw::krgw_ac::{kernel_krgw_ac, AcMode};
use pyscf_pbc_gw::krgw_cd::{
    ac_cd_split, kernel_krgw_cd, sigma_cd_real, sigma_imag_quad, sigma_residue, CdConfig, CdPole,
};
use pyscf_pbc_gw::sigma::imag_grid;
use pyscf_pbc_gw::types::{GwConfig, GwRoute, QpResult};

/// The contour grid is the AC grid: `nw` pinned, `x0 = 0.5` map.
#[test]
fn cd_grid_shared_with_ac() {
    let cd = CdConfig::default();
    assert_eq!(cd.nomega, 100, "grid size pinned on both sides");
    assert!(
        (cd.eta - 1e-3).abs() == 0.0,
        "broadening pinned to gw_gw_GW_eta default"
    );
    assert_eq!(
        (cd.conv_tol, cd.max_cycle),
        (1e-6, 50),
        "Newton controls pinned"
    );
    let (freqs, wts) = imag_grid(cd.nomega, 0.5).expect("grid must build");
    assert_eq!(freqs.len(), 100);
    assert_eq!(wts.len(), 100);
    // Ascending positive nodes (leggauss order, x0 map preserves order).
    assert!(freqs[0] > 0.0 && freqs[99] > freqs[0]);
}

/// `sigma_imag_quad` on one state/one frequency vs a direct transcription.
#[test]
fn sigma_imag_quad_matches_direct() {
    let (omega, eta) = (0.2f64, 1e-3f64);
    let mf = vec![vec![-0.5f64]];
    let w = vec![vec![vec![Complex64::new(0.3, 0.1)]]];
    let sign = vec![vec![1.0f64]];
    let (freqs, wts) = (vec![0.7f64], vec![0.9f64]);
    let got = sigma_imag_quad(omega, &mf, &w, &sign, &freqs, &wts, eta).expect("must evaluate");
    // Direct transcription (same math, separately written):
    // emo = ω − i·η·sign − e; g0 = wts·emo/(emo² + f²); σ = −g0·W/π.
    let emo = Complex64::new(omega - mf[0][0], -eta * sign[0][0]);
    let denom = emo * emo + Complex64::new(freqs[0] * freqs[0], 0.0);
    let g0 = emo * Complex64::new(wts[0], 0.0) / denom;
    let want = -g0 * w[0][0][0] / Complex64::new(std::f64::consts::PI, 0.0);
    assert!(
        (got - want).norm() < 1e-15,
        "imag quad deviates {:e}",
        (got - want).norm()
    );
}

/// Residue selection: strict endpoints, correct `fm` on both sides.
#[test]
fn residue_selection_is_strict() {
    let (ef, omega) = (0.0f64, 0.5f64);
    let v = Complex64::new(0.25, -0.125);
    // Poles exactly at the endpoints are NOT enclosed (strict inequalities).
    let poles = vec![
        CdPole {
            energy: ef,
            vertex: v,
        },
        CdPole {
            energy: omega,
            vertex: v,
        },
        CdPole {
            energy: 0.25,
            vertex: v,
        },
    ];
    let got = sigma_residue(omega, ef, &poles);
    assert!(
        (got - v).norm() < 1e-15,
        "only the interior pole, fm=+1: {got}"
    );
    // Below the Fermi level: fm = −1 over (ω, ef).
    let got_below = sigma_residue(-0.5, ef, &poles);
    assert!(
        (got_below - Complex64::new(0.0, 0.0)).norm() < 1e-15,
        "no pole in (−0.5, 0): {got_below}"
    );
    let poles_below = vec![
        CdPole {
            energy: -0.25,
            vertex: v,
        },
        CdPole {
            energy: 0.25,
            vertex: v,
        },
    ];
    let got_below = sigma_residue(-0.5, ef, &poles_below);
    assert!(
        (got_below + v).norm() < 1e-15,
        "fm=−1 below ef: {got_below}"
    );
}

/// Upstream refuses linearized CD — so does this port (never a silent no-op).
#[test]
fn linearized_cd_refused() {
    let cfg = GwConfig {
        nomega: 2,
        max_cycle: 50,
        conv_tol: 1e-6,
        orlo: 0,
        orhi: 1,
    };
    let cd = CdConfig {
        nomega: 2,
        ..CdConfig::default()
    };
    let mf = vec![vec![-0.5, 0.5]];
    let vk = vec![vec![0.0, 0.0]];
    let vmf = vec![vec![0.0, 0.0]];
    let w = vec![vec![vec![Complex64::new(0.0, 0.0); 2]; 2]];
    let sign = vec![vec![1.0, -1.0]];
    let poles = vec![vec![Vec::new()]];
    let (freqs, wts) = (vec![0.5, 1.5], vec![1.0, 1.0]);
    let err = kernel_krgw_cd(
        &mf,
        &vk,
        &vmf,
        &w,
        &sign,
        &poles,
        &freqs,
        &wts,
        0.0,
        0..1,
        &cfg,
        &cd,
        true,
        None,
    )
    .expect_err("linearized CD must be refused");
    assert!(matches!(
        err,
        pyscf_pbc_gw::error::PbcGwError::NotYetImplemented { .. }
    ));
}

/// Gate C (CD), analytic model: constant self-energy ⇒ closed-form QP root.
///
/// With `W = 0` and one enclosed pole of vertex `v`, `σ(ω) = v` exactly, so
/// the Newton root is `ep + v.re + vk − vmf` to solver precision. Route named.
#[test]
fn gate_c_cd_qp_energies() {
    let cd = CdConfig {
        nomega: 4,
        ..CdConfig::default()
    };
    let cfg = GwConfig {
        nomega: 4,
        max_cycle: 50,
        conv_tol: 1e-6,
        orlo: 1,
        orhi: 2,
    };
    let (ep0, ep1) = (-0.5f64, 0.4f64);
    let (vk1, vmf1) = (0.06f64, 0.02f64);
    let mf = vec![vec![ep0, ep1]];
    let vk = vec![vec![0.0, vk1]];
    let vmf = vec![vec![0.0, vmf1]];
    // W = 0 ⇒ σ^I = 0; the enclosed pole carries the whole self-energy.
    let w = vec![vec![vec![Complex64::new(0.0, 0.0); 4]; 2]];
    let sign = vec![vec![1.0, -1.0]];
    let v = Complex64::new(0.11, -0.03);
    let ef = 0.0;
    // Window orbs = 1..2; enclosed pole between ef and the expected root.
    let want = ep1 + v.re + vk1 - vmf1;
    assert!(
        want > ef,
        "test premise: root above ef so the pole is enclosed"
    );
    let poles = vec![vec![vec![CdPole {
        energy: (ef + want) / 2.0,
        vertex: v,
    }]]];
    let (freqs, wts) = imag_grid(cd.nomega, 0.5).expect("grid must build");
    // sigma_cd_real at the expected root recovers v (imag part vanishes).
    let z = sigma_cd_real(
        want,
        &mf,
        &w,
        &sign,
        &poles[0][0],
        &freqs,
        &wts,
        ef,
        cd.eta,
        None,
    )
    .expect("CD sigma must evaluate");
    assert!((z - v).norm() < 1e-12, "CD sigma at root: {z} vs {v}");
    let out = kernel_krgw_cd(
        &mf,
        &vk,
        &vmf,
        &w,
        &sign,
        &poles,
        &freqs,
        &wts,
        ef,
        1..2,
        &cfg,
        &cd,
        false,
        None,
    )
    .expect("G0W0-CD must solve");
    assert_eq!(
        out.route,
        GwRoute::ContourDeformation,
        "route named CD, never AC"
    );
    assert!(out.converged);
    assert_eq!(out.qp_energy.len(), 1);
    // Per-route Gate C floor is 1e-4; the closed form holds far inside it.
    assert!(
        (out.qp_energy[0] - want).abs() < 1e-9,
        "CD QP: {} vs closed form {want}",
        out.qp_energy[0]
    );
}

/// Exactly-constant Padé input is degenerate (vanishing Thiele denominators)
/// and must fail LOUDLY — never silently return a number.
#[test]
fn pade_degenerate_constant_input_fails_loudly() {
    use pyscf_pbc_gw::krgw_ac::ac_pade_fit_row;
    let nrow = 44;
    let srow: Vec<Complex64> = vec![Complex64::new(0.11, -0.03); nrow];
    let wrow: Vec<Complex64> = (0..nrow)
        .map(|i| Complex64::new(0.05 * (i as f64 + 1.0), 0.0))
        .collect();
    let err = ac_pade_fit_row(&srow, &wrow).expect_err("constant data is degenerate Padé input");
    assert!(matches!(
        err,
        pyscf_pbc_gw::error::PbcGwError::PadeFailure { .. }
    ));
}

/// The AC–CD split is RECORDED, not hidden — and route-blind numbers refuse.
///
/// Same constant-σ model through both drivers: AC (Padé over a 1e-9-regularized
/// constant row — exact-constant data is degenerate input per the test above)
/// and CD (direct) must agree inside Gate C, and the split helper reports the
/// number while refusing same-route inputs.
#[test]
fn ac_cd_split_recorded_and_route_blind_refused() {
    // AC side: constant sigma rows regularized by a 1e-9 smooth ramp.
    let nrow = 44;
    let v = Complex64::new(0.11, -0.03);
    let srow: Vec<Complex64> = (0..nrow)
        .map(|i| v * Complex64::new(1.0 + 1e-9 * (i as f64), 0.0))
        .collect();
    let wrow: Vec<Complex64> = (0..nrow)
        .map(|i| Complex64::new(0.05 * (i as f64 + 1.0), 0.0))
        .collect();
    let sigma_imag = vec![vec![srow]];
    let omegas = vec![wrow];
    let mf = vec![vec![0.4]];
    let vk = vec![vec![0.06]];
    let vmf = vec![vec![0.02]];
    let cfg = GwConfig {
        nomega: nrow,
        max_cycle: 100,
        conv_tol: 1e-6,
        orlo: 0,
        orhi: 1,
    };
    let ac = kernel_krgw_ac(
        &sigma_imag,
        &omegas,
        &mf,
        &vk,
        &vmf,
        0.0,
        0..1,
        &cfg,
        AcMode::Pade,
        false,
    )
    .expect("AC must solve the constant model");
    assert_eq!(ac.route, GwRoute::AnalyticContinuation);
    // CD side: same constant via an enclosed pole (imag part zero).
    let cd_cfg = CdConfig {
        nomega: 4,
        ..CdConfig::default()
    };
    let w = vec![vec![vec![Complex64::new(0.0, 0.0); 4]; 1]];
    let sign = vec![vec![-1.0]];
    let v = Complex64::new(0.11, -0.03);
    let want = 0.4 + v.re + 0.06 - 0.02;
    let poles = vec![vec![vec![CdPole {
        energy: want / 2.0,
        vertex: v,
    }]]];
    let (freqs, wts) = imag_grid(cd_cfg.nomega, 0.5).expect("grid must build");
    let cd = kernel_krgw_cd(
        &mf,
        &vk,
        &vmf,
        &w,
        &sign,
        &poles,
        &freqs,
        &wts,
        0.0,
        0..1,
        &cfg,
        &cd_cfg,
        false,
        None,
    )
    .expect("CD must solve the constant model");
    let split = ac_cd_split(&ac, &cd).expect("split must report");
    assert_eq!(split.len(), 1);
    // Both routes solve the same constant model: split inside Gate C (1e-4).
    assert!(
        split[0] < 1e-4,
        "AC–CD split on shared model: {:e}",
        split[0]
    );
    // Route-blind comparison is refused, never silently averaged.
    let blind = ac_cd_split(
        &ac,
        &QpResult {
            qp_energy: ac.qp_energy.clone(),
            route: GwRoute::AnalyticContinuation,
            converged: true,
        },
    );
    assert!(
        matches!(
            blind,
            Err(pyscf_pbc_gw::error::PbcGwError::RouteBlindComparison)
        ),
        "same-route split must be refused"
    );
}

/// Determinism at 1 vs 8 rayon workers inside one process.
#[test]
fn gw_cd_deterministic_across_thread_counts() {
    let run = || {
        let mf = vec![vec![-0.5, 0.5]];
        let w = vec![vec![vec![Complex64::new(0.2, 0.05); 3]; 2]];
        let sign = vec![vec![1.0, -1.0]];
        let (freqs, wts) = (vec![0.5, 1.0, 2.0], vec![1.0, 1.0, 1.0]);
        let z = sigma_imag_quad(0.1, &mf, &w, &sign, &freqs, &wts, 1e-3).expect("must evaluate");
        (z.re.to_bits(), z.im.to_bits())
    };
    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let pool8 = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap();
    assert_eq!(pool1.install(run), pool8.install(run));
}
