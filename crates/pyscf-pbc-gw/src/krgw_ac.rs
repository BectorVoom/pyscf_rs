//! k-point restricted G0W0 by analytic continuation (`pbc/gw/krgw_ac.py`, 644 l).
//!
//! Pipeline (mirroring `kernel`): imaginary-axis self-energy → Padé (or
//! two-pole) analytic continuation → quasiparticle equation per orbital
//! (linearized or Newton). The frequency grid is PART OF THE METHOD
//! (`_get_scaled_legendre_roots`, `x0 = 0.5` — same map as
//! [`crate::sigma::imag_grid`] with `omega0 = 0.5`, cross-checked against
//! `numpy.polynomial.legendre.leggauss` in tests); `nw` is pinned on both
//! sides (default 100).
//!
//! The continuation is ported LITERALLY (subsample indices, Thiele
//! reciprocal-difference table, continued-fraction evaluation order) because
//! a Padé fit is exquisitely sensitive to its own construction — a
//! "clean-room" Thiele would be a different approximation sharing only the
//! name. The two-pole FIT needs a least-squares optimizer and is refused
//! ([`AcMode::TwoPole`] evaluates given coefficients only); the gate runs
//! Padé. Results carry [`crate::types::GwRoute::AnalyticContinuation` and
//! never mix with CD numbers (19-01 Gate C, per route).

use num_complex::Complex64;

use crate::error::PbcGwError;
use crate::types::{GwConfig, GwRoute, QpResult};

/// Analytic-continuation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcMode {
    /// Padé–Thiele (`AC_pade_thiele_diag`).
    Pade,
    /// Two-pole model evaluation (`two_pole`) — coefficients supplied
    /// (the least-squares FIT is not ported: no optimizer in-tree).
    TwoPole,
}

/// Thiele reciprocal-difference coefficients (`thiele(fn, zn)`).
///
/// `vals`/`zn` have length `nfit ≥ 1` (complex — the grid lives on the
/// imaginary axis, `omega_occ[1:] = -1j*freqs`). LITERAL port of the table
/// recurrence `g[j,i] = (g[i-1,i-1] − g[j,i-1]) / ((zn[j] − zn[i-1]) ·
/// g[j,i-1])`; returns the diagonal. An in-place reciprocal-difference form
/// divides by value DIFFERENCES and blows up on smooth data (caught during
/// development) — this table form divides by values and does not.
pub fn thiele_coeffs(vals: &[Complex64], zn: &[Complex64]) -> Result<Vec<Complex64>, PbcGwError> {
    if vals.len() != zn.len() || vals.is_empty() {
        return Err(PbcGwError::ShapeMismatch { expected: zn.len(), got: vals.len() });
    }
    let n = vals.len();
    let mut g = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    for (i, v) in vals.iter().enumerate() {
        g[i][0] = *v;
    }
    for i in 1..n {
        for j in i..n {
            let denom = (zn[j] - zn[i - 1]) * g[j][i - 1];
            if denom.norm() == 0.0 || !denom.re.is_finite() || !denom.im.is_finite() {
                return Err(PbcGwError::PadeFailure {
                    reason: format!("vanishing Thiele denominator at i={i}, j={j}"),
                });
            }
            g[j][i] = (g[i - 1][i - 1] - g[j][i - 1]) / denom;
        }
    }
    Ok((0..n).map(|i| g[i][i]).collect())
}

/// Padé–Thiele evaluation (`pade_thiele(freqs, zn, coeff)`) at real `omega`.
/// Literal port of the continued-fraction ordering (needs `nfit ≥ 2`).
pub fn pade_eval(omega: f64, zn: &[Complex64], coeff: &[Complex64]) -> Result<Complex64, PbcGwError> {
    let nfit = coeff.len();
    if nfit < 2 || zn.len() != nfit {
        return Err(PbcGwError::ShapeMismatch { expected: nfit.max(2), got: zn.len().min(coeff.len()) });
    }
    let one = Complex64::new(1.0, 0.0);
    let w = Complex64::new(omega, 0.0);
    let mut x = coeff[nfit - 1] * (w - zn[nfit - 2]);
    for i in 0..nfit - 1 {
        let idx = nfit - i - 1;
        let denom = one + x;
        if denom.norm() == 0.0 {
            return Err(PbcGwError::PadeFailure { reason: "vanishing Padé denominator".into() });
        }
        x = coeff[idx] * (w - zn[idx - 1]) / denom;
    }
    Ok(coeff[0] / (one + x))
}

/// Two-pole model evaluation (`two_pole(freqs, coeff)`).
pub fn two_pole_eval(omega: f64, coeff: &[f64; 10]) -> Complex64 {
    let cf: Vec<Complex64> = (0..5).map(|i| Complex64::new(coeff[i], coeff[i + 5])).collect();
    cf[0] + cf[1] / (omega + cf[3]) + cf[2] / (omega + cf[4])
}

/// Padé fit row (`AC_pade_thiele_diag` for one orbital).
///
/// Subsamples literally (`idx = 1,7,…,37` then every 4th from 41) and fits
/// the first `npade*2` points. Grids are COMPLEX (imaginary axis).
/// Returns `(coeff, zn_fit)`. Requires at least 42 grid points (the hardcoded
/// subsample reaches index 41+).
pub fn ac_pade_fit_row(sigma: &[Complex64], omega: &[Complex64]) -> Result<(Vec<Complex64>, Vec<Complex64>), PbcGwError> {
    if sigma.len() != omega.len() || sigma.len() < 42 {
        return Err(PbcGwError::ShapeMismatch { expected: 42, got: sigma.len().min(omega.len()) });
    }
    let mut idx: Vec<usize> = (1..40).step_by(6).collect();
    let mut k = idx[idx.len() - 1] + 4;
    while k < sigma.len() {
        idx.push(k);
        k += 4;
    }
    let sub_s: Vec<Complex64> = idx.iter().map(|&i| sigma[i]).collect();
    let sub_w: Vec<Complex64> = idx.iter().map(|&i| omega[i]).collect();
    let nw2 = sub_s.len() / 2 * 2;
    let coeff = thiele_coeffs(&sub_s[..nw2], &sub_w[..nw2])?;
    Ok((coeff, sub_w[..nw2].to_vec()))
}

/// Linearized G0W0 QP energy: `e = ep + Z·(σR + vk − vmf)` with
/// `Z = 1/(1 − dσ/de)` by finite difference (`de = 1e-6`).
pub fn qp_linearized(
    ep: f64,
    sigma_r: impl Fn(f64) -> f64,
    vk: f64,
    vmf: f64,
) -> f64 {
    let de = 1e-6;
    let s0 = sigma_r(ep);
    let dsigma = sigma_r(ep + de) - s0;
    let z = 1.0 / (1.0 - dsigma / de);
    ep + z * (s0 + vk - vmf)
}

/// Full QP equation by Newton (`tol = 1e-6`, `maxiter = 100`):
/// `ω − ep − (σR(ω) + vk − vmf) = 0` from `ep`.
pub fn qp_newton(
    ep: f64,
    sigma_r: impl Fn(f64) -> f64,
    vk: f64,
    vmf: f64,
    tol: f64,
    maxiter: usize,
) -> Result<f64, PbcGwError> {
    let mut e = ep;
    for _ in 0..maxiter {
        let f = e - ep - (sigma_r(e) + vk - vmf);
        // Numerical derivative (upstream notes the analytic one as TODO).
        let h = 1e-6;
        let df = (sigma_r(e + h) - sigma_r(e - h)) / (2.0 * h);
        let denom = 1.0 - df;
        if denom == 0.0 || !denom.is_finite() {
            return Err(PbcGwError::QpNotConverged { cycles: maxiter });
        }
        let step = f / denom;
        e -= step;
        if step.abs() < tol {
            return Ok(e);
        }
    }
    Err(PbcGwError::QpNotConverged { cycles: maxiter })
}

/// G0W0-AC driver over an orbital window.
///
/// `sigma_imag[k][oi][n]` is the imaginary-axis correlation self-energy
/// (complex) on `omegas[oi][n]`; both rows are relative to the `orbs` window
/// (`sigmaI[k]`, `omega` rows as `get_sigma_diag` returns them — the trimmed
/// evaluation grid, NOT the full `nw`-point input grid; `cfg.nomega` records
/// the pinned input size and is asserted by the caller). `mf_energy[k][p]`
/// the mean-field energies (absolute orbital indices); `vk_diag`/`vmf_diag`
/// the MO-basis diagonals; `ef` the Fermi level. Solves the QP equation per
/// orbital (Newton unless `linearized`) with the Padé continuation.
#[allow(clippy::too_many_arguments)]
pub fn kernel_krgw_ac(
    sigma_imag: &[Vec<Vec<Complex64>>],
    omegas: &[Vec<Complex64>],
    mf_energy: &[Vec<f64>],
    vk_diag: &[Vec<f64>],
    vmf_diag: &[Vec<f64>],
    ef: f64,
    orbs: std::ops::Range<usize>,
    cfg: &GwConfig,
    mode: AcMode,
    linearized: bool,
) -> Result<QpResult, PbcGwError> {
    if mode != AcMode::Pade {
        return Err(PbcGwError::NotYetImplemented { module: "gw two-pole FIT (needs least-squares optimizer)" });
    }
    let nk = sigma_imag.len();
    if mf_energy.len() != nk || vk_diag.len() != nk || vmf_diag.len() != nk {
        return Err(PbcGwError::ShapeMismatch { expected: nk, got: mf_energy.len() });
    }
    if omegas.len() != orbs.len() {
        return Err(PbcGwError::ShapeMismatch { expected: orbs.len(), got: omegas.len() });
    }
    let mut qp = Vec::with_capacity(nk * orbs.len());
    // NOTE: single-k driver shape (19-10): the fixture gates k = 0..nklist
    // per call; multi-k batching reuses this loop unchanged.
    for k in 0..nk {
        for (oi, p) in orbs.clone().enumerate() {
            // sigma/omega rows are relative to the window.
            if oi >= sigma_imag[k].len() || oi >= omegas.len() || p >= mf_energy[k].len() {
                return Err(PbcGwError::ShapeMismatch { expected: p + 1, got: mf_energy[k].len() });
            }
            if sigma_imag[k][oi].len() != omegas[oi].len() {
                return Err(PbcGwError::ShapeMismatch {
                    expected: omegas[oi].len(),
                    got: sigma_imag[k][oi].len(),
                });
            }
            let (coeff, zn) = ac_pade_fit_row(&sigma_imag[k][oi], &omegas[oi])?;
            let sigma_r = |w: f64| {
                pade_eval(w - ef, &zn, &coeff).map(|z| z.re).unwrap_or(f64::NAN)
            };
            let ep = mf_energy[k][p];
            let e = if linearized {
                qp_linearized(ep, &sigma_r, vk_diag[k][p], vmf_diag[k][p])
            } else {
                qp_newton(ep, &sigma_r, vk_diag[k][p], vmf_diag[k][p], cfg.conv_tol, cfg.max_cycle)?
            };
            if !e.is_finite() {
                return Err(PbcGwError::QpNotConverged { cycles: cfg.max_cycle });
            }
            qp.push(e);
        }
    }
    Ok(QpResult { qp_energy: qp, route: GwRoute::AnalyticContinuation, converged: true })
}
