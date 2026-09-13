//! k-point unrestricted G0W0 by analytic continuation (`pbc/gw/kugw_ac.py`, 784 l).
//!
//! The largest single GW file — but NOT a new method: it is the unrestricted
//! generalisation of 19-10. The structure is 19-10's; what changes is that
//! `W` is **spin-summed** while the self-energies are per spin, so the two
//! channels couple through the screening even though their Green's functions
//! are separate (19-12 must-have).
//!
//! That coupling is STRUCTURAL here, not conventional: the driver takes ONE
//! shared screened interaction `w_imag` (the spin-summed `W`, built upstream
//! over both spins' particle-hole pairs) and uses it for BOTH channels. A
//! caller cannot supply per-spin screenings — the signature makes two `W`s
//! unrepresentable. Per-spin inputs are only the Green's-function halves
//! (mean-field spectra, `vk`/`vmf` diagonals).
//!
//! * [`sigma_row_from_screened`] builds one spin's imaginary-axis diagonal
//!   self-energy from the shared `W`: `σ^s_p(iω_n) = −(W_n/π)·Σ_m
//!   1/(iω_n − (e^s_m − ef) + i·η·s^s_m)` with `s^s_m = sign(ef − e^s_m)`.
//! * [`kernel_kugw_ac`] continues each spin's rows through 19-10's Padé
//!   machinery ([`crate::krgw_ac::ac_pade_fit_row`]) and solves each spin's
//!   QP equation (`linearized` or Newton, same tolerances).
//!
//! Every reduction routes through [`pyscf_algebra::oracle_sum`] (planar
//! parts). Host loops only (D-PBC-29 clause 2) — no `#[cube]`.

use num_complex::Complex64;

use crate::error::PbcGwError;
use crate::krgw_ac::{ac_pade_fit_row, pade_eval, qp_linearized, qp_newton, AcMode};
use crate::types::{GwConfig, GwRoute, QpResult};

/// Broadening for the Green's-function denominators (upstream
/// `gw_gw_GW_eta` default, pinned — same contour number as CD).
pub const GW_ETA: f64 = 1e-3;

/// One spin's imaginary-axis self-energy row from the shared screening and
/// this spin's spectrum, on the EXPLICIT imaginary grid `iw_nodes[n]`.
///
/// `σ^s(iω_n) = −(W_n/π)·Σ_m 1/(iω_n − (e^s_m − ef) + i·η·s^s_m)`,
/// `s^s_m = sign(ef − e^s_m)`. Ordered `oracle_sum` over planar parts.
pub fn sigma_row_on_grid(
    w_imag: &[Complex64],
    iw_nodes: &[f64],
    mf_spin: &[f64],
    ef: f64,
    eta: f64,
) -> Result<Vec<Complex64>, PbcGwError> {
    if w_imag.len() != iw_nodes.len() || w_imag.is_empty() {
        return Err(PbcGwError::ShapeMismatch {
            expected: iw_nodes.len(),
            got: w_imag.len(),
        });
    }
    if mf_spin.is_empty() {
        return Err(PbcGwError::ShapeMismatch {
            expected: 1,
            got: 0,
        });
    }
    let mut row = Vec::with_capacity(w_imag.len());
    for (n, w) in w_imag.iter().enumerate() {
        let z = Complex64::new(0.0, iw_nodes[n]);
        let mut re_t: Vec<f64> = Vec::with_capacity(mf_spin.len());
        let mut im_t: Vec<f64> = Vec::with_capacity(mf_spin.len());
        for e in mf_spin {
            let s = if ef >= *e { 1.0 } else { -1.0 };
            let denom = z - Complex64::new(e - ef, 0.0) + Complex64::new(0.0, eta * s);
            if denom.norm() == 0.0 {
                return Err(PbcGwError::PadeFailure {
                    reason: format!("vanishing UGW denominator at n={n}, e={e}"),
                });
            }
            let g = Complex64::new(1.0, 0.0) / denom;
            re_t.push(g.re);
            im_t.push(g.im);
        }
        let gsum = Complex64::new(
            pyscf_algebra::oracle_sum(&re_t),
            pyscf_algebra::oracle_sum(&im_t),
        );
        row.push(-*w * gsum / Complex64::new(std::f64::consts::PI, 0.0));
    }
    Ok(row)
}

/// Unrestricted G0W0-AC driver (per-spin QP energies, shared screening).
///
/// `w_imag_k[k][n]` is the spin-summed screened interaction (ONE per k,
/// used for both spins); `iw_nodes[n]` its imaginary grid;
/// `fit_grid[oi]` the window-relative complex evaluation grid for the Padé
/// fit (19-10 convention, `len ≥ 42`); per-spin `mf/vk/vmf` the
/// Green's-function halves. Returns `(alpha, beta)`, both
/// [`GwRoute::AnalyticContinuation`].
#[allow(clippy::too_many_arguments)]
pub fn kernel_kugw_ac(
    w_imag_k: &[Vec<Complex64>],
    iw_nodes: &[f64],
    fit_grid: &[Vec<Complex64>],
    mf_a: &[Vec<f64>],
    mf_b: &[Vec<f64>],
    vk_a: &[Vec<f64>],
    vk_b: &[Vec<f64>],
    vmf_a: &[Vec<f64>],
    vmf_b: &[Vec<f64>],
    ef: f64,
    orbs: std::ops::Range<usize>,
    cfg: &GwConfig,
    mode: AcMode,
    linearized: bool,
) -> Result<(QpResult, QpResult), PbcGwError> {
    if mode != AcMode::Pade {
        return Err(PbcGwError::NotYetImplemented {
            module: "gw two-pole FIT (needs least-squares optimizer)",
        });
    }
    let nk = mf_a.len();
    if mf_b.len() != nk
        || vk_a.len() != nk
        || vk_b.len() != nk
        || vmf_a.len() != nk
        || vmf_b.len() != nk
        || w_imag_k.len() != nk
    {
        return Err(PbcGwError::ShapeMismatch {
            expected: nk,
            got: mf_b.len(),
        });
    }
    if fit_grid.len() != orbs.len() {
        return Err(PbcGwError::ShapeMismatch {
            expected: orbs.len(),
            got: fit_grid.len(),
        });
    }
    let mut qa = Vec::with_capacity(nk * orbs.len());
    let mut qb = Vec::with_capacity(nk * orbs.len());
    for k in 0..nk {
        // BOTH channels from the SAME screening — the spin-summed coupling.
        let row_a = sigma_row_on_grid(&w_imag_k[k], iw_nodes, &mf_a[k], ef, GW_ETA)?;
        let row_b = sigma_row_on_grid(&w_imag_k[k], iw_nodes, &mf_b[k], ef, GW_ETA)?;
        for (oi, p) in orbs.clone().enumerate() {
            if p >= mf_a[k].len() || p >= mf_b[k].len() {
                return Err(PbcGwError::ShapeMismatch {
                    expected: p + 1,
                    got: mf_a[k].len(),
                });
            }
            if row_a.len() != fit_grid[oi].len() {
                return Err(PbcGwError::ShapeMismatch {
                    expected: fit_grid[oi].len(),
                    got: row_a.len(),
                });
            }
            for (spin, row, mf, vk, vmf, out) in [
                (0, &row_a, &mf_a[k], &vk_a[k], &vmf_a[k], &mut qa),
                (1, &row_b, &mf_b[k], &vk_b[k], &vmf_b[k], &mut qb),
            ] {
                let _ = spin;
                let (coeff, zn) = ac_pade_fit_row(row, &fit_grid[oi])?;
                let sigma_r = |w: f64| {
                    pade_eval(w - ef, &zn, &coeff)
                        .map(|z| z.re)
                        .unwrap_or(f64::NAN)
                };
                let ep = mf[p];
                let e = if linearized {
                    qp_linearized(ep, sigma_r, vk[p], vmf[p])
                } else {
                    qp_newton(ep, sigma_r, vk[p], vmf[p], cfg.conv_tol, cfg.max_cycle)?
                };
                if !e.is_finite() {
                    return Err(PbcGwError::QpNotConverged {
                        cycles: cfg.max_cycle,
                    });
                }
                out.push(e);
            }
        }
    }
    Ok((
        QpResult {
            qp_energy: qa,
            route: GwRoute::AnalyticContinuation,
            converged: true,
        },
        QpResult {
            qp_energy: qb,
            route: GwRoute::AnalyticContinuation,
            converged: true,
        },
    ))
}
