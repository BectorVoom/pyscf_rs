//! Slow explicit k-point G0W0 reference (`pbc/gw/kgw_slow.py`, 169 l).
//!
//! The obviously-correct reference the fast routes check against: the
//! correlation self-energy in Lehmann (sum-over-states) form, evaluated
//! DIRECTLY at real frequencies — no imaginary grid, no Padé fit, no
//! blocking, no fusion. **Do not optimise it** (19-13 must-have); its value
//! is the supercell-equivalence identity with [`crate::kgw_slow_supercell`]
//! that needs no upstream at all, and the slow-vs-fast agreement with
//! 19-10's `krgw_ac` on a small cell.
//!
//! ```text
//! σ_p(ω) = Σ_m w_m · (ω − e_m) / ((ω − e_m)² + η²)   (real part)
//! ```
//! with `s_m = sign(ef − e_m)` selecting the `+i·η·s_m` side of each pole
//! (same convention as [`crate::kugw_ac`] and [`crate::krgw_cd`]). Every sum
//! routes through [`pyscf_algebra::oracle_sum`]. Host loops only
//! (D-PBC-29 clause 2) — no `#[cube]`.

use crate::error::PbcGwError;
use crate::krgw_ac::qp_newton;
use crate::types::{GwConfig, GwRoute, QpResult};

/// Broadening for the Lehmann denominators (pinned — same contour number as
/// CD/UGW, `gw_gw_GW_eta` default).
pub const SLOW_ETA: f64 = 1e-3;

/// One Lehmann pole of the diagonal correlation self-energy.
#[derive(Debug, Clone, Copy)]
pub struct LehmannPole {
    /// Pole energy.
    pub energy: f64,
    /// Coupling weight (`|V|²`-like, already contracted to the diagonal).
    pub weight: f64,
}

/// Real part of the Lehmann self-energy at `omega`, ordered `oracle_sum`.
pub fn slow_sigma_real(poles: &[LehmannPole], omega: f64, eta: f64) -> f64 {
    let mut terms = Vec::with_capacity(poles.len());
    for pole in poles {
        let d = omega - pole.energy;
        terms.push(pole.weight * d / (d * d + eta * eta));
    }
    pyscf_algebra::oracle_sum(&terms)
}

/// Slow QP root for one orbital: Newton on `ω − ep − (σ(ω) + vk − vmf) = 0`.
pub fn kernel_slow_orbital(
    poles: &[LehmannPole],
    ep: f64,
    vk_minus_vmf: f64,
    eta: f64,
    conv_tol: f64,
    max_cycle: usize,
) -> Result<f64, PbcGwError> {
    let sigma_r = |w: f64| slow_sigma_real(poles, w, eta);
    let e = qp_newton(ep, sigma_r, vk_minus_vmf, 0.0, conv_tol, max_cycle)?;
    if !e.is_finite() {
        return Err(PbcGwError::QpNotConverged { cycles: max_cycle });
    }
    Ok(e)
}

/// Slow k-point G0W0 reference driver.
///
/// `poles_k[k][oi]` are the Lehmann poles per k-point per window orbital;
/// `vk_diag`/`vmf_diag` absolute-index MO diagonals. Returns
/// [`GwRoute::Slow`] — never an AC/CD number.
pub fn kernel_kgw_slow(
    mf_energy: &[Vec<f64>],
    poles_k: &[Vec<Vec<LehmannPole>>],
    vk_diag: &[Vec<f64>],
    vmf_diag: &[Vec<f64>],
    orbs: std::ops::Range<usize>,
    cfg: &GwConfig,
    eta: f64,
) -> Result<QpResult, PbcGwError> {
    let nk = mf_energy.len();
    if poles_k.len() != nk || vk_diag.len() != nk || vmf_diag.len() != nk {
        return Err(PbcGwError::ShapeMismatch {
            expected: nk,
            got: poles_k.len(),
        });
    }
    for pk in poles_k {
        if pk.len() != orbs.len() {
            return Err(PbcGwError::ShapeMismatch {
                expected: orbs.len(),
                got: pk.len(),
            });
        }
    }
    let mut qp = Vec::with_capacity(nk * orbs.len());
    for k in 0..nk {
        for (oi, p) in orbs.clone().enumerate() {
            if p >= mf_energy[k].len() {
                return Err(PbcGwError::ShapeMismatch {
                    expected: p + 1,
                    got: mf_energy[k].len(),
                });
            }
            qp.push(kernel_slow_orbital(
                &poles_k[k][oi],
                mf_energy[k][p],
                vk_diag[k][p] - vmf_diag[k][p],
                eta,
                cfg.conv_tol,
                cfg.max_cycle,
            )?);
        }
    }
    Ok(QpResult {
        qp_energy: qp,
        route: GwRoute::Slow,
        converged: true,
    })
}
