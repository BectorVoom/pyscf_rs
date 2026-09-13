//! Gamma-point slow G0W0 reference (`pbc/gw/gw_slow.py`, 24 l).
//!
//! Upstream this module is LITERALLY an alias of the molecular reference
//! (`IMDS = gw_slow.IMDS; kernel = gw_slow.kernel; GW = gw_slow.GW` —
//! `:21-24`): the single-k-point PBC reference IS the molecular slow GW.
//! This port mirrors that aliasing honestly: the gamma driver is the shared
//! Lehmann core ([`crate::kgw_slow`]) at `nk = 1`, not a second
//! implementation. **Do not optimise it.**

use crate::error::PbcGwError;
use crate::kgw_slow::{kernel_slow_orbital, LehmannPole, SLOW_ETA};
use crate::types::{GwConfig, GwRoute, QpResult};

/// Gamma-point slow G0W0 reference driver (the `gw_slow` alias shape).
///
/// `poles[oi]` are the Lehmann poles per window orbital at Γ. Identical
/// inputs to a one-k-point [`crate::kgw_slow::kernel_kgw_slow`] give
/// bit-identical outputs (asserted, not assumed).
pub fn kernel_gw_slow(
    mf_energy: &[f64],
    poles: &[Vec<LehmannPole>],
    vk_diag: &[f64],
    vmf_diag: &[f64],
    orbs: std::ops::Range<usize>,
    cfg: &GwConfig,
) -> Result<QpResult, PbcGwError> {
    if poles.len() != orbs.len() {
        return Err(PbcGwError::ShapeMismatch {
            expected: orbs.len(),
            got: poles.len(),
        });
    }
    let mut qp = Vec::with_capacity(orbs.len());
    for (oi, p) in orbs.enumerate() {
        if p >= mf_energy.len() || p >= vk_diag.len() || p >= vmf_diag.len() {
            return Err(PbcGwError::ShapeMismatch {
                expected: p + 1,
                got: mf_energy.len(),
            });
        }
        qp.push(kernel_slow_orbital(
            &poles[oi],
            mf_energy[p],
            vk_diag[p] - vmf_diag[p],
            SLOW_ETA,
            cfg.conv_tol,
            cfg.max_cycle,
        )?);
    }
    Ok(QpResult {
        qp_energy: qp,
        route: GwRoute::Slow,
        converged: true,
    })
}
