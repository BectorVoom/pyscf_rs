//! Supercell G0W0 reference (`pbc/gw/kgw_slow_supercell.py`, 135 l).
//!
//! Upstream runs the MOLECULAR slow code on a supercell model: several
//! k-points for the cost of discarding momentum conservation and using dense
//! matrices. A supercell Γ calculation and a k-mesh primitive-cell
//! calculation describe THE SAME CRYSTAL, so their agreement is a
//! supercell-equivalence identity needing no upstream at all (19-13
//! must-have — the same oracle-free structure Phase 11 used to catch a live
//! `super_cell` defect and Phase 16 used at `2.97e-8`).
//!
//! The identity at the Lehmann level: the supercell pole set is the union
//! over k of the primitive poles with weights scaled by `1/nk`
//! ([`replicate_supercell`]). Then `σ_super(ω) == mean_k σ_k(ω)` to summation
//! order (asserted at 1e-12, NOT bit-identity — concatenation vs per-k
//! accumulation order differs, and claiming bit-identity would be the
//! Phase-17 `ordered-reduction` trap in reverse). At `nk = 1` the two paths
//! are term-for-term identical and ARE bit-identical (asserted).
//!
//! **Do not optimise it.**

use crate::error::PbcGwError;
use crate::kgw_slow::{kernel_slow_orbital, slow_sigma_real, LehmannPole, SLOW_ETA};
use crate::types::{GwConfig, GwRoute, QpResult};

/// Replicate primitive k-mesh poles into the supercell pole set: union over
/// k with weights scaled by `1/nk` (the same crystal, Γ-only sampling).
pub fn replicate_supercell(poles_k: &[Vec<LehmannPole>]) -> Vec<LehmannPole> {
    let nk = poles_k.len().max(1) as f64;
    let mut out = Vec::new();
    for poles in poles_k {
        for pole in poles {
            out.push(LehmannPole {
                energy: pole.energy,
                weight: pole.weight / nk,
            });
        }
    }
    out
}

/// Mean over k of the primitive Lehmann self-energies (the other side of the
/// identity — ordered `oracle_sum` over the per-k values).
pub fn mean_primitive_sigma(poles_k: &[Vec<LehmannPole>], omega: f64, eta: f64) -> f64 {
    if poles_k.is_empty() {
        return 0.0;
    }
    let terms: Vec<f64> = poles_k
        .iter()
        .map(|poles| slow_sigma_real(poles, omega, eta))
        .collect();
    pyscf_algebra::oracle_sum(&terms) / (poles_k.len() as f64)
}

/// Supercell Γ-point slow G0W0 driver over the replicated pole set.
pub fn kernel_kgw_slow_supercell(
    mf_energy_super: &[f64],
    poles_super: &[Vec<LehmannPole>],
    vk_diag: &[f64],
    vmf_diag: &[f64],
    orbs: std::ops::Range<usize>,
    cfg: &GwConfig,
) -> Result<QpResult, PbcGwError> {
    if poles_super.len() != orbs.len() {
        return Err(PbcGwError::ShapeMismatch {
            expected: orbs.len(),
            got: poles_super.len(),
        });
    }
    let mut qp = Vec::with_capacity(orbs.len());
    for (oi, p) in orbs.enumerate() {
        if p >= mf_energy_super.len() {
            return Err(PbcGwError::ShapeMismatch {
                expected: p + 1,
                got: mf_energy_super.len(),
            });
        }
        qp.push(kernel_slow_orbital(
            &poles_super[oi],
            mf_energy_super[p],
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
