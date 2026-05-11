//! Aufbau-fill for closed-shell RHF. Body filled in plan 03-11.
//!
//! Source: `pyscf/scf/hf.py:1499-1517` — `def get_occ(self, mo_energy,
//! mo_coeff)`. Closed-shell RHF puts 2 electrons in each of the lowest
//! `nelec/2` MOs.
//!
//! UHF and GHF have their own occupation logic (alpha/beta split for
//! UHF, 2c-spinor for GHF); this default is the RHF closed-shell case
//! that the plan-03-03 OverrideHooks trait dispatches to. UHF override
//! impls (plan 03-04 follow-up) will provide their own.

use pyscf_core::PyscfRsError;

use crate::error::ScfError;

/// Default RHF Aufbau-fill: fill the lowest `nelec/2` MOs with occupancy
/// 2.0, the rest with 0.0. Returns an `Err` if `nelec` is odd (RHF
/// requires an even electron count — UHF / ROHF handle the odd case).
pub fn default_get_occ(mo_energy: &[f64], nelec: usize) -> Result<Vec<f64>, PyscfRsError> {
    if nelec % 2 != 0 {
        return Err(ScfError::Core(pyscf_core::CoreError::InvalidMolecule(format!(
            "RHF Aufbau requires even nelec; got {nelec} (use UHF / ROHF for odd-electron systems)"
        )))
        .into());
    }
    let n_occ = nelec / 2;
    let mut occ = vec![0.0_f64; mo_energy.len()];
    for i in 0..n_occ.min(mo_energy.len()) {
        occ[i] = 2.0;
    }
    Ok(occ)
}
