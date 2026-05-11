//! E_elec + E_tot. Body (with oracle_sum/oracle_dot per Pitfall 9) filled in plan 03-11.
use pyscf_core::{Density, Energy, PyscfRsError};

pub fn default_energy_elec(
    _dm: &Density,
    _h1e: &Density,
    _vhf: &Density,
) -> Result<(Energy, Energy), PyscfRsError> {
    unimplemented!("plan 03-11")
}

pub fn default_energy_tot(
    _dm: &Density,
    _h1e: &Density,
    _vhf: &Density,
) -> Result<Energy, PyscfRsError> {
    unimplemented!("plan 03-11")
}
