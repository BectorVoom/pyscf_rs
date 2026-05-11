//! Fock build. Body filled in plan 03-11 (kernel internals split per WARNING 3).
use pyscf_core::{Density, Mole, PyscfRsError};

pub fn default_get_hcore(_mol: &Mole) -> Result<Density, PyscfRsError> {
    unimplemented!("plan 03-11")
}
pub fn default_get_ovlp(_mol: &Mole) -> Result<Density, PyscfRsError> {
    unimplemented!("plan 03-11")
}
pub fn default_get_jk(
    _mol: &Mole,
    _dm: &Density,
) -> Result<(Density, Density), PyscfRsError> {
    unimplemented!("plan 03-11")
}
pub fn default_get_veff(_mol: &Mole, _dm: &Density) -> Result<Density, PyscfRsError> {
    unimplemented!("plan 03-11")
}
pub fn default_get_fock(
    _h1e: &Density,
    _s1e: &Density,
    _vhf: &Density,
    _dm: &Density,
    _cycle: i32,
    _diis_state: Option<&Density>,
) -> Result<Density, PyscfRsError> {
    unimplemented!("plan 03-11")
}
