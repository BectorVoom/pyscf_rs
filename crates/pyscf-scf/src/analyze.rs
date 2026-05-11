//! analyze/mulliken_pop/mulliken_meta/dip_moment. Bodies filled in plan 03-11.
use crate::RHF;
use pyscf_core::PyscfRsError;

pub fn analyze(_rhf: &RHF) -> Result<(), PyscfRsError> {
    unimplemented!("plan 03-11")
}
pub fn mulliken_pop(_rhf: &RHF) -> Result<MullikenResult, PyscfRsError> {
    unimplemented!("plan 03-11")
}
pub fn mulliken_meta(_rhf: &RHF) -> Result<MullikenResult, PyscfRsError> {
    unimplemented!("plan 03-11")
}
pub fn dip_moment(_rhf: &RHF) -> Result<[f64; 3], PyscfRsError> {
    unimplemented!("plan 03-11")
}

pub struct MullikenResult {
    pub atom_charges: Vec<f64>,
    pub ao_populations: Vec<f64>,
}
