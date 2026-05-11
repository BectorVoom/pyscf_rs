//! to_rhf/to_uhf/to_ghf/to_rks/to_uks. Bodies filled in plan 03-11.
use crate::{GHF, RHF, UHF};
use pyscf_core::PyscfRsError;

pub fn to_rhf(_uhf: &UHF) -> Result<RHF, PyscfRsError> {
    unimplemented!("plan 03-11")
}
pub fn to_uhf(_rhf: &RHF) -> Result<UHF, PyscfRsError> {
    unimplemented!("plan 03-11")
}
pub fn to_ghf(_rhf: &RHF) -> Result<GHF, PyscfRsError> {
    unimplemented!("plan 03-11")
}
pub fn to_uks_stub(_rhf: &RHF) -> Result<(), PyscfRsError> {
    Err(PyscfRsError::NotYetImplemented {
        phase: 4,
        what: "to_uks (Phase 4 DFT)",
    })
}
pub fn to_rks_stub(_rhf: &RHF) -> Result<(), PyscfRsError> {
    Err(PyscfRsError::NotYetImplemented {
        phase: 4,
        what: "to_rks (Phase 4 DFT)",
    })
}
