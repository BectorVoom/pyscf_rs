//! as_scanner. Closure body filled in plan 03-11.
use crate::RHF;
use pyscf_core::{Energy, Mole, PyscfRsError};

pub fn as_scanner(
    _rhf: &RHF,
) -> Box<dyn Fn(&Mole) -> Result<Energy, PyscfRsError> + Send + Sync> {
    unimplemented!("plan 03-11")
}
