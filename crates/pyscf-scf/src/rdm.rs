//! make_rdm1 = C diag(occ) C^T. Body filled in plan 03-11.
use pyscf_core::{Density, MOCoefficients, PyscfRsError};

pub fn default_make_rdm1(_mo: &MOCoefficients) -> Result<Density, PyscfRsError> {
    unimplemented!("plan 03-11")
}
