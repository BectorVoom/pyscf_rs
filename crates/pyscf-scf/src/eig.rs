//! eigh + canonicalize_signs. Body filled in plan 03-11.
use pyscf_core::{Density, MOCoefficients, PyscfRsError};

pub fn default_eig(
    _fock: &Density,
    _s1e: &Density,
) -> Result<MOCoefficients, PyscfRsError> {
    unimplemented!(
        "plan 03-11 — must call pyscf-algebra::eigh + pyscf_core::canonicalize_signs (SCF-13)"
    )
}
