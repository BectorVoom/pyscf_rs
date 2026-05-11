//! pyscf-df error variants (T-3-17 mitigation surface).
//!
//! Source: plan 03-05 Task 1. Maps to `PyscfRsError::Core(InvalidMolecule)`
//! (the only String-carrying catch-all variant on the pyscf-core enum;
//! `CoreError::Other` doesn't exist — see plan 03-03 SUMMARY deviation 1).

use thiserror::Error;

#[derive(Debug, Error)]
pub enum DfError {
    /// Auxiliary basis (P|Q) is rank-deficient — Cholesky decomposition
    /// failed. Threat T-3-17 mitigation: caller can fall back to canonical
    /// DF or HF (un-density-fitted) without panic.
    #[error("auxiliary basis (P|Q) is rank-deficient (Cholesky failed)")]
    SingularAux,

    /// Auxbasis name has no known default and 'weigend' fallback failed
    /// (e.g., the weigend basis file is missing from PYSCF_BASIS_PATH).
    #[error("auxbasis '{0}' has no known default and 'weigend' fallback failed")]
    UnknownAuxbasis(String),

    /// Lower-level algebra failure (host-faer Cholesky / linear-solve
    /// returned an error). Auto-converted from `pyscf_algebra::AlgebraError`.
    #[error("algebra: {0}")]
    Algebra(#[from] pyscf_algebra::AlgebraError),

    /// Lower-level pyscf-core error (Mole not built, dimension mismatch, ...).
    #[error("core: {0}")]
    Core(#[from] pyscf_core::CoreError),
}

impl From<DfError> for pyscf_core::PyscfRsError {
    fn from(e: DfError) -> Self {
        pyscf_core::PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(format!("{}", e)))
    }
}
