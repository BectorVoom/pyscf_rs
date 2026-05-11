//! DIIS error variants.
//!
//! Threat T-3-13 mitigation: singular B-matrix surfaces as
//! `DiisError::Singular`, not a panic. Caller (kernel cycle loop) may fall
//! back to a damped Fock for one cycle on this error.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum DiisError {
    /// B-matrix is rank-deficient (degenerate iterates produce a singular
    /// Pulay system). `pyscf_algebra::solve_linear` returned `Singular`.
    #[error("B-matrix is singular (rank-deficient subspace)")]
    Singular,
    /// Underlying algebra failure other than singularity.
    #[error("algebra: {0}")]
    Algebra(#[from] pyscf_algebra::AlgebraError),
}
