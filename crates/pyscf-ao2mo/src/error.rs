//! ao2mo-specific error variants. Composes via pyscf-core::PyscfRsError.
//!
//! Mirrors `crates/pyscf-scf/src/error.rs`: a thiserror enum with `#[from]`
//! bridges from the algebra/core error types plus a `From<Ao2moError> for
//! pyscf_core::PyscfRsError` routing through the `Core(InvalidMolecule(..))`
//! arm (avoids touching pyscf-core::error.rs this phase — a dedicated
//! `PyscfRsError::Ao2mo` variant is later-phase territory).
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Ao2moError {
    #[error("algebra: {0}")]
    Algebra(#[from] pyscf_algebra::AlgebraError),

    #[error("core: {0}")]
    Core(#[from] pyscf_core::CoreError),

    #[error("shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: usize, got: usize },

    #[error("ao2mo: not yet implemented (body lands in plan 05-02)")]
    NotYetImplemented,
}

impl From<Ao2moError> for pyscf_core::PyscfRsError {
    fn from(e: Ao2moError) -> Self {
        // Bridge Ao2moError → core::PyscfRsError via the
        // Core(InvalidMolecule(...)) arm, which carries an arbitrary String.
        // This mirrors the scf/error.rs bridge and avoids adding a dedicated
        // PyscfRsError::Ao2mo variant to pyscf-core this phase.
        pyscf_core::PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(format!("{}", e)))
    }
}
