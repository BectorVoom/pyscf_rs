//! MP2-specific error variants. Composes via pyscf-core::PyscfRsError.
//!
//! Mirrors `crates/pyscf-ao2mo/src/error.rs` (Task 1) and the scf/error.rs
//! precedent: a thiserror enum with `#[from]` bridges plus a
//! `From<Mp2Error> for pyscf_core::PyscfRsError` routing through the
//! `Core(InvalidMolecule(..))` arm (no dedicated `PyscfRsError::Mp2` variant
//! this phase).
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Mp2Error {
    #[error("algebra: {0}")]
    Algebra(#[from] pyscf_algebra::AlgebraError),

    #[error("core: {0}")]
    Core(#[from] pyscf_core::CoreError),

    #[error("ao2mo: {0}")]
    Ao2mo(#[from] pyscf_ao2mo::Ao2moError),

    #[error("shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: usize, got: usize },

    #[error("mp2: not yet implemented (body lands in plan 05-0{plan})")]
    NotYetImplemented { plan: u8 },
}

impl From<Mp2Error> for pyscf_core::PyscfRsError {
    fn from(e: Mp2Error) -> Self {
        // Bridge Mp2Error → core::PyscfRsError via the
        // Core(InvalidMolecule(...)) arm (carries an arbitrary String),
        // mirroring scf/error.rs and ao2mo/error.rs.
        pyscf_core::PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(format!("{}", e)))
    }
}
