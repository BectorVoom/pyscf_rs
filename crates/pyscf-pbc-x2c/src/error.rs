use thiserror::Error;

/// Errors for periodic exact-two-component relativity (`pbc/x2c`).
#[derive(Debug, Error)]
pub enum PbcX2cError {
    #[error(transparent)]
    Core(#[from] pyscf_core::PyscfRsError),

    #[error("algebra: {0}")]
    Algebra(#[from] pyscf_algebra::AlgebraError),

    #[error("shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: usize, got: usize },

    #[error("x2c/{module}: not yet implemented (lands in Phase 19)")]
    NotYetImplemented { module: &'static str },
}

impl From<PbcX2cError> for pyscf_core::PyscfRsError {
    fn from(e: PbcX2cError) -> Self {
        pyscf_core::PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(format!(
            "{e}"
        )))
    }
}
