use thiserror::Error;

/// Errors for periodic electron-phonon coupling (`pbc/eph`).
#[derive(Debug, Error)]
pub enum PbcEphError {
    #[error(transparent)]
    Core(#[from] pyscf_core::PyscfRsError),

    #[error("algebra: {0}")]
    Algebra(#[from] pyscf_algebra::AlgebraError),

    #[error("shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: usize, got: usize },

    #[error("eph/{module}: not yet implemented (lands in Phase 19)")]
    NotYetImplemented { module: &'static str },
}

impl From<PbcEphError> for pyscf_core::PyscfRsError {
    fn from(e: PbcEphError) -> Self {
        pyscf_core::PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(format!(
            "{e}"
        )))
    }
}
