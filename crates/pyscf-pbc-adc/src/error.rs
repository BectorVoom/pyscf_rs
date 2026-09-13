use thiserror::Error;

/// Errors for periodic ADC (`pbc/adc`).
#[derive(Debug, Error)]
pub enum PbcAdcError {
    #[error(transparent)]
    Core(#[from] pyscf_core::PyscfRsError),

    #[error("algebra: {0}")]
    Algebra(#[from] pyscf_algebra::AlgebraError),

    #[error("shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: usize, got: usize },

    #[error("adc/{module}: not yet implemented (lands in Phase 19)")]
    NotYetImplemented { module: &'static str },

    #[error("adc: Davidson did not converge in {cycles} cycles (last residual {last_residual:e})")]
    DavidsonNotConverged { cycles: usize, last_residual: f64 },

    #[error("adc: DF route and conventional route differ by {diff:e} — gate each against its own upstream number (19-17)")]
    DfRouteMismatch { diff: f64 },
}

impl From<PbcAdcError> for pyscf_core::PyscfRsError {
    fn from(e: PbcAdcError) -> Self {
        pyscf_core::PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(format!(
            "{e}"
        )))
    }
}
