use thiserror::Error;

/// Errors for periodic TDA/TDHF/TDDFT (`pbc/tdscf`).
#[derive(Debug, Error)]
pub enum PbcTdscfError {
    #[error(transparent)]
    Core(#[from] pyscf_core::PyscfRsError),

    #[error("algebra: {0}")]
    Algebra(#[from] pyscf_algebra::AlgebraError),

    #[error("shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: usize, got: usize },

    #[error("tdscf/{module}: not yet implemented (lands in Phase 19)")]
    NotYetImplemented { module: &'static str },

    #[error("tdscf: Davidson did not converge in {cycles} cycles (last residual {last_residual:e})")]
    DavidsonNotConverged { cycles: usize, last_residual: f64 },

    #[error("tdscf: nroots={nroots} exceeds response dimension {dim}")]
    TooManyRoots { nroots: usize, dim: usize },

    #[error("tdscf: A-B not positive definite (lowest {lowest:e}) - unstable reference, route to stability analysis")]
    UnstableReference { lowest: f64 },
}

impl From<PbcTdscfError> for pyscf_core::PyscfRsError {
    fn from(e: PbcTdscfError) -> Self {
        pyscf_core::PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(format!(
            "{e}"
        )))
    }
}
