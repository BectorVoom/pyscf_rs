use thiserror::Error;

/// Errors for periodic G0W0 (`pbc/gw`).
#[derive(Debug, Error)]
pub enum PbcGwError {
    #[error(transparent)]
    Core(#[from] pyscf_core::PyscfRsError),

    #[error("algebra: {0}")]
    Algebra(#[from] pyscf_algebra::AlgebraError),

    #[error("shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: usize, got: usize },

    #[error("gw/{module}: not yet implemented (lands in Phase 19)")]
    NotYetImplemented { module: &'static str },

    #[error("gw: AC and CD are different approximations — a route-blind comparison is refused (19-01 Task 3)")]
    RouteBlindComparison,

    #[error("gw: Padé continuation failed ({reason})")]
    PadeFailure { reason: String },

    #[error("gw: quasiparticle equation did not converge in {cycles} cycles")]
    QpNotConverged { cycles: usize },
}

impl From<PbcGwError> for pyscf_core::PyscfRsError {
    fn from(e: PbcGwError) -> Self {
        pyscf_core::PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(format!(
            "{e}"
        )))
    }
}
