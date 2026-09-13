use thiserror::Error;

#[derive(Debug, Error)]
pub enum PbcGradError {
    #[error(transparent)]
    Core(#[from] pyscf_core::PyscfRsError),

    /// A caller supplied an invalid central-difference displacement.  Validate
    /// this before evaluating either side of a finite difference.
    #[error("invalid finite-difference displacement: {disp} (must be finite and > 0)")]
    InvalidDisplacement { disp: f64 },

    /// Two periodic-gradient inputs disagree on their atom count or shape.
    #[error("shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: usize, got: usize },

    /// A deliberate Phase-18 seam.  This is a named error rather than a
    /// fabricated zero gradient while the method-specific body is unavailable.
    #[error("periodic gradient: not yet implemented (Phase {phase}): {what}")]
    NotYetImplemented { phase: u8, what: &'static str },

    /// Mirrors upstream `GradientsBase.optimizer`: only the ASE spelling is
    /// accepted at this entry point.
    #[error("periodic gradient optimizer '{solver}' is not supported (only 'ase')")]
    UnsupportedOptimizer { solver: String },
}

impl From<PbcGradError> for pyscf_core::PyscfRsError {
    fn from(error: PbcGradError) -> Self {
        match error {
            PbcGradError::Core(error) => error,
            other => pyscf_core::PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
                other.to_string(),
            )),
        }
    }
}
