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

    /// Mirrors `pbc/grad/rhf.py:42-47`: the gamma gradient has NO
    /// non-multigrid branch — the `else` is `raise NotImplementedError`.
    /// A Coulomb engine that is not `MultiGridNumInt2` is a named refusal,
    /// never a fallback to a different route.
    #[error(
        "gamma gradient requires MultiGridNumInt2 (pbc/grad/rhf.py:42-47 has no non-multigrid branch); got {detail}"
    )]
    NonMultigridCoulomb { detail: String },

    /// Mirrors `pbc/grad/rhf.py:78-79`: a non-gamma k-point raises
    /// `NotImplementedError`, exactly as upstream.
    #[error(
        "gamma gradient requires the gamma point (pbc/grad/rhf.py:78-79); got kpt [{0}, {1}, {2}]"
    )]
    NonGammaKpt(f64, f64, f64),
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
