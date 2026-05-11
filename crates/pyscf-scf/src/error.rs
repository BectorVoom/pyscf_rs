//! SCF-specific error variants. Composes via pyscf-core::PyscfRsError.
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ScfError {
    #[error("SCF did not converge after {cycles} cycles (last |ΔE|={last_diff:e})")]
    ConvergenceFailure { cycles: u32, last_diff: f64 },

    #[error("init_guess mode '{0}' not yet implemented (deferred to plan {1})")]
    InitGuessNotYetImplemented(&'static str, &'static str),

    #[error("algebra: {0}")]
    Algebra(#[from] pyscf_algebra::AlgebraError),

    #[error("core: {0}")]
    Core(#[from] pyscf_core::CoreError),

    #[error("py-override-failed: {cause}")]
    PythonOverrideFailed { cause: String },
}

impl From<ScfError> for pyscf_core::PyscfRsError {
    fn from(e: ScfError) -> Self {
        // Bridge ScfError → core::PyscfRsError via the Core(InvalidMolecule(...))
        // arm, which carries an arbitrary String. This avoids touching
        // pyscf-core::error.rs in plan 03-03 — adding a dedicated
        // PyscfRsError::Scf variant is plan-03-11 / plan-03-07 territory.
        pyscf_core::PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(format!("{}", e)))
    }
}
