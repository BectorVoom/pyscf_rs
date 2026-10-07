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
        // Non-convergence keeps its typed variant so the Python boundary can
        // report kind `ConvergenceFailure` (BIND-09; carryover
        // 20-molecular-python-suite-drift item 4). Everything else still
        // bridges via Core(InvalidMolecule(..)), which carries an arbitrary
        // String — adding a dedicated PyscfRsError::Scf variant remains
        // deferred as in plan 03-03.
        if let ScfError::ConvergenceFailure { cycles, .. } = &e {
            let iterations = *cycles;
            let reason = e.to_string();
            return pyscf_core::PyscfRsError::ConvergenceFailure { iterations, reason };
        }
        pyscf_core::PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(format!("{}", e)))
    }
}
