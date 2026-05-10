//! Project-wide error type. `PyscfRsError` is the public-facing error
//! enum returned by every method's `kernel()`. `CoreError` is the
//! pyscf-core-internal subset.

use thiserror::Error;

/// Top-level error returned by every pyscf-rs method.
/// Phase 1 ships only the foundational variants; Phases 2-7 add
/// method-specific variants via `#[from]` conversions.
#[derive(Debug, Error)]
pub enum PyscfRsError {
    #[error("core error: {0}")]
    Core(#[from] CoreError),

    #[error("not yet implemented (Phase {phase}): {what}")]
    NotYetImplemented { phase: u8, what: &'static str },

    #[error("convergence failure after {iterations} iterations: {reason}")]
    ConvergenceFailure {
        iterations: u32,
        reason: String,
    },
}

/// pyscf-core-internal error subset.
#[derive(Debug, Error)]
pub enum CoreError {
    #[error("invalid molecule: {0}")]
    InvalidMolecule(String),

    #[error("basis set parse error: {0}")]
    BasisParse(String),

    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}
