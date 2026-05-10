//! AlgebraError — algebra-specific errors.
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AlgebraError {
    #[error("backend error: {0}")]
    Backend(#[from] pyscf_runtime::BackendError),
    #[error("dimension mismatch in {op}: lhs {lhs:?}, rhs {rhs:?}")]
    DimensionMismatch { op: &'static str, lhs: Vec<usize>, rhs: Vec<usize> },
    #[error("dtype mismatch in {op}: lhs {lhs:?}, rhs {rhs:?}")]
    DtypeMismatch { op: &'static str, lhs: pyscf_runtime::DType, rhs: pyscf_runtime::DType },
    #[error("not yet implemented (Phase {phase}): {what}")]
    NotYetImplemented { phase: u8, what: &'static str },
    #[error("cubecl runtime error: {0}")]
    CubeclRuntime(String),
}
