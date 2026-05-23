//! AlgebraError — algebra-specific errors.
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AlgebraError {
    #[error("backend error: {0}")]
    Backend(#[from] pyscf_runtime::BackendError),
    #[error("dimension mismatch in {op}: lhs {lhs:?}, rhs {rhs:?}")]
    DimensionMismatch {
        op: &'static str,
        lhs: Vec<usize>,
        rhs: Vec<usize>,
    },
    #[error("dtype mismatch in {op}: lhs {lhs:?}, rhs {rhs:?}")]
    DtypeMismatch {
        op: &'static str,
        lhs: pyscf_runtime::DType,
        rhs: pyscf_runtime::DType,
    },
    #[error("not yet implemented (Phase {phase}): {what}")]
    NotYetImplemented { phase: u8, what: &'static str },
    #[error("cubecl runtime error: {0}")]
    CubeclRuntime(String),
    /// Phase 3 plan 03-01 — `solve_linear` shape validation. Distinct from
    /// `DimensionMismatch` (which carries `Vec<usize>` shapes for tensor
    /// ops) so the simpler row/col `expected vs actual` flat-slice check
    /// stays cheap and allocation-friendly.
    #[error("shape mismatch: expected {expected}, actual {actual}")]
    ShapeMismatch { expected: String, actual: String },
    /// Phase 3 plan 03-01 — `solve_linear` singular-matrix indicator.
    /// faer 0.24 `FullPivLu` does not panic on singular input; the
    /// solve produces non-finite entries which `solve_linear` detects.
    #[error("solve_linear: matrix is singular")]
    Singular,
}
