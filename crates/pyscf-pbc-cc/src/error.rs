use thiserror::Error;

#[derive(Debug, Error)]
pub enum PbcCcError {
    #[error(transparent)]
    Core(#[from] pyscf_core::PyscfRsError),
    /// A shape, rank or subscript violation in the `ZArr`/`einsum` layer.
    #[error("shape error: {0}")]
    Shape(String),
    /// The complex arena refused, or a spill failed.
    #[error(transparent)]
    Backend(#[from] pyscf_runtime::BackendError),
    /// An algebra-layer failure (the Davidson solver, a dense eigensolve).
    #[error("algebra error: {0}")]
    Algebra(String),
    /// A surface upstream PySCF 2.12.1 does not implement either. The payload
    /// names the upstream file and line that refuses, so the refusal cannot
    /// outlive its reason (the `15-CONTEXT §1.3` discipline).
    #[error("not implemented upstream either ({upstream}): {what}")]
    NotImplementedUpstream {
        upstream: &'static str,
        what: &'static str,
    },
    /// A convergence failure with the residual it reached.
    #[error("{what} did not converge: {detail}")]
    NotConverged { what: &'static str, detail: String },
}
