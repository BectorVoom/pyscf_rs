//! BackendError — backend-selection and resource errors.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("backend {backend} requested with {dtype} but unsatisfiable: {reason}")]
    Unsatisfiable {
        backend: &'static str,
        dtype: &'static str,
        reason: String,
    },

    #[error("memory limit exceeded: requested {requested} bytes, limit {limit} bytes")]
    MemoryLimitExceeded {
        requested: usize,
        limit: usize,
    },

    #[error("backend not compiled in: {0}")]
    FeatureNotEnabled(&'static str),

    #[error("probe failed for backend {backend}: {reason}")]
    ProbeFailed {
        backend: &'static str,
        reason: String,
    },
}
