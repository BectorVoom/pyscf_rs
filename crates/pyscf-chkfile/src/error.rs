//! pyscf-chkfile error type.
//!
//! Source: D-06 — per-method dump/load surface. Errors fall into three
//! buckets:
//! - HDF5 layer (open/read/write/group/dataset)
//! - mol JSON parse (Tampering threat T-3-03 mitigation)
//! - I/O (path resolution / std::io)
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ChkfileError {
    #[error("HDF5: {0}")]
    Hdf5(#[from] hdf5_metno::Error),

    /// Pitfall: untrusted chkfile may contain malformed `mol.dumps()` JSON.
    /// T-3-03 mitigation — `serde_json::from_str` is panic-free.
    #[error("malformed mol JSON: {0}")]
    MalformedMol(#[from] serde_json::Error),

    #[error("invalid UTF-8 in chkfile string")]
    InvalidUtf8,

    #[error("missing key {0} in chkfile")]
    MissingKey(String),

    #[error("shape mismatch reading {key}: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        key: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}
