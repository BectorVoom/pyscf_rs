//! Project-wide error type. `PyscfRsError` is the public-facing error
//! enum returned by every method's `kernel()`. `CoreError` is the
//! pyscf-core-internal subset.
//!
//! Phase 2 (GTO-01..11) adds basis-load + ECP-load + ECP-engine-not-available
//! variants so the gto crate's parsers and the EcpEngine trait stub can
//! propagate errors via `?`.

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
    ConvergenceFailure { iterations: u32, reason: String },

    /// Phase 2 GTO-02/GTO-03 — failure to resolve / parse a basis-set
    /// definition (file not found, ALIAS unknown, parse error, IO error).
    #[error("basis load error: {0}")]
    BasisLoad(#[from] BasisLoadError),

    /// Phase 2 GTO-05 (loading half) — failure to resolve / parse an ECP
    /// definition (NWChem-format ECP block).
    #[error("ECP load error: {0}")]
    EcpLoad(#[from] EcpLoadError),

    /// Phase 2 D-06 — `EcpEngine` trait shipped, but the cintx ECP
    /// workstream hasn't landed `cint1e_ecp` yet. Returned by the stub
    /// `EcpEngine` impl until the gap-closure plan wires the real engine.
    #[error(
        "ECP engine not available — pending cintx ECP merge (Phase 2 D-06 gap-closure plan 02-10)"
    )]
    EcpEngineNotAvailable,

    /// Phase 4 D-08 / CR-02 gap-closure — a value in the f32 precision path
    /// (`PYSCF_DTYPE=f32`) exceeded `f32::MAX` (~3.4e38) during an f64→f32
    /// conversion. Surfaced instead of silently substituting `0.0`, so the
    /// "honest f32 path" reports data corruption rather than producing a
    /// wrong-but-quiet result. `context` names the exact conversion site.
    #[error("numeric overflow in f32 precision path: {context} (value exceeds f32::MAX ~3.4e38)")]
    NumericOverflow { context: &'static str },
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

/// Phase 2 GTO-02/GTO-03 basis-loading failures. Surfaced via
/// `PyscfRsError::BasisLoad` (#[from]).
#[derive(Debug, Error)]
pub enum BasisLoadError {
    #[error("basis path not found; tried: {tried}. Set PYSCF_BASIS_PATH to override")]
    PathNotFound { tried: String },

    #[error("unknown basis name '{name}' (not in ALIAS, GTH_ALIAS, or PP_ALIAS)")]
    UnknownName { name: String },

    /// The basis set is known but defines nothing for this element — e.g.
    /// `def2-svp` covers no lanthanide, so `Eu` has no block in
    /// `def2-svp.dat`. Distinct from [`BasisLoadError::UnknownName`] because
    /// the two route differently: both may be retried against the Basis Set
    /// Exchange, but only this one means "the name was right".
    #[error("basis '{name}' has no entry for element '{symbol}' in {file}")]
    ElementAbsent {
        name: String,
        symbol: String,
        file: String,
    },

    #[error("parse error in {file}:{line}: {reason}")]
    Parse {
        file: String,
        line: usize,
        reason: String,
    },

    #[error("io error reading {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    /// The Basis Set Exchange fallback was reached and did not yield a basis.
    /// Mirrors the `BasisNotFoundError` upstream raises after its own BSE
    /// attempt fails (`pyscf/gto/basis/__init__.py:707-714`).
    #[error("Basis Set Exchange lookup failed for basis '{name}', element '{symbol}': {reason}")]
    Bse {
        name: String,
        symbol: String,
        reason: String,
    },
}

/// Phase 2 GTO-05 (loading half) ECP-loading failures. Surfaced via
/// `PyscfRsError::EcpLoad` (#[from]).
#[derive(Debug, Error)]
pub enum EcpLoadError {
    #[error("unknown ECP name '{0}'")]
    UnknownName(String),

    #[error("ECP parse error in {file}:{line}: {reason}")]
    Parse {
        file: String,
        line: usize,
        reason: String,
    },

    /// Counterpart of [`BasisLoadError::Bse`] on the ECP surface
    /// (`pyscf/gto/basis/__init__.py:765-779`).
    #[error("Basis Set Exchange lookup failed for ECP '{name}', element '{symbol}': {reason}")]
    Bse {
        name: String,
        symbol: String,
        reason: String,
    },
}
