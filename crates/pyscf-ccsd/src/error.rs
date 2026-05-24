//! CCSD-specific error variants. Composes via `pyscf_core::PyscfRsError`.
//!
//! Mirrors `crates/pyscf-mp2/src/error.rs` (the exact bridge shape, renamed
//! `Mp2`→`Ccsd`): a `thiserror` enum with `#[from]` bridges plus a
//! `From<CcsdError> for pyscf_core::PyscfRsError` routing through the
//! `Core(InvalidMolecule(..))` arm (no dedicated `PyscfRsError::Ccsd` variant
//! this phase).
//!
//! Beyond the MP2 arm set, CCSD carries two extra `#[from]` bridges that later
//! waves consume: `pyscf_runtime::BackendError` (the D-01 `try_reserve`
//! memory pre-flight — `MemoryLimitExceeded` propagates here) and
//! `pyscf_diis::DiisError` (the amplitude-DIIS subspace, Wave 1).
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CcsdError {
    #[error("algebra: {0}")]
    Algebra(#[from] pyscf_algebra::AlgebraError),

    #[error("core: {0}")]
    Core(#[from] pyscf_core::CoreError),

    #[error("ao2mo: {0}")]
    Ao2mo(#[from] pyscf_ao2mo::Ao2moError),

    /// The D-01 `WorkspacePool::try_reserve` memory pre-flight surfaces
    /// `BackendError::MemoryLimitExceeded { requested, limit }` here, so a hard
    /// refusal `?`-propagates to the caller (no silent OOM, no downgrade).
    #[error("backend: {0}")]
    Backend(#[from] pyscf_runtime::BackendError),

    /// Amplitude-DIIS (`pyscf_diis::Diis<AmplitudeSubspace>`, Wave 1) surfaces
    /// its singular-subspace / algebra failures here.
    #[error("diis: {0}")]
    Diis(#[from] pyscf_diis::DiisError),

    /// Load-bearing for the V5 shape-validation pattern: every hook-returned
    /// ERI/amplitude block length is checked BEFORE indexing, never panicking
    /// (`#![forbid(unsafe_code)]` means no UB on a logic error).
    #[error("shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: usize, got: usize },

    /// Staged-scaffold marker. `wave` is the Phase-6 wave that fills the body
    /// (NOT MP2's `plan` field): ccsd/rintermediates/update_amps → 1,
    /// uccsd/uintermediates/diis_amps → 2, lambda/ulambda/rdm/urdm/diagnostics
    /// → 3, dfccsd/direct → 4.
    #[error("ccsd: not yet implemented (body lands in Wave {wave})")]
    NotYetImplemented { wave: u8 },
}

impl From<CcsdError> for pyscf_core::PyscfRsError {
    fn from(e: CcsdError) -> Self {
        // Bridge CcsdError → core::PyscfRsError via the
        // Core(InvalidMolecule(...)) arm (carries an arbitrary String),
        // mirroring mp2/error.rs and scf/error.rs verbatim.
        pyscf_core::PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(format!("{}", e)))
    }
}
