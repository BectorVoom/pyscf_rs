//! Gradient-specific error variants. Composes via `pyscf_core::PyscfRsError`.
//!
//! Mirrors `crates/pyscf-ccsd/src/error.rs` (the exact bridge shape, renamed
//! `Ccsd`→`Grad`): a `thiserror` enum with `#[from]` bridges plus a
//! `From<GradError> for pyscf_core::PyscfRsError` routing through the
//! `Core(InvalidMolecule(..))` arm (no dedicated `PyscfRsError::Grad` variant
//! this phase — the locked precedent across mp2/ccsd/ao2mo/scf).
//!
//! Beyond the base Algebra/Core/Ao2mo arm set, GradError carries two extra
//! `#[from]` bridges because gradients CONSUME both post-SCF crates: the MP2
//! Z-vector gradient (GRAD-05) `?`-propagates `pyscf_mp2::Mp2Error`, and the
//! CCSD Λ-driven gradient (GRAD-06) `?`-propagates `pyscf_ccsd::CcsdError`.
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GradError {
    #[error("algebra: {0}")]
    Algebra(#[from] pyscf_algebra::AlgebraError),

    #[error("core: {0}")]
    Core(#[from] pyscf_core::CoreError),

    #[error("ao2mo: {0}")]
    Ao2mo(#[from] pyscf_ao2mo::Ao2moError),

    /// The MP2 Z-vector gradient (GRAD-05) consumes `pyscf-mp2`; its relaxed-
    /// density / amplitude failures `?`-propagate here.
    #[error("mp2: {0}")]
    Mp2(#[from] pyscf_mp2::Mp2Error),

    /// The CCSD Λ-driven gradient (GRAD-06) consumes `pyscf-ccsd`'s converged
    /// `solve_lambda` + `make_rdm1`/`make_rdm2`; their failures surface here.
    #[error("ccsd: {0}")]
    Ccsd(#[from] pyscf_ccsd::CcsdError),

    /// Load-bearing for the V5 shape-validation pattern (T-07-06): every
    /// caller-supplied `atmlst` index / hook-returned block length is checked
    /// BEFORE indexing, never panicking (`#![forbid(unsafe_code)]` means no UB
    /// on a logic error). An out-of-range `atmlst` index returns this rather
    /// than indexing out of bounds.
    #[error("shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: usize, got: usize },

    /// Caller-supplied finite-difference displacement (`verify_fd`) was
    /// non-positive or non-finite (T-07-05). Rejected before any energy eval.
    #[error("invalid finite-difference displacement: {disp} (must be finite and > 0)")]
    InvalidDisplacement { disp: f64 },

    /// Staged-scaffold marker. `wave` is the Phase-7 wave that fills the body:
    /// rhf → 3, uhf → 4, rks/uks → 5, mp2 → 6, ccsd → 7, ecp → 8, cphf → 2.
    #[error("grad: not yet implemented (body lands in Wave {wave})")]
    NotYetImplemented { wave: u8 },
}

impl From<GradError> for pyscf_core::PyscfRsError {
    fn from(e: GradError) -> Self {
        // Bridge GradError → core::PyscfRsError via the
        // Core(InvalidMolecule(...)) arm (carries an arbitrary String),
        // mirroring ccsd/error.rs and mp2/error.rs verbatim.
        pyscf_core::PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(format!("{}", e)))
    }
}
