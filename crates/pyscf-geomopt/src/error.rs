//! `GeomError` — the geometry-optimizer error type.
//!
//! Mirrors the locked error-bridge precedent across mp2/ccsd/grad
//! (`crates/pyscf-ccsd/src/error.rs`): every variant funnels into
//! `PyscfRsError::Core(CoreError::InvalidMolecule(..))` (no dedicated
//! `PyscfRsError::Geom` arm). Carries `#[from]` bridges for the algebra wall
//! (`AlgebraError`) and the gradient seam (`PyscfRsError` from the
//! `GradScanner`).

use thiserror::Error;

/// Errors raised by the native BFGS+RFO optimizer.
#[derive(Debug, Error)]
pub enum GeomError {
    /// A linear-algebra-wall operation (eigh_gen / solve_linear) failed.
    #[error("algebra: {0}")]
    Algebra(#[from] pyscf_algebra::AlgebraError),

    /// The driven `GradScanner` (energy/gradient evaluation) failed.
    #[error("scanner: {0}")]
    Scanner(#[from] pyscf_core::PyscfRsError),

    /// `maxsteps` was reached without satisfying the 5-criterion convergence.
    #[error("geomopt did not converge within {maxsteps} steps")]
    NotConverged { maxsteps: usize },

    /// A `constraints` kwarg was supplied — full shim parity lands in 07-06
    /// (D-07 / GEOMOPT-EXT-01 deferred). Raised as a clear error, never a
    /// silent no-op (T-07-11).
    #[error("constraints are not yet supported (GEOMOPT-EXT-01, lands in 07-06); pass constraints=None")]
    ConstraintsUnsupported,

    /// A user-supplied `maxsteps` failed validation (T-07-10: must be a
    /// bounded positive count).
    #[error("invalid maxsteps {got}: must be in 1..=10000")]
    InvalidMaxSteps { got: usize },

    /// The back-transform internal→Cartesian iteration did not converge
    /// (a pathological / singular B-matrix, T-07-13).
    #[error("internal→Cartesian back-transform failed to converge in {iters} iterations")]
    BacktransformDiverged { iters: usize },

    /// An HDF5 checkpoint read/write through the `pyscf-chkfile` alias failed
    /// (GEOMOPT-05). The optimizer-state spill reuses the sole-owner chkfile
    /// surface — it never declares its own `hdf5-metno` dep (D-07).
    #[error("checkpoint hdf5: {0}")]
    Chkfile(#[from] pyscf_chkfile::ChkfileError),

    /// A loaded optimizer-state checkpoint is corrupt / internally
    /// inconsistent (shape mismatch or an unknown schema version). Raised as a
    /// CLEAR error rather than resuming from garbage (T-07-19).
    #[error("corrupt optimizer-state checkpoint: {what}")]
    CheckpointCorrupt { what: String },
}

impl From<GeomError> for pyscf_core::PyscfRsError {
    fn from(e: GeomError) -> Self {
        pyscf_core::PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(format!("{e}")))
    }
}
