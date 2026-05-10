//! pyscf-rs: top-level façade re-exporting in-scope methods.
//!
//! Phase 1: re-exports pyscf-{core,runtime,algebra}. Phase 2+ adds
//! per-method re-exports as those crates land.
//!
//! End users do `cargo add pyscf-rs` and then `use pyscf_rs::scf::RHF;`
//! once Phase 3 lands the SCF module.
#![forbid(unsafe_code)]

pub use pyscf_core as core;
pub use pyscf_runtime as runtime;
pub use pyscf_algebra as algebra;

// Convenience re-exports of the most commonly used types.
pub use pyscf_core::{Density, Energy, MOCoefficients, Mole};
pub use pyscf_algebra::{select_backend, AlgebraClient, BackendSelection, DType, Tensor};
pub use pyscf_runtime::{BackendKind, WorkspacePool};
