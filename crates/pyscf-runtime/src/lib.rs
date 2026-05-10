//! pyscf-runtime: BackendKind, per-backend probes, WorkspacePool,
//! tracing init.
//!
//! ALG-06 carve-out: this crate (alongside pyscf-algebra) is permitted
//! to depend on cubecl-* runtime crates for low-level client probing.
//! All other workspace crates MUST consume algebra primitives via
//! pyscf-algebra and never name a cubecl-* type.
//!
//! Phase 1 status:
//!   * BackendKind enum + Default + from_env_str: COMPLETE (FOUND-03)
//!   * Per-backend probes: COMPLETE — cpu/cuda/wgpu/hip with OnceLock
//!     caching and catch_unwind discipline (Pitfall 5 + FOUND-07).
//!   * WorkspacePool: SKELETON — Phase 6 (CCSD-11) implements the body
//!     (RESEARCH §4).
//!   * tracing_init: COMPLETE — library helper, no subscriber install
//!     (RESEARCH §12).
//!
//! `select_backend()` lives in pyscf-algebra (returns AlgebraClient,
//! which is owned by that crate). pyscf-runtime exposes the building
//! blocks; pyscf-algebra wires them.
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::unwrap_used)]

pub mod backend;
pub mod error;
pub mod probe;
pub mod tracing_init;
pub mod workspace_pool;

pub use backend::{BackendKind, DType};
pub use error::BackendError;
pub use tracing_init::init_tracing;
pub use workspace_pool::WorkspacePool;
