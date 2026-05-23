//! pyscf-diis: generic Pulay extrapolation (CDIIS).
//!
//! Source: D-08 + D-09 + RESEARCH §"Pattern 7" lines 810-893. Mirrors
//! upstream `pyscf/scf/diis.py:40-87` (CDIIS class + `get_err_vec_orig`
//! SDF − FDS error vector method).
//!
//! Pitfall 9 mitigation (threat T-3-09): all reductions in the Pulay
//! solve go through `pyscf-algebra::oracle_dot` (B-matrix inner products)
//! and `pyscf-algebra::oracle_sum` (extrapolated-iterate cross-iterate
//! sums). Verified bit-identical across Linux/macOS / thread counts by
//! `tests/oracle_reductions.rs`.
//!
//! Phase 6 CCSD-04 amplitude DIIS reuses the same `Diis<S>` shell with an
//! `AmpsSubspace` impl of `DiisStorable` (deferred to Phase 6).
#![forbid(unsafe_code)]
#![warn(clippy::unwrap_used)]

pub mod cdiis;
pub mod error;
pub mod storable;

pub use cdiis::{Diis, err_vec_scf};
pub use error::DiisError;
pub use storable::DiisStorable;
