//! pyscf-pbc-symm: space group, Symmetry, k-point symmetry adapters
//!
//! Plan 17-02 lands the bottom of the symmetry stack: lattice point-group
//! detection (`geom`), the crystallographic classification tables
//! (`tables`) and the finite-group algebra (`group`) that both
//! `space_group.py` (17-03) and `basis.py` (17-04) read.
//!
//! See `.planning/pbc/PBC-MASTER-PLAN.md` D-PBC-25 for why `KPoints`
//! (17-05) will live in this crate, next to `Symmetry` (17-03), rather than
//! in `pyscf-pbc-lib`.
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::unwrap_used)]

pub mod basis;
pub mod error;
pub mod geom;
pub mod group;
pub mod kpts;
pub mod ktensor;
pub mod space_group;
pub mod symmetry;
pub mod tables;

pub use error::*;
