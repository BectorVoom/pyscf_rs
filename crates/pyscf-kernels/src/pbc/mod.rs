//! Periodic-boundary-condition device kernels (PBC-MASTER-PLAN §6).
//!
//! Phase 9 plan 09-02 seeds this module with K-04 (`zhadamard`), the one
//! complex operation in the §5.2 contract that has no real-primitive
//! decomposition. Later PBC phases add the remaining K-* kernels here.

//! Phase 9 plan 09-05 adds K-01 (`gv`) and K-02 (`struct_factor`).
//! Phase 9 plan 09-08 adds K-05 (`ewald_rlij`) and K-06 (`ewald_gs_terms`).

pub mod ewald;
pub mod gv;
pub mod struct_factor;
pub mod zhadamard;

pub use ewald::{EWALD_G0_SENTINEL, ewald_gs_terms, ewald_rlij};
pub use gv::gv;
pub use struct_factor::struct_factor;
pub use zhadamard::zhadamard;
