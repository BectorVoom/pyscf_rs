//! pyscf-pbc-scf: SCF (gamma) + KSCF/KRHF/KUHF/KROHF/KGHF, smearing, addons, stability, newton
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::unwrap_used)]

pub mod error;
pub use error::*;
