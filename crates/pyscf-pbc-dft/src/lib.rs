//! pyscf-pbc-dft: gen_grid, numint, KRKS/KUKS/KROKS/KGKS + gamma, DFT+U, multigrid, cdft
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::unwrap_used)]

pub mod error;
pub use error::*;
