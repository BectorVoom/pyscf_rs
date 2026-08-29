//! pyscf-pbc-dft: gen_grid, numint, KRKS/KUKS/KROKS/KGKS + gamma, DFT+U, multigrid, cdft
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::unwrap_used)]

pub mod error;
pub use error::*;

pub mod xc;
pub mod gen_grid;
pub mod numint;
pub mod numint2c;
pub mod veff;
pub mod krks;
pub mod kuks;
pub mod kroks;
pub mod kgks;
pub mod gamma;
pub mod kspu;
pub mod cdft;
