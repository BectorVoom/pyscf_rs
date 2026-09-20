//! pyscf-pbc-geomopt: periodic geometry optimization (plan 18-14).
//!
//! The atom-coordinate optimizer over `pyscf-geomopt`'s native BFGS+RFO
//! engine — `pyscf/pbc/geomopt/geometric_solver.py` (246 l) plus the
//! `pbc.geomopt.optimize` entry point (`__init__.py:18-23`). No lattice
//! degrees of freedom: upstream has none, and D-PBC-15 forbids inventing
//! them (`18-CONTEXT §1.7`).
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::unwrap_used)]

pub mod error;
pub mod geometric_solver;
pub use error::*;
pub use geometric_solver::{GeometryOptimizer, KernelInput, OptimizeOpts, kernel, optimize};
