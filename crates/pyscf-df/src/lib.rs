//! pyscf-df: density-fitting (DF) integrals.
//!
//! Source: D-10 — Phase 3 introduces this crate. Mirrors upstream `pyscf/df/`
//! (df.py / df_jk.py / incore.py / addons.py). Owns 3-center integrals via
//! `pyscf_gto::intor_with_auxmol("int3c2e_sph")` (plan 03-05 Task 0), 2-center
//! integrals via `pyscf_gto::intor_with_auxmol("int2c2e_sph")`, Cholesky of
//! `(P|Q)` (inline host-row-major Cholesky-Banachiewicz — see cholesky_eri.rs
//! module-level docs for the Tensor-API delegation rationale), and B-integral
//! assembly. Public surface: `DfIntegrals { b_uvq, naux, nao }` consumed
//! uniformly by SCF / DFT / MP2 / CCSD.
//!
//! Phase 3 ships in-memory only; HDF5 spill deferred to Phase 6 (D-11).
//! Plan 03-05 fills the body shipped as empty skeleton by plan 03-01.
#![forbid(unsafe_code)]
#![warn(clippy::unwrap_used)]

pub mod auxbasis;
pub mod cholesky_eri;
pub mod df_jk;
pub mod error;

pub use auxbasis::{DEFAULT_AUXBASIS, default_jkfit, default_ri};
pub use cholesky_eri::{DfIntegrals, cholesky_eri};
pub use df_jk::get_jk_df;
pub use error::DfError;
