//! pyscf-ao2mo: AO→MO integral transformation (general / full).
//!
//! Source: D-01/D-02 — Phase 5 introduces this crate as the 20th `pyscf-*`
//! workspace member. Mirrors upstream `pyscf/ao2mo/` (incore.py / addons.py /
//! the `general`/`full` 4-index quarter-transform entry points). The crate is
//! deliberately CCSD-reusable (D-01): MP2 and CCSD both consume the AO→MO
//! transform through this single surface.
//!
//! Algebra/pyo3 wall (D-01/D-03): this crate depends ONLY on
//! `pyscf-core`/`pyscf-algebra`/`pyscf-gto` — NO `pyo3`, NO `cubecl`. The
//! transform bodies are filled in plan 05-02; this plan ships compiling
//! skeletons so later waves have concrete targets.
#![forbid(unsafe_code)]
#![warn(clippy::unwrap_used)]

pub mod error;
pub mod incore;
pub mod transform;

pub use error::Ao2moError;
pub use incore::{full, general};
