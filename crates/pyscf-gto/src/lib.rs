//! pyscf-gto — Molecular structure & integrals.
//!
//! Phase 2 fills the bodies for GTO-01..11. Wave 0 (this commit, plan 02-01)
//! only lays out modules and imports cintx to prove reachability.

#![forbid(unsafe_code)]

pub mod layout_table; // Wave 0 (this plan); consumed by intor.rs in 02-05
// Plans 02-02..02-08 add: format_atom, format_basis, basis (mod), make_env,
// intor, eval_gto, ecp_engine_stub, dumps_loads
