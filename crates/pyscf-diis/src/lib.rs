//! pyscf-diis: generic Pulay extrapolation (CDIIS).
//!
//! Source: D-08 — Phase 3 introduces this crate. Generic over a `DiisStorable`
//! trait so SCF (FockSubspace) and Phase 6 CCSD (AmpsSubspace) share the same
//! Pulay machinery. Mirrors upstream `pyscf/scf/diis.py:40-87` (CDIIS class +
//! `get_err_vec_orig` SDF - FDS error vector).
//!
//! Pitfall 9 mitigation: all reductions go through pyscf-algebra::oracle_dot /
//! oracle_sum under `release-oracle` for bit-exact cross-platform DIIS.
//!
//! Phase 3 status: empty skeleton (Plan 03-01). Plan 03-04 fills.
#![forbid(unsafe_code)]
#![warn(clippy::unwrap_used)]
