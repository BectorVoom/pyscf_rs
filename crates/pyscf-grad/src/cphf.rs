//! The single CPHF/CPKS solver (D-03, GRAD-10). Body lands in Wave 2 (07-09+).
//!
//! Caller-shape analog only: `crates/pyscf-algebra/src/solve_linear.rs:29`
//! (`solve_linear(a, b, n)` — but that is DENSE LU; NO matrix-free Krylov exists
//! in `pyscf-algebra`, so this module BUILDS the Krylov, routing its gemm/dot
//! through `pyscf-algebra`). Upstream port ref: `pyscf/scf/cphf.py:29-148`
//! (`solve`/`solve_nos1`/`solve_withs1`) + `pyscf/lib/linalg_helper.py:1221`
//! (`lib.krylov`, Pople 1979).
//!
//! GRAD-10 contract: ONE solver. Both MP2 (`max_cycle=30`) and CCSD pass their
//! own `fvind` + RHS into this single `solve`. The `single_cphf_impl` structural
//! CI assertion guards against a second CPHF implementation.
use crate::error::GradError;
use pyscf_core::PyscfRsError;

/// Upstream `cphf.solve` default max iterations (`cphf.py:29`). MP2 grad
/// overrides this to 30 (Pitfall 5).
pub const DEFAULT_MAX_CYCLE: usize = 50;

/// Upstream `cphf.solve` default convergence tolerance (`cphf.py:30`).
pub const DEFAULT_TOL: f64 = 1e-9;

/// The single matrix-free CPHF/CPKS solver. `NotYetImplemented { wave: 2 }`
/// until 07-09 — the body ports `lib.krylov` (Pople 1979) and routes every
/// gemm/dot through `pyscf-algebra` (NO dense A-matrix — O(nocc²nvir²) memory
/// anti-pattern).
pub fn solve() -> Result<Vec<f64>, PyscfRsError> {
    Err(GradError::NotYetImplemented { wave: 2 }.into())
}
