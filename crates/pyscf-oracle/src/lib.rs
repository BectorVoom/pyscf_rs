//! pyscf-oracle: PySCF live-oracle harness (ORACLE-01, dev-deps only).
//!
//! Phase 1 declared `pyo3 = "=0.28.3"` with `auto-initialize` in dev-deps so
//! release wheels never link Python. Phase 3 plan 03-02 ships the macro stub;
//! plan 03-08 fills the body (chkfile round-trip + every SCF success criterion).
//!
//! Macro shape (final, per RESEARCH §Pattern 11):
//!   oracle_check!("scf_rhf_energy", H2O_CC_PVDZ, 1e-6)
//! → spawns Python via `Python::attach`, drives upstream PySCF, compares
//!   pyscf-rs to upstream within `tolerance`. Returns `Result<(), OracleError>`.
#![forbid(unsafe_code)]
#![warn(clippy::unwrap_used)]

pub mod fixtures;

pub use fixtures::{BENZENE_6_31GS, H2O_CC_PVDZ, H2O_TRIPLET_CCPVDZ, WATER_TRIMER_CC_PVDZ};

/// Stub macro — plan 03-08 fills the body. Call sites compile; runtime panics
/// with a clear "pending" message so test failures point to the right plan.
///
/// Bind the inputs to `let`-bindings of the documented types so callers get
/// helpful compile errors today (e.g., a non-`f64` tolerance fails right here
/// rather than at the eventual real call site).
#[macro_export]
macro_rules! oracle_check {
    ($method:literal, $fixture:expr, $tolerance:expr) => {{
        let _fixture = &$fixture;
        let _tolerance: f64 = $tolerance;
        panic!(
            "oracle_check!({:?}, _, {}) — body not yet implemented (plan 03-08)",
            $method, _tolerance
        );
    }};
}
