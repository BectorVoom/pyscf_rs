//! pyscf-oracle: PySCF live-oracle harness (ORACLE-01, dev-deps only).
//!
//! Macro shape: `oracle_check!("method_name", fixture, tolerance)`. Expands
//! to `runner::run_oracle_check(method, fixture, tolerance)`, which spawns
//! Python via `Python::attach`, builds both upstream and pyscf-rs results,
//! and asserts equality within `tolerance`.
//!
//! Covers ORACLE-02 (oracle_check macro — all 8 arms shipped per ROADMAP
//! success criterion 6) and ORACLE-08 (chkfile round-trip empirical seal —
//! STATE.md Blockers/Concerns line 90).
//!
//! Build-mode notes:
//!   - default features (no `python`): `cargo build -p pyscf-oracle`
//!     succeeds without a libpython.so install. The macro is invokable;
//!     each call returns `OracleError::PythonFeatureNotEnabled` at runtime
//!     (the `oracle_check!` macro turns that into a panic with that
//!     message, surfacing as a test failure rather than a silent skip).
//!   - `--features python`: enables the pyo3 `auto-initialize` runtime
//!     and exposes the 8 oracle arms. CI runners with libpython-dev +
//!     upstream `pyscf` installed opt in.
//!
//! ORACLE-01 ("release wheels never link Python") is preserved because
//! pyscf-oracle is never a normal dependency of pyscf-py (the wheel crate).
//! It only lives in dev-deps of consumer crates' tests.
#![forbid(unsafe_code)]
#![warn(clippy::unwrap_used)]

pub mod fixtures;
pub mod runner;

pub use fixtures::{BENZENE_6_31GS, H2O_CC_PVDZ, H2O_TRIPLET_CCPVDZ, WATER_TRIMER_CC_PVDZ};
pub use runner::{run_oracle_check, OracleError};

/// THE macro. Used in tests across pyscf-scf / pyscf-df / pyscf-chkfile /
/// pyscf-py and (in plan 03-10) inside Python pytest harness fixtures.
///
/// `($method:literal, $fixture:expr, $tolerance:expr)` — `$tolerance` is
/// bound to `f64` at the macro site so wrong types fail here, not deep in
/// `run_oracle_check`. `$fixture` is bound to `&str` for the same reason.
///
/// On failure: panics with a message listing pyscf-rs value, upstream
/// value, diff (via `OracleError::Display`).
/// On unknown method: panics with `"oracle: unknown method '<name>'"`.
/// On missing `python` feature: panics with `OracleError::PythonFeatureNotEnabled`.
#[macro_export]
macro_rules! oracle_check {
    ($method:literal, $fixture:expr, $tolerance:expr) => {{
        let fixture: &str = $fixture;
        let tolerance: f64 = $tolerance;
        match $crate::run_oracle_check($method, fixture, tolerance) {
            Ok(()) => {}
            Err(e) => panic!("oracle_check failed: {}", e),
        }
    }};
}
