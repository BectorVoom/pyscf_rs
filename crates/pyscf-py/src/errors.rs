//! Panic / Rust error → Python exception bridge (BIND-09).
//!
//! CRITICAL: abi3-py310 forbids `#[pyclass(extends=PyException)]` until
//! Python 3.12. Workaround per RESEARCH §Pattern 5 + xcfun-py errors.rs:
//! `create_exception!(_native, PyscfRsRuntimeError, PyException)` creates a
//! bare PyException-subclass at the C level. The Python overlay shim
//! (`python/pyscf/__init__.py`) grafts `.kind` + `.source_chain` attrs onto
//! the resulting exception.
//!
//! Source: `~/Documents/workspace/xcfun_rs/crates/xcfun-py/src/errors.rs:31-83`.
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyscf_core::{CoreError, PyscfRsError};

// The bare PyException-subclass at the C level. First arg `_native` matches
// the cdylib module name; PyO3 uses it to set `__module__` so `repr(e)`
// shows `pyscf._native.PyscfRsRuntimeError`.
create_exception!(_native, PyscfRsRuntimeError, PyException);

/// Convert a pyscf-core Rust error into a `PyErr` preserving variant kind
/// and the underlying error-source chain.
///
/// The exception is built with positional args `(message, kind, source_chain)`
/// so the Python `PyscfRsError` overlay subclass can unpack them onto
/// `.kind: str` and `.source_chain: list[str]` attributes.
pub fn pyscf_to_py(err: PyscfRsError) -> PyErr {
    let kind = error_kind(&err);
    let msg = format!("{}", err);
    let source_chain = collect_source_chain(&err);
    PyscfRsRuntimeError::new_err((msg, kind, source_chain))
}

/// Convert a Python error raised inside a subclass-override callback back
/// into a pyscf-core `PyscfRsError` so the SCF kernel can `?` propagate it.
///
/// Routes through `CoreError::InvalidMolecule(String)` (the only
/// String-carrying catch-all variant on `CoreError` — same convention as
/// plans 03-03, 03-04, 03-05, 03-06).
pub fn py_to_pyscf(err: PyErr) -> PyscfRsError {
    PyscfRsError::Core(CoreError::InvalidMolecule(format!(
        "python override raised: {}",
        err
    )))
}

/// Map a `PyscfRsError` variant to a short kind string usable from Python.
fn error_kind(err: &PyscfRsError) -> &'static str {
    match err {
        PyscfRsError::Core(_) => "Core",
        PyscfRsError::NotYetImplemented { .. } => "NotYetImplemented",
        PyscfRsError::ConvergenceFailure { .. } => "ConvergenceFailure",
        PyscfRsError::BasisLoad(_) => "BasisLoad",
        PyscfRsError::EcpLoad(_) => "EcpLoad",
        PyscfRsError::EcpEngineNotAvailable => "EcpEngineNotAvailable",
        PyscfRsError::NumericOverflow { .. } => "NumericOverflow",
    }
}

fn collect_source_chain(err: &dyn std::error::Error) -> Vec<String> {
    let mut chain = Vec::new();
    let mut cur: Option<&dyn std::error::Error> = err.source();
    while let Some(e) = cur {
        chain.push(format!("{}", e));
        cur = e.source();
    }
    chain
}
