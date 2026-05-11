//! pyscf-py: PyO3 wheel surface (BIND-01/02/04/05/06/07/09).
//!
//! Module structure:
//!   _native (root cdylib)
//!     ├── PyscfRsRuntimeError (create_exception! per BIND-09 + abi3 workaround)
//!     └── scf (submodule)
//!         ├── RHF (PyRHF — #[pyclass(subclass)])
//!         ├── UHF (PyUHF — #[pyclass(subclass)])
//!         ├── GHF (PyGHF — #[pyclass(subclass)])
//!         └── Scanner (PyScfScanner — Send+Sync closure wrapper, SCF-12)
//!
//! Algebra wall: pyscf-py is the ONLY workspace crate (besides pyscf-oracle
//! dev-deps) that depends on pyo3. The chemistry crates (pyscf-scf, -diis,
//! -df, -chkfile) stay pyo3-free; PyO3 dispatch happens here via the D-01
//! trait-callback bridge (`bridge::PyOverrideBridge`).
//!
//! NOTE: This crate is built as both `cdylib` (maturin produces the Python
//! `pyscf._native` extension module) and `rlib` (so integration tests can
//! name internal modules directly).
#![forbid(unsafe_code)]
#![warn(clippy::unwrap_used)]
#![allow(non_local_definitions)] // PyO3 macros emit non-local trait impls

pub mod bridge;
pub mod caches;
pub mod errors;
pub mod numpy_io;
pub mod scf;

use pyo3::prelude::*;

/// PyO3 entry point. The `_native` name MUST match the
/// `[tool.maturin] module-name = "pyscf._native"` value in `pyproject.toml`.
///
/// Module layout (BIND-02 contract — `from pyscf import scf` resolves to
/// `python/pyscf/scf/__init__.py` which re-exports `pyscf._native.scf.*`):
///   _native.PyscfRsRuntimeError    -- BIND-09 panic→exception
///   _native.scf.RHF / UHF / GHF    -- BIND-02 + SCF-01..03
///   _native.scf.Scanner            -- SCF-12 callable wrapper
#[pymodule]
fn _native(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // BIND-09 — register the panic→exception class.
    // Created via create_exception! in errors.rs — abi3-py310 workaround for
    // PyException subclassing (RESEARCH §Pattern 5). The Python overlay
    // `python/pyscf/__init__.py` grafts `.kind` + `.source_chain` attrs.
    m.add(
        "PyscfRsRuntimeError",
        py.get_type::<crate::errors::PyscfRsRuntimeError>(),
    )?;

    // BIND-02 — `scf` submodule containing PyRHF/PyUHF/PyGHF.
    let scf_mod = PyModule::new(py, "scf")?;
    crate::scf::register(py, &scf_mod)?;
    m.add_submodule(&scf_mod)?;

    Ok(())
}
