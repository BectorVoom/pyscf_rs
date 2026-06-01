//! pyscf-py: PyO3 wheel surface (BIND-01/02/04/05/06/07/09).
//!
//! Module structure:
//!   _native (root cdylib)
//!     ├── PyscfRsRuntimeError (create_exception! per BIND-09 + abi3 workaround)
//!     ├── scf (submodule)
//!     │   ├── RHF (PyRHF — #[pyclass(subclass)])
//!     │   ├── UHF (PyUHF — #[pyclass(subclass)])
//!     │   ├── GHF (PyGHF — #[pyclass(subclass)])
//!     │   └── Scanner (PyScfScanner — Send+Sync closure wrapper, SCF-12)
//!     └── dft (submodule, plan 04-09 — DFT-08 / D-08)
//!         ├── RKS (PyRKS — #[pyclass(subclass)])
//!         ├── UKS (PyUKS — #[pyclass(subclass)])
//!         └── NumInt (PyNumIntView — read-only precision view, D-08)
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
pub mod cc;
pub mod dft;
pub mod errors;
pub mod geomopt;
pub mod grad;
pub mod gto;
pub mod mp;
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

    // F-11 — arm the pyscf-core IoC builder hook so a direct
    // `pyscf_core::Mole::build()` works across the whole Python surface (the
    // `mol.copy(); mol.basis = aux; mol.build()` rebuild pattern). FOUND-02
    // stays intact: pyscf-core holds only a `fn(&mut Mole)` pointer.
    pyscf_gto::register_mole_builder();

    // BIND-02 — `scf` submodule containing PyRHF/PyUHF/PyGHF.
    let scf_mod = PyModule::new(py, "scf")?;
    crate::scf::register(py, &scf_mod)?;
    m.add_submodule(&scf_mod)?;

    // GTO overlay - minimal native Mole/M surface used by the Python SCF tests.
    let gto_mod = PyModule::new(py, "gto")?;
    crate::gto::register(py, &gto_mod)?;
    m.add_submodule(&gto_mod)?;

    // Plan 04-09 (DFT-08 / D-08) — `dft` submodule containing PyRKS/PyUKS.
    let dft_mod = PyModule::new(py, "dft")?;
    crate::dft::register(py, &dft_mod)?;
    m.add_submodule(&dft_mod)?;

    // Plan 05-07 (MP2 / D-07/D-08) — `mp` submodule containing
    // PyRMP2/PyUMP2/PyDFMP2 + the MP2() factory + PyMp2Scanner. The Python
    // overlay `python/pyscf/mp/__init__.py` re-exports `pyscf._native.mp.*`.
    let mp_mod = PyModule::new(py, "mp")?;
    crate::mp::register(py, &mp_mod)?;
    m.add_submodule(&mp_mod)?;

    // Plan 06-10 (CCSD / D-09) — `cc` submodule containing
    // PyRCCSD/PyUCCSD/PyDFCCSD + the CCSD() factory + PyCcsdScanner. The Python
    // overlay `python/pyscf/cc/__init__.py` re-exports `pyscf._native.cc.*` and
    // grafts the `mf.CCSD()` / `mf.density_fit().CCSD()` cross-module dispatch.
    let cc_mod = PyModule::new(py, "cc")?;
    crate::cc::register(py, &cc_mod)?;
    m.add_submodule(&cc_mod)?;

    // Plan 07-09 (gradients / D-07/D-09) — `grad` submodule containing
    // PyRhfGradients/PyUhfGradients/PyRksGradients/PyUksGradients/
    // PyMp2Gradients/PyCcsdGradients + the Gradients() factory + PyGradScanner
    // (the Mole -> (e_tot, de) geomopt seam). The Python overlay
    // `python/pyscf/grad/__init__.py` re-exports `pyscf._native.grad.*` and
    // grafts `mf.nuc_grad_method()` onto the Rust SCF classes (scf/hf.py:2484).
    let grad_mod = PyModule::new(py, "grad")?;
    crate::grad::register(py, &grad_mod)?;
    m.add_submodule(&grad_mod)?;

    // Plan 07-09 (geomopt / GEOMOPT-01/02/03 / D-06/D-07) — `geomopt` submodule
    // containing `optimize` + the `geometric_solver`/`berny_solver` nested shim
    // submodules, all delegating to the ONE native pyscf_geomopt engine (NO
    // geometric/pyberny runtime dep, GEOMOPT-01). The Python overlay
    // `python/pyscf/geomopt/__init__.py` re-exports `pyscf._native.geomopt.*`.
    let geomopt_mod = PyModule::new(py, "geomopt")?;
    crate::geomopt::register(py, &geomopt_mod)?;
    m.add_submodule(&geomopt_mod)?;

    Ok(())
}
