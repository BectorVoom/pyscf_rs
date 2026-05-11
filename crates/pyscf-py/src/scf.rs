//! `scf` submodule — PyRHF / PyUHF / PyGHF + density_fit factory.
//!
//! Plan 03-07 Task 1 ships placeholder pyclasses so the cdylib registers a
//! `_native.scf` submodule that imports cleanly. Task 2 fills the
//! 30-attribute getters/setters + kernel + 10 hook defaults with per-hook
//! `py.detach` wrappers (D-03, BIND-05).
use pyo3::prelude::*;

use pyscf_core::Mole;
use pyscf_scf::{GHF as GhfRust, RHF as RhfRust, UHF as UhfRust};

use crate::bridge::extract_mole_from_pyany;

/// Register PyRHF / PyUHF / PyGHF in the `_native.scf` submodule (BIND-02).
pub(crate) fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRHF>()?;
    m.add_class::<PyUHF>()?;
    m.add_class::<PyGHF>()?;
    Ok(())
}

/// Python-facing RHF (subclass-able per Pitfall 7 / BIND-07).
///
/// 30-attribute floor (SCF-14): mirrors `pyscf_scf::RHF` field-for-field;
/// each attribute is exposed via a `#[getter]` (+ `#[setter]` where the
/// field is user-mutable).
#[pyclass(subclass, name = "RHF", module = "pyscf._native.scf")]
pub struct PyRHF {
    pub(crate) inner: RhfRust,
    /// Reference to the user-supplied Python Mole object. Kept on the
    /// pyclass so the PyOverrideBridge can re-use it without re-serialising
    /// per hook call (saves O(N) JSON round-trips inside the SCF loop).
    pub(crate) py_mol: Py<PyAny>,
}

#[pymethods]
impl PyRHF {
    #[new]
    fn new(py: Python<'_>, mol: Py<PyAny>) -> PyResult<Self> {
        let mol_inner: Mole = extract_mole_from_pyany(py, &mol)?;
        Ok(PyRHF {
            inner: RhfRust::new(mol_inner),
            py_mol: mol,
        })
    }

    // ---- 30-attribute floor (SCF-14) — Task 2 fills the body. ----
    // Task 1 ships read-only e_tot and converged so the smoke test can
    // verify field reachability after `new(mol)`.

    #[getter]
    fn e_tot(&self) -> f64 {
        self.inner.e_tot
    }

    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }
}

#[pyclass(subclass, name = "UHF", module = "pyscf._native.scf")]
pub struct PyUHF {
    pub(crate) inner: UhfRust,
    pub(crate) py_mol: Py<PyAny>,
}

#[pymethods]
impl PyUHF {
    #[new]
    fn new(py: Python<'_>, mol: Py<PyAny>) -> PyResult<Self> {
        let mol_inner: Mole = extract_mole_from_pyany(py, &mol)?;
        Ok(PyUHF {
            inner: UhfRust::new(mol_inner),
            py_mol: mol,
        })
    }

    #[getter]
    fn e_tot(&self) -> f64 {
        self.inner.e_tot
    }
}

#[pyclass(subclass, name = "GHF", module = "pyscf._native.scf")]
pub struct PyGHF {
    pub(crate) inner: GhfRust,
    pub(crate) py_mol: Py<PyAny>,
}

#[pymethods]
impl PyGHF {
    #[new]
    fn new(py: Python<'_>, mol: Py<PyAny>) -> PyResult<Self> {
        let mol_inner: Mole = extract_mole_from_pyany(py, &mol)?;
        Ok(PyGHF {
            inner: GhfRust::new(mol_inner),
            py_mol: mol,
        })
    }

    #[getter]
    fn e_tot(&self) -> f64 {
        self.inner.e_tot
    }
}
