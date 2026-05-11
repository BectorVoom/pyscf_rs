//! PyOverrideBridge — D-01 trait-callback bridge.
//!
//! When a Python user subclasses `pyscf.scf.RHF` and overrides one of the
//! 10 SCF hooks, the Rust kernel must dispatch the override via the Python
//! C-API (`call_method1`). This module ships the `PyOverrideBridge` struct
//! which implements `pyscf_scf::OverrideHooks`; every hook method packages
//! its Rust arguments into Python objects, calls `slf.<hook>(...)`, and
//! converts the result back into Rust types.
//!
//! Source: RESEARCH §Pattern 1 lines 343-368; plan 03-07 Task 2 fills the
//! call_method1 dispatch bodies. Task 1 ships the struct with NoOverrides-
//! delegating bodies so the file compiles end-to-end against the SCF-08
//! trait surface.
//!
//! Pitfall 7 (subclass fidelity): every hook must route through
//! `slf.call_method1("hook_name", args)` so user-defined Python overrides
//! are invoked even when the Rust-side `PyRHF::<hook>` default exists.
//! `call_method1` resolves up the MRO, so subclass methods are found
//! transparently.
use pyo3::prelude::*;
use pyscf_core::{Density, Energy, MOCoefficients, Mole, PyscfRsError};
use pyscf_scf::{InitGuessMode, NoOverrides, OverrideHooks};

/// Bridge that owns a `Py<PyAny>` handle to the Python `self` and a
/// `Py<PyAny>` handle to the user-supplied Mole. The Rust SCF kernel calls
/// `bridge.<hook>(...)`; the impl re-attaches the GIL and dispatches via
/// `slf.call_method1`.
///
/// `Send + Sync` is NOT required: SCF kernel calls happen on the thread
/// that holds the GIL; the bridge is constructed inside a `PyRHF::kernel`
/// body and dropped before the GIL is released.
///
/// Plan 03-07 Task 1 ships the struct with no-op delegation to
/// `NoOverrides`; plan 03-07 Task 2 fills the call_method1 dispatch.
pub struct PyOverrideBridge {
    pub slf: Py<PyAny>,
    pub py_mol: Py<PyAny>,
}

impl PyOverrideBridge {
    /// Construct the bridge from a Python `self` handle + the user-supplied
    /// Mole handle. Both handles are reference-counted; the bridge owns them
    /// for the duration of one `PyRHF::kernel` call.
    pub fn new(slf: Py<PyAny>, py_mol: Py<PyAny>) -> Self {
        Self { slf, py_mol }
    }
}

impl OverrideHooks for PyOverrideBridge {
    // Plan 03-07 Task 1 stubs — delegate every hook to NoOverrides so the
    // pyscf-scf kernel can drive a no-override run end-to-end.
    //
    // Plan 03-07 Task 2 replaces each body with the `Python::attach(|py| {
    // let result = self.slf.bind(py).call_method1("hook_name", args)?;
    // ... })` pattern, preserving the BIND-07 subclass-override contract.
    fn get_hcore(&self, mol: &Mole) -> Result<Density, PyscfRsError> {
        NoOverrides.get_hcore(mol)
    }
    fn get_ovlp(&self, mol: &Mole) -> Result<Density, PyscfRsError> {
        NoOverrides.get_ovlp(mol)
    }
    fn get_init_guess(
        &self,
        mol: &Mole,
        mode: &InitGuessMode,
    ) -> Result<Density, PyscfRsError> {
        NoOverrides.get_init_guess(mol, mode)
    }
    fn get_jk(
        &self,
        mol: &Mole,
        dm: &Density,
    ) -> Result<(Density, Density), PyscfRsError> {
        NoOverrides.get_jk(mol, dm)
    }
    fn get_veff(&self, mol: &Mole, dm: &Density) -> Result<Density, PyscfRsError> {
        NoOverrides.get_veff(mol, dm)
    }
    fn get_fock(
        &self,
        h1e: &Density,
        s1e: &Density,
        vhf: &Density,
        dm: &Density,
        cycle: i32,
        diis_state: Option<&Density>,
    ) -> Result<Density, PyscfRsError> {
        NoOverrides.get_fock(h1e, s1e, vhf, dm, cycle, diis_state)
    }
    fn eig(
        &self,
        fock: &Density,
        s1e: &Density,
    ) -> Result<MOCoefficients, PyscfRsError> {
        NoOverrides.eig(fock, s1e)
    }
    fn get_occ(
        &self,
        mo_energy: &[f64],
        nelec: usize,
    ) -> Result<Vec<f64>, PyscfRsError> {
        NoOverrides.get_occ(mo_energy, nelec)
    }
    fn make_rdm1(&self, mo: &MOCoefficients) -> Result<Density, PyscfRsError> {
        NoOverrides.make_rdm1(mo)
    }
    fn energy_elec(
        &self,
        dm: &Density,
        h1e: &Density,
        vhf: &Density,
    ) -> Result<(Energy, Energy), PyscfRsError> {
        NoOverrides.energy_elec(dm, h1e, vhf)
    }
    fn energy_tot(
        &self,
        dm: &Density,
        h1e: &Density,
        vhf: &Density,
    ) -> Result<Energy, PyscfRsError> {
        NoOverrides.energy_tot(dm, h1e, vhf)
    }
}

/// Extract a pyscf-core `Mole` from any Python object.
///
/// Two paths, tried in order:
///   1. Python object has a `dumps()` method (upstream PySCF Mole) — call it
///      and pass through `pyscf_gto::loads(json)`.
///   2. Python object is a JSON string already — pass to `pyscf_gto::loads`.
///
/// Phase 2 D-08: `dumps()` / `loads()` is the canonical Mole interop seam.
/// This helper centralises the conversion so PyRHF::new / kernel / hook
/// defaults all share one extraction implementation.
pub fn extract_mole_from_pyany(py: Python<'_>, mol: &Py<PyAny>) -> PyResult<Mole> {
    let bound = mol.bind(py);
    // Path 1: object with .dumps() method (upstream pyscf Mole OR a pyscf-rs
    // wrapper that exposes the same interop method).
    if bound.hasattr("dumps")? {
        let json: String = bound.call_method0("dumps")?.extract()?;
        return pyscf_gto::loads(&json).map_err(crate::errors::pyscf_to_py);
    }
    // Path 2: a raw JSON string.
    if let Ok(s) = bound.extract::<String>() {
        return pyscf_gto::loads(&s).map_err(crate::errors::pyscf_to_py);
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "expected pyscf.gto.Mole (with .dumps() method) or a serialized Mole JSON string",
    ))
}
