//! PyOverrideBridge — D-01 trait-callback bridge.
//!
//! When a Python user subclasses `pyscf.scf.RHF` and overrides one of the
//! 10 SCF hooks, the Rust kernel must dispatch the override via the Python
//! C-API (`call_method1`). This module ships the `PyOverrideBridge` struct
//! which implements `pyscf_scf::OverrideHooks`; every hook method packages
//! its Rust arguments into Python objects, calls `slf.<hook>(...)`, and
//! converts the result back into Rust types.
//!
//! BIND-07 / Pitfall 7 (subclass fidelity): every hook routes through
//! `Bound::call_method1("hook_name", args)`. `call_method1` resolves up
//! the Python MRO, so subclass methods are found transparently — when no
//! subclass override exists, the PyRHF default (defined in `scf.rs` and
//! wrapping `pyscf_scf::default_*`) is invoked. When a subclass overrides
//! the method, that override is invoked instead. Either way, the Rust
//! kernel sees the SCF-08 trait surface.
//!
//! `Send + Sync` is NOT required: SCF kernel calls happen on the thread
//! that holds the GIL; the bridge is constructed inside `PyRHF::kernel`
//! and dropped at end-of-kernel.
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyscf_core::{Density, Energy, MOCoefficients, Mole, PyscfRsError};
use pyscf_dft::{KsOverrideHooks, KsVeff, NumInt, XcSpec};
use pyscf_grids::Grids;
use pyscf_scf::{InitGuessMode, OverrideHooks};

use crate::errors::{py_to_pyscf, pyscf_to_py};
use crate::numpy_io::{density_to_pyarray, mo_coeff_to_pyarray, to_density, to_mo_coeff};

/// Bridge owning a `Py<PyAny>` handle to the Python `self` (the
/// `PyRHF`-or-subclass instance) and a `Py<PyAny>` handle to the
/// user-supplied Python Mole. Constructed in `PyRHF::kernel`; dropped at
/// kernel end.
pub struct PyOverrideBridge {
    /// Python `self` of the PyRHF (or subclass). `call_method1` against
    /// this handle resolves the MRO so subclass overrides are invoked
    /// transparently.
    pub slf: Py<PyAny>,
    /// The user-supplied Python Mole object — re-used as the `mol` arg in
    /// every hook call so we don't serialize/deserialize per-cycle.
    pub py_mol: Py<PyAny>,
}

impl PyOverrideBridge {
    pub fn new(slf: Py<PyAny>, py_mol: Py<PyAny>) -> Self {
        Self { slf, py_mol }
    }
}

// -----------------------------------------------------------------------
// Helper: dispatch a hook via slf.call_method1 and return a typed result.
// -----------------------------------------------------------------------

/// Call `slf.<method_name>(*args)` and convert any `PyErr` into
/// `PyscfRsError` so the SCF kernel can `?` propagate it.
fn call_hook<'py, F, T>(
    slf: &Py<PyAny>,
    method: &str,
    args: Bound<'py, PyTuple>,
    extract: F,
) -> Result<T, PyscfRsError>
where
    F: FnOnce(Bound<'py, PyAny>) -> PyResult<T>,
{
    let py = args.py();
    let result = slf
        .bind(py)
        .call_method1(method, args)
        .map_err(py_to_pyscf)?;
    extract(result).map_err(py_to_pyscf)
}

impl OverrideHooks for PyOverrideBridge {
    fn get_hcore(&self, _mol: &Mole) -> Result<Density, PyscfRsError> {
        Python::attach(|py| {
            let args = PyTuple::new(py, [self.py_mol.bind(py).clone()]).map_err(py_to_pyscf)?;
            call_hook(&self.slf, "get_hcore", args, |r| {
                let arr: numpy::PyReadonlyArray2<f64> = r.extract()?;
                to_density(arr)
            })
        })
    }

    fn get_ovlp(&self, _mol: &Mole) -> Result<Density, PyscfRsError> {
        Python::attach(|py| {
            let args = PyTuple::new(py, [self.py_mol.bind(py).clone()]).map_err(py_to_pyscf)?;
            call_hook(&self.slf, "get_ovlp", args, |r| {
                let arr: numpy::PyReadonlyArray2<f64> = r.extract()?;
                to_density(arr)
            })
        })
    }

    fn get_init_guess(
        &self,
        _mol: &Mole,
        mode: &InitGuessMode,
    ) -> Result<Density, PyscfRsError> {
        // BIND-07 / Pitfall 7: dispatch through `slf.call_method1` so
        // a Python subclass override of `get_init_guess(mol, key)` is
        // invoked transparently via MRO resolution. The pyscf upstream
        // surface is `SCF.get_init_guess(mol, key='minao')`.
        Python::attach(|py| {
            let mode_str = match mode {
                InitGuessMode::Minao => "minao",
                InitGuessMode::Atom => "atom",
                InitGuessMode::OneElectron => "1e",
                InitGuessMode::Huckel => "huckel",
                InitGuessMode::Chkfile(_) => "chkfile",
                InitGuessMode::UserDM(d) => {
                    // UserDM is already a density — no Python round-trip needed.
                    return Ok(d.clone());
                }
            };
            let args = PyTuple::new(
                py,
                [
                    self.py_mol.bind(py).clone(),
                    pyo3::types::PyString::new(py, mode_str).into_any(),
                ],
            )
            .map_err(py_to_pyscf)?;
            call_hook(&self.slf, "get_init_guess", args, |r| {
                let arr: numpy::PyReadonlyArray2<f64> = r.extract()?;
                to_density(arr)
            })
        })
    }

    fn get_jk(
        &self,
        _mol: &Mole,
        dm: &Density,
    ) -> Result<(Density, Density), PyscfRsError> {
        Python::attach(|py| {
            let dm_py = density_to_pyarray(py, dm).map_err(py_to_pyscf)?;
            let args = PyTuple::new(
                py,
                [self.py_mol.bind(py).clone(), dm_py.into_any()],
            )
            .map_err(py_to_pyscf)?;
            call_hook(&self.slf, "get_jk", args, |r| {
                let (j_any, k_any): (Bound<'_, PyAny>, Bound<'_, PyAny>) = r.extract()?;
                let j_arr: numpy::PyReadonlyArray2<f64> = j_any.extract()?;
                let k_arr: numpy::PyReadonlyArray2<f64> = k_any.extract()?;
                Ok((to_density(j_arr)?, to_density(k_arr)?))
            })
        })
    }

    fn get_veff(&self, _mol: &Mole, dm: &Density) -> Result<Density, PyscfRsError> {
        Python::attach(|py| {
            let dm_py = density_to_pyarray(py, dm).map_err(py_to_pyscf)?;
            let args = PyTuple::new(
                py,
                [self.py_mol.bind(py).clone(), dm_py.into_any()],
            )
            .map_err(py_to_pyscf)?;
            call_hook(&self.slf, "get_veff", args, |r| {
                let arr: numpy::PyReadonlyArray2<f64> = r.extract()?;
                to_density(arr)
            })
        })
    }

    fn get_fock(
        &self,
        h1e: &Density,
        s1e: &Density,
        vhf: &Density,
        dm: &Density,
        cycle: i32,
        _diis_state: Option<&Density>,
    ) -> Result<Density, PyscfRsError> {
        Python::attach(|py| {
            let h1e_py = density_to_pyarray(py, h1e).map_err(py_to_pyscf)?;
            let s1e_py = density_to_pyarray(py, s1e).map_err(py_to_pyscf)?;
            let vhf_py = density_to_pyarray(py, vhf).map_err(py_to_pyscf)?;
            let dm_py = density_to_pyarray(py, dm).map_err(py_to_pyscf)?;
            let cycle_py = pyo3::types::PyInt::new(py, cycle as i64).into_any();
            let args = PyTuple::new(
                py,
                [
                    h1e_py.into_any(),
                    s1e_py.into_any(),
                    vhf_py.into_any(),
                    dm_py.into_any(),
                    cycle_py,
                ],
            )
            .map_err(py_to_pyscf)?;
            call_hook(&self.slf, "get_fock", args, |r| {
                let arr: numpy::PyReadonlyArray2<f64> = r.extract()?;
                to_density(arr)
            })
        })
    }

    fn eig(
        &self,
        fock: &Density,
        s1e: &Density,
    ) -> Result<MOCoefficients, PyscfRsError> {
        Python::attach(|py| {
            let fock_py = density_to_pyarray(py, fock).map_err(py_to_pyscf)?;
            let s1e_py = density_to_pyarray(py, s1e).map_err(py_to_pyscf)?;
            let args = PyTuple::new(py, [fock_py.into_any(), s1e_py.into_any()])
                .map_err(py_to_pyscf)?;
            call_hook(&self.slf, "eig", args, |r| {
                // Result is (mo_energy, mo_coeff): (np.ndarray[1d], np.ndarray[2d]).
                let (e_any, c_any): (Bound<'_, PyAny>, Bound<'_, PyAny>) = r.extract()?;
                let e_arr: numpy::PyReadonlyArray1<f64> = e_any.extract()?;
                let c_arr: numpy::PyReadonlyArray2<f64> = c_any.extract()?;
                let mut mc = to_mo_coeff(c_arr)?;
                mc.energies = e_arr.as_slice()?.to_vec();
                Ok(mc)
            })
        })
    }

    fn get_occ(
        &self,
        mo_energy: &[f64],
        _nelec: usize,
    ) -> Result<Vec<f64>, PyscfRsError> {
        Python::attach(|py| {
            let e_py = numpy::PyArray1::from_slice(py, mo_energy);
            let args = PyTuple::new(py, [e_py.into_any()]).map_err(py_to_pyscf)?;
            call_hook(&self.slf, "get_occ", args, |r| {
                let arr: numpy::PyReadonlyArray1<f64> = r.extract()?;
                Ok(arr.as_slice()?.to_vec())
            })
        })
    }

    fn make_rdm1(&self, mo: &MOCoefficients) -> Result<Density, PyscfRsError> {
        Python::attach(|py| {
            let c_py = mo_coeff_to_pyarray(py, mo).map_err(py_to_pyscf)?;
            let occ_py = numpy::PyArray1::from_slice(py, &mo.occupations);
            let args = PyTuple::new(py, [c_py.into_any(), occ_py.into_any()])
                .map_err(py_to_pyscf)?;
            call_hook(&self.slf, "make_rdm1", args, |r| {
                let arr: numpy::PyReadonlyArray2<f64> = r.extract()?;
                to_density(arr)
            })
        })
    }

    fn energy_elec(
        &self,
        dm: &Density,
        h1e: &Density,
        vhf: &Density,
    ) -> Result<(Energy, Energy), PyscfRsError> {
        Python::attach(|py| {
            let dm_py = density_to_pyarray(py, dm).map_err(py_to_pyscf)?;
            let h1e_py = density_to_pyarray(py, h1e).map_err(py_to_pyscf)?;
            let vhf_py = density_to_pyarray(py, vhf).map_err(py_to_pyscf)?;
            let args = PyTuple::new(
                py,
                [dm_py.into_any(), h1e_py.into_any(), vhf_py.into_any()],
            )
            .map_err(py_to_pyscf)?;
            call_hook(&self.slf, "energy_elec", args, |r| {
                let (e, ec): (f64, f64) = r.extract()?;
                Ok((Energy(e), Energy(ec)))
            })
        })
    }

    fn energy_tot(
        &self,
        dm: &Density,
        h1e: &Density,
        vhf: &Density,
    ) -> Result<Energy, PyscfRsError> {
        Python::attach(|py| {
            let dm_py = density_to_pyarray(py, dm).map_err(py_to_pyscf)?;
            let h1e_py = density_to_pyarray(py, h1e).map_err(py_to_pyscf)?;
            let vhf_py = density_to_pyarray(py, vhf).map_err(py_to_pyscf)?;
            let args = PyTuple::new(
                py,
                [dm_py.into_any(), h1e_py.into_any(), vhf_py.into_any()],
            )
            .map_err(py_to_pyscf)?;
            call_hook(&self.slf, "energy_tot", args, |r| {
                let e: f64 = r.extract()?;
                Ok(Energy(e))
            })
        })
    }
}

// -----------------------------------------------------------------------
// KS extension (DFT-08): PyOverrideBridge ALSO impls KsOverrideHooks so the
// KS SCF surface dispatches the two KS-specific seams through the SAME
// `slf.call_method1` MRO path (Pitfall 7). The inherited 11-method SCF
// surface is provided by the `impl OverrideHooks` above — including
// `get_veff`, which the KS cycle drives every iteration (its MRO resolution
// finds a Python subclass `get_veff` override transparently, else the PyRKS
// default KS `J + Vxc − hyb·K` #[pymethod]).
// -----------------------------------------------------------------------

impl KsOverrideHooks for PyOverrideBridge {
    /// KS effective potential. Reuses the `get_veff` template (the same
    /// `slf.call_method1("get_veff", (mol, dm))` dispatch as the inherited
    /// SCF hook — Pitfall 7) to obtain the (possibly subclass-overridden)
    /// effective potential matrix, then folds the consistent Exc/Ecoul energy
    /// terms from the Rust grid loop into the `KsVeff` bundle. Dispatch is via
    /// MRO, NEVER Rust dispatch.
    fn get_veff_ks(
        &self,
        ni: &NumInt,
        mol: &Mole,
        grids: &Grids,
        xc_code: &str,
        dm: &Density,
        j: &Density,
        k: &Density,
    ) -> Result<KsVeff, PyscfRsError> {
        // Energy decomposition (Exc/Ecoul/half_tr_d_vxc) from the Rust grid
        // loop — consistent with the active xc + dm.
        let mut bundle = pyscf_dft::default_get_veff(ni, mol, grids, xc_code, dm, j, k)?;
        // The effective-potential MATRIX comes from the Python `get_veff`
        // override (or the PyRKS default), dispatched via call_method1 (MRO).
        let veff = Python::attach(|py| {
            let dm_py = density_to_pyarray(py, dm).map_err(py_to_pyscf)?;
            let args = PyTuple::new(py, [self.py_mol.bind(py).clone(), dm_py.into_any()])
                .map_err(py_to_pyscf)?;
            call_hook(&self.slf, "get_veff", args, |r| {
                let arr: numpy::PyReadonlyArray2<f64> = r.extract()?;
                to_density(arr)
            })
        })?;
        bundle.veff = veff;
        Ok(bundle)
    }

    /// `define_xc_(description)` — string-recombination form (D-02). Packages
    /// the string and dispatches `slf.call_method1("define_xc_", (description,))`
    /// (MRO — a subclass override of `define_xc_` is invoked transparently).
    /// The Python side returns the resolved hybrid coefficient; we route the
    /// description through the Rust parser to build the canonical `XcSpec`.
    fn define_xc_(&self, description: &str) -> Result<XcSpec, PyscfRsError> {
        Python::attach(|py| {
            let desc_py = pyo3::types::PyString::new(py, description).into_any();
            let args = PyTuple::new(py, [desc_py]).map_err(py_to_pyscf)?;
            // Dispatch via MRO; the Python return value (hyb coeff) is observed
            // for the subclass-override-invoked assertion but the canonical
            // spec is the Rust parser result (string form, D-02).
            let _hyb: f64 = call_hook(&self.slf, "define_xc_", args, |r| r.extract())?;
            pyscf_dft::parser::xcfun::parse_xc(description).map_err(PyscfRsError::from)
        })
    }
}

/// Extract a pyscf-core `Mole` from any Python object.
///
/// Two paths, tried in order:
///   1. Python object has a `dumps()` method (upstream PySCF Mole) — call
///      it and pass through `pyscf_gto::loads(json)`.
///   2. Python object is a JSON string already — pass to `pyscf_gto::loads`.
///
/// Phase 2 D-08: `dumps()` / `loads()` is the canonical Mole interop seam.
pub fn extract_mole_from_pyany(py: Python<'_>, mol: &Py<PyAny>) -> PyResult<Mole> {
    let bound = mol.bind(py);
    if bound.hasattr("dumps")? {
        let json: String = bound.call_method0("dumps")?.extract()?;
        return pyscf_gto::loads(&json).map_err(pyscf_to_py);
    }
    if let Ok(s) = bound.extract::<String>() {
        return pyscf_gto::loads(&s).map_err(pyscf_to_py);
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "expected pyscf.gto.Mole (with .dumps() method) or a serialized Mole JSON string",
    ))
}
