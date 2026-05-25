//! `scf` submodule — PyRHF / PyUHF / PyGHF + 10 hook defaults + scanner.
//!
//! Plan 03-07 Task 2 — fills the 30-attribute floor (SCF-14), kernel/run
//! drivers, and 10 #[pymethods] hook defaults each wrapping the Rust
//! compute in `py.detach` (D-03, BIND-05). Subclass-override dispatch
//! goes through `bridge::PyOverrideBridge::call_method1` (BIND-07).
//!
//! Module layout:
//!   PyRHF  — primary 30-attr surface; full 10-hook default + kernel
//!   PyUHF  — alpha/beta variant, full surface
//!   PyGHF  — 2c spinor variant, minimum SCF-03 floor
//!   PyScfScanner — Send+Sync closure wrapper (SCF-12)

// PyO3 boundary: hook methods return multi-array tuples (e.g. the (J, K)
// pair as `(Bound<PyArray2>, Bound<PyArray2>)`). These FFI return shapes are
// inherent to the Python surface, so `type_complexity` is allowed module-wide.
#![allow(clippy::type_complexity)]

use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

use pyscf_core::{Density, Mole};
use pyscf_scf::{
    GHF as GhfRust, InitGuessMode, NoOverrides, OverrideHooks, RHF as RhfRust, UHF as UhfRust,
    default_eig, default_energy_elec, default_energy_tot, default_get_fock, default_get_hcore,
    default_get_jk, default_get_ovlp, default_get_veff, default_make_rdm1, kernel as scf_kernel,
};

use crate::bridge::{PyOverrideBridge, extract_mole_from_pyany};
use crate::errors::pyscf_to_py;
use crate::numpy_io::{
    density_to_pyarray, mo_coeff_to_pyarray, slice_to_pyarray1, to_density, to_mo_coeff,
};

/// Register PyRHF / PyUHF / PyGHF + PyScfScanner in the `_native.scf`
/// submodule (BIND-02).
pub(crate) fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRHF>()?;
    m.add_class::<PyUHF>()?;
    m.add_class::<PyGHF>()?;
    m.add_class::<PyScfScanner>()?;
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════
// PyRHF — Python-facing Restricted Hartree-Fock.
// 30-attribute floor (SCF-14) + 10 #[pymethods] hook defaults + kernel.
// ═══════════════════════════════════════════════════════════════════════

#[pyclass(subclass, name = "RHF", module = "pyscf._native.scf")]
pub struct PyRHF {
    pub(crate) inner: RhfRust,
    /// User-supplied Python Mole object. Held for PyOverrideBridge re-use
    /// (saves serialise/deserialise per hook call inside the SCF loop).
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

    // ───────────────────────────────────────────────────────────────────
    // SCF-14 30-attribute floor — #[getter] + #[setter] pairs.
    // ───────────────────────────────────────────────────────────────────

    // mol — read-only handle to the user-supplied Python Mole object.
    #[getter]
    fn mol<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.py_mol.bind(py).clone()
    }

    // Scalar SCF state (set by kernel()).
    #[getter]
    fn e_tot(&self) -> f64 {
        self.inner.e_tot
    }
    #[getter]
    fn e_elec(&self) -> f64 {
        self.inner.e_elec
    }
    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }
    #[getter]
    fn cycles(&self) -> u32 {
        self.inner.cycles
    }

    // mo_coeff / mo_energy / mo_occ — None until kernel() runs.
    #[getter]
    fn mo_coeff<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        match &self.inner.mo_coeff {
            Some(mc) => Ok(Some(mo_coeff_to_pyarray(py, mc)?)),
            None => Ok(None),
        }
    }
    #[getter]
    fn mo_energy<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .mo_energy
            .as_ref()
            .map(|v| slice_to_pyarray1(py, v))
    }
    #[getter]
    fn mo_occ<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.mo_occ.as_ref().map(|v| slice_to_pyarray1(py, v))
    }

    // User-tunable SCF parameters.
    #[getter]
    fn verbose(&self) -> u8 {
        self.inner.verbose
    }
    #[setter]
    fn set_verbose(&mut self, v: u8) {
        self.inner.verbose = v;
    }

    #[getter]
    fn chkfile(&self) -> Option<String> {
        self.inner.chkfile.as_ref().map(|p| p.display().to_string())
    }
    #[setter]
    fn set_chkfile(&mut self, v: Option<String>) {
        self.inner.chkfile = v.map(std::path::PathBuf::from);
    }

    #[getter]
    fn max_memory(&self) -> f64 {
        self.inner.max_memory
    }
    #[setter]
    fn set_max_memory(&mut self, v: f64) {
        self.inner.max_memory = v;
    }

    #[getter]
    fn direct_scf(&self) -> bool {
        self.inner.direct_scf
    }
    #[setter]
    fn set_direct_scf(&mut self, v: bool) {
        self.inner.direct_scf = v;
    }

    #[getter]
    fn direct_scf_tol(&self) -> f64 {
        self.inner.direct_scf_tol
    }
    #[setter]
    fn set_direct_scf_tol(&mut self, v: f64) {
        self.inner.direct_scf_tol = v;
    }

    #[getter]
    fn init_guess(&self) -> String {
        self.inner.init_guess.clone()
    }
    #[setter]
    fn set_init_guess(&mut self, v: String) {
        self.inner.init_guess = v;
    }

    #[getter]
    fn level_shift(&self) -> f64 {
        self.inner.level_shift
    }
    #[setter]
    fn set_level_shift(&mut self, v: f64) {
        self.inner.level_shift = v;
    }

    #[getter]
    fn damp(&self) -> f64 {
        self.inner.damp
    }
    #[setter]
    fn set_damp(&mut self, v: f64) {
        self.inner.damp = v;
    }

    #[getter]
    fn diis(&self) -> bool {
        self.inner.diis
    }
    #[setter]
    fn set_diis(&mut self, v: bool) {
        self.inner.diis = v;
    }

    #[getter]
    fn diis_space(&self) -> u32 {
        self.inner.diis_space
    }
    #[setter]
    fn set_diis_space(&mut self, v: u32) {
        self.inner.diis_space = v;
    }

    #[getter]
    fn diis_start_cycle(&self) -> u32 {
        self.inner.diis_start_cycle
    }
    #[setter]
    fn set_diis_start_cycle(&mut self, v: u32) {
        self.inner.diis_start_cycle = v;
    }

    #[getter]
    fn diis_damp(&self) -> f64 {
        self.inner.diis_damp
    }
    #[setter]
    fn set_diis_damp(&mut self, v: f64) {
        self.inner.diis_damp = v;
    }

    #[getter]
    fn diis_file(&self) -> Option<String> {
        self.inner
            .diis_file
            .as_ref()
            .map(|p| p.display().to_string())
    }
    #[setter]
    fn set_diis_file(&mut self, v: Option<String>) {
        self.inner.diis_file = v.map(std::path::PathBuf::from);
    }

    #[getter]
    fn max_cycle(&self) -> u32 {
        self.inner.max_cycle
    }
    #[setter]
    fn set_max_cycle(&mut self, v: u32) {
        self.inner.max_cycle = v;
    }

    #[getter]
    fn conv_tol(&self) -> f64 {
        self.inner.conv_tol
    }
    #[setter]
    fn set_conv_tol(&mut self, v: f64) {
        self.inner.conv_tol = v;
    }

    #[getter]
    fn conv_tol_grad(&self) -> Option<f64> {
        self.inner.conv_tol_grad
    }
    #[setter]
    fn set_conv_tol_grad(&mut self, v: Option<f64>) {
        self.inner.conv_tol_grad = v;
    }

    // with_df — true if RHF.density_fit() was called. We expose `.with_df`
    // as a bool flag (the inner Box<dyn Any> is opaque to Python).
    #[getter]
    fn with_df(&self) -> bool {
        self.inner.with_df.is_some()
    }

    #[getter]
    fn disp(&self) -> Option<String> {
        self.inner.disp.clone()
    }
    #[setter]
    fn set_disp(&mut self, v: Option<String>) {
        self.inner.disp = v;
    }

    #[getter]
    fn do_disp(&self) -> bool {
        self.inner.do_disp
    }
    #[setter]
    fn set_do_disp(&mut self, v: bool) {
        self.inner.do_disp = v;
    }

    #[getter]
    fn irrep_nelec(&self) -> std::collections::HashMap<String, u32> {
        self.inner.irrep_nelec.clone()
    }

    #[getter]
    fn nelec(&self) -> Option<(u32, u32)> {
        self.inner.nelec
    }

    // callback — exposed as bool (Box<dyn Fn> not Python-visible today).
    #[getter]
    fn callback(&self) -> bool {
        self.inner.callback.is_some()
    }

    #[getter]
    fn scf_summary(&self) -> std::collections::HashMap<String, f64> {
        self.inner.scf_summary.clone()
    }

    // ───────────────────────────────────────────────────────────────────
    // kernel + run + density_fit — drivers (BIND-02 idiom #2 / #3 / #6).
    // ───────────────────────────────────────────────────────────────────

    /// Drive SCF — Python: `mf.kernel()` or `mf.kernel(dm0)`.
    ///
    /// Subclass-override dispatch happens via `PyOverrideBridge::call_method1`
    /// on every overrideable hook (Pitfall 7 / BIND-07).
    #[pyo3(signature = (dm0=None))]
    fn kernel<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        dm0: Option<PyReadonlyArray2<'py, f64>>,
    ) -> PyResult<f64> {
        // Snapshot config + clone mol BEFORE acquiring any borrow that
        // would conflict with the bridge's call_method1 dispatches.
        let (mut cfg, mol) = {
            let me = slf.borrow();
            (me.inner.to_kernel_config(), me.inner.mol.clone())
        };
        if let Some(arr) = dm0 {
            let d = to_density(arr)?;
            cfg.init_guess = InitGuessMode::UserDM(d);
        }
        let class_name: String = slf.get_type().name()?.extract()?;
        let result = if class_name == "RHF" {
            // Exact base-class RHF has no Python hook overrides. Drive the
            // kernel through NoOverrides so CI parity runs do not repeatedly
            // bounce through upstream-PySCF Mole objects while holding the GIL.
            scf_kernel(&mol, &NoOverrides, cfg).map_err(pyscf_to_py)?
        } else {
            // The bridge needs a Py<PyAny> handle pointing at the same Python
            // object (so call_method1 resolves the MRO including subclass
            // overrides). `slf.clone()` increments the refcount; `.into_any()`
            // erases the typed Bound to a generic Py<PyAny>.
            let bridge_slf: Py<PyAny> = slf.clone().into_any().unbind();
            let py_mol = {
                let me = slf.borrow();
                me.py_mol.clone_ref(py)
            };
            let bridge = PyOverrideBridge::new(bridge_slf, py_mol);
            // NB: We DO NOT call `py.detach` here. The SCF kernel calls into
            // Python on every hook (via the bridge), so holding the GIL across
            // the kernel is mandatory — release+reacquire per hook would
            // exceed the cost of any GPU benefit. The hook-default methods
            // INSIDE this pyclass (get_hcore, get_ovlp, ...) DO release the
            // GIL via py.detach during the Rust compute (D-03 / BIND-05).
            scf_kernel(&mol, &bridge, cfg).map_err(pyscf_to_py)?
        };
        let e_tot = result.e_tot.0;

        // SCF-10 auto-write chkfile on convergence — done BEFORE the state
        // mirror below so we can pass `&result` directly to dump_scf_to_file
        // without cloning the MO matrix.
        let chkfile_path: Option<std::path::PathBuf> = {
            let me = slf.borrow();
            if result.converged {
                me.inner.chkfile.clone()
            } else {
                None
            }
        };
        if let Some(path) = chkfile_path {
            // Serialize Mole via the Python-side `.dumps()` if available
            // (preserves upstream-PySCF chkfile schema), else fall back to
            // the native pyscf_gto::dumps for pure-Rust callers.
            let mol_json: String = {
                let me = slf.borrow();
                let bound = me.py_mol.bind(py);
                if bound.hasattr("dumps")? {
                    bound.call_method0("dumps")?.extract::<String>()?
                } else {
                    pyscf_gto::dumps(&me.inner.mol).map_err(pyscf_to_py)?
                }
            };
            pyscf_scf::dump_scf_to_file(&path, &mol_json, &result).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("chkfile dump failed: {e}"))
            })?;
        }

        // Re-borrow mutably AFTER the bridge is dropped (bridge_slf only
        // borrows the Python object reference, not the Rust struct).
        {
            let mut me = slf.borrow_mut();
            me.inner.mo_coeff = Some(result.mo_coeff);
            me.inner.mo_energy = Some(result.mo_energy);
            me.inner.mo_occ = Some(result.mo_occ);
            me.inner.e_tot = e_tot;
            me.inner.converged = result.converged;
            me.inner.cycles = result.cycles;
        }
        Ok(e_tot)
    }

    /// `mf.run(*args, **kwargs)` — alias for kernel, returns self for chaining.
    /// BIND-02 idiom: `scf.RHF(mol).run()` and `scf.RHF(mol).density_fit().run()`.
    fn run<'py>(slf: Bound<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        let _ = PyRHF::kernel(slf.clone(), py, None)?;
        Ok(slf)
    }

    /// `RHF.from_chk(mol, path)` — reconstruct a PyRHF whose MO state is
    /// loaded from an existing chkfile (SCF-10). The Mole argument is the
    /// user-supplied Python Mole (held for getter access + future hook
    /// dispatch); the chkfile path is recorded so subsequent kernel calls
    /// will auto-overwrite on convergence.
    #[staticmethod]
    #[pyo3(signature = (mol, path))]
    fn from_chk(py: Python<'_>, mol: Py<PyAny>, path: String) -> PyResult<Self> {
        let mol_inner: Mole = extract_mole_from_pyany(py, &mol)?;
        let pathbuf = std::path::PathBuf::from(&path);
        let result = pyscf_scf::load_scf_from_file(&pathbuf).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("chkfile load failed: {e}"))
        })?;
        let mut rhf = RhfRust::new(mol_inner);
        rhf.e_tot = result.e_tot.0;
        rhf.converged = result.converged;
        rhf.cycles = result.cycles;
        rhf.mo_coeff = Some(result.mo_coeff);
        rhf.mo_energy = Some(result.mo_energy);
        rhf.mo_occ = Some(result.mo_occ);
        rhf.chkfile = Some(pathbuf);
        Ok(PyRHF {
            inner: rhf,
            py_mol: mol,
        })
    }

    // ───────────────────────────────────────────────────────────────────
    // density_fit — SCF-07 user-facing.
    // ───────────────────────────────────────────────────────────────────

    /// `mf.density_fit(auxbasis=None)` — DF-HF entry point (SCF-07).
    ///
    /// Returns self after populating `with_df`. The kernel call site
    /// (PyRHF::kernel) still drives via PyOverrideBridge; the `with_df`
    /// slot is consumed by pyscf-scf's `DfHooks` only when the pure-Rust
    /// caller wires `kernel(&mol, &DfHooks { df }, cfg)`. Phase 3 plan
    /// 03-10 oracle exercises the full end-to-end.
    #[pyo3(signature = (auxbasis=None))]
    fn density_fit<'py>(
        slf: Bound<'py, Self>,
        auxbasis: Option<&str>,
    ) -> PyResult<Bound<'py, Self>> {
        // RHF::density_fit consumes self by value; clone the mol BEFORE
        // taking a mutable borrow so the borrow checker doesn't see a
        // simultaneous &mut + & on slf.inner.
        let mol_clone = slf.borrow().inner.mol.clone();
        let mut me = slf.borrow_mut();
        let prior = std::mem::replace(&mut me.inner, RhfRust::new(mol_clone));
        let new_inner = prior.density_fit(auxbasis).map_err(pyscf_to_py)?;
        me.inner = new_inner;
        drop(me);
        Ok(slf)
    }

    // ───────────────────────────────────────────────────────────────────
    // 10 #[pymethods] hook defaults — D-03 + BIND-05 per-hook py.detach.
    // Each method extracts typed inputs, releases the GIL via py.detach,
    // runs the Rust default, and marshals output back to a NumPy array.
    // ───────────────────────────────────────────────────────────────────

    fn get_hcore<'py>(
        &self,
        py: Python<'py>,
        mol: Py<PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mol_ref = extract_mole_from_pyany(py, &mol)?;
        let h = py
            .detach(|| default_get_hcore(&mol_ref))
            .map_err(pyscf_to_py)?;
        density_to_pyarray(py, &h)
    }

    fn get_ovlp<'py>(
        &self,
        py: Python<'py>,
        mol: Py<PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mol_ref = extract_mole_from_pyany(py, &mol)?;
        let s = py
            .detach(|| default_get_ovlp(&mol_ref))
            .map_err(pyscf_to_py)?;
        density_to_pyarray(py, &s)
    }

    fn get_jk<'py>(
        &self,
        py: Python<'py>,
        mol: Py<PyAny>,
        dm: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let mol_ref = extract_mole_from_pyany(py, &mol)?;
        let dm_ref = to_density(dm)?;
        let (j, k) = py
            .detach(|| default_get_jk(&mol_ref, &dm_ref))
            .map_err(pyscf_to_py)?;
        Ok((density_to_pyarray(py, &j)?, density_to_pyarray(py, &k)?))
    }

    fn get_veff<'py>(
        &self,
        py: Python<'py>,
        mol: Py<PyAny>,
        dm: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mol_ref = extract_mole_from_pyany(py, &mol)?;
        let dm_ref = to_density(dm)?;
        let v = py
            .detach(|| default_get_veff(&mol_ref, &dm_ref))
            .map_err(pyscf_to_py)?;
        density_to_pyarray(py, &v)
    }

    fn get_fock<'py>(
        &self,
        py: Python<'py>,
        h1e: PyReadonlyArray2<'py, f64>,
        s1e: PyReadonlyArray2<'py, f64>,
        vhf: PyReadonlyArray2<'py, f64>,
        dm: PyReadonlyArray2<'py, f64>,
        cycle: i32,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let h1e_d = to_density(h1e)?;
        let s1e_d = to_density(s1e)?;
        let vhf_d = to_density(vhf)?;
        let dm_d = to_density(dm)?;
        let f = py
            .detach(|| default_get_fock(&h1e_d, &s1e_d, &vhf_d, &dm_d, cycle, None))
            .map_err(pyscf_to_py)?;
        density_to_pyarray(py, &f)
    }

    fn eig<'py>(
        &self,
        py: Python<'py>,
        fock: PyReadonlyArray2<'py, f64>,
        s1e: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
        let fock_d = to_density(fock)?;
        let s1e_d = to_density(s1e)?;
        let mc = py
            .detach(|| default_eig(&fock_d, &s1e_d))
            .map_err(pyscf_to_py)?;
        let energies = slice_to_pyarray1(py, &mc.energies);
        let coeffs = mo_coeff_to_pyarray(py, &mc)?;
        Ok((energies, coeffs))
    }

    fn get_occ<'py>(
        &self,
        py: Python<'py>,
        mo_energy: numpy::PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let energies: Vec<f64> = mo_energy.as_slice()?.to_vec();
        let nelec = self.inner.mol.nelectron;
        let occ = py
            .detach(|| pyscf_scf::default_get_occ(&energies, nelec))
            .map_err(pyscf_to_py)?;
        Ok(PyArray1::from_slice(py, &occ))
    }

    fn make_rdm1<'py>(
        &self,
        py: Python<'py>,
        mo_coeff: PyReadonlyArray2<'py, f64>,
        mo_occ: numpy::PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mut mc = to_mo_coeff(mo_coeff)?;
        let occ: Vec<f64> = mo_occ.as_slice()?.to_vec();
        mc.occupations = occ;
        let d = py.detach(|| default_make_rdm1(&mc)).map_err(pyscf_to_py)?;
        density_to_pyarray(py, &d)
    }

    fn energy_elec<'py>(
        &self,
        py: Python<'py>,
        dm: PyReadonlyArray2<'py, f64>,
        h1e: PyReadonlyArray2<'py, f64>,
        vhf: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(f64, f64)> {
        let dm_d = to_density(dm)?;
        let h1e_d = to_density(h1e)?;
        let vhf_d = to_density(vhf)?;
        let (e, ec) = py
            .detach(|| default_energy_elec(&dm_d, &h1e_d, &vhf_d))
            .map_err(pyscf_to_py)?;
        Ok((e.0, ec.0))
    }

    fn energy_tot<'py>(
        &self,
        py: Python<'py>,
        dm: PyReadonlyArray2<'py, f64>,
        h1e: PyReadonlyArray2<'py, f64>,
        vhf: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<f64> {
        let dm_d = to_density(dm)?;
        let h1e_d = to_density(h1e)?;
        let vhf_d = to_density(vhf)?;
        let e = py
            .detach(|| default_energy_tot(&dm_d, &h1e_d, &vhf_d))
            .map_err(pyscf_to_py)?;
        Ok(e.0)
    }

    // ───────────────────────────────────────────────────────────────────
    // Convert / analyze / scanner surface (SCF-09 / SCF-11 / SCF-12).
    // ───────────────────────────────────────────────────────────────────

    fn to_uhf(slf: PyRef<'_, Self>) -> PyResult<Py<PyUHF>> {
        let uhf_inner = pyscf_scf::to_uhf(&slf.inner).map_err(pyscf_to_py)?;
        let py = slf.py();
        let py_mol = slf.py_mol.clone_ref(py);
        Py::new(
            py,
            PyUHF {
                inner: uhf_inner,
                py_mol,
            },
        )
    }

    fn to_ghf(slf: PyRef<'_, Self>) -> PyResult<Py<PyGHF>> {
        let ghf_inner = pyscf_scf::to_ghf(&slf.inner).map_err(pyscf_to_py)?;
        let py = slf.py();
        let py_mol = slf.py_mol.clone_ref(py);
        Py::new(
            py,
            PyGHF {
                inner: ghf_inner,
                py_mol,
            },
        )
    }

    /// `mf.to_uks(xc=...)` — RHF → UKS conversion (DFT-11). Wired to the real
    /// `pyscf_dft::UKS` target via the `pyscf_scf::to_uks` KsConversion seam
    /// (the Phase 3 `NotYetImplemented{phase:4}` stub is retired). The carried
    /// scalar SCF settings are copied; MO coefficients are not (same
    /// convention as to_uhf).
    #[pyo3(signature = (xc="LDA,VWN"))]
    fn to_uks(slf: PyRef<'_, Self>, xc: &str) -> PyResult<Py<crate::dft::PyUKS>> {
        let conv = pyscf_scf::to_uks(&slf.inner).map_err(pyscf_to_py)?;
        let py = slf.py();
        let py_mol = slf.py_mol.clone_ref(py);
        let inner = crate::dft::uks_from_conversion(conv, xc);
        Py::new(py, crate::dft::PyUKS { inner, py_mol })
    }

    /// `mf.to_rks(xc=...)` — RHF → RKS conversion (DFT-11). Wired to the real
    /// `pyscf_dft::RKS` target via the `pyscf_scf::to_rks` KsConversion seam.
    #[pyo3(signature = (xc="LDA,VWN"))]
    fn to_rks(slf: PyRef<'_, Self>, xc: &str) -> PyResult<Py<crate::dft::PyRKS>> {
        let conv = pyscf_scf::to_rks(&slf.inner).map_err(pyscf_to_py)?;
        let py = slf.py();
        let py_mol = slf.py_mol.clone_ref(py);
        let inner = crate::dft::rks_from_conversion(conv, xc);
        Py::new(py, crate::dft::PyRKS { inner, py_mol })
    }

    // analyze / mulliken_pop / dip_moment hold a PyRef so they can't take
    // the GIL-release fast path (`Python` is !Send, so closures borrowing
    // slf can't satisfy `Ungil`). These are called once post-convergence
    // and not inside the SCF loop, so the latency cost is negligible.
    fn analyze(slf: PyRef<'_, Self>) -> PyResult<()> {
        pyscf_scf::analyze(&slf.inner).map_err(pyscf_to_py)
    }

    fn mulliken_pop<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let result = pyscf_scf::mulliken_pop(&slf.inner).map_err(pyscf_to_py)?;
        Ok((
            PyArray1::from_slice(py, &result.atom_charges),
            PyArray1::from_slice(py, &result.ao_populations),
        ))
    }

    fn dip_moment<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let d = pyscf_scf::dip_moment(&slf.inner).map_err(pyscf_to_py)?;
        Ok(PyArray1::from_slice(py, &d))
    }

    /// `mf.as_scanner()` — return a callable (PyScfScanner) that takes a
    /// Mole and returns the converged total energy (SCF-12).
    fn as_scanner(slf: PyRef<'_, Self>) -> PyScfScanner {
        let inner_scanner = pyscf_scf::as_scanner(&slf.inner);
        PyScfScanner {
            inner: Some(inner_scanner),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PyUHF — Unrestricted HF. Alpha/beta-paired MO slots.
// ═══════════════════════════════════════════════════════════════════════

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
    fn mol<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.py_mol.bind(py).clone()
    }

    #[getter]
    fn e_tot(&self) -> f64 {
        self.inner.e_tot
    }
    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }
    #[getter]
    fn cycles(&self) -> u32 {
        self.inner.cycles
    }
    #[getter]
    fn max_cycle(&self) -> u32 {
        self.inner.max_cycle
    }
    #[setter]
    fn set_max_cycle(&mut self, v: u32) {
        self.inner.max_cycle = v;
    }
    #[getter]
    fn conv_tol(&self) -> f64 {
        self.inner.conv_tol
    }
    #[setter]
    fn set_conv_tol(&mut self, v: f64) {
        self.inner.conv_tol = v;
    }
    #[getter]
    fn diis(&self) -> bool {
        self.inner.diis
    }
    #[setter]
    fn set_diis(&mut self, v: bool) {
        self.inner.diis = v;
    }
    #[getter]
    fn diis_space(&self) -> u32 {
        self.inner.diis_space
    }
    #[getter]
    fn init_guess(&self) -> String {
        self.inner.init_guess.clone()
    }

    fn to_rhf(slf: PyRef<'_, Self>) -> PyResult<Py<PyRHF>> {
        let rhf_inner = pyscf_scf::to_rhf(&slf.inner).map_err(pyscf_to_py)?;
        let py = slf.py();
        let py_mol = slf.py_mol.clone_ref(py);
        Py::new(
            py,
            PyRHF {
                inner: rhf_inner,
                py_mol,
            },
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PyGHF — Generalized HF. 2c-spinor MO.
// ═══════════════════════════════════════════════════════════════════════

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
    fn mol<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.py_mol.bind(py).clone()
    }

    #[getter]
    fn e_tot(&self) -> f64 {
        self.inner.e_tot
    }
    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }
    #[getter]
    fn max_cycle(&self) -> u32 {
        self.inner.max_cycle
    }
    #[setter]
    fn set_max_cycle(&mut self, v: u32) {
        self.inner.max_cycle = v;
    }
    #[getter]
    fn conv_tol(&self) -> f64 {
        self.inner.conv_tol
    }
    #[setter]
    fn set_conv_tol(&mut self, v: f64) {
        self.inner.conv_tol = v;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PyScfScanner — Send+Sync closure wrapper for SCF-12.
// Python: `e = mf.as_scanner()(new_mol)` returns f64.
// ═══════════════════════════════════════════════════════════════════════

#[pyclass(name = "Scanner", module = "pyscf._native.scf")]
pub struct PyScfScanner {
    /// `Option` so `__call__` can take by &mut self yet still hand the
    /// closure result back. The closure stays in place across calls — we
    /// don't move it out, just invoke it.
    inner: Option<
        Box<dyn Fn(&Mole) -> Result<pyscf_core::Energy, pyscf_core::PyscfRsError> + Send + Sync>,
    >,
}

#[pymethods]
impl PyScfScanner {
    /// Make the scanner callable from Python: `e = scanner(mol)`.
    fn __call__(&self, py: Python<'_>, mol: Py<PyAny>) -> PyResult<f64> {
        let mol_inner: Mole = extract_mole_from_pyany(py, &mol)?;
        let inner = self.inner.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("PyScfScanner: closure missing")
        })?;
        let e = py.detach(|| inner(&mol_inner)).map_err(pyscf_to_py)?;
        Ok(e.0)
    }
}

// Marker: silence unused-field warning on PyUHF/PyGHF/PyRHF py_mol when
// the only consumer is the `mol` getter — the getter does use it but
// Clippy/rustc occasionally miss the cfg-flagged usage.
#[allow(dead_code)]
fn _py_mol_use_marker(r: &PyRHF, u: &PyUHF, g: &PyGHF) -> usize {
    let _ = &r.py_mol;
    let _ = &u.py_mol;
    let _ = &g.py_mol;
    let _ = Density {
        nao: 0,
        data: vec![],
    };
    0
}

// Suppress unused-import warning for NoOverrides — it's named in
// documentation; the runtime path is via PyOverrideBridge. Keep the
// import in case future plans need direct delegation.
#[allow(dead_code)]
fn _no_overrides_use_marker() -> impl OverrideHooks {
    NoOverrides
}
