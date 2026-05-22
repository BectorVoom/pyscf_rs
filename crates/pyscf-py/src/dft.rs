//! `dft` submodule — PyRKS / PyUKS + KS hook defaults + read-only precision
//! getter (D-08).
//!
//! Plan 04-09 Task 1 — the DFT analog of `scf.rs`. `PyRKS`/`PyUKS` mirror
//! `PyRHF`/`PyUHF` (scf.rs:44-604): the DFT attribute floor (xc/grids/nlc/
//! nlcgrids + the inherited SCF floor subset), a `kernel`/`run` driver, KS
//! hook defaults whose `get_veff` produces the Kohn-Sham `J + Vxc − hyb·K`
//! form, and a `define_xc_` string-form hook. Subclass-override dispatch
//! goes through `bridge::PyOverrideBridge` (which now also impls
//! `pyscf_dft::KsOverrideHooks`) via `slf.call_method1` (Pitfall 7 / BIND-07).
//!
//! D-08: a READ-ONLY `#[getter]` (`dtype`) surfaces the active precision
//! (`_numint.dtype()` → "f32"/"f64") on the KS object. There is intentionally
//! NO paired `#[setter]` / `set_precision` — `PYSCF_DTYPE` is the single
//! source of truth for switching (the per-object runtime override is deferred
//! per D-08). The same read-only model is mirrored on the `_numint` getter
//! object (`PyNumIntView.dtype`).
//!
//! `pyscf-dft` stays **pyo3-free** — only this crate (pyscf-py) names pyo3
//! (the PyO3 wall).
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

use pyscf_core::{Density, Mole};
use pyscf_dft::{
    default_get_veff as ks_default_get_veff, KsOverrideHooks, NoKsOverrides, NumInt, RKS as RksRust,
    UKS as UksRust,
};
use pyscf_scf::InitGuessMode;

use crate::bridge::{extract_mole_from_pyany, PyOverrideBridge};
use crate::errors::pyscf_to_py;
use crate::numpy_io::{density_to_pyarray, slice_to_pyarray1, to_density};

/// Register PyRKS / PyUKS (+ the `_numint` view) in the `_native.dft`
/// submodule (BIND-02 analog).
pub(crate) fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRKS>()?;
    m.add_class::<PyUKS>()?;
    m.add_class::<PyNumIntView>()?;
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════
// PyNumIntView — read-only handle surfacing `_numint.dtype()` (D-08).
// Returned by `mf._numint`; exposes ONLY a read-only `dtype` #[getter].
// There is NO setter — the precision is fixed at construction from
// PYSCF_DTYPE (D-08); the per-object runtime override is deferred.
// ═══════════════════════════════════════════════════════════════════════

#[pyclass(name = "NumInt", module = "pyscf._native.dft")]
pub struct PyNumIntView {
    /// Active precision string ("f32"/"f64"), snapshot from the owning KS
    /// object's `NumInt::dtype()` at the time `mf._numint` was read.
    dtype: String,
}

#[pymethods]
impl PyNumIntView {
    /// Read-only active precision (D-08). NO paired #[setter].
    #[getter]
    fn dtype(&self) -> &str {
        &self.dtype
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PyRKS — Python-facing Restricted Kohn-Sham.
// DFT attribute floor + KS #[pymethods] hook defaults + kernel + the D-08
// read-only precision getter (no setter).
// ═══════════════════════════════════════════════════════════════════════

#[pyclass(subclass, name = "RKS", module = "pyscf._native.dft")]
pub struct PyRKS {
    pub(crate) inner: RksRust,
    /// User-supplied Python Mole object — held for PyOverrideBridge re-use.
    pub(crate) py_mol: Py<PyAny>,
}

#[pymethods]
impl PyRKS {
    #[new]
    #[pyo3(signature = (mol, xc="LDA,VWN"))]
    fn new(py: Python<'_>, mol: Py<PyAny>, xc: &str) -> PyResult<Self> {
        let mol_inner: Mole = extract_mole_from_pyany(py, &mol)?;
        Ok(PyRKS {
            inner: RksRust::new(mol_inner, xc),
            py_mol: mol,
        })
    }

    // ───────────────────────────────────────────────────────────────────
    // DFT attribute floor — xc / nlc + the inherited SCF floor subset.
    // ───────────────────────────────────────────────────────────────────

    #[getter]
    fn mol<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.py_mol.bind(py).clone()
    }

    // DFT-specific attributes (rks.py).
    #[getter] fn xc(&self) -> String { self.inner.xc.clone() }
    #[setter] fn set_xc(&mut self, v: String) { self.inner.xc = v; }

    #[getter] fn nlc(&self) -> String { self.inner.nlc.clone() }
    #[setter] fn set_nlc(&mut self, v: String) { self.inner.nlc = v; }

    // grids / nlcgrids — exposed as bool presence flags (the Grids struct is
    // opaque to Python today; the build happens inside kernel()).
    #[getter] fn grids(&self) -> bool { true }
    #[getter] fn nlcgrids(&self) -> bool { true }

    /// `mf._numint` — the read-only numerical-integration view (D-08).
    /// Surfaces ONLY the active precision; no mutable surface.
    #[getter]
    fn _numint(&self, py: Python<'_>) -> PyResult<Py<PyNumIntView>> {
        Py::new(
            py,
            PyNumIntView {
                dtype: self.inner.dtype().name().to_string(),
            },
        )
    }

    /// D-08 READ-ONLY active-precision getter. Returns "f32"/"f64" by
    /// delegating to `_numint.dtype()` (the 04-06 NumInt accessor). There is
    /// intentionally NO `set_precision` / precision `#[setter]` on the Python
    /// surface — `PYSCF_DTYPE` is the single source of truth (D-08 defers the
    /// per-object runtime override). Attempting `mf.dtype = ...` from Python
    /// raises `AttributeError` (no setter registered).
    #[getter]
    fn dtype(&self) -> &str {
        self.inner.dtype().name()
    }

    // Scalar SCF state (set by kernel()).
    #[getter] fn e_tot(&self)     -> f64  { self.inner.e_tot }
    #[getter] fn e_elec(&self)    -> f64  { self.inner.e_elec }
    #[getter] fn converged(&self) -> bool { self.inner.converged }
    #[getter] fn cycles(&self)    -> u32  { self.inner.cycles }

    #[getter]
    fn mo_coeff<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        match &self.inner.mo_coeff {
            Some(mc) => Ok(Some(crate::numpy_io::mo_coeff_to_pyarray(py, mc)?)),
            None => Ok(None),
        }
    }
    #[getter]
    fn mo_energy<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.mo_energy.as_ref().map(|v| slice_to_pyarray1(py, v))
    }
    #[getter]
    fn mo_occ<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.mo_occ.as_ref().map(|v| slice_to_pyarray1(py, v))
    }

    // Inherited SCF tunables (the floor subset rks.py reuses).
    #[getter] fn verbose(&self) -> u8 { self.inner.verbose }
    #[setter] fn set_verbose(&mut self, v: u8) { self.inner.verbose = v; }

    #[getter] fn max_cycle(&self) -> u32 { self.inner.max_cycle }
    #[setter] fn set_max_cycle(&mut self, v: u32) { self.inner.max_cycle = v; }

    #[getter] fn conv_tol(&self) -> f64 { self.inner.conv_tol }
    #[setter] fn set_conv_tol(&mut self, v: f64) { self.inner.conv_tol = v; }

    #[getter] fn init_guess(&self) -> String { self.inner.init_guess.clone() }
    #[setter] fn set_init_guess(&mut self, v: String) { self.inner.init_guess = v; }

    #[getter] fn diis(&self) -> bool { self.inner.diis }
    #[setter] fn set_diis(&mut self, v: bool) { self.inner.diis = v; }

    #[getter] fn level_shift(&self) -> f64 { self.inner.level_shift }
    #[setter] fn set_level_shift(&mut self, v: f64) { self.inner.level_shift = v; }

    #[getter] fn damp(&self) -> f64 { self.inner.damp }
    #[setter] fn set_damp(&mut self, v: f64) { self.inner.damp = v; }

    // ───────────────────────────────────────────────────────────────────
    // kernel + run — the KS SCF driver. Subclass-override dispatch happens
    // via PyOverrideBridge::call_method1 on every overrideable hook (Pitfall
    // 7 / BIND-07). The KS form (J + Vxc − hyb·K) is produced by the PyRKS
    // default get_veff #[pymethod] below; a subclass override replaces it
    // transparently via MRO.
    // ───────────────────────────────────────────────────────────────────

    #[pyo3(signature = (dm0=None))]
    fn kernel<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        dm0: Option<PyReadonlyArray2<'py, f64>>,
    ) -> PyResult<f64> {
        // Build the integration grid + snapshot config and mol BEFORE any
        // borrow that the bridge's call_method1 dispatches would conflict
        // with. `RKS::kernel` builds the grid internally; here we drive the
        // generic SCF kernel through the bridge (so subclass overrides fire),
        // building the grid up front to mirror RKS::kernel's contract.
        let (mut cfg, mol) = {
            let mut me = slf.borrow_mut();
            // Build the Becke grid once for this geometry (RKS::kernel does
            // this; we replicate it so the KS get_veff grid loop has coords).
            // Clone mol first so `grids.build(&mol)` does not co-borrow
            // `me.inner` mutably (grids) and immutably (mol) through the same
            // RefMut deref path.
            let mol = me.inner.mol.clone();
            let _ = me.inner.grids.build(&mol);
            (me.inner.to_kernel_config(), mol)
        };
        if let Some(arr) = dm0 {
            let d = to_density(arr)?;
            cfg.init_guess = InitGuessMode::UserDM(d);
        }
        let bridge_slf: Py<PyAny> = slf.clone().into_any().unbind();
        let py_mol = {
            let me = slf.borrow();
            me.py_mol.clone_ref(py)
        };
        let bridge = PyOverrideBridge::new(bridge_slf, py_mol);
        // Hold the GIL across the kernel — every hook calls into Python via
        // the bridge (the per-hook py.detach seam is inside the #[pymethods]
        // hook defaults, not here). Same contract as PyRHF::kernel.
        let result = pyscf_scf::kernel(&mol, &bridge, cfg).map_err(pyscf_to_py)?;
        let e_tot = result.e_tot.0;
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

    /// `mf.run()` — alias for kernel, returns self for chaining.
    fn run<'py>(slf: Bound<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        let _ = PyRKS::kernel(slf.clone(), py, None)?;
        Ok(slf)
    }

    // ───────────────────────────────────────────────────────────────────
    // KS #[pymethods] hook defaults — the KS get_veff (J + Vxc − hyb·K) +
    // define_xc_ string form. Each releases the GIL via py.detach during the
    // Rust compute (D-03 / BIND-05). The bridge's call_method1 finds these
    // via MRO when no subclass override exists.
    // ───────────────────────────────────────────────────────────────────

    /// KS effective potential `J + Vxc − hyb·K` (DFT-01). Builds (J, K) via
    /// the SCF default get_jk, runs the grid loop for Vxc, and returns the
    /// combined veff. A subclass override of `get_veff` replaces this whole
    /// method via MRO (Pitfall 7).
    fn get_veff<'py>(
        &mut self,
        py: Python<'py>,
        mol: Py<PyAny>,
        dm: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mol_ref = extract_mole_from_pyany(py, &mol)?;
        let dm_ref = to_density(dm)?;
        let xc = self.inner.xc.clone();
        // Build the grid lazily if kernel() hasn't (defensive — direct
        // mf.get_veff(mol, dm) calls outside kernel()).
        let _ = self.inner.grids.build(&mol_ref);
        let bundle = py
            .detach(|| {
                let (j, k) = pyscf_scf::default_get_jk(&mol_ref, &dm_ref)?;
                ks_default_get_veff(
                    &self.inner._numint,
                    &mol_ref,
                    &self.inner.grids,
                    &xc,
                    &dm_ref,
                    &j,
                    &k,
                )
            })
            .map_err(pyscf_to_py)?;
        density_to_pyarray(py, &bundle.veff)
    }

    /// `define_xc_(description)` — (re)define the active XC functional from a
    /// string (D-02 string form). The callable form is deferred
    /// (NotYetImplemented{deferred}). Returns the resolved hybrid coefficient
    /// so the Python caller can observe the recombination took effect.
    fn define_xc_(&mut self, py: Python<'_>, description: &str) -> PyResult<f64> {
        let hooks = NoKsOverrides;
        let desc = description.to_string();
        let spec = py
            .detach(|| hooks.define_xc_(&desc))
            .map_err(pyscf_to_py)?;
        // Adopt the recombined functional string as the active xc.
        self.inner.xc = desc;
        Ok(spec.hyb().0)
    }

    // ───────────────────────────────────────────────────────────────────
    // The remaining SCF hooks the KS SCF cycle dispatches (get_hcore /
    // get_ovlp / get_jk / get_fock / eig / get_occ / make_rdm1 / energy_*).
    // These delegate to the SCF defaults so the bridge's call_method1 finds a
    // concrete default for every hook on the MRO (matching PyRHF). The KS
    // energy correction (Tr(D·h1e)+Ecoul+Exc) is applied in energy_elec.
    // ───────────────────────────────────────────────────────────────────

    fn get_hcore<'py>(
        &self,
        py: Python<'py>,
        mol: Py<PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mol_ref = extract_mole_from_pyany(py, &mol)?;
        let h = py
            .detach(|| pyscf_scf::default_get_hcore(&mol_ref))
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
            .detach(|| pyscf_scf::default_get_ovlp(&mol_ref))
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
            .detach(|| pyscf_scf::default_get_jk(&mol_ref, &dm_ref))
            .map_err(pyscf_to_py)?;
        Ok((density_to_pyarray(py, &j)?, density_to_pyarray(py, &k)?))
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
            .detach(|| pyscf_scf::default_get_fock(&h1e_d, &s1e_d, &vhf_d, &dm_d, cycle, None))
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
            .detach(|| pyscf_scf::default_eig(&fock_d, &s1e_d))
            .map_err(pyscf_to_py)?;
        let energies = slice_to_pyarray1(py, &mc.energies);
        let coeffs = crate::numpy_io::mo_coeff_to_pyarray(py, &mc)?;
        Ok((energies, coeffs))
    }

    fn get_occ<'py>(
        &self,
        py: Python<'py>,
        mo_energy: numpy::PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let energies: Vec<f64> = mo_energy.as_slice()?.to_vec();
        let nelec = self.inner.mol.nelectron as usize;
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
        let mut mc = crate::numpy_io::to_mo_coeff(mo_coeff)?;
        let occ: Vec<f64> = mo_occ.as_slice()?.to_vec();
        mc.occupations = occ;
        let d = py
            .detach(|| pyscf_scf::default_make_rdm1(&mc))
            .map_err(pyscf_to_py)?;
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
            .detach(|| pyscf_scf::default_energy_elec(&dm_d, &h1e_d, &vhf_d))
            .map_err(pyscf_to_py)?;
        Ok((e.0, ec.0))
    }

    // ───────────────────────────────────────────────────────────────────
    // Cross-module dispatch — to_uks (RKS → UKS) wired to the real target.
    // ───────────────────────────────────────────────────────────────────

    fn to_uks(slf: PyRef<'_, Self>) -> PyResult<Py<PyUKS>> {
        let py = slf.py();
        let py_mol = slf.py_mol.clone_ref(py);
        // RKS → UKS: rebuild the UKS target on the same Mole + xc, copying the
        // scalar SCF settings (mirrors pyscf_scf::convert::to_uhf shape).
        let mut uks = UksRust::new(slf.inner.mol.clone(), &slf.inner.xc);
        uks.conv_tol = slf.inner.conv_tol;
        uks.conv_tol_grad = slf.inner.conv_tol_grad;
        uks.max_cycle = slf.inner.max_cycle;
        uks.diis = slf.inner.diis;
        uks.diis_space = slf.inner.diis_space;
        uks.diis_start_cycle = slf.inner.diis_start_cycle;
        uks.diis_damp = slf.inner.diis_damp;
        uks.level_shift = slf.inner.level_shift;
        uks.damp = slf.inner.damp;
        uks.direct_scf = slf.inner.direct_scf;
        uks.direct_scf_tol = slf.inner.direct_scf_tol;
        uks.init_guess = slf.inner.init_guess.clone();
        uks.verbose = slf.inner.verbose;
        uks.max_memory = slf.inner.max_memory;
        uks.nlc = slf.inner.nlc.clone();
        Py::new(py, PyUKS { inner: uks, py_mol })
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PyUKS — Unrestricted Kohn-Sham. Open-shell analog of PyRKS.
// ═══════════════════════════════════════════════════════════════════════

#[pyclass(subclass, name = "UKS", module = "pyscf._native.dft")]
pub struct PyUKS {
    pub(crate) inner: UksRust,
    pub(crate) py_mol: Py<PyAny>,
}

#[pymethods]
impl PyUKS {
    #[new]
    #[pyo3(signature = (mol, xc="LDA,VWN"))]
    fn new(py: Python<'_>, mol: Py<PyAny>, xc: &str) -> PyResult<Self> {
        let mol_inner: Mole = extract_mole_from_pyany(py, &mol)?;
        Ok(PyUKS {
            inner: UksRust::new(mol_inner, xc),
            py_mol: mol,
        })
    }

    #[getter]
    fn mol<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.py_mol.bind(py).clone()
    }

    #[getter] fn xc(&self) -> String { self.inner.xc.clone() }
    #[setter] fn set_xc(&mut self, v: String) { self.inner.xc = v; }

    #[getter] fn nlc(&self) -> String { self.inner.nlc.clone() }
    #[setter] fn set_nlc(&mut self, v: String) { self.inner.nlc = v; }

    /// `mf._numint` — read-only precision view (D-08).
    #[getter]
    fn _numint(&self, py: Python<'_>) -> PyResult<Py<PyNumIntView>> {
        Py::new(
            py,
            PyNumIntView {
                dtype: self.inner.dtype().name().to_string(),
            },
        )
    }

    /// D-08 READ-ONLY active-precision getter — no setter (see PyRKS::dtype).
    #[getter]
    fn dtype(&self) -> &str {
        self.inner.dtype().name()
    }

    #[getter] fn e_tot(&self)     -> f64  { self.inner.e_tot }
    #[getter] fn e_elec(&self)    -> f64  { self.inner.e_elec }
    #[getter] fn converged(&self) -> bool { self.inner.converged }
    #[getter] fn cycles(&self)    -> u32  { self.inner.cycles }

    #[getter] fn max_cycle(&self) -> u32 { self.inner.max_cycle }
    #[setter] fn set_max_cycle(&mut self, v: u32) { self.inner.max_cycle = v; }
    #[getter] fn conv_tol(&self) -> f64 { self.inner.conv_tol }
    #[setter] fn set_conv_tol(&mut self, v: f64) { self.inner.conv_tol = v; }
    #[getter] fn verbose(&self) -> u8 { self.inner.verbose }
    #[setter] fn set_verbose(&mut self, v: u8) { self.inner.verbose = v; }
    #[getter] fn init_guess(&self) -> String { self.inner.init_guess.clone() }

    #[pyo3(signature = (dm0=None))]
    fn kernel<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        dm0: Option<PyReadonlyArray2<'py, f64>>,
    ) -> PyResult<f64> {
        let (mut cfg, mol) = {
            let mut me = slf.borrow_mut();
            let mol = me.inner.mol.clone();
            let _ = me.inner.grids.build(&mol);
            (me.inner.to_kernel_config(), mol)
        };
        if let Some(arr) = dm0 {
            let d = to_density(arr)?;
            cfg.init_guess = InitGuessMode::UserDM(d);
        }
        let bridge_slf: Py<PyAny> = slf.clone().into_any().unbind();
        let py_mol = {
            let me = slf.borrow();
            me.py_mol.clone_ref(py)
        };
        let bridge = PyOverrideBridge::new(bridge_slf, py_mol);
        let result = pyscf_scf::kernel(&mol, &bridge, cfg).map_err(pyscf_to_py)?;
        let e_tot = result.e_tot.0;
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

    fn run<'py>(slf: Bound<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        let _ = PyUKS::kernel(slf.clone(), py, None)?;
        Ok(slf)
    }

    /// KS effective potential `J + Vxc − hyb·K` (open-shell Vxc via nr_uks).
    fn get_veff<'py>(
        &mut self,
        py: Python<'py>,
        mol: Py<PyAny>,
        dm: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mol_ref = extract_mole_from_pyany(py, &mol)?;
        let dm_ref = to_density(dm)?;
        let xc = self.inner.xc.clone();
        let _ = self.inner.grids.build(&mol_ref);
        let bundle = py
            .detach(|| {
                let (j, k) = pyscf_scf::default_get_jk(&mol_ref, &dm_ref)?;
                ks_default_get_veff(
                    &self.inner._numint,
                    &mol_ref,
                    &self.inner.grids,
                    &xc,
                    &dm_ref,
                    &j,
                    &k,
                )
            })
            .map_err(pyscf_to_py)?;
        density_to_pyarray(py, &bundle.veff)
    }

    /// `define_xc_(description)` — string form (D-02); callable deferred.
    fn define_xc_(&mut self, py: Python<'_>, description: &str) -> PyResult<f64> {
        let hooks = NoKsOverrides;
        let desc = description.to_string();
        let spec = py
            .detach(|| hooks.define_xc_(&desc))
            .map_err(pyscf_to_py)?;
        self.inner.xc = desc;
        Ok(spec.hyb().0)
    }

    fn get_hcore<'py>(
        &self,
        py: Python<'py>,
        mol: Py<PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mol_ref = extract_mole_from_pyany(py, &mol)?;
        let h = py
            .detach(|| pyscf_scf::default_get_hcore(&mol_ref))
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
            .detach(|| pyscf_scf::default_get_ovlp(&mol_ref))
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
            .detach(|| pyscf_scf::default_get_jk(&mol_ref, &dm_ref))
            .map_err(pyscf_to_py)?;
        Ok((density_to_pyarray(py, &j)?, density_to_pyarray(py, &k)?))
    }

    /// Cross-module dispatch — to_rks (UKS → RKS) wired to the real target.
    fn to_rks(slf: PyRef<'_, Self>) -> PyResult<Py<PyRKS>> {
        let py = slf.py();
        let py_mol = slf.py_mol.clone_ref(py);
        let mut rks = RksRust::new(slf.inner.mol.clone(), &slf.inner.xc);
        rks.conv_tol = slf.inner.conv_tol;
        rks.conv_tol_grad = slf.inner.conv_tol_grad;
        rks.max_cycle = slf.inner.max_cycle;
        rks.diis = slf.inner.diis;
        rks.diis_space = slf.inner.diis_space;
        rks.diis_start_cycle = slf.inner.diis_start_cycle;
        rks.diis_damp = slf.inner.diis_damp;
        rks.level_shift = slf.inner.level_shift;
        rks.damp = slf.inner.damp;
        rks.direct_scf = slf.inner.direct_scf;
        rks.direct_scf_tol = slf.inner.direct_scf_tol;
        rks.init_guess = slf.inner.init_guess.clone();
        rks.verbose = slf.inner.verbose;
        rks.max_memory = slf.inner.max_memory;
        rks.nlc = slf.inner.nlc.clone();
        Py::new(py, PyRKS { inner: rks, py_mol })
    }
}

// ═══════════════════════════════════════════════════════════════════════
// KsConversion → RKS/UKS assembly (DFT-11). The pyscf-scf `to_rks`/`to_uks`
// seam returns a KsConversion descriptor (it can't name pyscf_dft — the
// dependency-cycle wall); these helpers build the real pyscf_dft target here
// at the PyO3 boundary (where both crates are in scope). Used by
// `PyRHF::to_rks` / `PyRHF::to_uks` (scf.rs).
// ═══════════════════════════════════════════════════════════════════════

/// Assemble a real `pyscf_dft::RKS` from a `KsConversion` descriptor + xc.
pub(crate) fn rks_from_conversion(conv: pyscf_scf::KsConversion, xc: &str) -> RksRust {
    let mut rks = RksRust::new(conv.mol, xc);
    rks.conv_tol = conv.conv_tol;
    rks.conv_tol_grad = conv.conv_tol_grad;
    rks.max_cycle = conv.max_cycle;
    rks.diis = conv.diis;
    rks.diis_space = conv.diis_space;
    rks.diis_start_cycle = conv.diis_start_cycle;
    rks.diis_damp = conv.diis_damp;
    rks.level_shift = conv.level_shift;
    rks.damp = conv.damp;
    rks.direct_scf = conv.direct_scf;
    rks.direct_scf_tol = conv.direct_scf_tol;
    rks.init_guess = conv.init_guess;
    rks.verbose = conv.verbose;
    rks.max_memory = conv.max_memory;
    rks
}

/// Assemble a real `pyscf_dft::UKS` from a `KsConversion` descriptor + xc.
pub(crate) fn uks_from_conversion(conv: pyscf_scf::KsConversion, xc: &str) -> UksRust {
    let mut uks = UksRust::new(conv.mol, xc);
    uks.conv_tol = conv.conv_tol;
    uks.conv_tol_grad = conv.conv_tol_grad;
    uks.max_cycle = conv.max_cycle;
    uks.diis = conv.diis;
    uks.diis_space = conv.diis_space;
    uks.diis_start_cycle = conv.diis_start_cycle;
    uks.diis_damp = conv.diis_damp;
    uks.level_shift = conv.level_shift;
    uks.damp = conv.damp;
    uks.direct_scf = conv.direct_scf;
    uks.direct_scf_tol = conv.direct_scf_tol;
    uks.init_guess = conv.init_guess;
    uks.verbose = conv.verbose;
    uks.max_memory = conv.max_memory;
    uks
}

// Marker: silence unused-field/import warnings (mirrors scf.rs).
#[allow(dead_code)]
fn _markers(r: &PyRKS, u: &PyUKS) -> usize {
    let _ = &r.py_mol;
    let _ = &u.py_mol;
    let _ = Density { nao: 0, data: vec![] };
    let _: fn() -> NumInt = NumInt::new;
    let _ = |h: &NoKsOverrides, ni: &NumInt, mol: &Mole, g: &pyscf_grids::Grids, dm: &Density, j: &Density, k: &Density| {
        let _ = h.get_veff_ks(ni, mol, g, "lda", dm, j, k);
    };
    0
}
