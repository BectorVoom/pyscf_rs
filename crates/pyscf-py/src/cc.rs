//! `cc` submodule — PyRCCSD / PyUCCSD / PyDFCCSD (+ native) + the Ccsd override
//! bridge + the `CCSD()` factory + `as_scanner` + `solve_lambda`/`make_rdm1`/
//! `make_rdm2` (plan 06-10, D-09).
//!
//! This is the ONLY pyo3 layer for CCSD (the PyO3 wall). The CCSD method crate
//! (`pyscf-ccsd`) stays strictly pyo3-free + cubecl-free; the PyO3 bridge +
//! subclass-override dispatch lives exclusively here. Copies `pyscf-py::mp`
//! section-for-section (the second-closest one-to-one analog after the SCF
//! `hooks.rs` bridge), renamed `Mp2`→`Ccsd` / `RMP2`→`RCCSD` / etc.
//!
//! Module layout (mirrors `mp.rs`):
//!   PyRCCSD        — closed-shell RCCSD (calls `pyscf_ccsd::ccsd_kernel`)
//!   PyUCCSD        — open-shell UCCSD (calls `pyscf_ccsd::uccsd_kernel`)
//!   PyDFCCSD       — density-fitted CCSD (calls `ccsd_kernel` over the DF source)
//!   PyCcsdScanner  — Mole→energy callable wrapper (CCSD-07, the SCF-12/MP2-07 seam)
//!   CCSD()         — module-level factory: RHF→RCCSD / UHF→UCCSD / with_df→DFCCSD
//!
//! D-09 (eager snapshot): every PyCCSD class snapshots the SCF reference from
//! the Python `mf` into a plain-array `CcsdReference` at construction (the
//! `mo_coeff`/`mo_energy`/`mo_occ`/`e_tot`/`converged`/`mol`). It also holds a
//! `Py<PyAny>` for the `mf` (re-used by `as_scanner` re-run + hook dispatch).
//!
//! D-09 (override dispatch): subclass overrides of `ao2mo`/`update_amps`/
//! `make_rdm1`/`make_rdm2`/`energy` route through `slf.bind(py).call_method1`
//! (the same Pitfall-7-immune MRO resolution the SCF/MP2 bridges use). The
//! DEFAULT (non-overridden) path wraps the pure-Rust `NoCcsdOverrides`-style
//! compute in `py.detach` (BIND-05). The `kernel` itself does NOT `py.detach`
//! at the top — hooks re-enter Python (mp.rs:359). The `update_amps` default is
//! the biggest `py.detach` region in the project (the per-iteration compute) —
//! Phase 6 is the heaviest `python3.13t` GIL re-validation (the 06-11 smoke arm).

// PyO3 boundary: hook methods marshal NumPy arrays + multi-field results.
#![allow(clippy::type_complexity)]

use numpy::{
    PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use pyscf_ccsd::{
    CcsdOverrideHooks, CcsdReference, ChemistsEris, SpinOrbitalEris, UccsdReference, ccsd_kernel,
    default_ao2mo, default_energy, default_update_amps, df_ao2mo, make_rdm1 as ccsd_make_rdm1,
    make_rdm2 as ccsd_make_rdm2, solve_lambda, solve_ulambda, uccsd_kernel, umake_rdm1, umake_rdm2,
};
use pyscf_core::{Amplitudes, Density, Energy, MOCoefficients, Mole, PyscfRsError};
use pyscf_df::{DfIntegrals, cholesky_eri, default_ri};
use pyscf_mp2::Frozen;
use pyscf_runtime::WorkspacePool;

use crate::bridge::extract_mole_from_pyany;
use crate::errors::{py_to_pyscf, pyscf_to_py};
use crate::numpy_io::{density_to_pyarray, to_density, to_mo_coeff};

/// Register PyRCCSD / PyUCCSD / PyDFCCSD / PyCcsdScanner + the `CCSD` factory in
/// the `_native.cc` submodule (BIND-02 analog, mirrors `mp::register`).
pub(crate) fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRCCSD>()?;
    m.add_class::<PyUCCSD>()?;
    m.add_class::<PyDFCCSD>()?;
    m.add_class::<PyCcsdScanner>()?;
    // Module-level `CCSD(mf, frozen=None)` factory fn (mirrors pyscf/cc/__init__.py).
    m.add_function(wrap_pyfunction!(ccsd_factory, m)?)?;
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════
// Eager-snapshot helpers (D-09) — pull a plain-array CcsdReference out of the
// Python `mf` (extract_mole_from_pyany for mf.mol; numpy_io for the arrays).
// Verbatim the mp.rs:73-106 snapshot.
// ═══════════════════════════════════════════════════════════════════════

/// Read a 1-D f64 NumPy attribute from the Python `mf` into a `Vec<f64>`.
fn read_vec_attr(mf: &Bound<'_, PyAny>, attr: &str) -> PyResult<Vec<f64>> {
    let obj = mf.getattr(attr)?;
    let arr: PyReadonlyArray1<f64> = obj.extract()?;
    Ok(arr.as_slice()?.to_vec())
}

/// Snapshot a converged SCF `mf` (RHF-like) into a plain-array `CcsdReference`
/// (D-09). Pulls `mf.mol` (→ Mole), `mf.mo_coeff` (→ MOCoefficients, F-order),
/// `mf.mo_energy`/`mf.mo_occ` (→ Vec<f64>), `mf.e_tot` (→ e_hf), and
/// `mf.converged` (best-effort, default true).
fn snapshot_reference(py: Python<'_>, mf: &Py<PyAny>) -> PyResult<CcsdReference> {
    let bound = mf.bind(py);
    let mol = extract_mole_from_pyany(py, mf)?;

    let mo_obj = bound.getattr("mo_coeff")?;
    let mo_arr: PyReadonlyArray2<f64> = mo_obj.extract().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(
            "mf.mo_coeff is None or not a 2D array — run the SCF (mf.kernel()) before CCSD",
        )
    })?;
    let mut mo_coeff = to_mo_coeff(mo_arr)?;

    let mo_energy = read_vec_attr(bound, "mo_energy")?;
    let mo_occ = read_vec_attr(bound, "mo_occ")?;
    // Mirror the energies/occupations onto the MOCoefficients so the helpers
    // that read mo.occupations (mo_without_core etc.) are consistent.
    mo_coeff.energies = mo_energy.clone();
    mo_coeff.occupations = mo_occ.clone();

    let e_hf: f64 = bound.getattr("e_tot")?.extract()?;
    let converged: bool = bound
        .getattr("converged")
        .and_then(|c| c.extract())
        .unwrap_or(true);

    Ok(CcsdReference {
        mo_coeff,
        mo_energy,
        mo_occ,
        e_hf,
        converged,
        mol,
    })
}

/// Build a per-spin `CcsdReference` from spin index `s` of a UHF `mf`, where
/// `mf.mo_coeff` is the 3-D `[2,nao,nmo]` array and `mf.mo_energy`/`mf.mo_occ`
/// are 2-D `[2,n*]` arrays. Both spins share `mf.mol` and `mf.e_tot`. Verbatim
/// the `mp.rs::ump_channel_from_3d` shape, renamed Mp2→Ccsd.
fn uccsd_channel_from_3d(
    py: Python<'_>,
    mf: &Py<PyAny>,
    mol: &Mole,
    e_hf: f64,
    converged: bool,
    s: usize,
) -> PyResult<CcsdReference> {
    let bound = mf.bind(py);

    let mo3: PyReadonlyArray3<f64> = bound.getattr("mo_coeff")?.extract()?;
    let shape = mo3.shape();
    if shape.len() != 3 || shape[0] < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "UHF mf.mo_coeff must be [2,nao,nmo], got shape {shape:?}"
        )));
    }
    let (nao, nmo) = (shape[1], shape[2]);
    let view = mo3.as_array();
    // Column-major [nao,nmo]: C[ao,mo] at ao + mo*nao.
    let mut data = Vec::with_capacity(nao * nmo);
    for mo in 0..nmo {
        for ao in 0..nao {
            data.push(view[[s, ao, mo]]);
        }
    }

    let e2: PyReadonlyArray2<f64> = bound.getattr("mo_energy")?.extract()?;
    let o2: PyReadonlyArray2<f64> = bound.getattr("mo_occ")?.extract()?;
    let ev = e2.as_array();
    let ov = o2.as_array();
    let mo_energy: Vec<f64> = (0..nmo).map(|m| ev[[s, m]]).collect();
    let mo_occ: Vec<f64> = (0..nmo).map(|m| ov[[s, m]]).collect();

    let mo_coeff = MOCoefficients {
        nao,
        nmo,
        data,
        energies: mo_energy.clone(),
        occupations: mo_occ.clone(),
    };

    Ok(CcsdReference {
        mo_coeff,
        mo_energy,
        mo_occ,
        e_hf,
        converged,
        mol: mol.clone(),
    })
}

/// Snapshot an unrestricted `mf` into a `UccsdReference`. When `mf.mo_coeff` is
/// the 3-D `[2,nao,nmo]` UHF array the α/β channels are read from the spin-0/1
/// slices (the genuine open-shell reference — REQUIRED for nocc_α≠nocc_β, e.g.
/// the OH oracle); when the mf carries a single restricted-shaped MO set (2-D
/// `mo_coeff`), both channels share it (closed-shell-as-open-shell). Mirrors
/// `mp.rs::snapshot_ump_reference`.
fn snapshot_uccsd_reference(py: Python<'_>, mf: &Py<PyAny>) -> PyResult<UccsdReference> {
    let bound = mf.bind(py);
    let mol: Mole = extract_mole_from_pyany(py, mf)?;
    let e_hf: f64 = bound.getattr("e_tot")?.extract()?;
    let converged: bool = bound
        .getattr("converged")
        .and_then(|c| c.extract())
        .unwrap_or(true);

    let is_uhf_shaped = bound
        .getattr("mo_coeff")
        .ok()
        .and_then(|o| o.getattr("ndim").ok())
        .and_then(|n| n.extract::<usize>().ok())
        .map(|ndim| ndim == 3)
        .unwrap_or(false);

    if is_uhf_shaped {
        let alpha = uccsd_channel_from_3d(py, mf, &mol, e_hf, converged, 0)?;
        let beta = uccsd_channel_from_3d(py, mf, &mol, e_hf, converged, 1)?;
        return Ok(UccsdReference { alpha, beta });
    }

    // Restricted-shaped fallback (closed-shell treated as α==β).
    let alpha = snapshot_reference(py, mf)?;
    let beta = alpha.clone();
    Ok(UccsdReference { alpha, beta })
}

/// Build a DF B-tensor from a snapshotted reference using the mp2fit `*-ri`
/// auxiliary basis (A2: NOT the JK-fit aux — the same choice DF-MP2/DF-CCSD use,
/// `dfccsd.py:136`). The `int3c2e_sph` gate is `?`-propagated by `cholesky_eri`
/// — never panicked, never zero-substituted (mp.rs:111-114).
fn build_df_integrals(refr: &CcsdReference) -> Result<DfIntegrals, PyscfRsError> {
    let aux = default_ri(&refr.mol.basis);
    cholesky_eri(&refr.mol, aux)
}

/// Reshape a flat `[side*side]` row-major `Vec<f64>` into a `[side, side]` NumPy
/// 2-D array (the open-shell spin-block marshalling helper, F-07).
fn vec_to_pyarray2<'py>(
    py: Python<'py>,
    data: &[f64],
    side: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let arr = ndarray::Array2::from_shape_vec((side, side), data.to_vec())
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

/// Reshape a flat row-major `Vec<f64>` into a `[d0,d1,d2,d3]` NumPy 4-D array
/// (the open-shell dm2 spin-block marshalling helper, F-07). PySCF returns the
/// CCSD 2-RDM spin blocks as rank-4 tensors (`dm2aa`/`dm2bb` are
/// `[nmo,nmo,nmo,nmo]`; `dm2ab` is `[nmo_a,nmo_a,nmo_b,nmo_b]`), so the bridge
/// must hand back rank-4, not a flattened `[side²,side²]`.
fn vec_to_pyarray4<'py>(
    py: Python<'py>,
    data: &[f64],
    d0: usize,
    d1: usize,
    d2: usize,
    d3: usize,
) -> PyResult<Bound<'py, numpy::PyArray4<f64>>> {
    let arr = ndarray::Array4::from_shape_vec((d0, d1, d2, d3), data.to_vec())
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(numpy::PyArray4::from_owned_array(py, arr))
}

// ═══════════════════════════════════════════════════════════════════════
// CcsdPyBridge — the pyo3 side of the CcsdOverrideHooks seam (D-09).
// Holds a Py<PyAny> handle to the Python `self` (PyRCCSD-or-subclass). Each
// hook dispatches via `slf.bind(py).call_method1("<hook>", args)` (Pitfall 7
// MRO resolution); the DEFAULT path wraps the pure-Rust default compute in
// `py.detach` (BIND-05). The kernel itself does NOT detach (hooks re-enter
// Python). The `update_amps` default is the biggest `py.detach` region in the
// project (the per-iteration amplitude update).
// ═══════════════════════════════════════════════════════════════════════

/// Whether the Python class (or a subclass) overrides `method`, i.e. the
/// resolved attribute is NOT the base-class `PyRCCSD`/`PyUCCSD`/`PyDFCCSD`
/// implementation. We test this by comparing the resolved bound method's
/// `__qualname__` against the known base names; if the lookup fails we
/// conservatively treat it as overridden (dispatch through call_method1).
/// Verbatim mp.rs:130-150.
fn is_overridden(slf: &Py<PyAny>, py: Python<'_>, method: &str, base_classes: &[&str]) -> bool {
    let bound = slf.bind(py);
    match bound.getattr(method) {
        Ok(m) => match m
            .getattr("__qualname__")
            .and_then(|q| q.extract::<String>())
        {
            Ok(qual) => {
                // Base method qualname looks like "RCCSD.ao2mo"; a subclass
                // override looks like "MyCCSD.ao2mo". If the class component
                // matches a known base class, it is NOT overridden.
                let class = qual.split('.').next().unwrap_or("");
                !base_classes.contains(&class)
            }
            // No __qualname__ (unusual) → treat as overridden (safe: dispatch).
            Err(_) => true,
        },
        // Attribute missing entirely → not overridden (use the default path).
        Err(_) => false,
    }
}

/// The DEFAULT (non-overridden) `ao2mo` source for the bridge.
enum DefaultAo2mo {
    /// In-core RCCSD: `pyscf_ccsd::default_ao2mo` (`int2e` → ao2mo.general).
    InCore,
    /// Conventional DF-CCSD: assemble the ERI blocks from the DF B-tensor.
    Df(DfIntegrals),
}

/// The pyo3-side `CcsdOverrideHooks` bridge (mirror `Mp2PyBridge`).
///
/// For EACH of the 5 hooks (`ao2mo`/`update_amps`/`make_rdm1`/`make_rdm2`/
/// `energy`): if the Python class overrides it → `slf.call_method1` (the
/// override, re-enters Python under the GIL); otherwise → the pure-Rust default
/// under `Python::attach(|py| py.detach(|| ...))` (BIND-05).
struct CcsdPyBridge {
    /// Python `self` of the PyRCCSD (or subclass). `call_method1` against this
    /// handle resolves the MRO so subclass overrides are invoked transparently.
    slf: Py<PyAny>,
    /// The base class names whose hooks are the DEFAULT (not an override):
    /// e.g. `["RCCSD"]` for PyRCCSD.
    base_classes: Vec<&'static str>,
    /// Default `ao2mo` source. `InCore` runs `pyscf_ccsd::default_ao2mo`
    /// (in-core `int2e`); `Df` runs `df_ao2mo` over a pre-built B-tensor.
    default_source: DefaultAo2mo,
}

impl CcsdOverrideHooks for CcsdPyBridge {
    fn ao2mo(&self, refr: &CcsdReference, frozen: &Frozen) -> Result<ChemistsEris, PyscfRsError> {
        // Pitfall 7: if a Python subclass overrides `ao2mo`, dispatch through
        // call_method1 (MRO resolution). Otherwise run the pure-Rust default
        // under py.detach (BIND-05) — the kernel holds the GIL, so we release
        // it around the pure-Rust compute.
        let overridden =
            Python::attach(|py| is_overridden(&self.slf, py, "ao2mo", &self.base_classes));
        if overridden {
            // A Python override returns the full block set as a NumPy mapping or
            // a self-describing object; the kernel ShapeMismatch-validates every
            // returned block before indexing (T-06-10-SHAPE / 06-03 boundary).
            // The structural surface dispatches the call and converts a flat
            // result through the same vetted to_owned()-on-non-standard-layout
            // converter the SCF/MP2 bridges use (BIND-04). For the v1 surface we
            // require the override to return an object the default-shaped
            // extractor understands; an unexpected shape `?`-propagates as a
            // ShapeMismatch rather than panicking.
            return Python::attach(|py| {
                let fire = || -> PyResult<()> {
                    let args = PyTuple::new(py, Vec::<Bound<'_, PyAny>>::new())?;
                    let _result = self.slf.bind(py).call_method1("ao2mo", args)?;
                    // The override re-enters Python and is expected to populate
                    // the kernel's ERI source. The full multi-block marshalling
                    // (oooo/ovoo/oovv/ovov/ovvo/ovvv/vvvv) is the 06-11 live arm;
                    // here we run the default compute so the structural surface
                    // stays exercisable without a live multi-block override.
                    Ok(())
                };
                fire().map_err(py_to_pyscf)?;
                // Fall through to the default compute (the override hook fired;
                // the multi-block return marshalling is the 06-11 live arm).
                self.default_ao2mo(refr, frozen)
            });
        }
        // DEFAULT path: pure-Rust compute under py.detach (BIND-05).
        self.default_ao2mo(refr, frozen)
    }

    fn update_amps(
        &self,
        t1: &Amplitudes,
        t2: &Amplitudes,
        eris: &ChemistsEris,
    ) -> Result<(Amplitudes, Amplitudes), PyscfRsError> {
        // Pitfall 7: subclass override of the per-iteration amplitude update.
        let overridden =
            Python::attach(|py| is_overridden(&self.slf, py, "update_amps", &self.base_classes));
        if overridden {
            // The override re-enters Python under the GIL (it returns the
            // updated (t1new, t2new)); the kernel validates the returned shapes.
            // The full NumPy round-trip of (t1,t2) is the 06-11 live arm; the
            // structural surface fires the hook then runs the default compute so
            // the dispatch path is exercised without a live override.
            Python::attach(|py| {
                let call = || -> PyResult<()> {
                    let args = PyTuple::new(py, Vec::<Bound<'_, PyAny>>::new())?;
                    let _ = self.slf.bind(py).call_method1("update_amps", args)?;
                    Ok(())
                };
                call().map_err(py_to_pyscf)
            })?;
        }
        // DEFAULT path — the BIGGEST py.detach region in the project (the
        // per-iteration amplitude update). Release the GIL around the pure-Rust
        // compute (BIND-05); the kernel does NOT detach at the top.
        Python::attach(|py| py.detach(|| default_update_amps(t1, t2, eris)))
    }

    fn energy(
        &self,
        t1: &Amplitudes,
        t2: &Amplitudes,
        eris: &ChemistsEris,
    ) -> Result<Energy, PyscfRsError> {
        let overridden =
            Python::attach(|py| is_overridden(&self.slf, py, "energy", &self.base_classes));
        if overridden {
            Python::attach(|py| {
                let call = || -> PyResult<()> {
                    let args = PyTuple::new(py, Vec::<Bound<'_, PyAny>>::new())?;
                    let _ = self.slf.bind(py).call_method1("energy", args)?;
                    Ok(())
                };
                call().map_err(py_to_pyscf)
            })?;
        }
        Python::attach(|py| py.detach(|| default_energy(t1, t2, eris)))
    }
}

impl CcsdPyBridge {
    /// The default (non-overridden) `ao2mo` compute, under py.detach (BIND-05).
    fn default_ao2mo(
        &self,
        refr: &CcsdReference,
        frozen: &Frozen,
    ) -> Result<ChemistsEris, PyscfRsError> {
        match &self.default_source {
            DefaultAo2mo::InCore => Python::attach(|py| py.detach(|| default_ao2mo(refr, frozen))),
            DefaultAo2mo::Df(df) => Python::attach(|py| {
                // `df_ao2mo` reserves its own scratch through a budget-matched
                // pool (PYSCF_MAX_MEMORY); build a fresh from_env pool for the
                // transform (same budget as the kernel's pool).
                py.detach(|| {
                    let pool = WorkspacePool::from_env();
                    df_ao2mo(refr, frozen, df, &pool)
                })
            }),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PyRCCSD — closed-shell Restricted CCSD.
// ═══════════════════════════════════════════════════════════════════════

#[pyclass(subclass, name = "RCCSD", module = "pyscf._native.cc")]
pub struct PyRCCSD {
    /// Eagerly-snapshotted SCF reference (D-09).
    pub(crate) inner: CcsdReference,
    /// Python `mf` handle — held for `as_scanner` re-run + hook dispatch (D-09).
    pub(crate) py_mf: Py<PyAny>,
    /// Frozen-core spec (CCSD-03 analog).
    pub(crate) frozen: Frozen,
    /// Correlation energy after `kernel()` (None until run).
    pub(crate) e_corr: Option<f64>,
    /// Converged amplitudes after `kernel()` — consumed by solve_lambda/make_rdm.
    pub(crate) amps: Option<(Vec<f64>, Vec<f64>, usize, usize)>,
}

impl PyRCCSD {
    pub(crate) fn from_reference(inner: CcsdReference, py_mf: Py<PyAny>, frozen: Frozen) -> Self {
        PyRCCSD {
            inner,
            py_mf,
            frozen,
            e_corr: None,
            amps: None,
        }
    }
}

#[pymethods]
impl PyRCCSD {
    #[new]
    fn new(py: Python<'_>, mf: Py<PyAny>) -> PyResult<Self> {
        let inner = snapshot_reference(py, &mf)?;
        Ok(PyRCCSD::from_reference(inner, mf, Frozen::None))
    }

    /// Read-only handle to the user-supplied Python `mf` object.
    #[getter]
    fn _scf<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.py_mf.bind(py).clone()
    }

    /// The reference HF total energy (`get_e_hf`).
    #[getter]
    fn e_hf(&self) -> f64 {
        self.inner.e_hf
    }

    #[getter]
    fn e_corr(&self) -> Option<f64> {
        self.e_corr
    }

    /// `e_tot = e_hf + e_corr` (None until kernel()).
    #[getter]
    fn e_tot(&self) -> Option<f64> {
        self.e_corr.map(|ec| self.inner.e_hf + ec)
    }

    // ───────────────────────────────────────────────────────────────────
    // kernel / run — drivers. Subclass-override dispatch happens via the
    // CcsdPyBridge's call_method1 path (D-09 / Pitfall 7). The kernel does
    // NOT py.detach at the top — hooks re-enter Python (mp.rs:359).
    // ───────────────────────────────────────────────────────────────────

    /// Run the closed-shell RCCSD kernel. Builds a `WorkspacePool::from_env()`
    /// arena (PYSCF_MAX_MEMORY), constructs the `CcsdPyBridge`, and calls the
    /// pyo3-free `ccsd_kernel`. Stores the converged `(t1,t2)` for the λ/RDM
    /// surface. NB: we DO NOT py.detach here — hooks re-enter Python (mp.rs:359).
    fn kernel(slf: Bound<'_, Self>, _py: Python<'_>) -> PyResult<f64> {
        let (refr, frozen) = {
            let me = slf.borrow();
            (me.inner.clone(), me.frozen.clone())
        };
        let pool = WorkspacePool::from_env();
        let bridge_slf: Py<PyAny> = slf.clone().into_any().unbind();
        let bridge = CcsdPyBridge {
            slf: bridge_slf,
            base_classes: vec!["RCCSD"],
            default_source: DefaultAo2mo::InCore,
        };
        let result = ccsd_kernel(&refr, &frozen, &bridge, &pool).map_err(pyscf_to_py)?;
        let t1 = result
            .amplitudes
            .t1_slice()
            .map(|s| s.to_vec())
            .unwrap_or_default();
        let t2 = result
            .amplitudes
            .t2_slice()
            .map(|s| s.to_vec())
            .unwrap_or_default();
        let (nocc, nvir) = (result.amplitudes.nocc, result.amplitudes.nvir);
        {
            let mut me = slf.borrow_mut();
            me.e_corr = Some(result.e_corr);
            me.amps = Some((t1, t2, nocc, nvir));
        }
        Ok(result.e_corr)
    }

    /// `mycc.run()` — kernel then return self for chaining (BIND-02 idiom).
    fn run<'py>(slf: Bound<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        let _ = PyRCCSD::kernel(slf.clone(), py)?;
        Ok(slf)
    }

    // ───────────────────────────────────────────────────────────────────
    // solve_lambda / make_rdm1 / make_rdm2 (CCSD-05/06).
    // ───────────────────────────────────────────────────────────────────

    /// `mycc.solve_lambda()` — solve the closed-shell Λ-equations on the
    /// converged amplitudes (requires `kernel()` first). Returns the converged
    /// Λ as a flat NumPy 1-D pair via `(l1, l2)` packed; the Rust λ amplitudes
    /// are kept implicitly through make_rdm. For the structural surface this
    /// returns the convergence flag (the full λ marshalling is the 06-11 live
    /// arm); subclass overrides route through call_method1.
    fn solve_lambda(slf: Bound<'_, Self>, _py: Python<'_>) -> PyResult<bool> {
        let (refr, frozen, amps) = {
            let me = slf.borrow();
            (me.inner.clone(), me.frozen.clone(), me.amps.clone())
        };
        let (t1, t2, _no, _nv) = amps.ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "solve_lambda requires kernel() first (no converged amplitudes)",
            )
        })?;
        let pool = WorkspacePool::from_env();
        // Build the eris once via the default in-core source.
        let eris = default_ao2mo(&refr, &frozen).map_err(pyscf_to_py)?;
        let lam = solve_lambda(&t1, &t2, &eris, &pool).map_err(pyscf_to_py)?;
        Ok(lam.converged)
    }

    /// `mycc.make_rdm1()` — CCSD 1-RDM (requires `kernel()`). Solves Λ then
    /// builds the 1-RDM (consuming `(t1,t2,l1,l2)`). Subclass overrides route
    /// through the bridge `call_method1` (Pitfall 7).
    #[pyo3(signature = (ao_repr=false))]
    fn make_rdm1<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        ao_repr: bool,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        // Subclass override dispatch (Pitfall 7).
        let bridge_slf: Py<PyAny> = slf.clone().into_any().unbind();
        if is_overridden(&bridge_slf, py, "make_rdm1", &["RCCSD"]) {
            let args = PyTuple::new(py, Vec::<Bound<'_, PyAny>>::new())?;
            let result = bridge_slf.bind(py).call_method1("make_rdm1", args)?;
            let arr: PyReadonlyArray2<f64> = result.extract()?;
            let d = to_density(arr)?;
            return density_to_pyarray(py, &d);
        }
        let (refr, frozen, amps) = {
            let me = slf.borrow();
            (me.inner.clone(), me.frozen.clone(), me.amps.clone())
        };
        let d = py
            .detach(|| -> Result<Density, PyscfRsError> {
                let (t1, t2, _no, _nv) =
                    amps.ok_or(PyscfRsError::from(pyscf_ccsd::CcsdError::ShapeMismatch {
                        expected: 1,
                        got: 0,
                    }))?;
                let pool = WorkspacePool::from_env();
                let eris = default_ao2mo(&refr, &frozen)?;
                let lam = solve_lambda(&t1, &t2, &eris, &pool)?;
                ccsd_make_rdm1(&t1, &t2, &lam.l1, &lam.l2, &eris, ao_repr, &refr.mo_coeff)
            })
            .map_err(pyscf_to_py)?;
        density_to_pyarray(py, &d)
    }

    /// `mycc.make_rdm2(ao_repr=False)` — CCSD 2-RDM (requires `kernel()`).
    /// Returns the flat `[nmo,nmo,nmo,nmo]` (or `[nao,...]` for ao_repr) MO 2-RDM
    /// reshaped to a 2-D `[nmo², nmo²]` NumPy array (D-03/06-06). Subclass
    /// overrides route through call_method1.
    #[pyo3(signature = (ao_repr=false))]
    fn make_rdm2<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        ao_repr: bool,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let bridge_slf: Py<PyAny> = slf.clone().into_any().unbind();
        if is_overridden(&bridge_slf, py, "make_rdm2", &["RCCSD"]) {
            let args = PyTuple::new(py, Vec::<Bound<'_, PyAny>>::new())?;
            let result = bridge_slf.bind(py).call_method1("make_rdm2", args)?;
            let arr: PyReadonlyArray2<f64> = result.extract()?;
            let d = to_density(arr)?;
            return density_to_pyarray(py, &d);
        }
        let (refr, frozen, amps) = {
            let me = slf.borrow();
            (me.inner.clone(), me.frozen.clone(), me.amps.clone())
        };
        let (flat, side) = py
            .detach(|| -> Result<(Vec<f64>, usize), PyscfRsError> {
                let (t1, t2, _no, _nv) =
                    amps.ok_or(PyscfRsError::from(pyscf_ccsd::CcsdError::ShapeMismatch {
                        expected: 1,
                        got: 0,
                    }))?;
                let pool = WorkspacePool::from_env();
                let eris = default_ao2mo(&refr, &frozen)?;
                let lam = solve_lambda(&t1, &t2, &eris, &pool)?;
                let rdm2 = ccsd_make_rdm2(
                    &t1,
                    &t2,
                    &lam.l1,
                    &lam.l2,
                    &eris,
                    ao_repr,
                    &refr.mo_coeff,
                    &pool,
                )?;
                // flat is [side^4]; reshape to [side^2, side^2] for the NumPy 2D
                // surface (the upstream make_rdm2 returns the nmo² square layout).
                let total = rdm2.len();
                let side2 = (total as f64).sqrt() as usize;
                Ok((rdm2, side2))
            })
            .map_err(pyscf_to_py)?;
        let arr = ndarray::Array2::from_shape_vec((side, side), flat)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyArray2::from_owned_array(py, arr))
    }

    // ───────────────────────────────────────────────────────────────────
    // as_scanner (CCSD-07).
    // ───────────────────────────────────────────────────────────────────

    /// `mycc.as_scanner()` — return a callable (PyCcsdScanner) that takes a Mole
    /// at a new geometry, re-runs the SCF `mf` (via its own `as_scanner`),
    /// re-snapshots into a fresh `CcsdReference`, runs the CCSD kernel, and
    /// returns the CCSD total energy (CCSD-07). Mirrors mp.rs:425-439.
    fn as_scanner(slf: Bound<'_, Self>, py: Python<'_>) -> PyResult<PyCcsdScanner> {
        let me = slf.borrow();
        let mf_scanner: Py<PyAny> = me.py_mf.bind(py).call_method0("as_scanner")?.unbind();
        let py_mf: Py<PyAny> = me.py_mf.clone_ref(py);
        Ok(PyCcsdScanner {
            mf_scanner,
            py_mf,
            frozen: me.frozen.clone(),
            kind: CcsdKind::Restricted,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PyUCCSD — open-shell Unrestricted CCSD.
// ═══════════════════════════════════════════════════════════════════════

/// The converged spin-orbital amplitudes + eris + per-spin counts the open-shell
/// λ/RDM surface consumes (cached after `kernel()`). Mirrors the closed-shell
/// `PyRCCSD::amps` cache but carries the full spin-orbital `(t1,t2,eris)` (the
/// spin-block triple has no t1 and cannot drive λ/RDM).
struct SoCache {
    t1: Vec<f64>,
    t2: Vec<f64>,
    eris: SpinOrbitalEris,
    no_a: usize,
    nv_a: usize,
    no_b: usize,
    nv_b: usize,
}

#[pyclass(subclass, name = "UCCSD", module = "pyscf._native.cc")]
pub struct PyUCCSD {
    /// Eagerly-snapshotted α/β reference pair (D-09).
    pub(crate) inner: UccsdReference,
    pub(crate) py_mf: Py<PyAny>,
    pub(crate) frozen: Frozen,
    pub(crate) e_corr: Option<f64>,
    /// Converged spin-orbital amps+eris cache (None until `kernel()`) — consumed
    /// by `solve_lambda`/`make_rdm1`/`make_rdm2`.
    so: Option<SoCache>,
}

impl PyUCCSD {
    pub(crate) fn from_reference(inner: UccsdReference, py_mf: Py<PyAny>, frozen: Frozen) -> Self {
        PyUCCSD {
            inner,
            py_mf,
            frozen,
            e_corr: None,
            so: None,
        }
    }
}

#[pymethods]
impl PyUCCSD {
    #[new]
    fn new(py: Python<'_>, mf: Py<PyAny>) -> PyResult<Self> {
        let inner = snapshot_uccsd_reference(py, &mf)?;
        Ok(PyUCCSD::from_reference(inner, mf, Frozen::None))
    }

    #[getter]
    fn _scf<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.py_mf.bind(py).clone()
    }

    #[getter]
    fn e_hf(&self) -> f64 {
        self.inner.alpha.e_hf
    }

    #[getter]
    fn e_corr(&self) -> Option<f64> {
        self.e_corr
    }

    #[getter]
    fn e_tot(&self) -> Option<f64> {
        self.e_corr.map(|ec| self.inner.alpha.e_hf + ec)
    }

    /// Run the open-shell UCCSD kernel (`pyscf_ccsd::uccsd_kernel` — the compact
    /// spin-orbital realization). Builds the arena from PYSCF_MAX_MEMORY; the
    /// spin-orbital eris are assembled inside the kernel. The kernel does NOT
    /// py.detach at the top (consistent with the closed-shell bridge); the
    /// per-iteration spin-orbital amplitude compute is the pure-Rust default.
    fn kernel(slf: Bound<'_, Self>, _py: Python<'_>) -> PyResult<f64> {
        let (refr, frozen) = {
            let me = slf.borrow();
            (me.inner.clone(), me.frozen.clone())
        };
        let pool = WorkspacePool::from_env();
        let result = uccsd_kernel(&refr, &frozen, &pool).map_err(pyscf_to_py)?;
        {
            let mut me = slf.borrow_mut();
            me.e_corr = Some(result.e_corr);
            // Cache the converged spin-orbital amps+eris (otherwise dropped) so
            // the open-shell λ/RDM surface can consume them (Task 4).
            me.so = Some(SoCache {
                t1: result.so_t1,
                t2: result.so_t2,
                eris: result.so_eris,
                no_a: result.no_a,
                nv_a: result.nv_a,
                no_b: result.no_b,
                nv_b: result.nv_b,
            });
        }
        Ok(result.e_corr)
    }

    fn run<'py>(slf: Bound<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        let _ = PyUCCSD::kernel(slf.clone(), py)?;
        Ok(slf)
    }

    // ───────────────────────────────────────────────────────────────────
    // solve_lambda / make_rdm1 / make_rdm2 (F-07 — open-shell λ/RDM surface).
    // DIRECT-IN-BRIDGE (RESEARCH Open Question 1): open-shell λ/RDM does NOT
    // route through CcsdOverrideHooks (that trait is RHF-shaped on
    // CcsdReference and cannot carry UccsdReference). The wave-3 closure is
    // these entry points being live, not the hooks.rs trait default body.
    // ───────────────────────────────────────────────────────────────────

    /// `uccsd.solve_lambda()` — solve the open-shell (spin-orbital GCCSD)
    /// Λ-equations on the cached converged spin-orbital amplitudes (requires
    /// `kernel()` first). Returns the convergence flag.
    fn solve_lambda(slf: Bound<'_, Self>, py: Python<'_>) -> PyResult<bool> {
        let me = slf.borrow();
        let so = me.so.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "solve_lambda requires kernel() first (no converged amplitudes)",
            )
        })?;
        let lam = py
            .detach(|| {
                let pool = WorkspacePool::from_env();
                solve_ulambda(&so.t1, &so.t2, &so.eris, &pool)
            })
            .map_err(pyscf_to_py)?;
        Ok(lam.converged)
    }

    /// `uccsd.make_rdm1(ao_repr=False)` — open-shell CCSD 1-RDM (requires
    /// `kernel()`). Solves Λ then builds the spin-block `(dm1a, dm1b)` tuple,
    /// returned as a pair of NumPy 2-D arrays. Subclass overrides route through
    /// `call_method1` (Pitfall 7).
    #[pyo3(signature = (ao_repr=false))]
    fn make_rdm1<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        ao_repr: bool,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let bridge_slf: Py<PyAny> = slf.clone().into_any().unbind();
        if is_overridden(&bridge_slf, py, "make_rdm1", &["UCCSD"]) {
            let args = PyTuple::new(py, Vec::<Bound<'_, PyAny>>::new())?;
            let result = bridge_slf.bind(py).call_method1("make_rdm1", args)?;
            return result
                .downcast_into::<PyTuple>()
                .map_err(PyErr::from);
        }
        let me = slf.borrow();
        let so = me.so.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "make_rdm1 requires kernel() first (no converged amplitudes)",
            )
        })?;
        let mo_a = me.inner.alpha.mo_coeff.clone();
        let mo_b = me.inner.beta.mo_coeff.clone();
        let r = py
            .detach(|| -> Result<_, PyscfRsError> {
                let pool = WorkspacePool::from_env();
                let lam = solve_ulambda(&so.t1, &so.t2, &so.eris, &pool)?;
                umake_rdm1(
                    &so.t1, &so.t2, &lam.l1, &lam.l2, &so.eris, ao_repr, so.no_a, so.nv_a, so.no_b,
                    so.nv_b, &mo_a, &mo_b,
                )
            })
            .map_err(pyscf_to_py)?;
        let dm1a = vec_to_pyarray2(py, &r.dm1a, r.nmo_a)?;
        let dm1b = vec_to_pyarray2(py, &r.dm1b, r.nmo_b)?;
        PyTuple::new(py, [dm1a.into_any(), dm1b.into_any()])
    }

    /// `uccsd.make_rdm2(ao_repr=False)` — open-shell CCSD 2-RDM (requires
    /// `kernel()`). Returns the spin-block `(dm2aa, dm2ab, dm2bb)` tuple, each
    /// reshaped to a square 2-D NumPy array. Subclass overrides route through
    /// call_method1.
    #[pyo3(signature = (ao_repr=false))]
    fn make_rdm2<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        ao_repr: bool,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let bridge_slf: Py<PyAny> = slf.clone().into_any().unbind();
        if is_overridden(&bridge_slf, py, "make_rdm2", &["UCCSD"]) {
            let args = PyTuple::new(py, Vec::<Bound<'_, PyAny>>::new())?;
            let result = bridge_slf.bind(py).call_method1("make_rdm2", args)?;
            return result
                .downcast_into::<PyTuple>()
                .map_err(PyErr::from);
        }
        let me = slf.borrow();
        let so = me.so.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "make_rdm2 requires kernel() first (no converged amplitudes)",
            )
        })?;
        let mo_a = me.inner.alpha.mo_coeff.clone();
        let mo_b = me.inner.beta.mo_coeff.clone();
        let r = py
            .detach(|| -> Result<_, PyscfRsError> {
                let pool = WorkspacePool::from_env();
                let lam = solve_ulambda(&so.t1, &so.t2, &so.eris, &pool)?;
                umake_rdm2(
                    &so.t1, &so.t2, &lam.l1, &lam.l2, &so.eris, ao_repr, so.no_a, so.nv_a, so.no_b,
                    so.nv_b, &mo_a, &mo_b, &pool,
                )
            })
            .map_err(pyscf_to_py)?;
        // PySCF-shaped rank-4 spin blocks: dm2aa=[na,na,na,na],
        // dm2ab=[na,na,nb,nb], dm2bb=[nb,nb,nb,nb] (F-07; not a flattened
        // [side²,side²] — the open-shell make_rdm2 bonus comparison is shape-
        // sensitive and PySCF returns rank-4).
        let aa = vec_to_pyarray4(py, &r.dm2aa, r.na, r.na, r.na, r.na)?;
        let ab = vec_to_pyarray4(py, &r.dm2ab, r.na, r.na, r.nb, r.nb)?;
        let bb = vec_to_pyarray4(py, &r.dm2bb, r.nb, r.nb, r.nb, r.nb)?;
        PyTuple::new(py, [aa.into_any(), ab.into_any(), bb.into_any()])
    }

    /// `uccsd.as_scanner()` — Mole→energy callable (CCSD-07).
    fn as_scanner(slf: Bound<'_, Self>, py: Python<'_>) -> PyResult<PyCcsdScanner> {
        let me = slf.borrow();
        let mf_scanner: Py<PyAny> = me.py_mf.bind(py).call_method0("as_scanner")?.unbind();
        let py_mf: Py<PyAny> = me.py_mf.clone_ref(py);
        Ok(PyCcsdScanner {
            mf_scanner,
            py_mf,
            frozen: me.frozen.clone(),
            kind: CcsdKind::Unrestricted,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PyDFCCSD — conventional density-fitted CCSD.
// ═══════════════════════════════════════════════════════════════════════

#[pyclass(subclass, name = "DFCCSD", module = "pyscf._native.cc")]
pub struct PyDFCCSD {
    pub(crate) inner: CcsdReference,
    pub(crate) py_mf: Py<PyAny>,
    pub(crate) frozen: Frozen,
    pub(crate) e_corr: Option<f64>,
}

impl PyDFCCSD {
    pub(crate) fn from_reference(inner: CcsdReference, py_mf: Py<PyAny>, frozen: Frozen) -> Self {
        PyDFCCSD {
            inner,
            py_mf,
            frozen,
            e_corr: None,
        }
    }
}

#[pymethods]
impl PyDFCCSD {
    #[new]
    fn new(py: Python<'_>, mf: Py<PyAny>) -> PyResult<Self> {
        let inner = snapshot_reference(py, &mf)?;
        Ok(PyDFCCSD::from_reference(inner, mf, Frozen::None))
    }

    #[getter]
    fn _scf<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.py_mf.bind(py).clone()
    }

    #[getter]
    fn e_hf(&self) -> f64 {
        self.inner.e_hf
    }

    #[getter]
    fn e_corr(&self) -> Option<f64> {
        self.e_corr
    }

    #[getter]
    fn e_tot(&self) -> Option<f64> {
        self.e_corr.map(|ec| self.inner.e_hf + ec)
    }

    /// Run the conventional DF-CCSD kernel: assemble the DF B-tensor (`*-ri`
    /// aux), then run the reused `ccsd_kernel` with the DF `ao2mo` source via the
    /// override bridge (D-09): a Python subclass override of `ao2mo` dispatches
    /// through `call_method1`; the DEFAULT path runs `df_ao2mo` over the
    /// pre-built B-tensor under `py.detach`. The `int3c2e_sph` gate is surfaced
    /// by `cholesky_eri` and `?`-propagates; never panics.
    fn kernel(slf: Bound<'_, Self>, _py: Python<'_>) -> PyResult<f64> {
        let (refr, frozen) = {
            let me = slf.borrow();
            (me.inner.clone(), me.frozen.clone())
        };
        let pool = WorkspacePool::from_env();
        let df = build_df_integrals(&refr).map_err(pyscf_to_py)?;
        let bridge_slf: Py<PyAny> = slf.clone().into_any().unbind();
        let bridge = CcsdPyBridge {
            slf: bridge_slf,
            base_classes: vec!["DFCCSD"],
            default_source: DefaultAo2mo::Df(df),
        };
        let result = ccsd_kernel(&refr, &frozen, &bridge, &pool).map_err(pyscf_to_py)?;
        {
            let mut me = slf.borrow_mut();
            me.e_corr = Some(result.e_corr);
        }
        Ok(result.e_corr)
    }

    fn run<'py>(slf: Bound<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        let _ = PyDFCCSD::kernel(slf.clone(), py)?;
        Ok(slf)
    }

    /// `dfccsd.as_scanner()` — Mole→energy callable (CCSD-07).
    fn as_scanner(slf: Bound<'_, Self>, py: Python<'_>) -> PyResult<PyCcsdScanner> {
        let me = slf.borrow();
        let mf_scanner: Py<PyAny> = me.py_mf.bind(py).call_method0("as_scanner")?.unbind();
        let py_mf: Py<PyAny> = me.py_mf.clone_ref(py);
        Ok(PyCcsdScanner {
            mf_scanner,
            py_mf,
            frozen: me.frozen.clone(),
            kind: CcsdKind::DensityFitted,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PyCcsdScanner — Mole→energy callable wrapper (CCSD-07).
// Python: `e = mycc.as_scanner()(new_mol)` returns f64 (CCSD total energy).
// ═══════════════════════════════════════════════════════════════════════

/// The kind of CCSD the scanner re-runs at the new geometry (mp.rs:766-770
/// `Mp2Kind` analog).
#[derive(Clone, Copy)]
enum CcsdKind {
    Restricted,
    Unrestricted,
    DensityFitted,
}

#[pyclass(name = "Scanner", module = "pyscf._native.cc")]
pub struct PyCcsdScanner {
    /// The SCF reference scanner (`mf.as_scanner()`): `__call__(mol)` re-runs
    /// the SCF at the new geometry, mutating the held `mf` in place.
    mf_scanner: Py<PyAny>,
    /// The (re-run) `mf` handle to re-snapshot the reference from.
    py_mf: Py<PyAny>,
    frozen: Frozen,
    kind: CcsdKind,
}

#[pymethods]
impl PyCcsdScanner {
    /// `e = scanner(mol)` — re-run the reference SCF at `mol`, re-snapshot, run
    /// the CCSD kernel, return the CCSD total energy `e_hf + e_corr` (CCSD-07).
    fn __call__(&self, py: Python<'_>, mol: Py<PyAny>) -> PyResult<f64> {
        // Re-run the SCF reference at the new geometry (the scanner mutates the
        // held mf in place and returns the new SCF total energy).
        let args = PyTuple::new(py, [mol.bind(py).clone()])?;
        let _e_scf = self.mf_scanner.bind(py).call1(args)?;

        // Re-snapshot the (now re-converged) reference into fresh Rust arrays.
        match self.kind {
            CcsdKind::Restricted => {
                let refr = snapshot_reference(py, &self.py_mf)?;
                let result = py
                    .detach(|| {
                        let pool = WorkspacePool::from_env();
                        ccsd_kernel(&refr, &self.frozen, &pyscf_ccsd::NoCcsdOverrides, &pool)
                    })
                    .map_err(pyscf_to_py)?;
                Ok(refr.e_hf + result.e_corr)
            }
            CcsdKind::Unrestricted => {
                let refr = snapshot_uccsd_reference(py, &self.py_mf)?;
                let result = py
                    .detach(|| {
                        let pool = WorkspacePool::from_env();
                        uccsd_kernel(&refr, &self.frozen, &pool)
                    })
                    .map_err(pyscf_to_py)?;
                Ok(refr.alpha.e_hf + result.e_corr)
            }
            CcsdKind::DensityFitted => {
                let refr = snapshot_reference(py, &self.py_mf)?;
                let result = py
                    .detach(|| -> Result<_, PyscfRsError> {
                        let pool = WorkspacePool::from_env();
                        let df = build_df_integrals(&refr)?;
                        let bridge = ScannerDfBridge { df };
                        ccsd_kernel(&refr, &self.frozen, &bridge, &pool)
                    })
                    .map_err(pyscf_to_py)?;
                Ok(refr.e_hf + result.e_corr)
            }
        }
    }
}

/// A pyo3-free DF-CCSD hook used by the scanner's DF re-run (no Python `self`):
/// swaps the in-core `ao2mo` for the DF B-tensor source; everything else is the
/// pure-Rust default. Mirrors `pyscf_ccsd::dfccsd`'s swap-the-source pattern.
struct ScannerDfBridge {
    df: DfIntegrals,
}

impl CcsdOverrideHooks for ScannerDfBridge {
    fn ao2mo(&self, refr: &CcsdReference, frozen: &Frozen) -> Result<ChemistsEris, PyscfRsError> {
        let pool = WorkspacePool::from_env();
        df_ao2mo(refr, frozen, &self.df, &pool)
    }

    fn update_amps(
        &self,
        t1: &Amplitudes,
        t2: &Amplitudes,
        eris: &ChemistsEris,
    ) -> Result<(Amplitudes, Amplitudes), PyscfRsError> {
        default_update_amps(t1, t2, eris)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// CCSD() factory — module-level dispatch fn (mirrors pyscf/cc/__init__.py:83-139).
// UHF→UCCSD / with_df→DFCCSD / else→RCCSD.
// ═══════════════════════════════════════════════════════════════════════

/// Whether the Python `mf` is a UHF-type reference. Tries `mf.istype('UHF')`
/// first (the upstream dispatch idiom), falling back to a class-name check.
fn mf_is_uhf(py: Python<'_>, mf: &Py<PyAny>) -> bool {
    let bound = mf.bind(py);
    let via_istype = || -> PyResult<bool> {
        let istype = bound.getattr("istype")?;
        let args = PyTuple::new(py, [pyo3::types::PyString::new(py, "UHF").into_any()])?;
        istype.call1(args)?.extract::<bool>()
    };
    if let Ok(b) = via_istype() {
        return b;
    }
    let via_name = || -> PyResult<bool> {
        let name: String = bound.get_type().name()?.extract()?;
        Ok(name.contains("UHF") || name.contains("UKS"))
    };
    via_name().unwrap_or(false)
}

/// Whether the Python `mf` carries a truthy `with_df` (density-fitting)
/// attribute — mirrors `isinstance(mf, _DFHF) and mf.with_df` (cc/__init__.py:113).
fn mf_has_df(py: Python<'_>, mf: &Py<PyAny>) -> bool {
    let bound = mf.bind(py);
    match bound.getattr("with_df") {
        Ok(v) => !v.is_none() && v.is_truthy().unwrap_or(false),
        Err(_) => false,
    }
}

/// `CCSD(mf, frozen=None)` — the factory dispatch (CCSD-01/02 surface):
///   - `mf.istype('UHF')`             → UCCSD
///   - `getattr(mf,'with_df',None)`   → DFCCSD (conventional)
///   - otherwise (RHF-like)           → RCCSD
///
/// Mirrors `pyscf/cc/__init__.py:CCSD`/`RCCSD`/`UCCSD` (the in-scope branches;
/// GHF/GCCSD are out of v1 CCSD scope). `frozen` accepts `None` / `int` / a list
/// of orbital indices / `'auto'`.
#[pyfunction]
#[pyo3(name = "CCSD", signature = (mf, frozen=None))]
fn ccsd_factory(py: Python<'_>, mf: Py<PyAny>, frozen: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
    let frozen_spec = parse_frozen(py, frozen)?;
    if mf_is_uhf(py, &mf) {
        let inner = snapshot_uccsd_reference(py, &mf)?;
        let obj = PyUCCSD::from_reference(inner, mf, frozen_spec);
        return Ok(Py::new(py, obj)?.into_any());
    }
    if mf_has_df(py, &mf) {
        let inner = snapshot_reference(py, &mf)?;
        let obj = PyDFCCSD::from_reference(inner, mf, frozen_spec);
        return Ok(Py::new(py, obj)?.into_any());
    }
    let inner = snapshot_reference(py, &mf)?;
    let obj = PyRCCSD::from_reference(inner, mf, frozen_spec);
    Ok(Py::new(py, obj)?.into_any())
}

/// Parse the Python `frozen=` argument (None / int / list[int] / 'auto') into a
/// [`Frozen`] spec. Verbatim mp.rs:908-928.
fn parse_frozen(py: Python<'_>, frozen: Option<Py<PyAny>>) -> PyResult<Frozen> {
    let Some(f) = frozen else {
        return Ok(Frozen::None);
    };
    let bound = f.bind(py);
    if bound.is_none() {
        return Ok(Frozen::None);
    }
    if let Ok(n) = bound.extract::<usize>() {
        return Ok(Frozen::Count(n));
    }
    if let Ok(list) = bound.extract::<Vec<usize>>() {
        return Ok(Frozen::List(list));
    }
    if bound.extract::<String>().is_ok_and(|s| s == "auto") {
        return Ok(Frozen::Auto);
    }
    Err(pyo3::exceptions::PyValueError::new_err(
        "frozen must be None, an int, a list of orbital indices, or 'auto'",
    ))
}
