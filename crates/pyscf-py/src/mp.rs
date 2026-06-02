//! `mp` submodule — PyRMP2 / PyUMP2 / PyDFMP2 (+ native) + the Mp2 override
//! bridge + the `MP2()` factory + `as_scanner` (plan 05-07, D-07/D-08).
//!
//! This is the ONLY pyo3 layer for MP2 (the PyO3 wall). The MP2 method crates
//! (`pyscf-mp2`, `pyscf-ao2mo`) stay strictly pyo3-free + cubecl-free; the
//! PyO3 bridge + subclass-override dispatch lives exclusively here.
//!
//! Module layout (mirrors `scf.rs`):
//!   PyRMP2        — closed-shell RMP2 (calls `pyscf_mp2::rmp2_kernel`)
//!   PyUMP2        — open-shell UMP2 (calls `pyscf_mp2::ump2_kernel`)
//!   PyDFMP2       — conventional density-fitted MP2 (calls `dfrmp2_kernel`)
//!   PyMp2Scanner  — Mole→energy callable wrapper (MP2-07)
//!   MP2()         — module-level factory: RHF→RMP2 / UHF→UMP2 / with_df→DFMP2
//!
//! D-07 (eager snapshot): every PyMP2 class snapshots the SCF reference from
//! the Python `mf` into a plain-array `Mp2Reference` at construction (the
//! `mo_coeff`/`mo_energy`/`mo_occ`/`e_tot`/`converged`/`mol`). It also holds a
//! `Py<PyAny>` for the `mf` (re-used by `as_scanner` re-run + hook dispatch).
//!
//! D-08 (override dispatch): subclass overrides of `ao2mo`/`make_rdm1`/
//! `make_rdm2`/`energy` route through `slf.bind(py).call_method1` (the same
//! Pitfall-7-immune MRO resolution the SCF bridge uses). The DEFAULT
//! (non-overridden) path wraps the pure-Rust `pyscf_mp2::default_*` /
//! `NoMp2Overrides` compute in `py.detach` (BIND-05). The `kernel` itself does
//! NOT `py.detach` at the top — hooks re-enter Python (scf.rs:345-350).

// PyO3 boundary: hook methods marshal NumPy arrays + multi-field results.
#![allow(clippy::type_complexity)]

use numpy::{
    PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use pyscf_core::{Density, MOCoefficients, Mole, PyscfRsError};
use pyscf_df::{DfIntegrals, cholesky_eri, default_ri};
use pyscf_mp2::{
    ChemistsEris, Frozen, Mp2OverrideHooks, Mp2Reference, NoMp2Overrides, UmpReference,
    cross_spin_ao2mo, default_ao2mo, dfrmp2_kernel, dfump2_kernel, rmp2_kernel, scs_energy,
    ump2_kernel,
};

use crate::bridge::extract_mole_from_pyany;
use crate::errors::{py_to_pyscf, pyscf_to_py};
use crate::numpy_io::{density_to_pyarray, to_density, to_mo_coeff};

/// Register PyRMP2 / PyUMP2 / PyDFMP2 / PyMp2Scanner + the `MP2` factory in the
/// `_native.mp` submodule (BIND-02 analog).
pub(crate) fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRMP2>()?;
    m.add_class::<PyUMP2>()?;
    m.add_class::<PyDFMP2>()?;
    m.add_class::<PyMp2Scanner>()?;
    // Module-level `MP2(mf, frozen=None)` factory fn (mirrors pyscf/mp/__init__.py).
    m.add_function(wrap_pyfunction!(mp2_factory, m)?)?;
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════
// Eager-snapshot helpers (D-07) — pull a plain-array Mp2Reference out of the
// Python `mf` (extract_mole_from_pyany for mf.mol; numpy_io for the arrays).
// ═══════════════════════════════════════════════════════════════════════

/// Read a 1-D f64 NumPy attribute from the Python `mf` into a `Vec<f64>`.
fn read_vec_attr(mf: &Bound<'_, PyAny>, attr: &str) -> PyResult<Vec<f64>> {
    let obj = mf.getattr(attr)?;
    let arr: PyReadonlyArray1<f64> = obj.extract()?;
    Ok(arr.as_slice()?.to_vec())
}

/// Snapshot a converged SCF `mf` (RHF-like) into a plain-array `Mp2Reference`
/// (D-07). Pulls `mf.mol` (→ Mole), `mf.mo_coeff` (→ MOCoefficients, F-order),
/// `mf.mo_energy`/`mf.mo_occ` (→ Vec<f64>), `mf.e_tot` (→ e_hf), and
/// `mf.converged` (best-effort, default true).
fn snapshot_reference(py: Python<'_>, mf: &Py<PyAny>) -> PyResult<Mp2Reference> {
    let bound = mf.bind(py);
    let mol: Mole = extract_mole_from_pyany(py, mf)?;

    let mo_obj = bound.getattr("mo_coeff")?;
    let mo_arr: PyReadonlyArray2<f64> = mo_obj.extract().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(
            "mf.mo_coeff is None or not a 2D array — run the SCF (mf.kernel()) before MP2",
        )
    })?;
    let mut mo_coeff = to_mo_coeff(mo_arr)?;

    let mo_energy = read_vec_attr(bound, "mo_energy")?;
    let mo_occ = read_vec_attr(bound, "mo_occ")?;
    // Mirror the energies/occupations onto the MOCoefficients so the MP2-08
    // helpers (mo_without_core etc.) that read mo.occupations are consistent.
    mo_coeff.energies = mo_energy.clone();
    mo_coeff.occupations = mo_occ.clone();

    let e_hf: f64 = bound.getattr("e_tot")?.extract()?;
    let converged: bool = bound
        .getattr("converged")
        .and_then(|c| c.extract())
        .unwrap_or(true);

    Ok(Mp2Reference {
        mo_coeff,
        mo_energy,
        mo_occ,
        e_hf,
        converged,
        mol,
    })
}

/// Build a DF B-tensor from a snapshotted reference using the mp2fit `*-ri`
/// auxiliary basis (A2: NOT the JK-fit aux). The `int3c2e_sph` cintx#11 gate is
/// `?`-propagated by `cholesky_eri` — never panicked, never zero-substituted.
fn build_df_integrals(refr: &Mp2Reference) -> Result<DfIntegrals, PyscfRsError> {
    let aux = default_ri(&refr.mol.basis);
    cholesky_eri(&refr.mol, aux)
}

// ═══════════════════════════════════════════════════════════════════════
// Mp2PyBridge — the pyo3 side of the Mp2OverrideHooks seam (D-08).
// Holds a Py<PyAny> handle to the Python `self` (PyRMP2-or-subclass). Each
// hook dispatches via `slf.bind(py).call_method1("<hook>", args)` (Pitfall 7
// MRO resolution); the DEFAULT path wraps the pure-Rust default_* compute in
// `py.detach` (BIND-05). The kernel itself does NOT detach (hooks re-enter
// Python).
// ═══════════════════════════════════════════════════════════════════════

/// Whether the Python class (or a subclass) overrides `method`, i.e. the
/// resolved attribute is NOT the base-class `PyRMP2`/`PyUMP2`/`PyDFMP2`
/// implementation. We test this by comparing the resolved bound method's
/// `__qualname__` against the known base names; if the lookup fails we
/// conservatively treat it as overridden (dispatch through call_method1).
fn is_overridden(slf: &Py<PyAny>, py: Python<'_>, method: &str, base_classes: &[&str]) -> bool {
    let bound = slf.bind(py);
    match bound.getattr(method) {
        Ok(m) => match m
            .getattr("__qualname__")
            .and_then(|q| q.extract::<String>())
        {
            Ok(qual) => {
                // Base method qualname looks like "RMP2.ao2mo"; a subclass
                // override looks like "MyMP2.ao2mo". If the class component
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

/// The pyo3-side `Mp2OverrideHooks` bridge for the closed-shell path.
///
/// `ao2mo` is the main override point; when the Python class does NOT override
/// it the bridge runs the pure-Rust default (`default_ao2mo` for the in-core
/// path, or a caller-supplied `default` closure for DF) under `py.detach`.
struct Mp2PyBridge {
    /// Python `self` of the PyRMP2 (or subclass). `call_method1` against this
    /// handle resolves the MRO so subclass overrides are invoked transparently.
    slf: Py<PyAny>,
    /// The base class names whose `ao2mo`/`make_rdm1`/... are the DEFAULT (not
    /// an override): e.g. `["RMP2"]` for PyRMP2.
    base_classes: Vec<&'static str>,
    /// Default `ao2mo` source. `InCore` runs `pyscf_mp2::default_ao2mo`
    /// (in-core `int2e`); `Df` runs `df_ao2mo` over a pre-built B-tensor.
    default_source: DefaultAo2mo,
}

/// The DEFAULT (non-overridden) `ao2mo` source for the bridge.
enum DefaultAo2mo {
    /// In-core RMP2: `pyscf_mp2::default_ao2mo` (`int2e` → ao2mo.general).
    InCore,
    /// Conventional DF-MP2: assemble `(ia|jb)` from the DF B-tensor.
    Df(DfIntegrals),
}

impl Mp2OverrideHooks for Mp2PyBridge {
    fn ao2mo(&self, refr: &Mp2Reference, frozen: &Frozen) -> Result<ChemistsEris, PyscfRsError> {
        // Pitfall 7: if a Python subclass overrides `ao2mo`, dispatch through
        // call_method1 (MRO resolution). Otherwise run the pure-Rust default
        // under py.detach (BIND-05) — the kernel holds the GIL, so we release
        // it around the pure-Rust compute.
        let overridden =
            Python::attach(|py| is_overridden(&self.slf, py, "ao2mo", &self.base_classes));
        if overridden {
            let nocc = pyscf_mp2::get_nocc(&refr.mo_occ, frozen)?;
            return Python::attach(|py| {
                // Do all the Python marshalling inside a PyResult closure, then
                // convert any PyErr into PyscfRsError at the boundary.
                let assemble = || -> PyResult<ChemistsEris> {
                    let args = PyTuple::new(py, Vec::<Bound<'_, PyAny>>::new())?;
                    let result = self.slf.bind(py).call_method1("ao2mo", args)?;
                    // A Python override returns the flat (ia|jb) ovov block as a
                    // 1D NumPy array; nocc/nvir are taken from the active ref.
                    let arr: PyReadonlyArray1<f64> = result.extract()?;
                    let ovov = arr.as_slice()?.to_vec();
                    let nvir_sq = ovov.len() / (nocc * nocc).max(1);
                    let nvir = (nvir_sq as f64).sqrt() as usize;
                    Ok(ChemistsEris { ovov, nocc, nvir })
                };
                assemble().map_err(py_to_pyscf)
            });
        }
        // DEFAULT path: pure-Rust compute under py.detach (BIND-05).
        match &self.default_source {
            DefaultAo2mo::InCore => Python::attach(|py| py.detach(|| default_ao2mo(refr, frozen))),
            DefaultAo2mo::Df(df) => {
                Python::attach(|py| py.detach(|| pyscf_mp2::df_ao2mo(refr, frozen, df)))
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PyRMP2 — closed-shell Restricted MP2.
// ═══════════════════════════════════════════════════════════════════════

#[pyclass(subclass, name = "RMP2", module = "pyscf._native.mp")]
pub struct PyRMP2 {
    /// Eagerly-snapshotted SCF reference (D-07).
    pub(crate) inner: Mp2Reference,
    /// Python `mf` handle — held for `as_scanner` re-run + hook dispatch (D-07).
    pub(crate) py_mf: Py<PyAny>,
    /// Frozen-core spec (MP2-03).
    pub(crate) frozen: Frozen,
    /// SCS same-spin scale factor (MP2-06; 1.0 = plain MP2).
    pub(crate) ss_factor: f64,
    /// SCS opposite-spin scale factor (MP2-06; 1.0 = plain MP2).
    pub(crate) os_factor: f64,
    /// Correlation energy after `kernel()` (None until run).
    pub(crate) e_corr: Option<f64>,
    /// Same-spin component after `kernel()`.
    pub(crate) e_ss: Option<f64>,
    /// Opposite-spin component after `kernel()`.
    pub(crate) e_os: Option<f64>,
}

impl PyRMP2 {
    /// Construct from a snapshotted reference + the held `mf` (shared by the
    /// factory + the `#[new]` constructor).
    pub(crate) fn from_reference(inner: Mp2Reference, py_mf: Py<PyAny>, frozen: Frozen) -> Self {
        PyRMP2 {
            inner,
            py_mf,
            frozen,
            ss_factor: 1.0,
            os_factor: 1.0,
            e_corr: None,
            e_ss: None,
            e_os: None,
        }
    }
}

#[pymethods]
impl PyRMP2 {
    #[new]
    fn new(py: Python<'_>, mf: Py<PyAny>) -> PyResult<Self> {
        let inner = snapshot_reference(py, &mf)?;
        Ok(PyRMP2::from_reference(inner, mf, Frozen::None))
    }

    // ───────────────────────────────────────────────────────────────────
    // Attribute surface.
    // ───────────────────────────────────────────────────────────────────

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

    /// SCS-scaled correlation energy (MP2-06): `e_ss·ss + e_os·os`.
    #[getter]
    fn emp2_scs(&self) -> Option<f64> {
        match (self.e_ss, self.e_os) {
            (Some(ss), Some(os)) => Some(scs_energy(ss, os, self.ss_factor, self.os_factor)),
            _ => None,
        }
    }

    #[getter]
    fn emp2_ss_factor(&self) -> f64 {
        self.ss_factor
    }
    #[setter]
    fn set_emp2_ss_factor(&mut self, v: f64) {
        self.ss_factor = v;
    }

    #[getter]
    fn emp2_os_factor(&self) -> f64 {
        self.os_factor
    }
    #[setter]
    fn set_emp2_os_factor(&mut self, v: f64) {
        self.os_factor = v;
    }

    /// `mp2.set(emp2_ss_factor=, emp2_os_factor=)` — fluent SCS configuration
    /// (MP2-06), returns self for chaining.
    #[pyo3(signature = (emp2_ss_factor=None, emp2_os_factor=None))]
    fn set<'py>(
        slf: Bound<'py, Self>,
        emp2_ss_factor: Option<f64>,
        emp2_os_factor: Option<f64>,
    ) -> Bound<'py, Self> {
        {
            let mut me = slf.borrow_mut();
            if let Some(ss) = emp2_ss_factor {
                me.ss_factor = ss;
            }
            if let Some(os) = emp2_os_factor {
                me.os_factor = os;
            }
        }
        slf
    }

    // ───────────────────────────────────────────────────────────────────
    // kernel / run — drivers. Subclass-override dispatch happens via the
    // Mp2PyBridge's call_method1 path (D-08 / Pitfall 7).
    // ───────────────────────────────────────────────────────────────────

    /// Run the closed-form RMP2 kernel. Returns `(e_corr, t2=None)` (the t2
    /// block is not marshalled to Python here; `make_rdm1`/`make_rdm2` consume
    /// it Rust-side once arity-4 int2e lands — cintx#11-gated).
    #[pyo3(signature = (with_t2=false))]
    fn kernel(slf: Bound<'_, Self>, _py: Python<'_>, with_t2: bool) -> PyResult<f64> {
        // Snapshot the config BEFORE constructing the bridge (the bridge holds
        // a Py<PyAny> to slf, so we must release the Rust borrow first).
        let (frozen, ss, os) = {
            let me = slf.borrow();
            (me.frozen.clone(), me.ss_factor, me.os_factor)
        };
        let refr = {
            let me = slf.borrow();
            me.inner.clone()
        };
        // The bridge needs a Py<PyAny> pointing at the same Python object so
        // call_method1 resolves the MRO (subclass overrides). NB: we DO NOT
        // py.detach here — hooks re-enter Python (scf.rs:345-350).
        let bridge_slf: Py<PyAny> = slf.clone().into_any().unbind();
        let bridge = Mp2PyBridge {
            slf: bridge_slf,
            base_classes: vec!["RMP2"],
            default_source: DefaultAo2mo::InCore,
        };
        let result = rmp2_kernel(&refr, &frozen, &bridge, with_t2).map_err(pyscf_to_py)?;
        let e_corr = scs_energy(result.e_ss, result.e_os, ss, os);
        {
            let mut me = slf.borrow_mut();
            me.e_corr = Some(e_corr);
            me.e_ss = Some(result.e_ss);
            me.e_os = Some(result.e_os);
        }
        Ok(e_corr)
    }

    /// `mp2.run()` — kernel then return self for chaining (BIND-02 idiom).
    fn run<'py>(slf: Bound<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        let _ = PyRMP2::kernel(slf.clone(), py, false)?;
        Ok(slf)
    }

    // ───────────────────────────────────────────────────────────────────
    // make_rdm1 / make_rdm2 — route through the bridge (MP2-05).
    // ───────────────────────────────────────────────────────────────────

    /// MP2 1-RDM. Routes through the bridge's `make_rdm1` seam (default returns
    /// the cintx#11-gated `NotYetImplemented` until arity-4 int2e lands; a
    /// Python subclass override is dispatched via call_method1).
    fn make_rdm1<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (refr, frozen) = {
            let me = slf.borrow();
            (me.inner.clone(), me.frozen.clone())
        };
        let bridge_slf: Py<PyAny> = slf.clone().into_any().unbind();
        let d = rdm_via_bridge(&bridge_slf, py, "make_rdm1", &["RMP2"], &refr, &frozen)?;
        density_to_pyarray(py, &d)
    }

    /// MP2 2-RDM. Routes through the bridge's `make_rdm2` seam (same gating).
    fn make_rdm2<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (refr, frozen) = {
            let me = slf.borrow();
            (me.inner.clone(), me.frozen.clone())
        };
        let bridge_slf: Py<PyAny> = slf.clone().into_any().unbind();
        let d = rdm_via_bridge(&bridge_slf, py, "make_rdm2", &["RMP2"], &refr, &frozen)?;
        density_to_pyarray(py, &d)
    }

    // ───────────────────────────────────────────────────────────────────
    // as_scanner (MP2-07).
    // ───────────────────────────────────────────────────────────────────

    /// `mp2.as_scanner()` — return a callable (PyMp2Scanner) that takes a Mole
    /// at a new geometry, re-runs the SCF `mf` (via its own `as_scanner`),
    /// re-snapshots into a fresh `Mp2Reference`, runs the MP2 kernel, and
    /// returns the MP2 total energy (MP2-07). Mirrors scf.rs:694-699.
    fn as_scanner(slf: Bound<'_, Self>, py: Python<'_>) -> PyResult<PyMp2Scanner> {
        let me = slf.borrow();
        // The SCF scanner: mf.as_scanner() returns a callable; we hold it so
        // __call__(mol) re-runs the reference at the new geometry.
        let mf_scanner: Py<PyAny> = me.py_mf.bind(py).call_method0("as_scanner")?.unbind();
        let py_mf: Py<PyAny> = me.py_mf.clone_ref(py);
        Ok(PyMp2Scanner {
            mf_scanner,
            py_mf,
            frozen: me.frozen.clone(),
            ss_factor: me.ss_factor,
            os_factor: me.os_factor,
            kind: Mp2Kind::Restricted,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PyUMP2 — open-shell Unrestricted MP2.
// ═══════════════════════════════════════════════════════════════════════

#[pyclass(subclass, name = "UMP2", module = "pyscf._native.mp")]
pub struct PyUMP2 {
    /// Eagerly-snapshotted α/β reference pair (D-07).
    pub(crate) inner: UmpReference,
    pub(crate) py_mf: Py<PyAny>,
    pub(crate) frozen: Frozen,
    pub(crate) ss_factor: f64,
    pub(crate) os_factor: f64,
    pub(crate) e_corr: Option<f64>,
    /// Whether this UMP2 was built from a density-fitted (`with_df`) UHF
    /// reference — selects `dfump2_kernel` over the conventional cross-spin
    /// `int2e` path (F-06).
    pub(crate) df: bool,
}

impl PyUMP2 {
    pub(crate) fn from_reference(inner: UmpReference, py_mf: Py<PyAny>, frozen: Frozen) -> Self {
        Self::from_reference_df(inner, py_mf, frozen, false)
    }

    /// Construct with an explicit DF flag (the factory sets `df=true` for a
    /// `with_df` UHF reference).
    pub(crate) fn from_reference_df(
        inner: UmpReference,
        py_mf: Py<PyAny>,
        frozen: Frozen,
        df: bool,
    ) -> Self {
        PyUMP2 {
            inner,
            py_mf,
            frozen,
            ss_factor: 1.0,
            os_factor: 1.0,
            e_corr: None,
            df,
        }
    }
}

/// Build a per-spin `Mp2Reference` from spin index `s` of a UHF `mf`, where
/// `mf.mo_coeff` is the 3-D `[2, nao, nmo]` array and `mf.mo_energy`/`mf.mo_occ`
/// are 2-D `[2, n*]` arrays. Both spins share `mf.mol` and `mf.e_tot`.
fn ump_channel_from_3d(
    py: Python<'_>,
    mf: &Py<PyAny>,
    mol: &Mole,
    e_hf: f64,
    converged: bool,
    s: usize,
) -> PyResult<Mp2Reference> {
    let bound = mf.bind(py);

    // mo_coeff[s] — slice the [2,nao,nmo] array into a column-major [nao,nmo].
    let mo3: PyReadonlyArray3<f64> = bound.getattr("mo_coeff")?.extract()?;
    let shape = mo3.shape();
    if shape.len() != 3 || shape[0] < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "UHF mf.mo_coeff must be [2,nao,nmo], got shape {shape:?}"
        )));
    }
    let (nao, nmo) = (shape[1], shape[2]);
    let view = mo3.as_array();
    // Column-major [nao, nmo]: C[ao, mo] at ao + mo*nao.
    let mut data = Vec::with_capacity(nao * nmo);
    for mo in 0..nmo {
        for ao in 0..nao {
            data.push(view[[s, ao, mo]]);
        }
    }

    // mo_energy[s] / mo_occ[s] — slice the [2,n*] arrays.
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

    Ok(Mp2Reference {
        mo_coeff,
        mo_energy,
        mo_occ,
        e_hf,
        converged,
        mol: mol.clone(),
    })
}

/// Snapshot an unrestricted `mf` into a `UmpReference`. Upstream UHF carries
/// `mo_coeff[0/1]` (α/β); when `mf.mo_coeff` is the 3-D `[2,nao,nmo]` UHF array
/// the α/β channels are read from the spin-0/spin-1 slices (the genuine
/// open-shell reference). When the mf carries a single restricted-shaped MO set
/// (2-D `mo_coeff`), both channels share it (closed-shell-as-open-shell).
fn snapshot_ump_reference(py: Python<'_>, mf: &Py<PyAny>) -> PyResult<UmpReference> {
    let bound = mf.bind(py);
    let mol: Mole = extract_mole_from_pyany(py, mf)?;
    let e_hf: f64 = bound.getattr("e_tot")?.extract()?;
    let converged: bool = bound
        .getattr("converged")
        .and_then(|c| c.extract())
        .unwrap_or(true);

    // Detect the UHF [2,nao,nmo] mo_coeff: read the α/β split when present.
    let is_uhf_shaped = bound
        .getattr("mo_coeff")
        .ok()
        .and_then(|o| o.getattr("ndim").ok())
        .and_then(|n| n.extract::<usize>().ok())
        .map(|ndim| ndim == 3)
        .unwrap_or(false);

    if is_uhf_shaped {
        let alpha = ump_channel_from_3d(py, mf, &mol, e_hf, converged, 0)?;
        let beta = ump_channel_from_3d(py, mf, &mol, e_hf, converged, 1)?;
        return Ok(UmpReference { alpha, beta });
    }

    // Restricted-shaped fallback (closed-shell treated as α==β).
    let alpha = snapshot_reference(py, mf)?;
    let beta = alpha.clone();
    Ok(UmpReference { alpha, beta })
}

#[pymethods]
impl PyUMP2 {
    #[new]
    fn new(py: Python<'_>, mf: Py<PyAny>) -> PyResult<Self> {
        let inner = snapshot_ump_reference(py, &mf)?;
        Ok(PyUMP2::from_reference(inner, mf, Frozen::None))
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

    #[getter]
    fn emp2_ss_factor(&self) -> f64 {
        self.ss_factor
    }
    #[setter]
    fn set_emp2_ss_factor(&mut self, v: f64) {
        self.ss_factor = v;
    }
    #[getter]
    fn emp2_os_factor(&self) -> f64 {
        self.os_factor
    }
    #[setter]
    fn set_emp2_os_factor(&mut self, v: f64) {
        self.os_factor = v;
    }

    /// Run the open-shell UMP2 kernel.
    ///
    /// Conventional (non-DF) path: builds the per-spin αα/ββ `(ia|jb)` blocks
    /// via the in-core `default_ao2mo` and the opposite-spin αβ `(ia|JB)` block
    /// via `pyscf_mp2::cross_spin_ao2mo` — the genuine cross-spin
    /// `(o_α v_α | o_β v_β)` transform with the explicit F-order→C-order repack
    /// (F-06). The αβ term is the dominant MP2 correlation contribution; it is
    /// no longer gated. Errors from `int2e`/`ao2mo` `?`-propagate; never panics.
    ///
    /// DF path (when this UMP2 was built from a `with_df` UHF reference): routes
    /// through `pyscf_mp2::dfump2_kernel`, which assembles all three spin blocks
    /// (αα/ββ same-spin DF ao2mo + the αβ cross-spin Q-contraction) and returns
    /// the correct C-order αβ block via `assemble_cross_spin`.
    #[pyo3(signature = (with_t2=false))]
    fn kernel(slf: Bound<'_, Self>, _py: Python<'_>, with_t2: bool) -> PyResult<f64> {
        let (refr, frozen, df_mode) = {
            let me = slf.borrow();
            (me.inner.clone(), me.frozen.clone(), me.df)
        };
        let result = if df_mode {
            // DF-unrestricted: dfump2_kernel owns all three spin blocks (the αβ
            // block via assemble_cross_spin) — NOT the RHF dfrmp2_kernel.
            let df = build_df_integrals(&refr.alpha).map_err(pyscf_to_py)?;
            dfump2_kernel(&refr, &frozen, &df, with_t2).map_err(pyscf_to_py)?
        } else {
            // Conventional: per-spin αα/ββ + cross-spin αβ (F-06).
            let eris_aa = default_ao2mo(&refr.alpha, &frozen).map_err(pyscf_to_py)?;
            let eris_bb = default_ao2mo(&refr.beta, &frozen).map_err(pyscf_to_py)?;
            let eris_ab =
                cross_spin_ao2mo(&refr.alpha, &refr.beta, &frozen).map_err(pyscf_to_py)?;
            ump2_kernel(&refr, &frozen, &eris_aa, &eris_ab, &eris_bb, with_t2)
                .map_err(pyscf_to_py)?
        };
        {
            let mut me = slf.borrow_mut();
            me.e_corr = Some(result.e_corr);
        }
        Ok(result.e_corr)
    }

    fn run<'py>(slf: Bound<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        let _ = PyUMP2::kernel(slf.clone(), py, false)?;
        Ok(slf)
    }

    /// `ump2.as_scanner()` — Mole→energy callable (MP2-07).
    fn as_scanner(slf: Bound<'_, Self>, py: Python<'_>) -> PyResult<PyMp2Scanner> {
        let me = slf.borrow();
        let mf_scanner: Py<PyAny> = me.py_mf.bind(py).call_method0("as_scanner")?.unbind();
        let py_mf: Py<PyAny> = me.py_mf.clone_ref(py);
        Ok(PyMp2Scanner {
            mf_scanner,
            py_mf,
            frozen: me.frozen.clone(),
            ss_factor: me.ss_factor,
            os_factor: me.os_factor,
            kind: if me.df {
                Mp2Kind::DensityFittedUnrestricted
            } else {
                Mp2Kind::Unrestricted
            },
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PyDFMP2 — conventional density-fitted MP2.
// ═══════════════════════════════════════════════════════════════════════

#[pyclass(subclass, name = "DFMP2", module = "pyscf._native.mp")]
pub struct PyDFMP2 {
    pub(crate) inner: Mp2Reference,
    pub(crate) py_mf: Py<PyAny>,
    pub(crate) frozen: Frozen,
    pub(crate) ss_factor: f64,
    pub(crate) os_factor: f64,
    pub(crate) e_corr: Option<f64>,
    pub(crate) e_ss: Option<f64>,
    pub(crate) e_os: Option<f64>,
}

impl PyDFMP2 {
    pub(crate) fn from_reference(inner: Mp2Reference, py_mf: Py<PyAny>, frozen: Frozen) -> Self {
        PyDFMP2 {
            inner,
            py_mf,
            frozen,
            ss_factor: 1.0,
            os_factor: 1.0,
            e_corr: None,
            e_ss: None,
            e_os: None,
        }
    }
}

#[pymethods]
impl PyDFMP2 {
    #[new]
    fn new(py: Python<'_>, mf: Py<PyAny>) -> PyResult<Self> {
        let inner = snapshot_reference(py, &mf)?;
        Ok(PyDFMP2::from_reference(inner, mf, Frozen::None))
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

    #[getter]
    fn emp2_scs(&self) -> Option<f64> {
        match (self.e_ss, self.e_os) {
            (Some(ss), Some(os)) => Some(scs_energy(ss, os, self.ss_factor, self.os_factor)),
            _ => None,
        }
    }

    #[getter]
    fn emp2_ss_factor(&self) -> f64 {
        self.ss_factor
    }
    #[setter]
    fn set_emp2_ss_factor(&mut self, v: f64) {
        self.ss_factor = v;
    }
    #[getter]
    fn emp2_os_factor(&self) -> f64 {
        self.os_factor
    }
    #[setter]
    fn set_emp2_os_factor(&mut self, v: f64) {
        self.os_factor = v;
    }

    /// Run the conventional DF-MP2 kernel: assemble the DF B-tensor (`*-ri`
    /// aux), then run the reused `rmp2_kernel` with the DF `(ia|jb)` source via
    /// the override bridge (D-08): a Python subclass override of `ao2mo`
    /// dispatches through `call_method1`; the DEFAULT path runs `df_ao2mo` over
    /// the pre-built B-tensor under `py.detach`. The `int3c2e_sph` cintx#11 gate
    /// is surfaced by `cholesky_eri` and `?`-propagates; never panics.
    #[pyo3(signature = (with_t2=false))]
    fn kernel(slf: Bound<'_, Self>, _py: Python<'_>, with_t2: bool) -> PyResult<f64> {
        let (refr, frozen, ss, os) = {
            let me = slf.borrow();
            (
                me.inner.clone(),
                me.frozen.clone(),
                me.ss_factor,
                me.os_factor,
            )
        };
        let df = build_df_integrals(&refr).map_err(pyscf_to_py)?;
        // Route through the override bridge so a subclass `ao2mo` override is
        // dispatched via call_method1; the default DF source is the pre-built
        // B-tensor. `rmp2_kernel` reuses the closed-form RMP2 math verbatim with
        // the DF block in place (dfmp2.py "swap the ERI source").
        let bridge_slf: Py<PyAny> = slf.clone().into_any().unbind();
        let bridge = Mp2PyBridge {
            slf: bridge_slf,
            base_classes: vec!["DFMP2"],
            default_source: DefaultAo2mo::Df(df),
        };
        let result = rmp2_kernel(&refr, &frozen, &bridge, with_t2).map_err(pyscf_to_py)?;
        let e_corr = scs_energy(result.e_ss, result.e_os, ss, os);
        {
            let mut me = slf.borrow_mut();
            me.e_corr = Some(e_corr);
            me.e_ss = Some(result.e_ss);
            me.e_os = Some(result.e_os);
        }
        Ok(e_corr)
    }

    fn run<'py>(slf: Bound<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        let _ = PyDFMP2::kernel(slf.clone(), py, false)?;
        Ok(slf)
    }

    /// `dfmp2.as_scanner()` — Mole→energy callable (MP2-07).
    fn as_scanner(slf: Bound<'_, Self>, py: Python<'_>) -> PyResult<PyMp2Scanner> {
        let me = slf.borrow();
        let mf_scanner: Py<PyAny> = me.py_mf.bind(py).call_method0("as_scanner")?.unbind();
        let py_mf: Py<PyAny> = me.py_mf.clone_ref(py);
        Ok(PyMp2Scanner {
            mf_scanner,
            py_mf,
            frozen: me.frozen.clone(),
            ss_factor: me.ss_factor,
            os_factor: me.os_factor,
            kind: Mp2Kind::DensityFitted,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════
// make_rdm1 / make_rdm2 bridge dispatch helper (shared by PyRMP2).
// ═══════════════════════════════════════════════════════════════════════

/// Dispatch a `make_rdm1`/`make_rdm2` call through the bridge: a Python
/// subclass override is invoked via call_method1 (Pitfall 7); otherwise the
/// default `NoMp2Overrides` seam runs under py.detach (which returns the
/// cintx#11-gated `NotYetImplemented` until arity-4 int2e lands).
fn rdm_via_bridge(
    slf: &Py<PyAny>,
    py: Python<'_>,
    method: &str,
    base_classes: &[&str],
    refr: &Mp2Reference,
    frozen: &Frozen,
) -> PyResult<Density> {
    if is_overridden(slf, py, method, base_classes) {
        let args = PyTuple::new(py, Vec::<Bound<'_, PyAny>>::new())?;
        let result = slf.bind(py).call_method1(method, args)?;
        let arr: PyReadonlyArray2<f64> = result.extract()?;
        return to_density(arr);
    }
    // DEFAULT path: NoMp2Overrides seam under py.detach.
    let no = NoMp2Overrides;
    let d = py.detach(|| match method {
        "make_rdm1" => no.make_rdm1(refr, frozen),
        _ => no.make_rdm2(refr, frozen),
    });
    d.map_err(pyscf_to_py)
}

// ═══════════════════════════════════════════════════════════════════════
// PyMp2Scanner — Mole→energy callable wrapper (MP2-07).
// Python: `e = mp2.as_scanner()(new_mol)` returns f64 (MP2 total energy).
// ═══════════════════════════════════════════════════════════════════════

/// The kind of MP2 the scanner re-runs at the new geometry.
#[derive(Clone, Copy)]
enum Mp2Kind {
    Restricted,
    Unrestricted,
    DensityFitted,
    /// Open-shell density-fitted: re-run UHF, snapshot the α/β pair, route
    /// through `dfump2_kernel` (NOT the RHF `dfrmp2_kernel`).
    DensityFittedUnrestricted,
}

#[pyclass(name = "Scanner", module = "pyscf._native.mp")]
pub struct PyMp2Scanner {
    /// The SCF reference scanner (`mf.as_scanner()`): `__call__(mol)` re-runs
    /// the SCF at the new geometry, mutating the held `mf` in place.
    mf_scanner: Py<PyAny>,
    /// The (re-run) `mf` handle to re-snapshot the reference from.
    py_mf: Py<PyAny>,
    frozen: Frozen,
    ss_factor: f64,
    os_factor: f64,
    kind: Mp2Kind,
}

#[pymethods]
impl PyMp2Scanner {
    /// `e = scanner(mol)` — re-run the reference SCF at `mol`, re-snapshot, run
    /// the MP2 kernel, return the MP2 total energy `e_hf + e_corr` (MP2-07).
    fn __call__(&self, py: Python<'_>, mol: Py<PyAny>) -> PyResult<f64> {
        // Re-run the SCF reference at the new geometry (the scanner mutates the
        // held mf in place and returns the new SCF total energy).
        let args = PyTuple::new(py, [mol.bind(py).clone()])?;
        let _e_scf = self.mf_scanner.bind(py).call1(args)?;

        // Re-snapshot the (now re-converged) reference into fresh Rust arrays.
        match self.kind {
            Mp2Kind::Restricted => {
                let refr = snapshot_reference(py, &self.py_mf)?;
                let bridge = NoMp2Overrides;
                let result = py
                    .detach(|| rmp2_kernel(&refr, &self.frozen, &bridge, false))
                    .map_err(pyscf_to_py)?;
                let e_corr = scs_energy(result.e_ss, result.e_os, self.ss_factor, self.os_factor);
                Ok(refr.e_hf + e_corr)
            }
            Mp2Kind::Unrestricted => {
                let refr = snapshot_ump_reference(py, &self.py_mf)?;
                let (eris_aa, eris_ab, eris_bb) = py
                    .detach(|| -> Result<_, PyscfRsError> {
                        let aa = default_ao2mo(&refr.alpha, &self.frozen)?;
                        let bb = default_ao2mo(&refr.beta, &self.frozen)?;
                        // αβ via the genuine cross-spin (o_α v_α | o_β v_β)
                        // transform with the explicit F→C repack (F-06) — NOT a
                        // reuse of the α same-spin block.
                        let ab = cross_spin_ao2mo(&refr.alpha, &refr.beta, &self.frozen)?;
                        Ok((aa, ab, bb))
                    })
                    .map_err(pyscf_to_py)?;
                let result = py
                    .detach(|| {
                        ump2_kernel(&refr, &self.frozen, &eris_aa, &eris_ab, &eris_bb, false)
                    })
                    .map_err(pyscf_to_py)?;
                Ok(refr.alpha.e_hf + result.e_corr)
            }
            Mp2Kind::DensityFitted => {
                let refr = snapshot_reference(py, &self.py_mf)?;
                let result = py
                    .detach(|| -> Result<_, PyscfRsError> {
                        let df = build_df_integrals(&refr)?;
                        dfrmp2_kernel(&refr, &self.frozen, &df, false)
                    })
                    .map_err(pyscf_to_py)?;
                let e_corr = scs_energy(result.e_ss, result.e_os, self.ss_factor, self.os_factor);
                Ok(refr.e_hf + e_corr)
            }
            Mp2Kind::DensityFittedUnrestricted => {
                // Open-shell DF: route the αα/ββ + αβ cross-spin assembly through
                // dfump2_kernel (NOT the RHF dfrmp2_kernel, which would silently
                // drop the opposite-spin cross block on a UHF reference).
                let refr = snapshot_ump_reference(py, &self.py_mf)?;
                let result = py
                    .detach(|| -> Result<_, PyscfRsError> {
                        let df = build_df_integrals(&refr.alpha)?;
                        dfump2_kernel(&refr, &self.frozen, &df, false)
                    })
                    .map_err(pyscf_to_py)?;
                Ok(refr.alpha.e_hf + result.e_corr)
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// MP2() factory — module-level dispatch fn (mirrors pyscf/mp/__init__.py:27-65).
// RHF→RMP2 / UHF→UMP2 / with_df→DFMP2.
// ═══════════════════════════════════════════════════════════════════════

/// Whether the Python `mf` is a UHF-type reference. Tries `mf.istype('UHF')`
/// first (the upstream dispatch idiom), falling back to a class-name check.
fn mf_is_uhf(py: Python<'_>, mf: &Py<PyAny>) -> bool {
    let bound = mf.bind(py);
    // Path 1: mf.istype('UHF') (the upstream dispatch idiom).
    let via_istype = || -> PyResult<bool> {
        let istype = bound.getattr("istype")?;
        let args = PyTuple::new(py, [pyo3::types::PyString::new(py, "UHF").into_any()])?;
        istype.call1(args)?.extract::<bool>()
    };
    if let Ok(b) = via_istype() {
        return b;
    }
    // Path 2 (fallback): class name contains "UHF" / "UKS".
    let via_name = || -> PyResult<bool> {
        let name: String = bound.get_type().name()?.extract()?;
        Ok(name.contains("UHF") || name.contains("UKS"))
    };
    via_name().unwrap_or(false)
}

/// Whether the Python `mf` carries a `with_df` (density-fitting) attribute that
/// is truthy — mirrors `getattr(mf, 'with_df', None)`.
fn mf_has_df(py: Python<'_>, mf: &Py<PyAny>) -> bool {
    let bound = mf.bind(py);
    match bound.getattr("with_df") {
        Ok(v) => !v.is_none() && v.is_truthy().unwrap_or(false),
        Err(_) => false,
    }
}

/// `MP2(mf, frozen=None)` — the factory dispatch (MP2-01):
///   - `mf.istype('UHF')`             → UMP2
///   - `getattr(mf,'with_df',None)`   → DFMP2 (conventional)
///   - otherwise (RHF-like)           → RMP2
///
/// Mirrors `pyscf/mp/__init__.py:MP2`/`RMP2`/`UMP2` (the in-scope branches;
/// GHF/GMP2 are out of v1 MP2 scope). `frozen` accepts `None` / `int` / a list
/// of orbital indices (MP2-03).
#[pyfunction]
#[pyo3(name = "MP2", signature = (mf, frozen=None))]
fn mp2_factory(py: Python<'_>, mf: Py<PyAny>, frozen: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
    let frozen_spec = parse_frozen(py, frozen)?;
    if mf_is_uhf(py, &mf) {
        // UHF → UMP2; a `with_df` UHF reference selects the DF-unrestricted
        // path (dfump2_kernel), NOT the RHF DFMP2 (which would drop the αβ
        // cross-spin block). UHF is checked before with_df so an open-shell DF
        // reference never misroutes through dfrmp2_kernel.
        let df = mf_has_df(py, &mf);
        let inner = snapshot_ump_reference(py, &mf)?;
        let obj = PyUMP2::from_reference_df(inner, mf, frozen_spec, df);
        return Ok(Py::new(py, obj)?.into_any());
    }
    if mf_has_df(py, &mf) {
        let inner = snapshot_reference(py, &mf)?;
        let obj = PyDFMP2::from_reference(inner, mf, frozen_spec);
        return Ok(Py::new(py, obj)?.into_any());
    }
    let inner = snapshot_reference(py, &mf)?;
    let obj = PyRMP2::from_reference(inner, mf, frozen_spec);
    Ok(Py::new(py, obj)?.into_any())
}

/// Parse the Python `frozen=` argument (None / int / list[int]) into a
/// [`Frozen`] spec (MP2-03). The `'auto'` / window forms are not exposed
/// through this factory surface (they need extra kwargs upstream).
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
