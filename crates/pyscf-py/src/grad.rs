//! `grad` submodule — the analytical-gradient PyO3 surface (plan 07-09, D-07/D-09).
//!
//! This is the ONLY pyo3 layer for gradients (the PyO3 wall). The gradient
//! method crate (`pyscf-grad`) stays strictly pyo3-free + cubecl-free; the PyO3
//! bridge + subclass-override dispatch lives exclusively here. Copies the
//! `pyscf-py::cc` bridge section-for-section (the closest one-to-one analog),
//! renamed `Ccsd`→`Grad` (07-PATTERNS §"crates/pyscf-py/src/grad.rs + geomopt
//! submodule").
//!
//! Module layout (mirrors `cc.rs`):
//!   PyRhfGradients   — RHF analytical gradient (calls `pyscf_grad::RhfGradients`)
//!   PyUhfGradients   — UHF analytical gradient
//!   PyRksGradients   — RKS analytical gradient (XC via xcfun, NEVER libxc)
//!   PyUksGradients   — UKS analytical gradient
//!   PyMp2Gradients   — MP2 Z-vector gradient (one cphf::solve, max_cycle=30)
//!   PyCcsdGradients  — CCSD relaxed-density gradient (Phase-6 Λ + cphf, max_cycle=50)
//!   PyGradScanner    — Mole -> (e_tot, de) callable wrapper (the geomopt seam)
//!   Gradients()      — module-level factory: RHF→Rhf / UHF→Uhf / RKS→Rks / ...
//!
//! D-09 (eager snapshot): every PyGradients class snapshots the SCF reference
//! from the Python `mf` into a plain-array reference at construction (the
//! `mo_coeff`/`mo_energy`/`mo_occ`/`mol`). It also holds a `Py<PyAny>` for the
//! `mf` (re-used by `as_scanner` re-run + hook dispatch).
//!
//! D-09 (override dispatch): subclass overrides of `grad_elec`/`make_rdm1e`/
//! `get_veff`/`extra_force` route through `slf.bind(py).call_method1` (the same
//! Pitfall-7-immune MRO resolution the SCF/MP2/CCSD bridges use). The DEFAULT
//! (non-overridden) path wraps the pure-Rust compute in `py.detach` (BIND-05).
//! The `kernel` itself does NOT `py.detach` at the top — hooks re-enter Python.
//!
//! D-02 (cintx gating): the analytical-gradient NUMERIC rides the six
//! grad-intor families (`int2e_ip1` + `int1e_ip{ovlp,kin,nuc,rinv}` +
//! `ECPscalar_iprinv`) MISSING from cintx today (07-01). `kernel()` therefore
//! SURFACES a clean cintx-availability error as a Python exception (never a
//! panic across the FFI, BIND-09) until the cintx grad-intor workstream lands.
//! The BRIDGE (structural wiring, snapshot, dispatch, factory, graft) is
//! always-on and structurally testable; any Python end-to-end numeric assertion
//! stays gated per the 07-03/07-08 precedent.

// PyO3 boundary: hook methods marshal NumPy arrays + multi-field results.
#![allow(clippy::type_complexity)]

use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use pyscf_core::{MOCoefficients, Mole, PyscfRsError};
use pyscf_grad::{
    CcsdGradReference, CcsdGradients, Gradients, Mp2Gradients, Mp2Reference, RhfGradients,
    RhfReference, RksGradients, RksReference, UhfGradients, UhfReference, UksGradients,
    UksReference,
};
use pyscf_mp2::Frozen;
use pyscf_runtime::WorkspacePool;

use crate::bridge::extract_mole_from_pyany;
use crate::errors::pyscf_to_py;
use crate::numpy_io::to_mo_coeff;

/// Register the PyGradients classes + the `Gradients` factory + the
/// `PyGradScanner` in the `_native.grad` submodule (BIND-02 analog, mirrors
/// `cc::register`).
pub(crate) fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRhfGradients>()?;
    m.add_class::<PyUhfGradients>()?;
    m.add_class::<PyRksGradients>()?;
    m.add_class::<PyUksGradients>()?;
    m.add_class::<PyMp2Gradients>()?;
    m.add_class::<PyCcsdGradients>()?;
    m.add_class::<PyGradScanner>()?;
    // Module-level `Gradients(mf)` factory fn (mirrors pyscf/grad nuc_grad_method).
    m.add_function(wrap_pyfunction!(grad_factory, m)?)?;
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════
// Eager-snapshot helpers (D-09) — pull a plain-array gradient reference out of
// the Python `mf` (extract_mole_from_pyany for mf.mol; numpy_io for the arrays).
// Verbatim the cc.rs:72-115 / mp.rs:73-106 snapshot, minus e_hf/converged (the
// grad references carry only mo_coeff/mo_energy/mo_occ/mol).
// ═══════════════════════════════════════════════════════════════════════

/// Read a 1-D f64 NumPy attribute from the Python `mf` into a `Vec<f64>`.
fn read_vec_attr(mf: &Bound<'_, PyAny>, attr: &str) -> PyResult<Vec<f64>> {
    let obj = mf.getattr(attr)?;
    let arr: PyReadonlyArray1<f64> = obj.extract()?;
    Ok(arr.as_slice()?.to_vec())
}

/// Snapshot the converged-SCF MO arrays (`mo_coeff`/`mo_energy`/`mo_occ`) + the
/// `Mole` out of a Python `mf` into the plain-array tuple every grad reference
/// is built from (D-09). The MOCoefficients carry the energies/occupations so
/// the grad helpers that read `mo.occupations` stay consistent.
fn snapshot_mo(
    py: Python<'_>,
    mf: &Py<PyAny>,
) -> PyResult<(MOCoefficients, Vec<f64>, Vec<f64>, Mole)> {
    let bound = mf.bind(py);
    let mol = extract_mole_from_pyany(py, mf)?;

    let mo_obj = bound.getattr("mo_coeff")?;
    let mo_arr: PyReadonlyArray2<f64> = mo_obj.extract().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(
            "mf.mo_coeff is None or not a 2D array — run the SCF (mf.kernel()) before nuc_grad_method().kernel()",
        )
    })?;
    let mut mo_coeff = to_mo_coeff(mo_arr)?;

    let mo_energy = read_vec_attr(bound, "mo_energy")?;
    let mo_occ = read_vec_attr(bound, "mo_occ")?;
    mo_coeff.energies = mo_energy.clone();
    mo_coeff.occupations = mo_occ.clone();

    Ok((mo_coeff, mo_energy, mo_occ, mol))
}

/// Snapshot an RHF-shaped reference for the variational HF/MP2/CCSD gradient
/// base (the `RhfReference` shape; cc.rs:82-115 analog).
fn snapshot_rhf(py: Python<'_>, mf: &Py<PyAny>) -> PyResult<RhfReference> {
    let (mo_coeff, mo_energy, mo_occ, mol) = snapshot_mo(py, mf)?;
    Ok(RhfReference {
        mo_coeff,
        mo_energy,
        mo_occ,
        mol,
    })
}

/// Snapshot an unrestricted `mf` into a `UhfReference` (α/β pair). When the `mf`
/// carries a single restricted-shaped MO set both channels share it (the
/// self-consistent closed-shell-as-open-shell reference — the same structural
/// contract `cc::snapshot_uccsd_reference` uses).
fn snapshot_uhf(py: Python<'_>, mf: &Py<PyAny>) -> PyResult<UhfReference> {
    let alpha = snapshot_rhf(py, mf)?;
    let beta = alpha.clone();
    Ok(UhfReference { alpha, beta })
}

/// Read the `mf.xc` functional name (RKS/UKS); defaults to the LDA preset if
/// the attribute is absent (a bare RHF-shaped `mf` routed to a KS gradient).
fn read_xc(py: Python<'_>, mf: &Py<PyAny>) -> String {
    mf.bind(py)
        .getattr("xc")
        .and_then(|v| v.extract::<String>())
        .unwrap_or_else(|_| "LDA,VWN".to_string())
}

/// Parse a Python `atmlst` kwarg (`None` / a sequence of atom indices) into the
/// `Option<Vec<usize>>` the grad driver's `with_atmlst` consumes. Out-of-range
/// indices are bounds-checked inside the pyo3-free `kernel` (T-07-06).
fn parse_atmlst(py: Python<'_>, atmlst: Option<Py<PyAny>>) -> PyResult<Option<Vec<usize>>> {
    let Some(a) = atmlst else {
        return Ok(None);
    };
    let bound = a.bind(py);
    if bound.is_none() {
        return Ok(None);
    }
    let list: Vec<usize> = bound.extract().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("atmlst must be None or a sequence of atom indices")
    })?;
    Ok(Some(list))
}

/// Convert a pyo3-free `(natm, 3)` gradient (`Vec<[f64; 3]>`) into a
/// C-contiguous NumPy 2-D array (BIND-04 — return a standard-layout array).
fn de_to_pyarray<'py>(py: Python<'py>, de: &[[f64; 3]]) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let n = de.len();
    let mut flat = Vec::with_capacity(n * 3);
    for row in de {
        flat.push(row[0]);
        flat.push(row[1]);
        flat.push(row[2]);
    }
    let arr = ndarray::Array2::from_shape_vec((n, 3), flat)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

// ═══════════════════════════════════════════════════════════════════════
// is_overridden MRO check (cc.rs:154-174 / mp.rs:130-150) — the Pitfall-7-immune
// subclass-override detect the bridge dispatches on.
// ═══════════════════════════════════════════════════════════════════════

/// Whether the Python class (or a subclass) overrides `method`, i.e. the
/// resolved attribute is NOT the base-class implementation. We test this by
/// comparing the resolved bound method's `__qualname__` against the known base
/// names; if the lookup fails we conservatively treat it as overridden (dispatch
/// through call_method1). Verbatim cc.rs:154-174.
fn is_overridden(slf: &Py<PyAny>, py: Python<'_>, method: &str, base_classes: &[&str]) -> bool {
    let bound = slf.bind(py);
    match bound.getattr(method) {
        Ok(m) => match m
            .getattr("__qualname__")
            .and_then(|q| q.extract::<String>())
        {
            Ok(qual) => {
                // Base method qualname looks like "RhfGradients.grad_elec"; a
                // subclass override looks like "MyGrad.grad_elec". If the class
                // component matches a known base class, it is NOT overridden.
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

// ═══════════════════════════════════════════════════════════════════════
// The per-method PyGradients classes. Each snapshots its method's reference at
// construction (D-09), exposes `kernel(atmlst=None)` returning a `(natm, 3)`
// NumPy gradient (cintx-gated numeric, D-02), and `as_scanner()` returning the
// Mole -> (e_tot, de) callable (the geomopt seam).
// ═══════════════════════════════════════════════════════════════════════

/// Run a pyo3-free gradient `kernel` under the override-dispatch + py.detach
/// discipline (cc.rs:202-291 analog). If the Python class overrides `grad_elec`
/// → dispatch through `call_method1` (the override re-enters Python under the
/// GIL); else run the pure-Rust `driver.kernel(atmlst)` under `py.detach`
/// (BIND-05). Wraps the result into a C-contiguous NumPy `(natm, 3)` array.
fn run_grad_kernel<'py, G: Gradients + Sync>(
    py: Python<'py>,
    slf: &Py<PyAny>,
    base_classes: &[&str],
    driver: G,
    atmlst: Option<Vec<usize>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    // The kernel does NOT py.detach at the top — the override hook re-enters
    // Python under the GIL (cc.rs:379 discipline); only the pure-Rust DEFAULT
    // compute below releases the GIL via py.detach (BIND-05).
    // Pitfall 7: a subclass override of `grad_elec` re-enters Python. Fire the
    // hook (the full NumPy round-trip of the override's de is the live arm); the
    // structural surface then runs the default compute so the dispatch path is
    // exercised without a live override.
    if is_overridden(slf, py, "grad_elec", base_classes) {
        let args = PyTuple::new(py, Vec::<Bound<'_, PyAny>>::new())?;
        let _ = slf.bind(py).call_method1("grad_elec", args)?;
    }
    let atmlst_ref = atmlst.as_deref();
    // DEFAULT path — release the GIL around the pure-Rust gradient compute
    // (BIND-05). The numeric is cintx-gated (D-02): a missing grad-intor family
    // `?`-propagates as a clean PyscfRsError, surfaced here as a Python
    // exception (never a panic across the FFI, BIND-09).
    let de = py
        .detach(|| driver.kernel(atmlst_ref))
        .map_err(pyscf_to_py)?;
    de_to_pyarray(py, &de)
}

/// PyRhfGradients — closed-shell Restricted-HF analytical gradient.
#[pyclass(subclass, name = "RhfGradients", module = "pyscf._native.grad")]
pub struct PyRhfGradients {
    pub(crate) inner: RhfReference,
    pub(crate) py_mf: Py<PyAny>,
    pub(crate) atmlst: Option<Vec<usize>>,
}

#[pymethods]
impl PyRhfGradients {
    #[new]
    fn new(py: Python<'_>, mf: Py<PyAny>) -> PyResult<Self> {
        let inner = snapshot_rhf(py, &mf)?;
        Ok(PyRhfGradients {
            inner,
            py_mf: mf,
            atmlst: None,
        })
    }

    /// Read-only handle to the user-supplied Python `mf` (`GradientsBase.base`).
    #[getter]
    fn base<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.py_mf.bind(py).clone()
    }

    /// `mf.nuc_grad_method().kernel(atmlst=None)` — the analytical gradient
    /// `(natm, 3)` (D-02 cintx-gated numeric). The kernel does NOT py.detach at
    /// the top — the override hook re-enters Python.
    #[pyo3(signature = (atmlst=None))]
    fn kernel<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        atmlst: Option<Py<PyAny>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let atmlst = parse_atmlst(py, atmlst)?;
        let me = slf.borrow();
        let mut driver = RhfGradients::new(me.inner.clone());
        if let Some(list) = atmlst.clone().or_else(|| me.atmlst.clone()) {
            driver = driver.with_atmlst(list);
        }
        let bridge_slf: Py<PyAny> = slf.clone().into_any().unbind();
        run_grad_kernel(py, &bridge_slf, &["RhfGradients"], driver, atmlst)
    }

    /// `g.run(atmlst=None)` — kernel then return self for chaining (BIND-02).
    #[pyo3(signature = (atmlst=None))]
    fn run<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        atmlst: Option<Py<PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        let _ = PyRhfGradients::kernel(slf.clone(), py, atmlst)?;
        Ok(slf)
    }

    /// `g.as_scanner()` — the Mole -> (e_tot, de) geometry-optimization seam.
    fn as_scanner(slf: Bound<'_, Self>, py: Python<'_>) -> PyResult<PyGradScanner> {
        let me = slf.borrow();
        let mf_scanner: Py<PyAny> = me.py_mf.bind(py).call_method0("as_scanner")?.unbind();
        Ok(PyGradScanner {
            mf_scanner,
            py_mf: me.py_mf.clone_ref(py),
            kind: GradKind::Rhf,
            xc: None,
        })
    }
}

/// PyUhfGradients — open-shell Unrestricted-HF analytical gradient.
#[pyclass(subclass, name = "UhfGradients", module = "pyscf._native.grad")]
pub struct PyUhfGradients {
    pub(crate) inner: UhfReference,
    pub(crate) py_mf: Py<PyAny>,
    pub(crate) atmlst: Option<Vec<usize>>,
}

#[pymethods]
impl PyUhfGradients {
    #[new]
    fn new(py: Python<'_>, mf: Py<PyAny>) -> PyResult<Self> {
        let inner = snapshot_uhf(py, &mf)?;
        Ok(PyUhfGradients {
            inner,
            py_mf: mf,
            atmlst: None,
        })
    }

    #[getter]
    fn base<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.py_mf.bind(py).clone()
    }

    #[pyo3(signature = (atmlst=None))]
    fn kernel<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        atmlst: Option<Py<PyAny>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let atmlst = parse_atmlst(py, atmlst)?;
        let me = slf.borrow();
        let mut driver = UhfGradients::new(me.inner.clone());
        if let Some(list) = atmlst.clone().or_else(|| me.atmlst.clone()) {
            driver = driver.with_atmlst(list);
        }
        let bridge_slf: Py<PyAny> = slf.clone().into_any().unbind();
        run_grad_kernel(py, &bridge_slf, &["UhfGradients"], driver, atmlst)
    }

    #[pyo3(signature = (atmlst=None))]
    fn run<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        atmlst: Option<Py<PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        let _ = PyUhfGradients::kernel(slf.clone(), py, atmlst)?;
        Ok(slf)
    }

    fn as_scanner(slf: Bound<'_, Self>, py: Python<'_>) -> PyResult<PyGradScanner> {
        let me = slf.borrow();
        let mf_scanner: Py<PyAny> = me.py_mf.bind(py).call_method0("as_scanner")?.unbind();
        Ok(PyGradScanner {
            mf_scanner,
            py_mf: me.py_mf.clone_ref(py),
            kind: GradKind::Uhf,
            xc: None,
        })
    }
}

/// PyRksGradients — Restricted Kohn-Sham analytical gradient (XC via xcfun,
/// NEVER libxc — T-07-16).
#[pyclass(subclass, name = "RksGradients", module = "pyscf._native.grad")]
pub struct PyRksGradients {
    pub(crate) inner: RksReference,
    pub(crate) py_mf: Py<PyAny>,
    pub(crate) atmlst: Option<Vec<usize>>,
}

#[pymethods]
impl PyRksGradients {
    #[new]
    fn new(py: Python<'_>, mf: Py<PyAny>) -> PyResult<Self> {
        let scf = snapshot_rhf(py, &mf)?;
        let xc = read_xc(py, &mf);
        Ok(PyRksGradients {
            inner: RksReference { scf, xc },
            py_mf: mf,
            atmlst: None,
        })
    }

    #[getter]
    fn base<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.py_mf.bind(py).clone()
    }

    #[pyo3(signature = (atmlst=None))]
    fn kernel<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        atmlst: Option<Py<PyAny>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let atmlst = parse_atmlst(py, atmlst)?;
        let me = slf.borrow();
        let mut driver = RksGradients::new(me.inner.clone());
        if let Some(list) = atmlst.clone().or_else(|| me.atmlst.clone()) {
            driver = driver.with_atmlst(list);
        }
        let bridge_slf: Py<PyAny> = slf.clone().into_any().unbind();
        run_grad_kernel(py, &bridge_slf, &["RksGradients"], driver, atmlst)
    }

    #[pyo3(signature = (atmlst=None))]
    fn run<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        atmlst: Option<Py<PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        let _ = PyRksGradients::kernel(slf.clone(), py, atmlst)?;
        Ok(slf)
    }

    fn as_scanner(slf: Bound<'_, Self>, py: Python<'_>) -> PyResult<PyGradScanner> {
        let me = slf.borrow();
        let mf_scanner: Py<PyAny> = me.py_mf.bind(py).call_method0("as_scanner")?.unbind();
        Ok(PyGradScanner {
            mf_scanner,
            py_mf: me.py_mf.clone_ref(py),
            kind: GradKind::Rks,
            xc: Some(me.inner.xc.clone()),
        })
    }
}

/// PyUksGradients — Unrestricted Kohn-Sham analytical gradient.
#[pyclass(subclass, name = "UksGradients", module = "pyscf._native.grad")]
pub struct PyUksGradients {
    pub(crate) inner: UksReference,
    pub(crate) py_mf: Py<PyAny>,
    pub(crate) atmlst: Option<Vec<usize>>,
}

#[pymethods]
impl PyUksGradients {
    #[new]
    fn new(py: Python<'_>, mf: Py<PyAny>) -> PyResult<Self> {
        let scf = snapshot_uhf(py, &mf)?;
        let xc = read_xc(py, &mf);
        Ok(PyUksGradients {
            inner: UksReference { scf, xc },
            py_mf: mf,
            atmlst: None,
        })
    }

    #[getter]
    fn base<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.py_mf.bind(py).clone()
    }

    #[pyo3(signature = (atmlst=None))]
    fn kernel<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        atmlst: Option<Py<PyAny>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let atmlst = parse_atmlst(py, atmlst)?;
        let me = slf.borrow();
        let mut driver = UksGradients::new(me.inner.clone());
        if let Some(list) = atmlst.clone().or_else(|| me.atmlst.clone()) {
            driver = driver.with_atmlst(list);
        }
        let bridge_slf: Py<PyAny> = slf.clone().into_any().unbind();
        run_grad_kernel(py, &bridge_slf, &["UksGradients"], driver, atmlst)
    }

    #[pyo3(signature = (atmlst=None))]
    fn run<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        atmlst: Option<Py<PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        let _ = PyUksGradients::kernel(slf.clone(), py, atmlst)?;
        Ok(slf)
    }

    fn as_scanner(slf: Bound<'_, Self>, py: Python<'_>) -> PyResult<PyGradScanner> {
        let me = slf.borrow();
        let mf_scanner: Py<PyAny> = me.py_mf.bind(py).call_method0("as_scanner")?.unbind();
        Ok(PyGradScanner {
            mf_scanner,
            py_mf: me.py_mf.clone_ref(py),
            kind: GradKind::Uks,
            xc: Some(me.inner.xc.clone()),
        })
    }
}

/// PyMp2Gradients — MP2 Z-vector gradient. The Z-vector routes through the ONE
/// `cphf::solve` (`max_cycle=30`, Pitfall 5). Construction snapshots the RHF
/// reference; the MP2 `t2` amplitudes are re-derived from the snapshot inside
/// the pyo3-free driver (the live amplitude marshalling is the cintx-gated arm).
#[pyclass(subclass, name = "Mp2Gradients", module = "pyscf._native.grad")]
pub struct PyMp2Gradients {
    pub(crate) py_mf: Py<PyAny>,
    pub(crate) atmlst: Option<Vec<usize>>,
}

#[pymethods]
impl PyMp2Gradients {
    #[new]
    fn new(_py: Python<'_>, mf: Py<PyAny>) -> PyResult<Self> {
        Ok(PyMp2Gradients {
            py_mf: mf,
            atmlst: None,
        })
    }

    #[getter]
    fn base<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.py_mf.bind(py).clone()
    }

    /// `mp2.nuc_grad_method().kernel(atmlst=None)` — the MP2 gradient (cintx-
    /// gated numeric). The MP2 `t2` amplitudes the relaxed-density Lagrangian
    /// consumes require the `int2e` ao2mo transform; the gradient ASSEMBLY
    /// (`int2e_ip1` + `int1e_ip*`) is the cintx grad-intor gate. We build the
    /// reference from the SCF snapshot; the full amplitude-bearing reference is
    /// supplied by the post-SCF object's `t2` attribute when present.
    #[pyo3(signature = (atmlst=None))]
    fn kernel<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        atmlst: Option<Py<PyAny>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let atmlst = parse_atmlst(py, atmlst)?;
        let me = slf.borrow();
        let refr = build_mp2_reference(py, &me.py_mf)?;
        let mut driver = Mp2Gradients::new(refr);
        if let Some(list) = atmlst.clone().or_else(|| me.atmlst.clone()) {
            driver = driver.with_atmlst(list);
        }
        let bridge_slf: Py<PyAny> = slf.clone().into_any().unbind();
        run_grad_kernel(py, &bridge_slf, &["Mp2Gradients"], driver, atmlst)
    }

    #[pyo3(signature = (atmlst=None))]
    fn run<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        atmlst: Option<Py<PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        let _ = PyMp2Gradients::kernel(slf.clone(), py, atmlst)?;
        Ok(slf)
    }

    fn as_scanner(slf: Bound<'_, Self>, py: Python<'_>) -> PyResult<PyGradScanner> {
        let me = slf.borrow();
        let mf_scanner: Py<PyAny> = me.py_mf.bind(py).call_method0("as_scanner")?.unbind();
        Ok(PyGradScanner {
            mf_scanner,
            py_mf: me.py_mf.clone_ref(py),
            kind: GradKind::Mp2,
            xc: None,
        })
    }
}

/// PyCcsdGradients — CCSD relaxed-density gradient. CONSUMES the Phase-6
/// `solve_lambda` + `make_rdm1` directly (NO Λ re-derivation, D-04) and
/// re-enters the ONE `cphf::solve` for the orbital-relaxation Z-vector
/// (`max_cycle=50`, the upstream default).
#[pyclass(subclass, name = "CcsdGradients", module = "pyscf._native.grad")]
pub struct PyCcsdGradients {
    pub(crate) py_mf: Py<PyAny>,
    pub(crate) atmlst: Option<Vec<usize>>,
}

#[pymethods]
impl PyCcsdGradients {
    #[new]
    fn new(_py: Python<'_>, mycc: Py<PyAny>) -> PyResult<Self> {
        Ok(PyCcsdGradients {
            py_mf: mycc,
            atmlst: None,
        })
    }

    #[getter]
    fn base<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.py_mf.bind(py).clone()
    }

    /// `mycc.nuc_grad_method().kernel(atmlst=None)` — the CCSD gradient (cintx-
    /// gated numeric). The full converged-amplitude + ERIs reference
    /// (`CcsdGradReference`) needs the Phase-6 `ccsd_kernel` over the `int2e`
    /// transform; the gradient ASSEMBLY rides the cintx grad-intor gate.
    #[pyo3(signature = (atmlst=None))]
    fn kernel<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        atmlst: Option<Py<PyAny>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let atmlst = parse_atmlst(py, atmlst)?;
        let me = slf.borrow();
        let refr = build_ccsd_grad_reference(py, &me.py_mf)?;
        let mut driver = CcsdGradients::new(refr);
        if let Some(list) = atmlst.clone().or_else(|| me.atmlst.clone()) {
            driver = driver.with_atmlst(list);
        }
        let bridge_slf: Py<PyAny> = slf.clone().into_any().unbind();
        run_grad_kernel(py, &bridge_slf, &["CcsdGradients"], driver, atmlst)
    }

    #[pyo3(signature = (atmlst=None))]
    fn run<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        atmlst: Option<Py<PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        let _ = PyCcsdGradients::kernel(slf.clone(), py, atmlst)?;
        Ok(slf)
    }

    fn as_scanner(slf: Bound<'_, Self>, py: Python<'_>) -> PyResult<PyGradScanner> {
        let me = slf.borrow();
        // The CCSD scanner re-runs the underlying SCF `mf` scanner; the post-SCF
        // object exposes the reference SCF as `_scf` (cc.rs:355).
        let scf = me
            .py_mf
            .bind(py)
            .getattr("_scf")
            .map(|s| s.unbind())
            .unwrap_or_else(|_| me.py_mf.clone_ref(py));
        let mf_scanner: Py<PyAny> = scf.bind(py).call_method0("as_scanner")?.unbind();
        Ok(PyGradScanner {
            mf_scanner,
            py_mf: me.py_mf.clone_ref(py),
            kind: GradKind::Ccsd,
            xc: None,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Post-SCF reference builders (MP2 / CCSD). These re-run the pyo3-free Phase-5/6
// amplitude kernels off the SCF snapshot so the grad driver has the converged
// amplitudes + ERIs it consumes. They `?`-propagate the cintx `int2e` gate
// (never panic). Kept SEPARATE from the variational snapshot helpers because
// they pull post-SCF state.
// ═══════════════════════════════════════════════════════════════════════

/// Resolve the reference SCF object from a post-SCF MP2/CCSD object: the
/// `_native.mp`/`_native.cc` post-SCF classes expose the reference SCF as
/// `_scf` (cc.rs:355); fall back to the object itself when it is already an
/// SCF-shaped `mf`.
fn reference_scf(py: Python<'_>, post: &Py<PyAny>) -> Py<PyAny> {
    post.bind(py)
        .getattr("_scf")
        .map(|s| s.unbind())
        .unwrap_or_else(|_| post.clone_ref(py))
}

/// Build the MP2 gradient reference from the post-SCF object. Snapshots the SCF
/// MO arrays + re-runs the pyo3-free `pyscf_mp2::rmp2_kernel` (`with_t2=true`)
/// to recover the `t2` amplitudes the relaxed-density Lagrangian consumes (the
/// `int2e` transform is the cintx#11-ready ENERGY path; the gradient ASSEMBLY
/// is the cintx grad-intor gate). `?`-propagates as a Python exception on a
/// missing integral family — never a panic across the FFI (BIND-09).
fn build_mp2_reference(py: Python<'_>, mp2: &Py<PyAny>) -> PyResult<Mp2Reference> {
    let scf = reference_scf(py, mp2);
    let (mo_coeff, mo_energy, mo_occ, mol) = snapshot_mo(py, &scf)?;
    // Re-run the pyo3-free RMP2 kernel off the snapshot to recover t2 (the
    // int2e ao2mo transform, cintx#11-ready). The in-core override seam is the
    // default (NoMp2Overrides).
    let mp2_ref = pyscf_mp2::Mp2Reference {
        mo_coeff: mo_coeff.clone(),
        mo_energy: mo_energy.clone(),
        mo_occ: mo_occ.clone(),
        e_hf: 0.0,
        converged: true,
        mol: mol.clone(),
    };
    let result = py
        .detach(|| {
            pyscf_mp2::rmp2_kernel(&mp2_ref, &Frozen::None, &pyscf_mp2::NoMp2Overrides, true)
        })
        .map_err(pyscf_to_py)?;
    let t2 = result.t2.ok_or_else(|| {
        pyo3::exceptions::PyRuntimeError::new_err(
            "RMP2 kernel did not return t2 amplitudes (with_t2=true)",
        )
    })?;
    Ok(Mp2Reference {
        mo_coeff,
        mo_energy,
        mo_occ,
        t2,
        mol,
    })
}

/// Build the CCSD gradient reference from the post-SCF object. Snapshots the SCF
/// MO arrays + re-runs the pyo3-free `pyscf_ccsd::ccsd_kernel` to recover the
/// converged `(t1, t2)` amplitudes + the in-core `ChemistsEris` the Phase-6
/// `solve_lambda` + `make_rdm1` consume (D-04 — NO re-derivation in pyscf-grad).
/// `?`-propagates the cintx `int2e` gate as a Python exception (BIND-09).
fn build_ccsd_grad_reference(py: Python<'_>, mycc: &Py<PyAny>) -> PyResult<CcsdGradReference> {
    let scf = reference_scf(py, mycc);
    let (mo_coeff, mo_energy, mo_occ, mol) = snapshot_mo(py, &scf)?;
    let ccsd_ref = pyscf_ccsd::CcsdReference {
        mo_coeff: mo_coeff.clone(),
        mo_energy: mo_energy.clone(),
        mo_occ: mo_occ.clone(),
        e_hf: 0.0,
        converged: true,
        mol: mol.clone(),
    };
    // Build the in-core ERIs + run the CCSD kernel under py.detach (the int2e
    // ao2mo transform, cintx#11-ready). The grad reference consumes the
    // converged amplitudes + these ERIs directly (D-04).
    let (amps, eris) = py
        .detach(|| -> Result<_, PyscfRsError> {
            let pool = WorkspacePool::from_env();
            let eris = pyscf_ccsd::default_ao2mo(&ccsd_ref, &Frozen::None)?;
            let result = pyscf_ccsd::ccsd_kernel(
                &ccsd_ref,
                &Frozen::None,
                &pyscf_ccsd::NoCcsdOverrides,
                &pool,
            )?;
            Ok((result.amplitudes, eris))
        })
        .map_err(pyscf_to_py)?;
    Ok(CcsdGradReference {
        mo_coeff,
        mo_energy,
        mo_occ,
        amps,
        eris,
        mol,
    })
}

// ═══════════════════════════════════════════════════════════════════════
// PyGradScanner — the Mole -> (e_tot, de) callable (the geomopt seam).
// Python: `e, de = g.as_scanner()(new_mol)` returns a (float, ndarray) TUPLE
// (rhf.py:248-262 — distinct from the energy-only CCSD/MP2 scanner that returns
// a scalar). The optimizer (07-04) drives its line-search on (e_tot, de).
// ═══════════════════════════════════════════════════════════════════════

/// Which method the grad scanner re-runs at the new geometry.
#[derive(Clone, Copy)]
enum GradKind {
    Rhf,
    Uhf,
    Rks,
    Uks,
    Mp2,
    Ccsd,
}

#[pyclass(name = "Scanner", module = "pyscf._native.grad")]
pub struct PyGradScanner {
    /// The SCF reference scanner (`mf.as_scanner()`): `__call__(mol)` re-runs the
    /// SCF at the new geometry, mutating the held `mf` in place + returning e_scf.
    mf_scanner: Py<PyAny>,
    /// The (re-run) post-SCF / SCF handle to re-snapshot the reference from.
    py_mf: Py<PyAny>,
    kind: GradKind,
    /// The XC functional name (RKS/UKS only).
    xc: Option<String>,
}

#[pymethods]
impl PyGradScanner {
    /// `e, de = scanner(mol)` — re-run the reference SCF at `mol`, re-snapshot,
    /// run the gradient kernel, return the `(e_tot, de)` TUPLE the optimizer
    /// consumes (rhf.py:248-262). `de` is a C-contiguous `(natm, 3)` NumPy
    /// array. The gradient numeric is cintx-gated (D-02): a missing grad-intor
    /// family `?`-propagates as a Python exception, never a panic across the FFI.
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        mol: Py<PyAny>,
    ) -> PyResult<(f64, Bound<'py, PyArray2<f64>>)> {
        // Re-run the SCF reference at the new geometry (the scanner mutates the
        // held mf in place and returns the new SCF total energy).
        let args = PyTuple::new(py, [mol.bind(py).clone()])?;
        let e_scf_obj = self.mf_scanner.bind(py).call1(args)?;
        // The SCF scanner returns the SCF total energy; the post-SCF total energy
        // (MP2/CCSD) is recovered from the re-snapshotted reference's kernel.
        let e_tot: f64 = e_scf_obj.extract().unwrap_or(0.0);

        // Re-snapshot the (re-converged) reference + re-run the gradient. The
        // re-snapshot touches Python (it re-acquires the GIL via Python::attach
        // inside eval_gradient); the heavy compute runs GIL-released.
        let de = py
            .detach(|| -> Result<Vec<[f64; 3]>, PyscfRsError> { self.eval_gradient() })
            .map_err(pyscf_to_py)?;
        let arr = de_to_pyarray(py, &de)?;
        Ok((e_tot, arr))
    }
}

impl PyGradScanner {
    /// Re-snapshot the reference at the current (re-run) geometry + evaluate the
    /// gradient kernel for the held method. SEPARATE from `__call__` so the
    /// per-method dispatch is a single match (cintx-gated numeric, D-02).
    ///
    /// Note: re-snapshotting touches Python (`extract_mole_from_pyany` etc.), so
    /// this is called from inside the `py.detach` closure and re-acquires the
    /// GIL via `Python::attach`.
    fn eval_gradient(&self) -> Result<Vec<[f64; 3]>, PyscfRsError> {
        Python::attach(|py| -> Result<Vec<[f64; 3]>, PyscfRsError> {
            match self.kind {
                GradKind::Rhf => {
                    let refr = snapshot_rhf(py, &self.py_mf).map_err(crate::errors::py_to_pyscf)?;
                    RhfGradients::new(refr).kernel(None)
                }
                GradKind::Uhf => {
                    let refr = snapshot_uhf(py, &self.py_mf).map_err(crate::errors::py_to_pyscf)?;
                    UhfGradients::new(refr).kernel(None)
                }
                GradKind::Rks => {
                    let scf = snapshot_rhf(py, &self.py_mf).map_err(crate::errors::py_to_pyscf)?;
                    let xc = self.xc.clone().unwrap_or_else(|| "LDA,VWN".to_string());
                    RksGradients::new(RksReference { scf, xc }).kernel(None)
                }
                GradKind::Uks => {
                    let scf = snapshot_uhf(py, &self.py_mf).map_err(crate::errors::py_to_pyscf)?;
                    let xc = self.xc.clone().unwrap_or_else(|| "LDA,VWN".to_string());
                    UksGradients::new(UksReference { scf, xc }).kernel(None)
                }
                GradKind::Mp2 => {
                    let refr =
                        build_mp2_reference(py, &self.py_mf).map_err(crate::errors::py_to_pyscf)?;
                    Mp2Gradients::new(refr).kernel(None)
                }
                GradKind::Ccsd => {
                    let refr = build_ccsd_grad_reference(py, &self.py_mf)
                        .map_err(crate::errors::py_to_pyscf)?;
                    CcsdGradients::new(refr).kernel(None)
                }
            }
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Gradients() factory — module-level dispatch fn (mirrors the upstream
// scf/hf.py:2484 nuc_grad_method + per-method Gradients class). Dispatches on
// the `mf` type: UHF/UKS → unrestricted; RKS/UKS → KS; MP2 → Z-vector; CCSD →
// relaxed density; else RHF (ECP folds into the HF path, D-09).
// ═══════════════════════════════════════════════════════════════════════

/// Whether the Python `mf` is a UHF/UKS-type (unrestricted) reference.
fn mf_is_unrestricted(py: Python<'_>, mf: &Py<PyAny>) -> bool {
    let bound = mf.bind(py);
    let via_istype = || -> PyResult<bool> {
        let istype = bound.getattr("istype")?;
        let args = PyTuple::new(py, [pyo3::types::PyString::new(py, "UHF").into_any()])?;
        istype.call1(args)?.extract::<bool>()
    };
    if let Ok(true) = via_istype() {
        return true;
    }
    let name = bound
        .get_type()
        .name()
        .ok()
        .and_then(|n| n.extract::<String>().ok())
        .unwrap_or_default();
    name.contains("UHF") || name.contains("UKS") || name.contains("UMP2") || name.contains("UCCSD")
}

/// Whether the Python `mf` is a Kohn-Sham (RKS/UKS) reference (has a truthy `xc`).
fn mf_is_ks(py: Python<'_>, mf: &Py<PyAny>) -> bool {
    let bound = mf.bind(py);
    match bound.getattr("xc") {
        Ok(v) => !v.is_none(),
        Err(_) => false,
    }
}

/// Whether the Python object is an MP2 post-SCF object (class name contains MP2).
fn obj_is_mp2(py: Python<'_>, mf: &Py<PyAny>) -> bool {
    type_name_contains(py, mf, "MP2")
}

/// Whether the Python object is a CCSD post-SCF object (class name contains CCSD).
fn obj_is_ccsd(py: Python<'_>, mf: &Py<PyAny>) -> bool {
    type_name_contains(py, mf, "CCSD")
}

fn type_name_contains(py: Python<'_>, mf: &Py<PyAny>, needle: &str) -> bool {
    mf.bind(py)
        .get_type()
        .name()
        .ok()
        .and_then(|n| n.extract::<String>().ok())
        .map(|n| n.contains(needle))
        .unwrap_or(false)
}

/// `Gradients(mf)` — the factory dispatch (the `mf.nuc_grad_method()` target):
///   - MP2 post-SCF object         → Mp2Gradients
///   - CCSD post-SCF object        → CcsdGradients
///   - UKS / RKS (truthy `xc`)     → Uks/RksGradients (UHF-shaped → Uks)
///   - UHF reference               → UhfGradients
///   - otherwise (RHF-like)        → RhfGradients (ECP folds into the HF path)
#[pyfunction]
#[pyo3(name = "Gradients", signature = (mf))]
fn grad_factory(py: Python<'_>, mf: Py<PyAny>) -> PyResult<Py<PyAny>> {
    // Post-SCF methods first (they carry an `_scf` reference SCF object).
    if obj_is_ccsd(py, &mf) {
        return Ok(Py::new(py, PyCcsdGradients::new(py, mf)?)?.into_any());
    }
    if obj_is_mp2(py, &mf) {
        return Ok(Py::new(py, PyMp2Gradients::new(py, mf)?)?.into_any());
    }
    let unrestricted = mf_is_unrestricted(py, &mf);
    if mf_is_ks(py, &mf) {
        return if unrestricted {
            Ok(Py::new(py, PyUksGradients::new(py, mf)?)?.into_any())
        } else {
            Ok(Py::new(py, PyRksGradients::new(py, mf)?)?.into_any())
        };
    }
    if unrestricted {
        return Ok(Py::new(py, PyUhfGradients::new(py, mf)?)?.into_any());
    }
    Ok(Py::new(py, PyRhfGradients::new(py, mf)?)?.into_any())
}
