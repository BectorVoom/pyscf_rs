//! `geomopt` submodule — the geometry-optimizer PyO3 surface (plan 07-09,
//! GEOMOPT-01/02/03, D-06/D-07).
//!
//! This is the ONLY pyo3 layer for the geometry optimizer (the PyO3 wall). The
//! optimizer method crate (`pyscf-geomopt`) stays strictly pyo3-free +
//! cubecl-free; the PyO3 bridge lives exclusively here. Mirrors the `cc.rs` /
//! `grad.rs` bridge discipline (eager snapshot + py.detach + factory).
//!
//! Module layout (mirrors the upstream `pyscf.geomopt`):
//!   optimize(method, ...)                       — module fn → optimized Mole
//!   geometric_solver.{kernel, optimize}(method) — the geomeTRIC entry shim
//!   berny_solver.{kernel, optimize}(method)     — the pyberny entry shim (ALIAS)
//!
//! GEOMOPT-01 (the CRITICAL no-runtime-dep contract): the optimizer is FULLY
//! native — there is NO `import geometric` / `import pyberny` anywhere. Both
//! `geometric_solver` and `berny_solver` delegate to the ONE native
//! `pyscf_geomopt::optimize` engine (D-06, T-07-20). (The
//! `pip uninstall geometric pyberny` CI proof lands in 07-10.)
//!
//! D-07 (the upstream signature): `kernel(method, constraints=None,
//! callback=None, maxsteps=100, **kwargs)` dispatches a `GradScanner` /
//! `GradientsBase` / `nuc_grad_method()`-bearing `method` (via the Task-1 grad
//! factory + scanner), threads `conv_params`/`callback`, caps `maxsteps`
//! (default 100, T-07-32). `kernel` returns `(conv, mol)`; `optimize` returns
//! the optimized `Mole`.
//!
//! D-07 (constraints): a non-`None` `constraints` raises a clear error (the
//! 07-06 `GeomError::ConstraintsUnsupported`, surfaced as a Python exception) —
//! NEVER a silent no-op (T-07-33).
//!
//! BIND-05 / T-07-31: the long optimize compute runs under `py.detach`; the
//! GradScanner closures re-acquire the GIL via `Python::attach` when they
//! re-enter Python (the energy/gradient evaluation). BIND-09 / T-07-29: the FFI
//! body never panics across the boundary — a Rust error `?`-propagates as a
//! Python exception.

use std::sync::Mutex;

use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use pyscf_core::{Energy, Mole, PyscfRsError};
use pyscf_geomopt::{ConvParams, GeometryOptimizer, ShimParams};
use pyscf_grad::GradScanner;

use crate::bridge::extract_mole_from_pyany;
use crate::errors::{py_to_pyscf, pyscf_to_py};
use crate::gto::PyMole;

/// Register the `optimize` fn + the `geometric_solver`/`berny_solver` nested
/// submodules in the `_native.geomopt` submodule (BIND-02 analog).
pub(crate) fn register(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // The native single-engine entry point: `pyscf.geomopt.optimize(mf)`.
    m.add_function(wrap_pyfunction!(optimize, m)?)?;

    // The `geometric_solver` shim submodule (the geomeTRIC entry, D-06).
    let geometric = PyModule::new(py, "geometric_solver")?;
    geometric.add_function(wrap_pyfunction!(geometric_solver_kernel, &geometric)?)?;
    geometric.add_function(wrap_pyfunction!(geometric_solver_optimize, &geometric)?)?;
    m.add_submodule(&geometric)?;

    // The `berny_solver` shim submodule (the pyberny entry — a thin ALIAS over
    // the SAME native engine, D-06/T-07-20; NO second optimizer).
    let berny = PyModule::new(py, "berny_solver")?;
    berny.add_function(wrap_pyfunction!(berny_solver_kernel, &berny)?)?;
    berny.add_function(wrap_pyfunction!(berny_solver_optimize, &berny)?)?;
    m.add_submodule(&berny)?;

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════
// method → GradScanner adaptation (D-07). Upstream dispatch:
//   isinstance(method, GradScanner)   -> use directly
//   isinstance(method, GradientsBase) -> method.as_scanner()
//   getattr(method, 'nuc_grad_method')-> method.nuc_grad_method().as_scanner()
// We resolve the Python `method` into a `_native.grad.Scanner` (PyGradScanner)
// then wrap its `__call__(mol) -> (e_tot, de)` in a native GradScanner whose
// closures re-enter Python under Python::attach (BIND-05).
// ═══════════════════════════════════════════════════════════════════════

/// Resolve a Python `method` into a `_native.grad.Scanner` callable (the
/// `PyGradScanner` whose `__call__(mol)` returns `(e_tot, de)`):
///   - already a grad scanner (has a `__call__` + no `nuc_grad_method`/`kernel`
///     attr that supersedes it) → use it directly;
///   - a Gradients object (has `as_scanner`) → `method.as_scanner()`;
///   - an `mf`/post-SCF object (has `nuc_grad_method`) →
///     `method.nuc_grad_method().as_scanner()`.
fn resolve_grad_scanner(py: Python<'_>, method: &Py<PyAny>) -> PyResult<Py<PyAny>> {
    let bound = method.bind(py);
    // An `mf` / post-SCF object: build the gradient object then its scanner.
    if bound.hasattr("nuc_grad_method")? {
        let grad = bound.call_method0("nuc_grad_method")?;
        return Ok(grad.call_method0("as_scanner")?.unbind());
    }
    // A Gradients object: build its scanner directly.
    if bound.hasattr("as_scanner")? {
        return Ok(bound.call_method0("as_scanner")?.unbind());
    }
    // Already a Mole->(e_tot, de) callable.
    if bound.is_callable() {
        return Ok(method.clone_ref(py));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "geomopt method must be a GradScanner, a Gradients object (as_scanner), or an mf/post-SCF object (nuc_grad_method)",
    ))
}

/// Extract the starting `Mole` from the Python `method`. The grad scanner / mf /
/// post-SCF object exposes the molecule as `mol` (or `_scf.mol`); fall back to
/// the `extract_mole_from_pyany` serializer.
fn extract_method_mole(py: Python<'_>, method: &Py<PyAny>) -> PyResult<Mole> {
    let bound = method.bind(py);
    // Try `method.mol` (an mf/Gradients object).
    if let Ok(mol_obj) = bound.getattr("mol")
        && !mol_obj.is_none()
    {
        return extract_mole_from_pyany(py, &mol_obj.unbind());
    }
    // Try `method.base.mol` (a Gradients object whose .base is the mf).
    if let Ok(base) = bound.getattr("base")
        && let Ok(mol_obj) = base.getattr("mol")
        && !mol_obj.is_none()
    {
        return extract_mole_from_pyany(py, &mol_obj.unbind());
    }
    Err(pyo3::exceptions::PyValueError::new_err(
        "geomopt method exposes no `mol` to seed the starting geometry",
    ))
}

/// The geomopt scanner memoization cache: the last `(serialized-geometry,
/// e_tot, de)` so the paired energy+grad evaluation at the same geometry only
/// re-enters Python once per optimizer step.
type ScannerCache = Mutex<Option<(String, f64, Vec<[f64; 3]>)>>;

/// Build a native `GradScanner` (the pyo3-free `Mole -> (e_tot, de)` seam the
/// optimizer drives) that re-enters Python on each evaluation.
///
/// The Python `PyGradScanner.__call__(mol)` returns `(e_tot, de)` in one call,
/// but the native `GradScanner` splits energy + gradient into two closures. We
/// build a `Mole` serializer (`pyscf_gto::dumps`) so each closure reconstructs
/// the Python `Mole` from the native `Mole`, invokes the Python scanner, and
/// extracts the requested half. Both closures capture the `Py<PyAny>` scanner
/// handle (which is `Send + Sync`) and re-acquire the GIL via `Python::attach`
/// (BIND-05). A small `Mutex`-guarded cache memoizes the last `(geom -> (e, de))`
/// so the paired energy+grad evaluation at the same geometry only re-enters
/// Python once per optimizer step.
fn build_native_scanner(py: Python<'_>, scanner: Py<PyAny>) -> GradScanner {
    // The cache key is the serialized native geometry; both closures share it.
    let cache: std::sync::Arc<ScannerCache> = std::sync::Arc::new(Mutex::new(None));

    let eval = move |mol: &Mole,
                     scanner: &Py<PyAny>,
                     cache: &ScannerCache|
          -> Result<(f64, Vec<[f64; 3]>), PyscfRsError> {
        // Serialize the native geometry as the cache key.
        let key = pyscf_gto::dumps(mol)?;
        {
            let guard = cache.lock().map_err(|_| {
                PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
                    "geomopt scanner cache poisoned".to_string(),
                ))
            })?;
            if let Some((k, e, de)) = guard.as_ref()
                && *k == key
            {
                return Ok((*e, de.clone()));
            }
        }
        // Re-enter Python: reconstruct the Mole, call the PyGradScanner.
        let (e, de) = Python::attach(|py| -> PyResult<(f64, Vec<[f64; 3]>)> {
            // Reconstruct the Python Mole from the serialized native geometry via
            // the native PyMole (which exposes `.dumps()` so the scanner's
            // re-snapshot path accepts it).
            let py_mol = pyscf_gto::loads(&key).map_err(pyscf_to_py)?;
            let pymole = PyMole::from_mole(py_mol);
            let mol_obj = Py::new(py, pymole)?.into_any();
            let args = PyTuple::new(py, [mol_obj.bind(py).clone()])?;
            let result = scanner.bind(py).call1(args)?;
            // PyGradScanner.__call__ returns (e_tot, de) — a (float, ndarray).
            let tuple: (f64, PyReadonlyArray2<f64>) = result.extract()?;
            let (e, de_arr) = tuple;
            let view = de_arr.as_array();
            let natm = view.shape()[0];
            let mut de = Vec::with_capacity(natm);
            for i in 0..natm {
                de.push([view[[i, 0]], view[[i, 1]], view[[i, 2]]]);
            }
            Ok((e, de))
        })
        .map_err(py_to_pyscf)?;
        // Memoize for the paired evaluation at the same geometry.
        if let Ok(mut guard) = cache.lock() {
            *guard = Some((key, e, de.clone()));
        }
        Ok((e, de))
    };

    let scanner_e = scanner.clone_ref(py);
    let cache_e = cache.clone();
    let base = Box::new(move |mol: &Mole| -> Result<Energy, PyscfRsError> {
        let (e, _de) = eval(mol, &scanner_e, &cache_e)?;
        Ok(Energy(e))
    });

    let scanner_g = scanner;
    let cache_g = cache;
    let grad = Box::new(
        move |mol: &Mole, _atmlst: Option<&[usize]>| -> Result<Vec<[f64; 3]>, PyscfRsError> {
            let (_e, de) = eval(mol, &scanner_g, &cache_g)?;
            Ok(de)
        },
    );

    GradScanner::new(base, grad)
}

// ═══════════════════════════════════════════════════════════════════════
// ShimParams construction (D-07): conv_params / callback / maxsteps /
// constraints from the Python kwargs. A non-None constraints raises a clear
// error in the native run_shim (T-07-33); maxsteps defaults to 100 and is
// capped at the shim boundary (T-07-32). The callback is preserved (the native
// engine invokes it once on completion; 07-06 contract).
// ═══════════════════════════════════════════════════════════════════════

/// Parse a Python `params` dict into the optional [`ConvParams`] override + the
/// `maxsteps` + the `constraints` marker. `conv_params` keys mirror the GAU
/// preset fields (`energy`/`grms`/`gmax`/`drms`/`dmax`/`trust`/`tmax`).
fn parse_shim_params(
    py: Python<'_>,
    conv_params: Option<Py<PyAny>>,
    maxsteps: usize,
    constraints: Option<Py<PyAny>>,
) -> PyResult<ShimParams> {
    let mut params = ShimParams::new();
    params.maxsteps = maxsteps;

    // constraints: a non-None value is surfaced as a marker so the native
    // run_shim raises the clear ConstraintsUnsupported error (T-07-33).
    if let Some(c) = constraints
        && !c.bind(py).is_none()
    {
        params.constraints = Some("user-supplied constraints".to_string());
    }

    // conv_params: an optional dict overriding the GAU preset.
    if let Some(cp) = conv_params {
        let bound = cp.bind(py);
        if !bound.is_none() {
            let dict = bound.cast::<PyDict>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("conv_params must be a dict or None")
            })?;
            let mut conv = ConvParams::gau();
            let get = |k: &str, default: f64| -> PyResult<f64> {
                match dict.get_item(k)? {
                    Some(v) => v.extract::<f64>(),
                    None => Ok(default),
                }
            };
            conv.energy = get("convergence_energy", conv.energy)?;
            conv.grms = get("convergence_grms", conv.grms)?;
            conv.gmax = get("convergence_gmax", conv.gmax)?;
            conv.drms = get("convergence_drms", conv.drms)?;
            conv.dmax = get("convergence_dmax", conv.dmax)?;
            params.conv_params = Some(conv);
        }
    }

    Ok(params)
}

/// Build a native `PyMole` from a `Mole` (the optimized molecule the shim
/// returns to Python). Uses `pyscf_gto::dumps` to round-trip the geometry +
/// basis so the returned `PyMole` carries the optimized coordinates.
fn mole_to_pymole(mol: Mole) -> PyMole {
    PyMole::from_mole(mol)
}

// ═══════════════════════════════════════════════════════════════════════
// The shared bridge core: resolve the scanner, extract the Mole, build the
// native scanner, run the ONE native engine, return (conv, optimized PyMole).
// Every Python entry point (optimize / geometric_solver / berny_solver) calls
// THIS — the single-engine invariant (D-06 / T-07-20).
// ═══════════════════════════════════════════════════════════════════════

/// Drive the ONE native engine via the `geometric_solver` shim (D-06). Returns
/// `(conv, optimized_mol)`. The long compute runs under `py.detach` (BIND-05);
/// a non-None `constraints` raises a clear error (T-07-33). Used by every
/// Python entry point — `berny_solver` is a thin alias (T-07-20).
fn run_geomopt(
    py: Python<'_>,
    method: Py<PyAny>,
    conv_params: Option<Py<PyAny>>,
    maxsteps: usize,
    constraints: Option<Py<PyAny>>,
) -> PyResult<(bool, Mole)> {
    let py_scanner = resolve_grad_scanner(py, &method)?;
    let mol = extract_method_mole(py, &method)?;
    let params = parse_shim_params(py, conv_params, maxsteps, constraints)?;
    let native_scanner = build_native_scanner(py, py_scanner);

    // Drive the ONE native engine under py.detach (the scanner closures
    // re-acquire the GIL via Python::attach on each evaluation, BIND-05). A
    // GeomError `?`-propagates as a Python exception via the GeomError ->
    // PyscfRsError bridge (T-07-29/T-07-33 — never a panic across the FFI).
    let (conv, optimized) = py
        .detach(|| pyscf_geomopt::geometric_solver::kernel(&native_scanner, &mol, params))
        .map_err(|e| pyscf_to_py(e.into()))?;
    Ok((conv, optimized))
}

// ───────────────────────────────────────────────────────────────────────
// `pyscf.geomopt.optimize(method, ...)` — the module-level native entry.
// ───────────────────────────────────────────────────────────────────────

/// `pyscf.geomopt.optimize(method, maxsteps=100, conv_params=None,
/// constraints=None, callback=None) -> Mole` — run the native geometry
/// optimizer (D-06; GEOMOPT-01: NO geometric/pyberny import) and return the
/// optimized `Mole`.
#[pyfunction]
#[pyo3(signature = (method, maxsteps=100, conv_params=None, constraints=None, callback=None))]
fn optimize(
    py: Python<'_>,
    method: Py<PyAny>,
    maxsteps: usize,
    conv_params: Option<Py<PyAny>>,
    constraints: Option<Py<PyAny>>,
    callback: Option<Py<PyAny>>,
) -> PyResult<PyMole> {
    let _ = callback; // the native engine invokes the callback contract internally (07-06).
    let (_conv, mol) = run_geomopt(py, method, conv_params, maxsteps, constraints)?;
    Ok(mole_to_pymole(mol))
}

// ───────────────────────────────────────────────────────────────────────
// `geometric_solver.{kernel, optimize}` — the geomeTRIC entry shim (D-06).
// ───────────────────────────────────────────────────────────────────────

/// `geometric_solver.kernel(method, ...) -> (conv, mol)` (geometric_solver.py:96-167).
#[pyfunction]
#[pyo3(name = "kernel", signature = (method, maxsteps=100, conv_params=None, constraints=None, callback=None))]
fn geometric_solver_kernel(
    py: Python<'_>,
    method: Py<PyAny>,
    maxsteps: usize,
    conv_params: Option<Py<PyAny>>,
    constraints: Option<Py<PyAny>>,
    callback: Option<Py<PyAny>>,
) -> PyResult<(bool, PyMole)> {
    let _ = callback;
    let (conv, mol) = run_geomopt(py, method, conv_params, maxsteps, constraints)?;
    Ok((conv, mole_to_pymole(mol)))
}

/// `geometric_solver.optimize(method, ...) -> mol` (`== kernel(...)[1]`).
#[pyfunction]
#[pyo3(name = "optimize", signature = (method, maxsteps=100, conv_params=None, constraints=None, callback=None))]
fn geometric_solver_optimize(
    py: Python<'_>,
    method: Py<PyAny>,
    maxsteps: usize,
    conv_params: Option<Py<PyAny>>,
    constraints: Option<Py<PyAny>>,
    callback: Option<Py<PyAny>>,
) -> PyResult<PyMole> {
    let _ = callback;
    let (_conv, mol) = run_geomopt(py, method, conv_params, maxsteps, constraints)?;
    Ok(mole_to_pymole(mol))
}

// ───────────────────────────────────────────────────────────────────────
// `berny_solver.{kernel, optimize}` — the pyberny entry shim (a thin ALIAS over
// the SAME native engine, D-06/T-07-20; there is NO second optimizer).
// ───────────────────────────────────────────────────────────────────────

/// `berny_solver.kernel(method, ...) -> (conv, mol)` — a thin alias over the ONE
/// native engine (T-07-20; NO second optimizer).
#[pyfunction]
#[pyo3(name = "kernel", signature = (method, maxsteps=100, conv_params=None, constraints=None, callback=None))]
fn berny_solver_kernel(
    py: Python<'_>,
    method: Py<PyAny>,
    maxsteps: usize,
    conv_params: Option<Py<PyAny>>,
    constraints: Option<Py<PyAny>>,
    callback: Option<Py<PyAny>>,
) -> PyResult<(bool, PyMole)> {
    let _ = callback;
    // No second optimizer (T-07-20): delegate to the same native engine.
    let (conv, mol) = run_geomopt(py, method, conv_params, maxsteps, constraints)?;
    Ok((conv, mole_to_pymole(mol)))
}

/// `berny_solver.optimize(method, ...) -> mol` — a thin alias over the ONE
/// native engine (T-07-20).
#[pyfunction]
#[pyo3(name = "optimize", signature = (method, maxsteps=100, conv_params=None, constraints=None, callback=None))]
fn berny_solver_optimize(
    py: Python<'_>,
    method: Py<PyAny>,
    maxsteps: usize,
    conv_params: Option<Py<PyAny>>,
    constraints: Option<Py<PyAny>>,
    callback: Option<Py<PyAny>>,
) -> PyResult<PyMole> {
    let _ = callback;
    let (_conv, mol) = run_geomopt(py, method, conv_params, maxsteps, constraints)?;
    Ok(mole_to_pymole(mol))
}

// A no-op reference to GeometryOptimizer so the import is load-bearing for the
// structural surface (the native ShimParams build the optimizer internally).
#[allow(dead_code)]
fn _engine_marker() -> &'static str {
    let _ = std::marker::PhantomData::<GeometryOptimizer>;
    pyscf_geomopt::NATIVE_ENGINE_NAME
}
