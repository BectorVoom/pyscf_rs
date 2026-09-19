//! `pyscf._native.pbc.lib` — `kpts_helper` and `kpts` (plan 20-14).
//!
//! Upstream scripts import these as modules (`from pyscf.pbc.lib import
//! kpts_helper`; `from pyscf.pbc.lib.kpts import KPoints`), so both are real
//! nested modules — `pyscf._native.pbc.lib.kpts_helper` and
//! `pyscf._native.pbc.lib.kpts` — registered in `sys.modules` with the same
//! three-step pattern `pbc/mod.rs` uses.
//!
//! Nothing is bound twice:
//! * `kpts.KPoints` / `kpts.make_kpts` ARE `pyscf._native.pbc.symm`'s objects;
//! * `kpts_helper.get_kconserv` IS `pyscf._native.pbc.gto.get_kconserv` (20-09);
//! * `is_gamma_point` / `gamma_point` ARE `is_zero` (`kpts_helper.py:37`).
//!
//! The Rust core (`pyscf_pbc_lib::kpts_helper`) takes scaled k-points or the
//! lattice instead of a `Cell` (it sits below `pyscf-pbc-gto`); the conversions
//! here are exactly upstream's `cell.get_scaled_kpts` / `get_abs_kpts` calls.

use numpy::ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use pyscf_pbc_lib::kpts_helper::{self as kh, KIdx, KPT_DIFF_TOL};

use crate::bridge::extract_cell_from_pyany;
use crate::pbc::convert::{extract_kpts, kpts_to_pyarray};

/// Register `kpts_helper` and `kpts` as children of `pyscf._native.pbc.lib`.
/// `symm` and `gto` must already be in `sys.modules`.
pub fn register(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let sys_modules = PyModule::import(py, "sys")?.getattr("modules")?;
    let parent: String = m.name()?.extract()?;

    let helper = PyModule::new(py, &format!("{parent}.kpts_helper"))?;
    helper.setattr(
        "__doc__",
        "pyscf.pbc.lib.kpts_helper — k-point helpers (plan 20-14).",
    )?;
    helper.add("KPT_DIFF_TOL", KPT_DIFF_TOL)?;
    helper.add_function(wrap_pyfunction!(is_zero, &helper)?)?;
    helper.add("is_gamma_point", helper.getattr("is_zero")?)?;
    helper.add("gamma_point", helper.getattr("is_zero")?)?;
    helper.add_function(wrap_pyfunction!(is_trim, &helper)?)?;
    helper.add_function(wrap_pyfunction!(member, &helper)?)?;
    helper.add_function(wrap_pyfunction!(intersection, &helper)?)?;
    helper.add_function(wrap_pyfunction!(unique, &helper)?)?;
    helper.add_function(wrap_pyfunction!(unique_with_wrap_around, &helper)?)?;
    helper.add_function(wrap_pyfunction!(group_by_conj_pairs, &helper)?)?;
    helper.add_function(wrap_pyfunction!(kk_adapted_iter, &helper)?)?;
    helper.add_function(wrap_pyfunction!(get_kconserv3, &helper)?)?;
    let gto = sys_modules.get_item("pyscf._native.pbc.gto")?;
    helper.add("get_kconserv", gto.getattr("get_kconserv")?)?;

    let kpts = PyModule::new(py, &format!("{parent}.kpts"))?;
    kpts.setattr(
        "__doc__",
        "pyscf.pbc.lib.kpts — the KPoints type (same object as pyscf._native.pbc.symm.KPoints).",
    )?;
    let symm = sys_modules.get_item("pyscf._native.pbc.symm")?;
    kpts.add("KPoints", symm.getattr("KPoints")?)?;
    kpts.add("make_kpts", symm.getattr("make_kpts")?)?;
    kpts.add("KPT_DIFF_TOL", KPT_DIFF_TOL)?;

    for child in [&helper, &kpts] {
        m.add_submodule(child)?;
        let full: String = child.name()?.extract()?;
        sys_modules.set_item(full, child)?;
    }
    Ok(())
}

fn idx_i64<'py>(py: Python<'py>, v: &[usize]) -> Bound<'py, PyArray1<i64>> {
    PyArray1::from_vec(py, v.iter().map(|&x| x as i64).collect())
}

fn idx_i32<'py>(py: Python<'py>, v: &[usize]) -> Bound<'py, PyArray1<i32>> {
    PyArray1::from_vec(py, v.iter().map(|&x| x as i32).collect())
}

/// `is_zero(kpt)` — `sum |k| < KPT_DIFF_TOL`, any shape.
#[pyfunction]
fn is_zero(py: Python<'_>, kpt: &Bound<'_, PyAny>) -> PyResult<bool> {
    let np = PyModule::import(py, "numpy")?;
    let flat = np
        .call_method1("asarray", (kpt, "float64"))?
        .call_method0("ravel")?;
    let v: Vec<f64> = flat.extract()?;
    Ok(kh::is_zero(&v))
}

/// `is_trim(cell, kpts, tol=KPT_DIFF_TOL)` → bool array.
#[pyfunction(signature = (cell, kpts, tol = KPT_DIFF_TOL))]
fn is_trim<'py>(
    py: Python<'py>,
    cell: &Bound<'py, PyAny>,
    kpts: &Bound<'py, PyAny>,
    tol: f64,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let c = extract_cell_from_pyany(py, cell)?;
    let (k, _) = extract_kpts(kpts)?;
    Ok(PyArray1::from_vec(
        py,
        pyscf_pbc_gto::kpts_mesh::is_trim(&c, &k, tol),
    ))
}

/// `member(kpt, kpts)` → ascending int64 indices.
#[pyfunction]
fn member<'py>(
    kpt: &Bound<'py, PyAny>,
    kpts: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let (k, _) = extract_kpts(kpt)?;
    if k.len() != 1 {
        return Err(PyValueError::new_err("member: kpt must be ONE 3-vector"));
    }
    let (ks, _) = extract_kpts(kpts)?;
    Ok(idx_i64(kpt.py(), &kh::member(&k[0], &ks)))
}

/// `intersection(kpts1, kpts2)` → ascending int64 indices into `kpts1`.
#[pyfunction]
fn intersection<'py>(
    kpts1: &Bound<'py, PyAny>,
    kpts2: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let (a, _) = extract_kpts(kpts1)?;
    let (b, _) = extract_kpts(kpts2)?;
    Ok(idx_i64(kpts1.py(), &kh::intersection(&a, &b)))
}

/// `unique(kpts)` → `(uniq_kpts, uniq_index, uniq_inverse)`, first-occurrence order.
#[pyfunction]
fn unique<'py>(py: Python<'py>, kpts: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyTuple>> {
    let (k, _) = extract_kpts(kpts)?;
    let u = kh::unique(&k);
    PyTuple::new(
        py,
        [
            kpts_to_pyarray(py, &u.kpts)?.into_any(),
            idx_i64(py, &u.index).into_any(),
            idx_i64(py, &u.inverse).into_any(),
        ],
    )
}

/// `unique_with_wrap_around(cell, kpts)` → `(uniq_kpts, uniq_index, uniq_inverse)`.
#[pyfunction]
fn unique_with_wrap_around<'py>(
    py: Python<'py>,
    cell: &Bound<'py, PyAny>,
    kpts: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    let c = extract_cell_from_pyany(py, cell)?;
    let (k, _) = extract_kpts(kpts)?;
    let (index, inverse) = kh::unique_with_wrap_around(&c.get_scaled_kpts(&k));
    let uniq: Vec<[f64; 3]> = index.iter().map(|&i| k[i]).collect();
    PyTuple::new(
        py,
        [
            kpts_to_pyarray(py, &uniq)?.into_any(),
            idx_i64(py, &index).into_any(),
            idx_i64(py, &inverse).into_any(),
        ],
    )
}

/// `np.round(x, 5)` — the rounding `group_by_conj_pairs` applies.
fn round5(x: f64) -> f64 {
    (x * 1e5).round_ties_even() / 1e5
}

/// `group_by_conj_pairs(cell, kpts, wrap_around=True, return_kpts_pairs=True)`
/// — `idx_pairs` (list of `(k, k_conj or None)`), plus `kpts_pairs` when asked.
///
/// The `kpts_pairs` of the `wrap_around` branch are upstream's
/// `cell.get_abs_kpts(scaled)` of the folded scaled k-points
/// (`kpts_helper.py:179-187`); the fold is repeated here operation for
/// operation because the Rust core returns indices only.
#[pyfunction(signature = (cell, kpts, wrap_around = true, return_kpts_pairs = true))]
fn group_by_conj_pairs<'py>(
    py: Python<'py>,
    cell: &Bound<'py, PyAny>,
    kpts: &Bound<'py, PyAny>,
    wrap_around: bool,
    return_kpts_pairs: bool,
) -> PyResult<Py<PyAny>> {
    let c = extract_cell_from_pyany(py, cell)?;
    let (k, _) = extract_kpts(kpts)?;
    let scaled = c.get_scaled_kpts(&k);
    let pairs = kh::group_by_conj_pairs(&scaled, wrap_around);
    let idx_pairs = PyList::new(py, pairs.iter().map(|&(i, j)| (i, j)))?;
    if !return_kpts_pairs {
        return Ok(idx_pairs.into_any().unbind());
    }
    let kk: Vec<[f64; 3]> = if wrap_around {
        let folded: Vec<[f64; 3]> = scaled
            .iter()
            .map(|s| {
                s.map(|x| {
                    let mut v = x - x.trunc(); // np.modf(scaled)[0]
                    let r = round5(v);
                    if r > 0.5 {
                        v -= 1.0;
                    } else if r <= -0.5 {
                        v += 1.0;
                    }
                    v
                })
            })
            .collect();
        c.get_abs_kpts(&folded)
            .map_err(crate::errors::pyscf_to_py)?
    } else {
        k.clone()
    };
    let np1 = |v: [f64; 3]| PyArray1::from_vec(py, v.to_vec()).into_any();
    let kpts_pairs = PyList::new(
        py,
        pairs
            .iter()
            .map(|&(i, j)| -> PyResult<Bound<'py, PyTuple>> {
                let second = match j {
                    Some(j) => np1(kk[j]),
                    None => py.None().into_bound(py),
                };
                PyTuple::new(py, [np1(kk[i]), second])
            })
            .collect::<PyResult<Vec<_>>>()?,
    )?;
    Ok(
        PyTuple::new(py, [idx_pairs.into_any(), kpts_pairs.into_any()])?
            .into_any()
            .unbind(),
    )
}

/// `kk_adapted_iter(cell, kpts, kk_idx=None, time_reversal_symmetry=True)` —
/// an iterator of `(kpt, kpti_idx, kptj_idx, self_conj)` (int32 index arrays),
/// in upstream's group order. `dk[i*nk+j] = kpts[j] - kpts[i]`
/// (`kpts_helper.py:228`), optionally subset by `kk_idx`, then
/// `cell.get_scaled_kpts(dk)`.
#[pyfunction(signature = (cell, kpts, kk_idx = None, time_reversal_symmetry = true))]
fn kk_adapted_iter<'py>(
    py: Python<'py>,
    cell: &Bound<'py, PyAny>,
    kpts: &Bound<'py, PyAny>,
    kk_idx: Option<&Bound<'py, PyAny>>,
    time_reversal_symmetry: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let kk_idx: Option<Vec<usize>> = match kk_idx {
        Some(v) if !v.is_none() => Some(v.extract()?),
        _ => None,
    };
    if kk_idx.is_some() && time_reversal_symmetry {
        return Err(PyNotImplementedError::new_err(
            "Time reversal symmetry with custom ki-kj pairs",
        ));
    }
    let c = extract_cell_from_pyany(py, cell)?;
    let (k, _) = extract_kpts(kpts)?;
    let nk = k.len();
    let mut dk = Vec::with_capacity(nk * nk);
    for ki in &k {
        for kj in &k {
            dk.push([kj[0] - ki[0], kj[1] - ki[1], kj[2] - ki[2]]);
        }
    }
    if let Some(sel) = &kk_idx {
        if let Some(bad) = sel.iter().find(|&&p| p >= nk * nk) {
            return Err(PyValueError::new_err(format!(
                "kk_idx entry {bad} out of range for nkpts^2 = {}",
                nk * nk
            )));
        }
        dk = sel.iter().map(|&p| dk[p]).collect();
    }
    let scaled = c.get_scaled_kpts(&dk);
    let groups = kh::kk_adapted_iter(nk, &scaled, &dk, kk_idx.as_deref(), time_reversal_symmetry)
        .map_err(|()| {
        PyNotImplementedError::new_err("Time reversal symmetry with custom ki-kj pairs")
    })?;
    let items = groups
        .iter()
        .map(|g| {
            PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, g.kpt.to_vec()).into_any(),
                    idx_i32(py, &g.ki_idx).into_any(),
                    idx_i32(py, &g.kj_idx).into_any(),
                    pyo3::types::PyBool::new(py, g.self_conj)
                        .to_owned()
                        .into_any(),
                ],
            )
        })
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, items)?.try_iter().map(|it| it.into_any())
}

/// `get_kconserv3(cell, kpts, kijkab)` — an integer index is a pinned axis
/// (dropped from the shape), anything else an index array.
#[pyfunction]
fn get_kconserv3<'py>(
    py: Python<'py>,
    cell: &Bound<'py, PyAny>,
    kpts: &Bound<'py, PyAny>,
    kijkab: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let c = extract_cell_from_pyany(py, cell)?;
    let (k, _) = extract_kpts(kpts)?;
    let np = PyModule::import(py, "numpy")?;
    let integer = np.getattr("integer")?;
    let mut idx = Vec::with_capacity(5);
    for x in kijkab.try_iter()? {
        let x = x?;
        let is_int = x.is_instance_of::<pyo3::types::PyInt>() || x.is_instance(&integer)?;
        let e = if is_int {
            KIdx::One(x.extract()?)
        } else {
            KIdx::Many(np.call_method1("ravel", (x,))?.extract()?)
        };
        if e.indices().iter().any(|&i| i >= k.len()) {
            return Err(PyValueError::new_err(
                "get_kconserv3: k-point index out of range",
            ));
        }
        idx.push(e);
    }
    if idx.len() != 5 {
        return Err(PyValueError::new_err(
            "get_kconserv3: kijkab must have 5 entries (ki, kj, kk, ka, kb)",
        ));
    }
    let r = pyscf_pbc_gto::kpts_mesh::get_kconserv3(&c, &k, &idx);
    let data: Vec<i64> = r.data.iter().map(|&x| i64::from(x)).collect();
    let arr = ArrayD::from_shape_vec(IxDyn(&r.shape), data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py).into_any())
}
