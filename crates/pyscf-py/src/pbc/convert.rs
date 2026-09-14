//! Small argument/return converters shared by the periodic bindings
//! (`pbc::gto`, `pbc::df`, and the driver plans after them).
//!
//! Everything complex goes through 20-07's `numpy_io` helpers with an explicit
//! [`BufOrder`]; this file only adds the real-valued k-point / lattice shapes and
//! the one layout the periodic integral outputs share (`comp * a * b`, F-order per
//! component).

use numpy::ndarray::{Array2, ArrayD, IxDyn, ShapeBuilder};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyscf_algebra::CTensor;

use crate::numpy_io::{BufOrder, ctensor_to_array};

/// A k-point argument: `(kpts, single)` where `single` is `true` when the caller
/// passed ONE 3-vector (upstream then returns a single matrix, not a stack).
pub fn extract_kpts(obj: &Bound<'_, PyAny>) -> PyResult<(Vec<[f64; 3]>, bool)> {
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, f64>>() {
        let v = arr.as_array();
        if v.ncols() != 3 {
            return Err(PyValueError::new_err(format!(
                "kpts must have shape (nkpts, 3), got {:?}",
                v.shape()
            )));
        }
        return Ok((
            v.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect(),
            false,
        ));
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, f64>>() {
        let v = arr.as_array();
        if v.len() != 3 {
            return Err(PyValueError::new_err(format!(
                "a single kpt must have shape (3,), got {:?}",
                v.shape()
            )));
        }
        return Ok((vec![[v[0], v[1], v[2]]], true));
    }
    if let Ok(v) = obj.extract::<Vec<[f64; 3]>>() {
        return Ok((v, false));
    }
    if let Ok(v) = obj.extract::<[f64; 3]>() {
        return Ok((vec![v], true));
    }
    Err(PyTypeError::new_err(
        "kpts must be a (nkpts, 3) or (3,) float array / nested sequence",
    ))
}

/// `None` → `None`; otherwise [`extract_kpts`].
pub fn extract_kpts_opt(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<(Vec<[f64; 3]>, bool)>> {
    match obj {
        None => Ok(None),
        Some(o) if o.is_none() => Ok(None),
        Some(o) => extract_kpts(o).map(Some),
    }
}

/// `(n, 3)` float64 array, C-order.
pub fn kpts_to_pyarray<'py>(
    py: Python<'py>,
    kpts: &[[f64; 3]],
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let flat: Vec<f64> = kpts.iter().flatten().copied().collect();
    let arr = Array2::from_shape_vec((kpts.len(), 3), flat)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

/// `3 x 3` float64 array, one lattice/reciprocal vector per ROW.
pub fn mat3_to_pyarray<'py>(
    py: Python<'py>,
    m: &[[f64; 3]; 3],
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    kpts_to_pyarray(py, &m[..])
}

/// A length-3 integer triple (`mesh`, `nks`, `ncopy`). Accepts any sequence or
/// integer array.
pub fn extract_usize3(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<[usize; 3]> {
    if let Ok(v) = obj.extract::<[usize; 3]>() {
        return Ok(v);
    }
    let v: Vec<f64> = obj
        .extract()
        .map_err(|_| PyTypeError::new_err(format!("{what} must be a length-3 integer sequence")))?;
    if v.len() != 3 || v.iter().any(|x| *x < 0.0 || x.fract() != 0.0) {
        return Err(PyValueError::new_err(format!(
            "{what} must be three non-negative integers, got {v:?}"
        )));
    }
    Ok([v[0] as usize, v[1] as usize, v[2] as usize])
}

/// A `3 x 3` real matrix from an ndarray or nested sequence.
pub fn extract_mat3(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<[[f64; 3]; 3]> {
    if let Ok(m) = obj.extract::<[[f64; 3]; 3]>() {
        return Ok(m);
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, f64>>() {
        let v = arr.as_array();
        if v.shape() == [3, 3] {
            return Ok(std::array::from_fn(|i| std::array::from_fn(|j| v[[i, j]])));
        }
    }
    if let Ok(v) = obj.extract::<Vec<f64>>() {
        if v.len() == 9 {
            return Ok(std::array::from_fn(|i| {
                std::array::from_fn(|j| v[i * 3 + j])
            }));
        }
    }
    Err(PyTypeError::new_err(format!(
        "{what} must be a 3x3 matrix (ndarray or nested sequence)"
    )))
}

/// One periodic integral / AO block laid out `comp * a * b`, F-order per
/// component (`c*a*b + i + j*a` — `PbcIntorOutput`, `EvalAoKptsOutput`), as a
/// numpy array shaped `(comp, a, b)` — or `(a, b)` when `comp == 1`.
///
/// That flat index IS the F-order index of shape `(a, b, comp)`, so the planes
/// are read with `BufOrder::F` over `[a, b, comp]` and the axes permuted to
/// `(comp, a, b)` (a stride change, no copy, no arithmetic). With `real` the
/// imaginary plane is dropped — upstream's `.real` on a gamma-point block.
pub fn comp_block_to_py(
    py: Python<'_>,
    t: &CTensor,
    a: usize,
    b: usize,
    comp: usize,
    real: bool,
) -> PyResult<Py<PyAny>> {
    let shape = [a, b, comp];
    if real {
        let n = a * b * comp;
        if t.re.len() != n {
            return Err(PyValueError::new_err(format!(
                "block has {} elements, expected {a}*{b}*{comp}",
                t.re.len()
            )));
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&shape).f(), t.re.clone())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let arr = if comp == 1 {
            arr.index_axis_move(numpy::ndarray::Axis(2), 0)
        } else {
            arr.permuted_axes(IxDyn(&[2, 0, 1]))
        };
        return Ok(arr.into_pyarray(py).into_any().unbind());
    }
    let arr = ctensor_to_array(t, &shape, BufOrder::F).map_err(PyValueError::new_err)?;
    let arr = if comp == 1 {
        arr.index_axis_move(numpy::ndarray::Axis(2), 0)
    } else {
        arr.permuted_axes(IxDyn(&[2, 0, 1]))
    };
    Ok(arr.into_pyarray(py).into_any().unbind())
}
