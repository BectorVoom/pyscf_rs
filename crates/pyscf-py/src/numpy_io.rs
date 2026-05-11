//! NumPy boundary discipline (D-04, BIND-04, Pitfall 5).
//!
//! Every `PyReadonlyArray2` input runs `is_c_contiguous()` and falls back to
//! `to_owned()` if not. The BIND-04 stride-fuzz test in plan 03-10 calls each
//! entry point with `a`, `a.T`, `a[::2]`, `a[:,1:5]` and asserts identical
//! answers.
//!
//! Source: `~/Documents/workspace/xcfun_rs/crates/xcfun-py/src/numpy_io.rs:9-76`.
//!
//! Density convention: C-contiguous (row-major).
//! MOCoefficients convention: F-contiguous (column-major, LAPACK) — Pitfall 8.

use numpy::{PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyscf_core::{Density, MOCoefficients};

/// Convert a NumPy 2D f64 array into a pyscf-core `Density` (C-order).
///
/// BIND-04: validates square-2D shape and standard-layout. Falls back to
/// `to_owned()` when the input is not C-contiguous (transposes, strided
/// slices, etc.) so downstream Rust code sees a flat row-major buffer.
pub fn to_density<'py>(arr: PyReadonlyArray2<'py, f64>) -> PyResult<Density> {
    let shape = arr.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "density must be square 2D, got shape {:?}",
            shape
        )));
    }
    let n = shape[0];
    let data: Vec<f64> = if arr.is_c_contiguous() {
        arr.as_slice()?.to_vec()
    } else {
        // BIND-04 fallback: re-materialise as default-order (C-contiguous)
        // owned ndarray, then drop into a flat Vec.
        let view = arr.as_array();
        let mut buf = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                buf.push(view[[i, j]]);
            }
        }
        buf
    };
    Ok(Density { nao: n, data })
}

/// Convert a NumPy 2D f64 array into `MOCoefficients` (F-order, LAPACK).
///
/// Pitfall 8: MO coefficient matrices are F-contiguous in upstream PySCF.
/// If the input is F-contiguous, we slice directly; otherwise we transpose-
/// copy via the iterator.
pub fn to_mo_coeff<'py>(arr: PyReadonlyArray2<'py, f64>) -> PyResult<MOCoefficients> {
    let shape = arr.shape();
    if shape.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "mo_coeff must be 2D, got shape {:?}",
            shape
        )));
    }
    let (nao, nmo) = (shape[0], shape[1]);
    let data: Vec<f64> = if is_f_contiguous(&arr) {
        arr.as_slice()?.to_vec()
    } else {
        let view = arr.as_array();
        let mut buf = Vec::with_capacity(nao * nmo);
        for j in 0..nmo {
            for i in 0..nao {
                buf.push(view[[i, j]]);
            }
        }
        buf
    };
    Ok(MOCoefficients {
        nao,
        nmo,
        data,
        energies: vec![],
        occupations: vec![],
    })
}

fn is_f_contiguous(arr: &PyReadonlyArray2<'_, f64>) -> bool {
    let strides = arr.strides();
    let shape = arr.shape();
    let f64_size = std::mem::size_of::<f64>() as isize;
    strides.len() == 2
        && strides[0] == f64_size
        && strides[1] == (shape[0] as isize) * f64_size
}

/// Convert a `Density` into a C-contiguous NumPy 2D array.
pub fn density_to_pyarray<'py>(
    py: Python<'py>,
    d: &Density,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let arr = ndarray::Array2::from_shape_vec((d.nao, d.nao), d.data.clone())
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

/// Convert `MOCoefficients` into a C-contiguous NumPy 2D array.
///
/// Internal `mc.data` is F-order flat (column-major). We rebuild it in
/// C-order layout because Python's downstream tools (h5py, plotting,
/// linear-algebra) work with C-contiguous NumPy arrays by default.
/// The F-order convention is preserved on the chkfile boundary (plan 03-06
/// `write_dataset_f_order`) and on the Rust internal data buffer.
pub fn mo_coeff_to_pyarray<'py>(
    py: Python<'py>,
    mc: &MOCoefficients,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (nao, nmo) = (mc.nao, mc.nmo);
    let mut c_data = Vec::with_capacity(nao * nmo);
    for i in 0..nao {
        for j in 0..nmo {
            // mc.data is F-order: index (i, j) → i + j*nao.
            c_data.push(mc.data[i + j * nao]);
        }
    }
    let arr = ndarray::Array2::from_shape_vec((nao, nmo), c_data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

/// Convert a slice of f64 into a 1D NumPy array (used for mo_energy / mo_occ).
pub fn slice_to_pyarray1<'py>(py: Python<'py>, v: &[f64]) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_slice(py, v)
}
