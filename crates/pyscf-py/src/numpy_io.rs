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

use numpy::ndarray::{ArrayD, ArrayViewD, IxDyn, ShapeBuilder};
use numpy::{
    Complex64, IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyReadonlyArray2, PyReadonlyArrayDyn,
    PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use pyscf_algebra::CTensor;
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
    strides.len() == 2 && strides[0] == f64_size && strides[1] == (shape[0] as isize) * f64_size
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

// ─────────────────────────────────────────────────────────────────────────────
// Plan 20-07 — the complex, k-resolved boundary (periodic quantities).
//
// Inside the workspace every complex quantity is PLANAR `CTensor { re, im }`
// (D-PBC-02); interleaved `Complex64` exists only here, at the PyO3 boundary.
// A `CTensor` carries no shape and no layout, so every conversion names both:
// the logical `shape` and the `BufOrder` of the flat planes. That is
// load-bearing — `pyscf_pbc_scf::types` keeps `nao x nao` matrices ROW-MAJOR
// but MO coefficients COLUMN-MAJOR, and reading one as the other transposes
// silently instead of erroring (`gto.rs::intor_spinor`'s `IxDyn(..).f()`).
//
// `KMats = Vec<CTensor>` crosses as a Python LIST of arrays and
// `KDms = Vec<KMats>` as a list of such lists — never a stacked ndarray: under
// k-symmetry per-k blocks need not share a shape.
//
// Conversions are pure element moves (no arithmetic), so a round trip is
// bitwise exact on both planes. The pyo3-free core (`ctensor_to_array`,
// `array_to_ctensor`, `kmats_to_arrays`, …) is what
// `tests/complex_roundtrip.rs` drives; the `*_pyarray` / `*_pylist` / `to_*`
// wrappers below are thin shells over it.
// ─────────────────────────────────────────────────────────────────────────────

/// Memory layout of a flat `CTensor` plane relative to its logical shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufOrder {
    /// Row-major: last index varies fastest (`nao x nao` k-matrices, `KMats`).
    C,
    /// Column-major: first index varies fastest (MO coefficients, cintx output).
    F,
}

/// Per-k shapes of one `KMats` channel (`shapes[k]`).
pub type KMatsShapes = Vec<Vec<usize>>;
/// Per-(set, k) shapes of a `KDms` (`shapes[set][k]`).
pub type KDmsShapes = Vec<KMatsShapes>;

/// pyo3-free core: planar `CTensor` → owned `ArrayD<Complex64>` laid out in
/// `order`, so element `(i, j, …)` reads the matching flat index of `t`.
///
/// Errors (never truncates) when the two planes differ in length or when
/// `shape` does not multiply out to the plane length.
pub fn ctensor_to_array(
    t: &CTensor,
    shape: &[usize],
    order: BufOrder,
) -> Result<ArrayD<Complex64>, String> {
    if t.re.len() != t.im.len() {
        return Err(format!(
            "CTensor planes differ in length: re {} vs im {}",
            t.re.len(),
            t.im.len()
        ));
    }
    let n: usize = shape.iter().product();
    if n != t.re.len() {
        return Err(format!(
            "cannot shape {} complex elements as {:?} ({} elements)",
            t.re.len(),
            shape,
            n
        ));
    }
    let data: Vec<Complex64> =
        t.re.iter()
            .zip(t.im.iter())
            .map(|(&re, &im)| Complex64::new(re, im))
            .collect();
    let dim = IxDyn(shape);
    let arr = match order {
        BufOrder::C => ArrayD::from_shape_vec(dim, data),
        BufOrder::F => ArrayD::from_shape_vec(dim.f(), data),
    };
    arr.map_err(|e| format!("cannot shape as {shape:?} ({order:?}-order): {e}"))
}

/// pyo3-free core: any `ArrayViewD<Complex64>` — C- or F-contiguous, strided,
/// transposed or negative-stride — → planar `CTensor` flattened in `order` by
/// LOGICAL index. The view's shape is `view.shape()`; it is not stored.
pub fn array_to_ctensor(view: ArrayViewD<'_, Complex64>, order: BufOrder) -> CTensor {
    let n = view.len();
    let mut re = Vec::with_capacity(n);
    let mut im = Vec::with_capacity(n);
    let mut push = |z: &Complex64| {
        re.push(z.re);
        im.push(z.im);
    };
    match order {
        BufOrder::C => view.iter().for_each(&mut push),
        // reversing the axes makes the logical C walk the original F walk
        BufOrder::F => view.t().iter().for_each(&mut push),
    }
    CTensor { re, im }
}

/// pyo3-free core: one `KMats` channel → one array per k-point.
/// `shapes.len()` must equal `kmats.len()`; a mismatch errors rather than
/// zipping short.
pub fn kmats_to_arrays(
    kmats: &[CTensor],
    shapes: &[Vec<usize>],
    order: BufOrder,
) -> Result<Vec<ArrayD<Complex64>>, String> {
    if kmats.len() != shapes.len() {
        return Err(format!(
            "KMats has {} k blocks but {} shapes were given",
            kmats.len(),
            shapes.len()
        ));
    }
    kmats
        .iter()
        .zip(shapes)
        .enumerate()
        .map(|(k, (t, s))| ctensor_to_array(t, s, order).map_err(|e| format!("k={k}: {e}")))
        .collect()
}

/// pyo3-free core: one view per k-point → `(KMats, per-k shapes)`.
pub fn arrays_to_kmats(
    views: &[ArrayViewD<'_, Complex64>],
    order: BufOrder,
) -> (Vec<CTensor>, KMatsShapes) {
    views
        .iter()
        .map(|v| (array_to_ctensor(v.view(), order), v.shape().to_vec()))
        .unzip()
}

/// pyo3-free core: `KDms` (`[set][k]`) → nested per-set, per-k arrays.
pub fn kdms_to_arrays(
    kdms: &[Vec<CTensor>],
    shapes: &[Vec<Vec<usize>>],
    order: BufOrder,
) -> Result<Vec<Vec<ArrayD<Complex64>>>, String> {
    if kdms.len() != shapes.len() {
        return Err(format!(
            "KDms has {} spin sets but {} shape sets were given",
            kdms.len(),
            shapes.len()
        ));
    }
    kdms.iter()
        .zip(shapes)
        .enumerate()
        .map(|(s, (km, sh))| kmats_to_arrays(km, sh, order).map_err(|e| format!("set={s}: {e}")))
        .collect()
}

/// pyo3-free core: nested `[set][k]` views → `(KDms, per-(set,k) shapes)`.
pub fn arrays_to_kdms(
    views: &[Vec<ArrayViewD<'_, Complex64>>],
    order: BufOrder,
) -> (Vec<Vec<CTensor>>, KDmsShapes) {
    views
        .iter()
        .map(|kset| arrays_to_kmats(kset, order))
        .unzip()
}

fn value_err(e: String) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e)
}

/// `CTensor` (planes laid out in `order`, logical `shape`) → numpy
/// `complex128`. With `BufOrder::F` the result is F-contiguous and
/// generalises `gto.rs::intor_spinor`'s `IxDyn(&shape).f()`.
pub fn ctensor_to_pyarray<'py>(
    py: Python<'py>,
    t: &CTensor,
    shape: &[usize],
    order: BufOrder,
) -> PyResult<Bound<'py, PyArrayDyn<Complex64>>> {
    Ok(ctensor_to_array(t, shape, order)
        .map_err(value_err)?
        .into_pyarray(py))
}

/// numpy `complex128` (any contiguity / strides) → `(CTensor, shape)` with the
/// planes normalised to `order`. A non-`complex128` dtype is rejected by the
/// `PyReadonlyArrayDyn<Complex64>` extraction itself (TypeError), never cast.
pub fn to_ctensor<'py>(
    arr: PyReadonlyArrayDyn<'py, Complex64>,
    order: BufOrder,
) -> PyResult<(CTensor, Vec<usize>)> {
    let view = arr.as_array();
    let shape = view.shape().to_vec();
    Ok((array_to_ctensor(view, order), shape))
}

/// One `KMats` channel → a Python `list` of `len(kmats)` arrays.
pub fn kmats_to_pylist<'py>(
    py: Python<'py>,
    kmats: &[CTensor],
    shapes: &[Vec<usize>],
    order: BufOrder,
) -> PyResult<Bound<'py, PyList>> {
    let arrays = kmats_to_arrays(kmats, shapes, order).map_err(value_err)?;
    PyList::new(py, arrays.into_iter().map(|a| a.into_pyarray(py)))
}

/// Python `list`/`tuple` of `complex128` arrays → `(KMats, per-k shapes)`.
///
/// A stacked ndarray is REJECTED (TypeError): the list form is the contract,
/// because k-symmetry blocks need not share a shape. Callers holding an
/// upstream `(nkpts, nao, nao)` array pass `list(a)`.
pub fn to_kmats<'py>(
    obj: &Bound<'py, PyAny>,
    order: BufOrder,
) -> PyResult<(Vec<CTensor>, KMatsShapes)> {
    let items = sequence_items(obj, "KMats")?;
    let mut kmats = Vec::with_capacity(items.len());
    let mut shapes = Vec::with_capacity(items.len());
    for (k, item) in items.iter().enumerate() {
        let arr: PyReadonlyArrayDyn<'py, Complex64> = item.extract().map_err(|e| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "KMats[{k}] must be a complex128 ndarray: {e}"
            ))
        })?;
        let (t, s) = to_ctensor(arr, order)?;
        kmats.push(t);
        shapes.push(s);
    }
    Ok((kmats, shapes))
}

/// `KDms` (`[set][k]`) → a Python `list` of per-set `list`s of arrays.
pub fn kdms_to_pylist<'py>(
    py: Python<'py>,
    kdms: &[Vec<CTensor>],
    shapes: &[Vec<Vec<usize>>],
    order: BufOrder,
) -> PyResult<Bound<'py, PyList>> {
    if kdms.len() != shapes.len() {
        return Err(value_err(format!(
            "KDms has {} spin sets but {} shape sets were given",
            kdms.len(),
            shapes.len()
        )));
    }
    let sets = kdms
        .iter()
        .zip(shapes)
        .map(|(km, sh)| kmats_to_pylist(py, km, sh, order))
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, sets)
}

/// Python `list`/`tuple` of per-set `list`s of arrays → `(KDms, shapes)`.
/// Each set must itself be a list/tuple (see [`to_kmats`]).
pub fn to_kdms<'py>(
    obj: &Bound<'py, PyAny>,
    order: BufOrder,
) -> PyResult<(Vec<Vec<CTensor>>, KDmsShapes)> {
    let sets = sequence_items(obj, "KDms")?;
    let mut kdms = Vec::with_capacity(sets.len());
    let mut shapes = Vec::with_capacity(sets.len());
    for (s, set) in sets.iter().enumerate() {
        let (km, sh) = to_kmats(set, order)
            .map_err(|e| pyo3::exceptions::PyTypeError::new_err(format!("KDms[{s}]: {e}")))?;
        kdms.push(km);
        shapes.push(sh);
    }
    Ok((kdms, shapes))
}

/// Items of a Python `list` or `tuple`; anything else (notably a stacked
/// ndarray) is a TypeError naming `what`.
fn sequence_items<'py>(obj: &Bound<'py, PyAny>, what: &str) -> PyResult<Vec<Bound<'py, PyAny>>> {
    if let Ok(list) = obj.cast::<PyList>() {
        Ok(list.iter().collect())
    } else if let Ok(tuple) = obj.cast::<PyTuple>() {
        Ok(tuple.iter().collect())
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "{what} must be a list (or tuple) of per-k complex128 arrays, got {}; \
             a stacked ndarray is not accepted because k-symmetry blocks need not share a shape",
            obj.get_type()
                .name()
                .map(|n| n.to_string())
                .unwrap_or_else(|_| "<unknown>".into())
        )))
    }
}

// Private self-test hooks (plan 20-07): the default `abi3-py310` feature turns
// on `pyo3/extension-module`, so no Rust test binary can host an interpreter;
// `python/pyscf/tests/test_complex_boundary.py` drives the numpy wrappers above
// through these. Underscore-prefixed, registered on the `_native` root only.

/// `_native._roundtrip_ctensor(a, order)` — `to_ctensor` then
/// `ctensor_to_pyarray` in the same `order` (`"C"` or `"F"`).
#[pyfunction]
fn _roundtrip_ctensor<'py>(
    py: Python<'py>,
    a: PyReadonlyArrayDyn<'py, Complex64>,
    order: &str,
) -> PyResult<Bound<'py, PyArrayDyn<Complex64>>> {
    let order = parse_order(order)?;
    let (t, shape) = to_ctensor(a, order)?;
    ctensor_to_pyarray(py, &t, &shape, order)
}

/// `_native._roundtrip_kmats(list, order)` — `to_kmats` then `kmats_to_pylist`.
#[pyfunction]
fn _roundtrip_kmats<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    order: &str,
) -> PyResult<Bound<'py, PyList>> {
    let order = parse_order(order)?;
    let (km, shapes) = to_kmats(obj, order)?;
    kmats_to_pylist(py, &km, &shapes, order)
}

/// `_native._roundtrip_kdms(list_of_lists, order)` — `to_kdms` then `kdms_to_pylist`.
#[pyfunction]
fn _roundtrip_kdms<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    order: &str,
) -> PyResult<Bound<'py, PyList>> {
    let order = parse_order(order)?;
    let (kd, shapes) = to_kdms(obj, order)?;
    kdms_to_pylist(py, &kd, &shapes, order)
}

/// `_native._planes(a, order)` — the raw `(re, im)` planes `to_ctensor`
/// produces, as float64 1-D arrays, so Python can check the F/C flattening.
#[pyfunction]
fn _planes<'py>(
    py: Python<'py>,
    a: PyReadonlyArrayDyn<'py, Complex64>,
    order: &str,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let (t, _) = to_ctensor(a, parse_order(order)?)?;
    Ok((PyArray1::from_vec(py, t.re), PyArray1::from_vec(py, t.im)))
}

fn parse_order(order: &str) -> PyResult<BufOrder> {
    match order {
        "C" => Ok(BufOrder::C),
        "F" => Ok(BufOrder::F),
        other => Err(value_err(format!(
            "order must be 'C' or 'F', got {other:?}"
        ))),
    }
}

/// Register the private boundary self-test hooks on the `_native` root.
pub fn register_selftest(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_roundtrip_ctensor, m)?)?;
    m.add_function(wrap_pyfunction!(_roundtrip_kmats, m)?)?;
    m.add_function(wrap_pyfunction!(_roundtrip_kdms, m)?)?;
    m.add_function(wrap_pyfunction!(_planes, m)?)?;
    Ok(())
}
