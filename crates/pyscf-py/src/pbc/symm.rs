//! `pyscf._native.pbc.symm` — `KPoints`, `Symmetry`, `SpaceGroup` (plan 20-14).
//!
//! Upstream's `KPoints` (`pyscf/pbc/lib/kpts.py:847`) is what `pbc.scf` /
//! `pbc.dft` dispatch on with `isinstance(kpts, libkpts.KPoints)`, so it is a
//! real `#[pyclass(subclass)]` here, and `pyscf._native.pbc.lib.kpts.KPoints`
//! is the SAME type object (`pbc/lib.rs`).
//!
//! Four facts shape this file:
//!
//! 1. **The port deliberately differs from upstream k-symmetry.** Phase 17 found
//!    defects in PySCF 2.12.1's own code (`17-VERIFICATION.md` §6): D-17-07-01
//!    (`little_cogroup_ops` indexes `k2opk`'s `2*nop` column space, upstream
//!    raises `IndexError`), D-17-09-01, D-17-09-02 (non-canonical orbitals on a
//!    k-mesh of lower symmetry than the lattice) and the `-1` index of
//!    `MORotationMatrix.build`. Every method below calls the Rust port as is;
//!    its refusals raise and its detector (`ops_outside_kmesh_subgroup`) is bound.
//! 2. **Detection is native.** There is no spglib: `SpaceGroup.backend` is
//!    `'pyscf'` and setting anything else raises `NotImplementedError`.
//! 3. **A Rust `KPoints` stores no `Cell`** (D-PBC-25). The binding keeps the
//!    Python cell object (`kpts.cell is cell`) plus the Rust cell it was built
//!    against, so `get_kconserv()` and the transforms need no argument.
//! 4. **`Cell.build(space_group_symmetry=True)` builds the lattice symmetry**
//!    (`cell.py:1770-1772`). [`ensure_lattice_symmetry`] is that step; the native
//!    `Cell.build` calls it (`pbc/gto.rs`), and every cell entering this file
//!    passes through it, so an upstream-shaped cell gets the same treatment.

use numpy::ndarray::{Array2, Array3};
use numpy::{Complex64, IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyscf_pbc_gto::Cell;
use pyscf_pbc_lib::kpts_helper::KPT_DIFF_TOL;
use pyscf_pbc_symm::PbcSymmError;
use pyscf_pbc_symm::geom;
use pyscf_pbc_symm::kpts::{KPoints, NO_MAP};
use pyscf_pbc_symm::space_group::{SPGElement, SYMPREC, SpaceGroup};
use pyscf_pbc_symm::symmetry::{DmatSet, Symmetry, build_lattice_symmetry};

use crate::bridge::extract_cell_from_pyany;
use crate::errors::{PyscfRsRuntimeError, pyscf_to_py};
use crate::pbc::convert::{extract_kpts, extract_kpts_opt, extract_usize3, kpts_to_pyarray};

/// Dotted module name every class in this file reports.
pub const MODULE: &str = "pyscf._native.pbc.symm";

/// Register `KPoints`, `Symmetry`, `SpaceGroup`, `SPGElement`, `make_kpts` and
/// `get_crystal_class` on the child module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKPoints>()?;
    m.add_class::<PySymmetry>()?;
    m.add_class::<PySpaceGroup>()?;
    m.add_class::<PySpgElement>()?;
    m.add_function(wrap_pyfunction!(make_kpts, m)?)?;
    m.add_function(wrap_pyfunction!(get_crystal_class, m)?)?;
    m.add("SYMPREC", SYMPREC)?;
    Ok(())
}

/// `PbcSymmError` → Python. A wrapped `PyscfRsError` keeps its kind. The `"s4"`
/// refusal of `make_k4_ibz` is a port refusal (`NotYetImplemented`); any other
/// unsupported symmetry string is upstream's own `NotImplementedError`
/// (`kpts.py:301`). Everything else reports kind `"PbcSymm"`.
pub fn pbc_symm_to_py(err: PbcSymmError) -> PyErr {
    let msg = err.to_string();
    match err {
        PbcSymmError::Core(e) => pyscf_to_py(e),
        PbcSymmError::UnsupportedK4Symmetry(s) if s == "s4" => {
            PyscfRsRuntimeError::new_err((msg, "NotYetImplemented", Vec::<String>::new()))
        }
        PbcSymmError::UnsupportedK4Symmetry(_) => PyNotImplementedError::new_err(msg),
        _ => PyscfRsRuntimeError::new_err((msg, "PbcSymm", Vec::<String>::new())),
    }
}

/// `cell.py:1770-1772` — a built cell with `space_group_symmetry` carries its
/// lattice symmetry, with `check_mesh_symmetry = not cell._mesh_from_build`
/// (an auto mesh is ENLARGED to the lattice symmetry; a user mesh drops ops).
/// A no-op when the symmetry is already there or not requested.
///
/// # Errors
/// As [`build_lattice_symmetry`].
pub fn ensure_lattice_symmetry(cell: &mut Cell) -> Result<(), PbcSymmError> {
    if cell._built && cell.natm > 0 && cell.space_group_symmetry && cell.lattice_symmetry.is_none()
    {
        let check = !cell._mesh_from_build;
        build_lattice_symmetry(cell, check)?;
    }
    Ok(())
}

/// Any cell `extract_cell_from_pyany` accepts, with [`ensure_lattice_symmetry`].
pub fn symmetry_cell(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Cell> {
    let mut cell = extract_cell_from_pyany(py, obj)?;
    ensure_lattice_symmetry(&mut cell).map_err(pbc_symm_to_py)?;
    Ok(cell)
}

// ─────────────────────────────────────────────────────────────────────────────
// small converters
// ─────────────────────────────────────────────────────────────────────────────

fn idx_array<'py>(py: Python<'py>, v: &[usize]) -> Bound<'py, PyArray1<i64>> {
    PyArray1::from_vec(py, v.iter().map(|&x| x as i64).collect())
}

fn idx_list<'py>(py: Python<'py>, v: &[Vec<usize>]) -> PyResult<Bound<'py, PyList>> {
    PyList::new(py, v.iter().map(|x| idx_array(py, x)))
}

fn f64_rows<'py>(py: Python<'py>, m: &[Vec<f64>]) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let nrow = m.len();
    let ncol = m.first().map_or(0, Vec::len);
    let flat: Vec<f64> = m.iter().flatten().copied().collect();
    let arr = Array2::from_shape_vec((nrow, ncol), flat)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

fn dmats_to_py<'py>(py: Python<'py>, dmats: &[DmatSet]) -> PyResult<Bound<'py, PyList>> {
    let mut per_op = Vec::with_capacity(dmats.len());
    for set in dmats {
        let mut per_l = Vec::with_capacity(set.len());
        for m in set {
            per_l.push(f64_rows(py, m)?);
        }
        per_op.push(PyList::new(py, per_l)?);
    }
    PyList::new(py, per_op)
}

fn ops_to_py<'py>(py: Python<'py>, ops: &[SPGElement]) -> PyResult<Bound<'py, PyList>> {
    let items = ops
        .iter()
        .map(|op| Py::new(py, PySpgElement { op: *op }))
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, items)
}

fn cell_obj_or_none(py: Python<'_>, c: &Option<Py<PyAny>>) -> Py<PyAny> {
    match c {
        Some(o) => o.clone_ref(py),
        None => py.None(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SPGElement
// ─────────────────────────────────────────────────────────────────────────────

/// `space_group.py:84` — one space-group operation `(rot | trans)` in the
/// direct-lattice basis. Read-only.
#[pyclass(
    frozen,
    name = "SPGElement",
    module = "pyscf._native.pbc.symm",
    skip_from_py_object
)]
pub struct PySpgElement {
    op: SPGElement,
}

#[pymethods]
impl PySpgElement {
    /// `rot`, int32 when every entry is integral (always, for the direct-lattice
    /// ops a `SpaceGroup` holds), float64 otherwise.
    #[getter]
    fn rot(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let r = self.op.rot;
        if r.iter().flatten().all(|x| x.fract() == 0.0) {
            let flat: Vec<i32> = r.iter().flatten().map(|&x| x as i32).collect();
            let arr = Array2::from_shape_vec((3, 3), flat)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            return Ok(arr.into_pyarray(py).into_any().unbind());
        }
        let rows: Vec<Vec<f64>> = r.iter().map(|row| row.to_vec()).collect();
        Ok(f64_rows(py, &rows)?.into_any().unbind())
    }

    #[getter]
    fn trans<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.op.trans.to_vec())
    }

    #[getter]
    fn is_eye(&self) -> bool {
        self.op.is_eye()
    }

    #[getter]
    fn is_inversion(&self) -> bool {
        self.op.is_inversion()
    }

    #[getter]
    fn rot_is_eye(&self) -> bool {
        self.op.rot_is_eye()
    }

    #[getter]
    fn rot_is_inversion(&self) -> bool {
        self.op.rot_is_inversion()
    }

    #[getter]
    fn trans_is_zero(&self) -> bool {
        self.op.trans_is_zero()
    }

    /// `op.a2b(cell)` — the same operation in the reciprocal basis.
    fn a2b(&self, py: Python<'_>, cell: &Bound<'_, PyAny>) -> PyResult<Self> {
        let c = extract_cell_from_pyany(py, cell)?;
        Ok(Self {
            op: self.op.a2b(&c).map_err(pbc_symm_to_py)?,
        })
    }

    /// `op.a2r(cell)` — the same operation in Cartesian coordinates.
    fn a2r(&self, py: Python<'_>, cell: &Bound<'_, PyAny>) -> PyResult<Self> {
        let c = extract_cell_from_pyany(py, cell)?;
        Ok(Self {
            op: self.op.a2r(&c).map_err(pbc_symm_to_py)?,
        })
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .cast::<PySpgElement>()
            .map(|o| o.get().op == self.op)
            .unwrap_or(false)
    }

    fn __hash__(&self) -> i64 {
        self.op.hash_key()
    }

    fn __repr__(&self) -> String {
        format!(
            "SPGElement(rot={:?}, trans={:?})",
            self.op.rot, self.op.trans
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SpaceGroup
// ─────────────────────────────────────────────────────────────────────────────

/// `space_group.py:250` — `SpaceGroup(cell, symprec=SYMPREC)`, native backend.
#[pyclass(
    subclass,
    name = "SpaceGroup",
    module = "pyscf._native.pbc.symm",
    skip_from_py_object
)]
pub struct PySpaceGroup {
    cell: Option<Py<PyAny>>,
    symprec: f64,
    inner: Option<SpaceGroup>,
}

#[pymethods]
impl PySpaceGroup {
    #[new]
    #[pyo3(signature = (cell = None, symprec = SYMPREC))]
    fn new(cell: Option<Py<PyAny>>, symprec: f64) -> Self {
        Self {
            cell,
            symprec,
            inner: None,
        }
    }

    /// `build(dump_info=True)` — runs the native space-group search; returns self.
    #[pyo3(signature = (dump_info = true))]
    fn build<'py>(slf: &Bound<'py, Self>, dump_info: bool) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        let (cell_obj, symprec) = {
            let b = slf.borrow();
            (cell_obj_or_none(py, &b.cell), b.symprec)
        };
        if cell_obj.is_none(py) {
            return Err(PyValueError::new_err("SpaceGroup.build: cell is None"));
        }
        let cell = extract_cell_from_pyany(py, cell_obj.bind(py))?;
        let sg = SpaceGroup::build(&cell, symprec).map_err(pbc_symm_to_py)?;
        if dump_info {
            sg.dump_info(None);
        }
        slf.borrow_mut().inner = Some(sg);
        Ok(slf.clone())
    }

    #[getter]
    fn cell(&self, py: Python<'_>) -> Py<PyAny> {
        cell_obj_or_none(py, &self.cell)
    }

    #[setter]
    fn set_cell(&mut self, v: Option<Py<PyAny>>) {
        self.cell = v;
    }

    #[getter]
    fn symprec(&self) -> f64 {
        self.symprec
    }

    #[setter]
    fn set_symprec(&mut self, v: f64) {
        self.symprec = v;
    }

    /// Always `'pyscf'`: detection is native, spglib is deliberately absent.
    #[getter]
    fn backend(&self) -> &'static str {
        "pyscf"
    }

    #[setter]
    fn set_backend(&mut self, v: &str) -> PyResult<()> {
        if v == "pyscf" {
            return Ok(());
        }
        Err(PyNotImplementedError::new_err(format!(
            "SpaceGroup.backend = {v:?}: only the native 'pyscf' backend is ported \
             (upstream's pyscf_spglib.py has no counterpart, deliberately)"
        )))
    }

    #[getter]
    fn ops<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        ops_to_py(py, self.inner.as_ref().map_or(&[][..], |s| &s.ops))
    }

    #[getter]
    fn nop(&self) -> usize {
        self.inner.as_ref().map_or(0, |s| s.nop)
    }

    /// `{'point_group_symbol': ...}` once built, `{}` before (upstream's native
    /// backend never fills the international symbol/number either).
    #[getter]
    fn groupname<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        if let Some(s) = &self.inner {
            d.set_item("point_group_symbol", s.point_group_symbol)?;
        }
        Ok(d)
    }

    #[pyo3(signature = (ops = None))]
    fn dump_info(&self, ops: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        if ops.is_some_and(|o| !o.is_none()) {
            return Err(PyNotImplementedError::new_err(
                "SpaceGroup.dump_info(ops=...) is not bound; dump_info() logs self.ops",
            ));
        }
        if let Some(s) = &self.inner {
            s.dump_info(None);
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Symmetry
// ─────────────────────────────────────────────────────────────────────────────

/// `symmetry.py:132` — `Symmetry(cell)`.
///
/// Not re-exported by the `pyscf.pbc.symm` overlay: upstream's own
/// `Cell.build_lattice_symmetry` (`cell.py:1567-1579`) constructs
/// `pyscf.pbc.symm.Symmetry` on an UPSTREAM cell and then `del`s attributes off
/// it, so the overlay keeps serving upstream's class under that name.
#[pyclass(
    subclass,
    name = "Symmetry",
    module = "pyscf._native.pbc.symm",
    skip_from_py_object
)]
pub struct PySymmetry {
    cell: Option<Py<PyAny>>,
    inner: Symmetry,
}

#[pymethods]
impl PySymmetry {
    #[new]
    #[pyo3(signature = (cell = None))]
    fn new(cell: Option<Py<PyAny>>) -> Self {
        Self {
            cell,
            inner: Symmetry::default(),
        }
    }

    /// `build(space_group_symmetry=True, symmorphic=True, check_mesh_symmetry=True)`.
    /// With `cell is None` only `_built` is set (`symmetry.py:168-170`).
    #[pyo3(signature = (space_group_symmetry = true, symmorphic = true, check_mesh_symmetry = true))]
    fn build<'py>(
        slf: &Bound<'py, Self>,
        space_group_symmetry: bool,
        symmorphic: bool,
        check_mesh_symmetry: bool,
    ) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        let cell_obj = cell_obj_or_none(py, &slf.borrow().cell);
        if cell_obj.is_none(py) {
            slf.borrow_mut().inner.built = true;
            return Ok(slf.clone());
        }
        let cell = extract_cell_from_pyany(py, cell_obj.bind(py))?;
        let s = Symmetry::build(&cell, space_group_symmetry, symmorphic, check_mesh_symmetry)
            .map_err(pbc_symm_to_py)?;
        slf.borrow_mut().inner = s;
        Ok(slf.clone())
    }

    /// `check_mesh_symmetry(cell=None, ops=None, mesh=None, tol=SYMPREC,
    /// return_mesh=False)` → `rm_list`, or `(rm_list, mesh)`. `ops` must be
    /// `None` (this object's own ops).
    #[pyo3(signature = (cell = None, ops = None, mesh = None, tol = SYMPREC, return_mesh = false))]
    fn check_mesh_symmetry(
        &self,
        py: Python<'_>,
        cell: Option<&Bound<'_, PyAny>>,
        ops: Option<&Bound<'_, PyAny>>,
        mesh: Option<&Bound<'_, PyAny>>,
        tol: f64,
        return_mesh: bool,
    ) -> PyResult<Py<PyAny>> {
        if ops.is_some_and(|o| !o.is_none()) {
            return Err(PyNotImplementedError::new_err(
                "Symmetry.check_mesh_symmetry(ops=...) is not bound; it uses self.ops",
            ));
        }
        let cell = match cell {
            Some(c) if !c.is_none() => extract_cell_from_pyany(py, c)?,
            _ => {
                let own = cell_obj_or_none(py, &self.cell);
                if own.is_none(py) {
                    return Err(PyValueError::new_err(
                        "Symmetry.check_mesh_symmetry: no cell",
                    ));
                }
                extract_cell_from_pyany(py, own.bind(py))?
            }
        };
        let mesh = match mesh {
            Some(m) if !m.is_none() => Some(extract_usize3(m, "mesh")?),
            _ => None,
        };
        let (rm, mesh1) = self
            .inner
            .check_mesh_symmetry(&cell, mesh, tol, return_mesh);
        let rm = PyList::new(py, rm)?;
        match mesh1 {
            Some(m) if return_mesh => Ok(PyTuple::new(
                py,
                [rm.into_any(), PyList::new(py, m)?.into_any()],
            )?
            .into_any()
            .unbind()),
            _ => Ok(rm.into_any().unbind()),
        }
    }

    fn dump_info(&self) {
        self.inner.dump_info();
    }

    /// `reset(cell=None)` — drops the build (`symmetry.py:220-223`).
    #[pyo3(signature = (cell = None))]
    fn reset<'py>(slf: &Bound<'py, Self>, cell: Option<&Bound<'py, PyAny>>) -> Bound<'py, Self> {
        let _ = cell;
        slf.borrow_mut().inner.reset();
        slf.clone()
    }

    #[getter]
    fn cell(&self, py: Python<'_>) -> Py<PyAny> {
        cell_obj_or_none(py, &self.cell)
    }

    #[setter]
    fn set_cell(&mut self, v: Option<Py<PyAny>>) {
        self.cell = v;
    }

    #[getter]
    fn spacegroup(&self, py: Python<'_>) -> PyResult<Option<Py<PySpaceGroup>>> {
        match &self.inner.spacegroup {
            None => Ok(None),
            Some(sg) => Py::new(
                py,
                PySpaceGroup {
                    cell: self.cell.as_ref().map(|c| c.clone_ref(py)),
                    symprec: sg.symprec,
                    inner: Some(sg.clone()),
                },
            )
            .map(Some),
        }
    }

    #[getter]
    fn symmorphic(&self) -> bool {
        self.inner.symmorphic
    }

    #[getter]
    fn ops<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        ops_to_py(py, &self.inner.ops)
    }

    #[getter]
    fn nop(&self) -> usize {
        self.inner.nop
    }

    #[getter]
    fn has_inversion(&self) -> bool {
        self.inner.has_inversion
    }

    /// `Dmats[iop][l]` — `(2l+1, 2l+1)` (spherical) Wigner-D blocks; `None`
    /// before `build`.
    #[getter(Dmats)]
    fn dmats<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyList>>> {
        if !self.inner.built {
            return Ok(None);
        }
        dmats_to_py(py, &self.inner.dmats).map(Some)
    }

    #[getter]
    fn l_max(&self) -> Option<usize> {
        self.inner.built.then_some(self.inner.l_max)
    }

    #[getter]
    fn _built(&self) -> bool {
        self.inner.built
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KPoints
// ─────────────────────────────────────────────────────────────────────────────

/// `kpts.py:847` — `KPoints(cell=None, kpts=np.zeros((1,3)))`.
///
/// Arrays are returned fresh on every attribute read (the Rust object owns the
/// data); index arrays are int64, `get_kconserv()` is int32 as in
/// `kpts_helper`. Transformed MO coefficients / density / Fock blocks are
/// complex128 row-major `(nao, nmo)`, read by LOGICAL index from any input view.
#[pyclass(
    subclass,
    name = "KPoints",
    module = "pyscf._native.pbc.symm",
    skip_from_py_object
)]
pub struct PyKPoints {
    cell_obj: Option<Py<PyAny>>,
    cell: Option<Cell>,
    inner: KPoints,
}

impl PyKPoints {
    /// The Rust `KPoints` (for the driver plans).
    pub fn kpoints(&self) -> &KPoints {
        &self.inner
    }

    /// The Rust cell this object was last built against.
    ///
    /// # Errors
    /// `ValueError` before `build()`.
    pub fn rust_cell(&self) -> PyResult<&Cell> {
        self.cell.as_ref().ok_or_else(|| {
            PyValueError::new_err("KPoints is not built against a cell: call build()")
        })
    }

    /// Wrap a built Rust `KPoints` together with its cell.
    pub fn from_built(cell_obj: Py<PyAny>, cell: Cell, inner: KPoints) -> Self {
        Self {
            cell_obj: Some(cell_obj),
            cell: Some(cell),
            inner,
        }
    }

    fn built_cell(&self) -> PyResult<&Cell> {
        self.rust_cell()
    }
}

/// `x[0][0]` has `ndim` dimensions → the input carries two spin channels
/// (upstream's `isinstance(x[0][0], np.ndarray) and x[0][0].ndim == ...`).
fn is_spin_pair(py: Python<'_>, obj: &Bound<'_, PyAny>, ndim: usize) -> PyResult<bool> {
    let np = PyModule::import(py, "numpy")?;
    let first = obj.get_item(0)?;
    let inner = match first.get_item(0) {
        Ok(v) => v,
        Err(_) => return Ok(false),
    };
    let n: usize = np.call_method1("ndim", (inner,))?.extract()?;
    Ok(n == ndim)
}

fn seq_items<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    obj.try_iter()?.collect()
}

fn read_f64_vecs(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
    let np = PyModule::import(py, "numpy")?;
    seq_items(obj)?
        .into_iter()
        .map(|item| {
            let a = np.call_method1("asarray", (item, "float64"))?;
            let a: PyReadonlyArray1<'_, f64> = a.extract()?;
            Ok(a.as_array().iter().copied().collect())
        })
        .collect()
}

/// Per-k complex matrices, read by logical (row-major) index. Returns the
/// blocks and the common `(rows, cols)`.
fn read_cmats(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
) -> PyResult<(Vec<Vec<Complex64>>, usize, usize)> {
    let np = PyModule::import(py, "numpy")?;
    let mut shape: Option<(usize, usize)> = None;
    let mut out = Vec::new();
    for item in seq_items(obj)? {
        let a = np.call_method1("asarray", (item, "complex128"))?;
        let a: PyReadonlyArray2<'_, Complex64> = a.extract()?;
        let v = a.as_array();
        let s = (v.nrows(), v.ncols());
        match shape {
            None => shape = Some(s),
            Some(p) if p != s => {
                return Err(PyValueError::new_err(format!(
                    "every k-point block must have the same shape: {p:?} vs {s:?}"
                )));
            }
            _ => {}
        }
        out.push(v.iter().copied().collect());
    }
    let (r, c) = shape.unwrap_or((0, 0));
    Ok((out, r, c))
}

fn cmats_to_list<'py>(
    py: Python<'py>,
    blocks: Vec<Vec<Complex64>>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyList>> {
    let arrays = blocks
        .into_iter()
        .map(|b| {
            Array2::from_shape_vec((rows, cols), b)
                .map(|a| a.into_pyarray(py))
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, arrays)
}

fn f64_vecs_to_list<'py>(py: Python<'py>, v: Vec<Vec<f64>>) -> PyResult<Bound<'py, PyList>> {
    PyList::new(py, v.into_iter().map(|x| PyArray1::from_vec(py, x)))
}

#[pymethods]
impl PyKPoints {
    #[new]
    #[pyo3(signature = (cell = None, kpts = None))]
    fn new(cell: Option<Py<PyAny>>, kpts: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let k = extract_kpts_opt(kpts)?.map_or_else(|| vec![[0.0; 3]], |(k, _)| k);
        Ok(Self {
            cell_obj: cell,
            cell: None,
            inner: KPoints::new(k),
        })
    }

    /// `build(space_group_symmetry=False, time_reversal_symmetry=False,
    /// symmorphic=True, check_mesh_symmetry=True)` → self (`kpts.py:1017`).
    /// A cell carrying a lattice symmetry lends it (`symmorphic`/
    /// `check_mesh_symmetry` then come from `Cell.build`). No cell: a no-op.
    #[pyo3(signature = (space_group_symmetry = false, time_reversal_symmetry = false,
                        symmorphic = true, check_mesh_symmetry = true))]
    fn build<'py>(
        slf: &Bound<'py, Self>,
        space_group_symmetry: bool,
        time_reversal_symmetry: bool,
        symmorphic: bool,
        check_mesh_symmetry: bool,
    ) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        let cell_obj = cell_obj_or_none(py, &slf.borrow().cell_obj);
        if cell_obj.is_none(py) {
            return Ok(slf.clone());
        }
        let cell = symmetry_cell(py, cell_obj.bind(py))?;
        {
            let mut b = slf.borrow_mut();
            b.inner
                .build(
                    &cell,
                    space_group_symmetry,
                    time_reversal_symmetry,
                    symmorphic,
                    check_mesh_symmetry,
                )
                .map_err(pbc_symm_to_py)?;
            b.cell = Some(cell);
        }
        Ok(slf.clone())
    }

    #[getter]
    fn cell(&self, py: Python<'_>) -> Py<PyAny> {
        cell_obj_or_none(py, &self.cell_obj)
    }

    #[getter]
    fn nkpts(&self) -> usize {
        self.inner.nkpts()
    }

    #[getter]
    fn nkpts_ibz(&self) -> usize {
        self.inner.nkpts_ibz()
    }

    fn __len__(&self) -> usize {
        self.inner.nkpts_ibz()
    }

    fn __repr__(&self) -> String {
        format!(
            "<pyscf._native.pbc.symm.KPoints nkpts={} nkpts_ibz={} nop={} time_reversal={}>",
            self.inner.nkpts(),
            self.inner.nkpts_ibz(),
            self.inner.nop(),
            self.inner.time_reversal
        )
    }

    #[getter]
    fn _built(&self) -> bool {
        self.cell.is_some()
    }

    #[getter]
    fn time_reversal(&self) -> bool {
        self.inner.time_reversal
    }

    #[getter]
    fn symmorphic(&self) -> bool {
        self.inner.symmetry.symmorphic
    }

    #[getter]
    fn has_inversion(&self) -> bool {
        self.inner.has_inversion()
    }

    #[getter]
    fn nop(&self) -> usize {
        self.inner.nop()
    }

    #[getter]
    fn ops<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        ops_to_py(py, self.inner.ops())
    }

    #[getter(Dmats)]
    fn dmats<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyList>>> {
        if !self.inner.symmetry.built {
            return Ok(None);
        }
        dmats_to_py(py, self.inner.dmats()).map(Some)
    }

    #[getter]
    fn l_max(&self) -> Option<usize> {
        self.inner
            .symmetry
            .built
            .then_some(self.inner.symmetry.l_max)
    }

    #[getter]
    fn kpts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        kpts_to_pyarray(py, &self.inner.kpts)
    }

    #[getter]
    fn kpts_ibz<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        kpts_to_pyarray(py, &self.inner.kpts_ibz)
    }

    /// `None` before `build()` (upstream's `__init__` value).
    #[getter]
    fn kpts_scaled(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        if self.cell.is_none() {
            return Ok(None);
        }
        Ok(Some(
            kpts_to_pyarray(py, &self.inner.kpts_scaled)?
                .into_any()
                .unbind(),
        ))
    }

    #[getter]
    fn kpts_scaled_ibz(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        if self.cell.is_none() {
            return Ok(None);
        }
        Ok(Some(
            kpts_to_pyarray(py, &self.inner.kpts_scaled_ibz)?
                .into_any()
                .unbind(),
        ))
    }

    #[getter]
    fn weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.weights.clone())
    }

    #[getter]
    fn weights_ibz<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.weights_ibz.clone())
    }

    #[getter]
    fn ibz2bz<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        idx_array(py, &self.inner.ibz2bz)
    }

    #[getter]
    fn bz2ibz<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        idx_array(py, &self.inner.bz2ibz)
    }

    /// `(nkpts, nop*(time_reversal+1))` int64 with `-1` for "not in the mesh";
    /// `None` before `build()`.
    #[getter]
    fn k2opk<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<i64>>>> {
        if self.inner.k2opk.is_empty() {
            return Ok(None);
        }
        let nrow = self.inner.k2opk.len();
        let ncol = self.inner.k2opk[0].len();
        let flat: Vec<i64> = self.inner.k2opk.iter().flatten().copied().collect();
        debug_assert!(flat.iter().all(|&x| x >= NO_MAP));
        let arr = Array2::from_shape_vec((nrow, ncol), flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Some(arr.into_pyarray(py)))
    }

    #[getter]
    fn stars<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        idx_list(py, &self.inner.stars)
    }

    #[getter]
    fn stars_ops<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        idx_list(py, &self.inner.stars_ops)
    }

    #[getter]
    fn stars_ops_bz<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        idx_array(py, &self.inner.stars_ops_bz)
    }

    #[getter]
    fn time_reversal_symm_bz<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        idx_array(py, &self.inner.time_reversal_symm_bz)
    }

    /// Indices into `k2opk`'s COLUMN space (`2*nop` with time reversal) — the
    /// representation D-17-07-01 is about.
    #[getter]
    fn little_cogroup_ops<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        idx_list(py, &self.inner.little_cogroup_ops)
    }

    /// `kpts.get_kconserv()` → int32 `(nkpts, nkpts, nkpts)`.
    fn get_kconserv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray3<i32>>> {
        let kc = self.inner.get_kconserv(self.built_cell()?);
        let n = kc.nkpts;
        let arr = Array3::from_shape_vec((n, n, n), kc.data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray(py))
    }

    /// `make_gdf_kptij_lst_jk()` → `(npair, 2, 3)`.
    fn make_gdf_kptij_lst_jk<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, numpy::PyArray3<f64>>> {
        let pairs = self.inner.make_gdf_kptij_lst_jk();
        let flat: Vec<f64> = pairs
            .iter()
            .flat_map(|(a, b)| a.iter().chain(b.iter()).copied().collect::<Vec<_>>())
            .collect();
        let arr = Array3::from_shape_vec((pairs.len(), 2, 3), flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray(py))
    }

    /// `make_ktuples_ibz(kpts_scaled=None, ntuple=2, tol=KPT_DIFF_TOL)` →
    /// `(ibz2bz, weight_ibz, bz2ibz, stars, stars_ops, stars_ops_bz)`. Only the
    /// `kpts_scaled is None` branch is ported (nothing reaches the other).
    #[pyo3(signature = (kpts_scaled = None, ntuple = 2, tol = KPT_DIFF_TOL))]
    fn make_ktuples_ibz<'py>(
        &self,
        py: Python<'py>,
        kpts_scaled: Option<&Bound<'py, PyAny>>,
        ntuple: usize,
        tol: f64,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let _ = tol;
        if kpts_scaled.is_some_and(|k| !k.is_none()) {
            return Err(PyNotImplementedError::new_err(
                "make_ktuples_ibz(kpts_scaled=...) is not ported (kpts.py:143-149, needs \
                 map_kpts_tuples at ntuple > 1; no caller reaches it)",
            ));
        }
        if self.inner.k2opk.is_empty() {
            return Err(PyValueError::new_err(
                "make_ktuples_ibz: call build() first",
            ));
        }
        let t = self.inner.make_ktuples_ibz(ntuple);
        PyTuple::new(
            py,
            [
                idx_array(py, &t.ibz2bz).into_any(),
                PyArray1::from_vec(py, t.weight_ibz).into_any(),
                idx_array(py, &t.bz2ibz).into_any(),
                idx_list(py, &t.stars)?.into_any(),
                idx_list(py, &t.stars_ops)?.into_any(),
                idx_array(py, &t.stars_ops_bz).into_any(),
            ],
        )
    }

    /// `make_k4_ibz(sym='s1', return_ops=False)`. `'s4'` raises
    /// `PyscfRsRuntimeError` (`NotYetImplemented`): upstream's own tree has no
    /// caller for it, so the port ships no number no oracle checks.
    #[pyo3(signature = (sym = "s1", return_ops = false))]
    fn make_k4_ibz<'py>(
        &self,
        py: Python<'py>,
        sym: &str,
        return_ops: bool,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let r = self
            .inner
            .make_k4_ibz(self.built_cell()?, sym)
            .map_err(pbc_symm_to_py)?;
        let flat: Vec<i64> = r.k4.iter().flatten().map(|&x| x as i64).collect();
        let k4 = Array2::from_shape_vec((r.k4.len(), 4), flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into_pyarray(py);
        let mut items = vec![
            k4.into_any(),
            PyArray1::from_vec(py, r.weight).into_any(),
            idx_array(py, &r.bz2ibz).into_any(),
        ];
        if sym == "s1" && return_ops {
            items.push(idx_array(py, &r.ibz2bz).into_any());
            items.push(idx_list(py, &r.stars_ops)?.into_any());
            items.push(idx_array(py, &r.stars_ops_bz).into_any());
        }
        PyTuple::new(py, items)
    }

    /// `little_cogroups(return_indices=True)` → `(copgs, indices)`: each
    /// co-group as its sorted `(order, 3, 3)` int32 rotation matrices, plus the
    /// sort permutation. Raises (kind `"PbcSymm"`) where upstream raises
    /// `IndexError` — D-17-07-01, time reversal on a cell without inversion.
    #[pyo3(signature = (return_indices = true))]
    fn little_cogroups<'py>(
        &self,
        py: Python<'py>,
        return_indices: bool,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let _ = return_indices;
        let (copgs, indices) = self.inner.little_cogroups().map_err(pbc_symm_to_py)?;
        let mut groups = Vec::with_capacity(copgs.len());
        for g in &copgs {
            let flat: Vec<i32> = g
                .elements
                .iter()
                .flat_map(|e| e.matrix.into_iter().flatten())
                .collect();
            let arr = Array3::from_shape_vec((g.elements.len(), 3, 3), flat)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            groups.push(arr.into_pyarray(py));
        }
        PyTuple::new(
            py,
            [
                PyList::new(py, groups)?.into_any(),
                idx_list(py, &indices)?.into_any(),
            ],
        )
    }

    /// D-17-09-02's detector: `True` for every `k2opk` column that moves some
    /// k-point out of the mesh.
    fn ops_outside_kmesh_subgroup<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        PyArray1::from_vec(py, self.inner.ops_outside_kmesh_subgroup())
    }

    /// `[(ibz_index, op_column), ...]` of little-co-group ops outside the
    /// k-mesh subgroup (D-17-09-02).
    fn little_cogroup_ops_outside_kmesh_subgroup(&self) -> Vec<(usize, usize)> {
        self.inner.little_cogroup_ops_outside_kmesh_subgroup()
    }

    /// `transform_mo_coeff(mo_coeff_ibz)` — `([2,] nkpts_ibz, nao, nmo)` in,
    /// `([2,] nkpts)` list of complex `(nao, nmo)` out.
    fn transform_mo_coeff<'py>(
        &self,
        py: Python<'py>,
        mo_coeff_ibz: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyList>> {
        let cell = self.built_cell()?;
        let one = |obj: &Bound<'py, PyAny>| -> PyResult<Bound<'py, PyList>> {
            let (blocks, nao, nmo) = read_cmats(py, obj)?;
            let out = py
                .detach(|| self.inner.transform_mo_coeff(cell, &blocks, nao, nmo))
                .map_err(pbc_symm_to_py)?;
            cmats_to_list(py, out, nao, nmo)
        };
        if is_spin_pair(py, mo_coeff_ibz, 2)? {
            return PyList::new(
                py,
                [
                    one(&mo_coeff_ibz.get_item(0)?)?,
                    one(&mo_coeff_ibz.get_item(1)?)?,
                ],
            );
        }
        one(mo_coeff_ibz)
    }

    /// `transform_single_mo_coeff(mo_coeff_ibz, k)` — one full-BZ k-point.
    fn transform_single_mo_coeff<'py>(
        &self,
        py: Python<'py>,
        mo_coeff_ibz: &Bound<'py, PyAny>,
        k: usize,
    ) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        let cell = self.built_cell()?;
        let (blocks, nao, nmo) = read_cmats(py, mo_coeff_ibz)?;
        if k >= self.inner.nkpts() {
            return Err(PyValueError::new_err(format!(
                "k = {k} out of range for nkpts = {}",
                self.inner.nkpts()
            )));
        }
        if blocks.len() != self.inner.nkpts_ibz() {
            return Err(pbc_symm_to_py(PbcSymmError::KptsSymmInputMismatch(
                format!(
                    "mo_coeff has {} blocks, expected nkpts_ibz = {}",
                    blocks.len(),
                    self.inner.nkpts_ibz()
                ),
            )));
        }
        let out = self
            .inner
            .transform_mo_coeff_k(cell, &blocks, nao, nmo, k)
            .map_err(pbc_symm_to_py)?;
        Array2::from_shape_vec((nao, nmo), out)
            .map(|a| a.into_pyarray(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// `transform_mo_occ(mo_occ_ibz)` — pure index map; `[2,]` spin form too.
    fn transform_mo_occ<'py>(
        &self,
        py: Python<'py>,
        mo_occ_ibz: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyList>> {
        let one = |obj: &Bound<'py, PyAny>| -> PyResult<Bound<'py, PyList>> {
            let v = read_f64_vecs(py, obj)?;
            f64_vecs_to_list(py, self.inner.transform_mo_occ(&v).map_err(pbc_symm_to_py)?)
        };
        if is_spin_pair(py, mo_occ_ibz, 1)? {
            return PyList::new(
                py,
                [
                    one(&mo_occ_ibz.get_item(0)?)?,
                    one(&mo_occ_ibz.get_item(1)?)?,
                ],
            );
        }
        one(mo_occ_ibz)
    }

    /// `transform_mo_energy(mo_energy_ibz)` — pure index map; `[2,]` spin form too.
    fn transform_mo_energy<'py>(
        &self,
        py: Python<'py>,
        mo_energy_ibz: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyList>> {
        let one = |obj: &Bound<'py, PyAny>| -> PyResult<Bound<'py, PyList>> {
            let v = read_f64_vecs(py, obj)?;
            f64_vecs_to_list(
                py,
                self.inner.transform_mo_energy(&v).map_err(pbc_symm_to_py)?,
            )
        };
        if is_spin_pair(py, mo_energy_ibz, 1)? {
            return PyList::new(
                py,
                [
                    one(&mo_energy_ibz.get_item(0)?)?,
                    one(&mo_energy_ibz.get_item(1)?)?,
                ],
            );
        }
        one(mo_energy_ibz)
    }

    /// `transform_dm(dm_ibz)` — `R dm R^H` per star; `[2,]` spin form too.
    fn transform_dm<'py>(
        &self,
        py: Python<'py>,
        dm_ibz: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyList>> {
        self.sandwich(py, dm_ibz, false)
    }

    /// `transform_1e_operator(fock_ibz)` (alias `transform_fock`).
    fn transform_1e_operator<'py>(
        &self,
        py: Python<'py>,
        fock_ibz: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyList>> {
        self.sandwich(py, fock_ibz, true)
    }

    fn transform_fock<'py>(
        &self,
        py: Python<'py>,
        fock_ibz: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyList>> {
        self.sandwich(py, fock_ibz, true)
    }

    /// `symmetrize_wavefunction` — upstream itself raises
    /// `RuntimeError('need verification')` (`kpts.py:415`).
    fn symmetrize_wavefunction(&self) -> PyResult<()> {
        self.inner
            .symmetrize_wavefunction()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

impl PyKPoints {
    fn sandwich<'py>(
        &self,
        py: Python<'py>,
        obj: &Bound<'py, PyAny>,
        fock: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        let cell = self.built_cell()?;
        let one = |o: &Bound<'py, PyAny>| -> PyResult<Bound<'py, PyList>> {
            let (blocks, nao, ncol) = read_cmats(py, o)?;
            if nao != ncol {
                return Err(PyValueError::new_err(format!(
                    "expected square (nao, nao) blocks, got ({nao}, {ncol})"
                )));
            }
            let out = py
                .detach(|| {
                    if fock {
                        self.inner.transform_1e_operator(cell, &blocks, nao)
                    } else {
                        self.inner.transform_dm(cell, &blocks, nao)
                    }
                })
                .map_err(pbc_symm_to_py)?;
            cmats_to_list(py, out, nao, nao)
        };
        if is_spin_pair(py, obj, 2)? {
            return PyList::new(py, [one(&obj.get_item(0)?)?, one(&obj.get_item(1)?)?]);
        }
        one(obj)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Free functions
// ─────────────────────────────────────────────────────────────────────────────

/// Build a `KPoints` for `cell` over the full-BZ mesh `kpts` —
/// `pyscf_pbc_symm::kpts::make_kpts` (the exact call Gate A gates). Shared with
/// `Cell.make_kpts(space_group_symmetry/time_reversal_symmetry=True)`.
pub fn kpoints_for_cell(
    py: Python<'_>,
    cell_obj: &Bound<'_, PyAny>,
    kpts: &[[f64; 3]],
    space_group_symmetry: bool,
    time_reversal_symmetry: bool,
) -> PyResult<Py<PyKPoints>> {
    let cell = symmetry_cell(py, cell_obj)?;
    let kp = py
        .detach(|| {
            pyscf_pbc_symm::kpts::make_kpts(
                &cell,
                kpts,
                space_group_symmetry,
                time_reversal_symmetry,
            )
        })
        .map_err(pbc_symm_to_py)?;
    Py::new(
        py,
        PyKPoints::from_built(cell_obj.clone().unbind(), cell, kp),
    )
}

/// `make_kpts(cell, kpts=np.zeros((1,3)), space_group_symmetry=False,
/// time_reversal_symmetry=False, **kwargs)` (`kpts.py:804`). A `KPoints` input
/// is rebuilt in place, as upstream does.
#[pyfunction(signature = (cell, kpts = None, space_group_symmetry = false,
                          time_reversal_symmetry = false, **kwargs))]
fn make_kpts<'py>(
    py: Python<'py>,
    cell: &Bound<'py, PyAny>,
    kpts: Option<&Bound<'py, PyAny>>,
    space_group_symmetry: bool,
    time_reversal_symmetry: bool,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Py<PyAny>> {
    if let Some(k) = kpts
        && let Ok(kp) = k.cast::<PyKPoints>()
    {
        let kw = PyDict::new(py);
        kw.set_item("space_group_symmetry", space_group_symmetry)?;
        kw.set_item("time_reversal_symmetry", time_reversal_symmetry)?;
        if let Some(extra) = kwargs {
            kw.update(extra.as_mapping())?;
        }
        return Ok(kp.call_method("build", (), Some(&kw))?.unbind());
    }
    if kwargs.is_some_and(|k| !k.is_empty()) {
        return Err(PyNotImplementedError::new_err(
            "make_kpts(**kwargs): extra build keywords are only forwarded to an existing KPoints",
        ));
    }
    let k = match kpts {
        Some(k) if !k.is_none() => extract_kpts(k)?.0,
        _ => vec![[0.0; 3]],
    };
    Ok(kpoints_for_cell(py, cell, &k, space_group_symmetry, time_reversal_symmetry)?.into_any())
}

/// `geom.get_crystal_class(cell, ops=None, tol=SYMPREC)` →
/// `(point_group_symbol, laue_class)`. `ops` must be `None`.
#[pyfunction(signature = (cell, ops = None, tol = SYMPREC))]
fn get_crystal_class(
    py: Python<'_>,
    cell: &Bound<'_, PyAny>,
    ops: Option<&Bound<'_, PyAny>>,
    tol: f64,
) -> PyResult<(&'static str, &'static str)> {
    if ops.is_some_and(|o| !o.is_none()) {
        return Err(PyNotImplementedError::new_err(
            "get_crystal_class(ops=...) is not bound; the ops are searched from the cell",
        ));
    }
    let c = extract_cell_from_pyany(py, cell)?;
    geom::get_crystal_class(&c, None, tol).map_err(pbc_symm_to_py)
}
