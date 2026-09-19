//! `pyscf._native.pbc.tools` — the FFT family, `madelung`, `ExxDiv`,
//! `cutoff_to_mesh` / `mesh_to_cutoff` (plan 20-14).
//!
//! * **`ExxDiv` is bound ONCE, here** — the canonical definition is
//!   `pyscf-pbc-tools/src/coulg.rs:14`, and `pyscf-pbc-gto` only re-exports it.
//!   [`register`] attaches the same type object to `pyscf._native.pbc.gto`, so
//!   `pbc.gto.ExxDiv is pbc.tools.ExxDiv`. Upstream has no such class (it passes
//!   the strings `'ewald'`/`'vcut_sph'`/`'vcut_ws'`); both spellings are accepted
//!   by `get_coulG`.
//! * **`get_coulG`, `super_cell`, `cell_plus_imgs`** are 20-09's objects from
//!   `pyscf._native.pbc.gto`, re-exported (identity), not rebound.
//! * **FFT layout.** `fft(f, mesh)` takes any array whose size is a multiple of
//!   `prod(mesh)`; it is read C-order as `(nbatch, ngrids)` rows (upstream's
//!   `f.reshape(-1, *mesh)`), transformed by `pyscf_pbc_tools::fft`, and returned
//!   complex128 in `f`'s shape. A real input is widened by numpy (exact).

use numpy::{Complex64, PyArray1, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyscf_algebra::CTensor;
use pyscf_pbc_tools::coulg::ExxDiv;
use pyscf_pbc_tools::{PbcToolsError, fft as fft_ops};

use crate::bridge::extract_cell_from_pyany;
use crate::errors::pyscf_to_py;
use crate::numpy_io::{BufOrder, ctensor_to_pyarray, to_ctensor};
use crate::pbc::convert::{extract_kpts_opt, extract_mat3, extract_usize3};

/// Register the tools surface; `gto` must already be in `sys.modules`.
pub fn register(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyExxDiv>()?;
    m.add_function(wrap_pyfunction!(fft, m)?)?;
    m.add_function(wrap_pyfunction!(ifft, m)?)?;
    m.add_function(wrap_pyfunction!(fftk, m)?)?;
    m.add_function(wrap_pyfunction!(ifftk, m)?)?;
    m.add_function(wrap_pyfunction!(madelung, m)?)?;
    m.add_function(wrap_pyfunction!(cutoff_to_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_to_cutoff, m)?)?;
    let sys_modules = PyModule::import(py, "sys")?.getattr("modules")?;
    let gto = sys_modules.get_item("pyscf._native.pbc.gto")?;
    for name in ["get_coulG", "super_cell", "cell_plus_imgs"] {
        m.add(name, gto.getattr(name)?)?;
    }
    gto.setattr("ExxDiv", m.getattr("ExxDiv")?)?;
    Ok(())
}

/// `PbcToolsError` → Python (it only wraps a `PyscfRsError`).
pub fn pbc_tools_to_py(err: PbcToolsError) -> PyErr {
    match err {
        PbcToolsError::Core(e) => pyscf_to_py(e),
    }
}

/// How the `G + k = 0` exchange divergence is treated — `coulg.rs:14`.
/// `str(ExxDiv.EWALD) == 'ewald'`; `ExxDiv.parse(s)` is `None` for
/// `''`/`'none'`/`'false'`/unknown, as upstream's fall-through `else`.
#[pyclass(
    eq,
    eq_int,
    frozen,
    from_py_object,
    name = "ExxDiv",
    module = "pyscf._native.pbc.tools"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyExxDiv {
    #[pyo3(name = "EWALD")]
    Ewald,
    #[pyo3(name = "VCUT_SPH")]
    VcutSph,
    #[pyo3(name = "VCUT_WS")]
    VcutWs,
}

impl From<PyExxDiv> for ExxDiv {
    fn from(v: PyExxDiv) -> Self {
        match v {
            PyExxDiv::Ewald => ExxDiv::Ewald,
            PyExxDiv::VcutSph => ExxDiv::VcutSph,
            PyExxDiv::VcutWs => ExxDiv::VcutWs,
        }
    }
}

impl From<ExxDiv> for PyExxDiv {
    fn from(v: ExxDiv) -> Self {
        match v {
            ExxDiv::Ewald => PyExxDiv::Ewald,
            ExxDiv::VcutSph => PyExxDiv::VcutSph,
            ExxDiv::VcutWs => PyExxDiv::VcutWs,
        }
    }
}

#[pymethods]
impl PyExxDiv {
    /// Parse upstream's string spelling (case-insensitive).
    #[staticmethod]
    fn parse(s: &str) -> Option<Self> {
        ExxDiv::parse(s).map(Self::from)
    }

    /// The upstream string spelling.
    #[getter]
    fn value(&self) -> &'static str {
        ExxDiv::from(*self).as_str()
    }

    fn __str__(&self) -> &'static str {
        ExxDiv::from(*self).as_str()
    }
}

/// `Option<ExxDiv>` from upstream's `exx`/`exxdiv` argument: `None`/`False` →
/// `None`, `True` → Ewald, a string → [`ExxDiv::parse`], an `ExxDiv` → itself.
/// Shared with `pbc::gto::get_coulG`.
pub fn extract_exxdiv(v: Option<&Bound<'_, PyAny>>) -> PyResult<Option<ExxDiv>> {
    let Some(v) = v else { return Ok(None) };
    if v.is_none() {
        return Ok(None);
    }
    if let Ok(e) = v.extract::<PyExxDiv>() {
        return Ok(Some(e.into()));
    }
    if let Ok(s) = v.extract::<String>() {
        return Ok(ExxDiv::parse(&s));
    }
    if let Ok(b) = v.extract::<bool>() {
        return Ok(b.then_some(ExxDiv::Ewald));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "exx must be None, a bool, 'ewald'/'vcut_sph'/'vcut_ws', or a pbc.tools.ExxDiv",
    ))
}

/// A grid array as `(CTensor rows, original shape)`, C-order.
fn grid_in(py: Python<'_>, f: &Bound<'_, PyAny>) -> PyResult<(CTensor, Vec<usize>)> {
    let np = PyModule::import(py, "numpy")?;
    let kw = PyDict::new(py);
    kw.set_item("dtype", np.getattr("complex128")?)?;
    let arr = np.call_method("ascontiguousarray", (f,), Some(&kw))?;
    let ro: PyReadonlyArrayDyn<'_, Complex64> = arr.extract()?;
    to_ctensor(ro, BufOrder::C)
}

/// A transform over `(rows, mesh, optional phase)`.
type GridOp = fn(&CTensor, [usize; 3], Option<&CTensor>) -> Result<CTensor, PbcToolsError>;

fn grid_call<'py>(
    py: Python<'py>,
    f: &Bound<'py, PyAny>,
    mesh: &Bound<'py, PyAny>,
    phase: Option<&Bound<'py, PyAny>>,
    op: GridOp,
) -> PyResult<Bound<'py, PyAny>> {
    let mesh = extract_usize3(mesh, "mesh")?;
    let (t, shape) = grid_in(py, f)?;
    let ph = match phase {
        Some(p) => Some(grid_in(py, p)?.0),
        None => None,
    };
    let out = py
        .detach(|| op(&t, mesh, ph.as_ref()))
        .map_err(pbc_tools_to_py)?;
    Ok(ctensor_to_pyarray(py, &out, &shape, BufOrder::C)?.into_any())
}

/// `fft(f, mesh)` — forward 3-D transform, normalisation 1 (`fft.rs:96`).
#[pyfunction]
fn fft<'py>(
    py: Python<'py>,
    f: &Bound<'py, PyAny>,
    mesh: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    grid_call(py, f, mesh, None, |t, m, _| fft_ops::fft(t, m))
}

/// `ifft(g, mesh)` — inverse transform, `1/ngrids` (`fft.rs:110`).
#[pyfunction]
fn ifft<'py>(
    py: Python<'py>,
    g: &Bound<'py, PyAny>,
    mesh: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    grid_call(py, g, mesh, None, |t, m, _| fft_ops::ifft(t, m))
}

fn phase_required(p: Option<&CTensor>) -> Result<&CTensor, PbcToolsError> {
    p.ok_or_else(|| {
        PbcToolsError::Core(pyscf_core::PyscfRsError::Core(
            pyscf_core::CoreError::InvalidMolecule("fftk/ifftk: phase array missing".into()),
        ))
    })
}

/// `fftk(f, mesh, expmikr)` — `fft(f * expmikr)` (`fft.rs:125`).
#[pyfunction]
fn fftk<'py>(
    py: Python<'py>,
    f: &Bound<'py, PyAny>,
    mesh: &Bound<'py, PyAny>,
    expmikr: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    grid_call(py, f, mesh, Some(expmikr), |t, m, p| {
        fft_ops::fftk(t, m, phase_required(p)?)
    })
}

/// `ifftk(g, mesh, expikr)` — `ifft(g) * expikr` (`fft.rs:134`).
#[pyfunction]
fn ifftk<'py>(
    py: Python<'py>,
    g: &Bound<'py, PyAny>,
    mesh: &Bound<'py, PyAny>,
    expikr: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    grid_call(py, g, mesh, Some(expikr), |t, m, p| {
        fft_ops::ifftk(t, m, phase_required(p)?)
    })
}

/// `madelung(cell, kpts, omega=None)` (`pyscf_pbc_gto::coulg::madelung`).
#[pyfunction(signature = (cell, kpts, omega = None))]
fn madelung(
    py: Python<'_>,
    cell: &Bound<'_, PyAny>,
    kpts: &Bound<'_, PyAny>,
    omega: Option<f64>,
) -> PyResult<f64> {
    let c = extract_cell_from_pyany(py, cell)?;
    let k = extract_kpts_opt(Some(kpts))?.map_or_else(|| vec![[0.0; 3]], |(k, _)| k);
    py.detach(|| pyscf_pbc_gto::madelung(&c, &k, omega))
        .map_err(pyscf_to_py)
}

/// `cutoff_to_mesh(a, cutoff)` → int64 `(3,)` (`mesh.rs:138`).
#[pyfunction]
fn cutoff_to_mesh<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    cutoff: f64,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let a = extract_mat3(a, "a")?;
    let m = pyscf_pbc_tools::mesh::cutoff_to_mesh(&a, cutoff).map_err(pyscf_to_py)?;
    Ok(PyArray1::from_vec(
        py,
        m.iter().map(|&x| x as i64).collect(),
    ))
}

/// `mesh_to_cutoff(a, mesh)` → float64 `(3,)` (`mesh.rs:168`).
#[pyfunction]
fn mesh_to_cutoff<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    mesh: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = extract_mat3(a, "a")?;
    let mesh = extract_usize3(mesh, "mesh")?;
    let ke = pyscf_pbc_tools::mesh::mesh_to_cutoff(&a, mesh).map_err(pyscf_to_py)?;
    Ok(PyArray1::from_vec(py, ke.to_vec()))
}
