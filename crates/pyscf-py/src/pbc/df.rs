//! `pyscf._native.pbc.df` — ONE wrapper over the five periodic DF builders
//! (plan 20-10).
//!
//! # Shape
//!
//! * [`PyPeriodicDf`] (`PeriodicDf`, the base class) holds the builder as a
//!   shared trait object and carries every method: `build`, `get_nuc`,
//!   `get_pp`, `get_hcore`, `get_jk`, the `_cderi` persistence surface and the
//!   ao2mo entry points. `trait PeriodicDf` (`traits.rs:102`) is object-safe, so
//!   one body serves all five builders.
//! * `FFTDF`, `AFTDF`, `GDF`, `MDF`, `RSDF` are thin subclasses that only
//!   construct — so `isinstance(mydf, GDF)` discriminates routes as upstream's
//!   drivers do. `PWDF`, `DF` and `RSGDF` are the SAME class objects as `AFTDF`,
//!   `GDF` and `RSDF` (module attributes bound to the existing type, not copies).
//!
//! # Ownership — the contract 20-12/20-13 drivers rely on
//!
//! A DF object is mutable state (`build()` fills `_cderi`, `mydf.auxbasis = …`
//! changes the fit), and `mf.with_df = mydf` must work AFTER driver construction
//! (`PBC-MASTER-PLAN:1909`). So:
//!
//! 1. the builder lives behind an `Arc` ([`PyPeriodicDf::shared`]); every
//!    configuration setter builds a FRESH builder from the stored configuration
//!    and swaps the `Arc` — stale caches (a `cderi` fitted with the old
//!    `auxbasis`) can never be served, and a driver still holding the previous
//!    `Arc` is unaffected until it re-fetches;
//! 2. a driver binding stores the Python object (`with_df: Py<PyAny>`, so
//!    `mf.with_df is mydf`), and at every `kernel`/`get_jk` entry calls
//!    [`extract_df`] (or [`PyPeriodicDf::boxed`]) to get a
//!    `Box<dyn PeriodicDf>` — a [`SharedDf`] adapter over the current `Arc` —
//!    which it hands to `Krhf::from_df` or assigns to the Rust driver's
//!    `pub with_df` field. Laziness is preserved: the adapter forwards to the
//!    same builder, so a `cderi` built through either handle is built once.

use std::path::PathBuf;
use std::sync::Arc;

use numpy::ndarray::{Array2, ArrayD};
use numpy::{Complex64, IntoPyArray, PyArray2};
use pyo3::exceptions::{PyAttributeError, PyNotImplementedError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use pyscf_algebra::CTensor;
use pyscf_pbc_df::df_jk::KMats;
use pyscf_pbc_df::traits::{JkOpts, JkResult, PeriodicDf};
use pyscf_pbc_df::{
    Aftdf, CoulGCache, DfKind, DfOpts, Eri, Eri7d, Fftdf, Gdf, Mdf, MoCoeff, MoKpts, PbcDfError,
    Rsdf, SrBlock,
};
use pyscf_pbc_gto::{Cell, ExxDiv};

use crate::errors::{PyscfRsRuntimeError, pyscf_to_py};
use crate::numpy_io::{
    BufOrder, ctensor_to_array, kdms_to_pylist, kmats_to_pylist, to_ctensor, to_kdms,
};
use crate::pbc::convert::{extract_kpts_opt, extract_usize3, kpts_to_pyarray};

/// Register the base class, the five builders, the three aliases and
/// `density_fit` on the child module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPeriodicDf>()?;
    m.add_class::<PyFftdf>()?;
    m.add_class::<PyAftdf>()?;
    m.add_class::<PyGdf>()?;
    m.add_class::<PyMdf>()?;
    m.add_class::<PyRsdf>()?;
    // Upstream `pbc/df/__init__.py:21,31` + `rsdf.py` — aliases are the SAME
    // type object, so `PWDF is AFTDF` holds.
    m.add("PWDF", m.getattr("AFTDF")?)?;
    m.add("DF", m.getattr("GDF")?)?;
    m.add("RSGDF", m.getattr("RSDF")?)?;
    m.add_function(wrap_pyfunction!(density_fit, m)?)?;
    m.add_function(wrap_pyfunction!(_driver_handle_get_jk, m)?)?;
    Ok(())
}

/// `PbcDfError` → Python. A wrapped `PyscfRsError` keeps its kind
/// (`NotYetImplemented` refusals stay recognisable); the FFT/backend variants
/// report kind `"PbcDf"`.
pub fn pbc_df_to_py(err: PbcDfError) -> PyErr {
    match err {
        PbcDfError::Core(e) => pyscf_to_py(e),
        other => PyscfRsRuntimeError::new_err((other.to_string(), "PbcDf", Vec::<String>::new())),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SharedDf — the Box<dyn PeriodicDf> a driver receives
// ─────────────────────────────────────────────────────────────────────────────

/// A `PeriodicDf` that forwards to a shared builder. This is what
/// [`PyPeriodicDf::boxed`] hands a driver: `Krhf::from_df(Box::new(SharedDf(..)))`.
///
/// `build` runs the builder's own `build` only when this handle is the sole
/// owner; otherwise it is a no-op, which is safe because every builder builds
/// lazily on first use (`Gdf::cderi`, `Mdf::gdf`, `Fftdf::ao_kpts`, `Aftdf`'s
/// G-vector cache are all `&self` + `OnceLock`/`Mutex`).
#[derive(Debug, Clone)]
pub struct SharedDf(pub Arc<dyn PeriodicDf>);

impl PeriodicDf for SharedDf {
    fn cell(&self) -> &Cell {
        self.0.cell()
    }
    fn mesh(&self) -> [usize; 3] {
        self.0.mesh()
    }
    fn kpts(&self) -> &[[f64; 3]] {
        self.0.kpts()
    }
    fn build(&mut self) -> Result<(), PbcDfError> {
        match Arc::get_mut(&mut self.0) {
            Some(df) => df.build(),
            None => Ok(()),
        }
    }
    fn get_nuc(&self, kpts: &[[f64; 3]]) -> Result<Vec<CTensor>, PbcDfError> {
        self.0.get_nuc(kpts)
    }
    fn get_pp(&self, kpts: &[[f64; 3]]) -> Result<Vec<CTensor>, PbcDfError> {
        self.0.get_pp(kpts)
    }
    fn name(&self) -> &'static str {
        self.0.name()
    }
    fn local_potential_r(&self) -> Result<Option<Vec<f64>>, PbcDfError> {
        self.0.local_potential_r()
    }
    fn get_jk(
        &self,
        dms: &[KMats],
        kpts: &[[f64; 3]],
        opts: JkOpts<'_>,
    ) -> Result<JkResult, PbcDfError> {
        self.0.get_jk(dms, kpts, opts)
    }
    fn ao2mo(
        &self,
        mos: [&MoCoeff; 4],
        kidx: [usize; 4],
        compact: bool,
    ) -> Result<Eri, PbcDfError> {
        self.0.ao2mo(mos, kidx, compact)
    }
    fn ao2mo_cached(
        &self,
        mos: [&MoCoeff; 4],
        kidx: [usize; 4],
        compact: bool,
        cache: Option<&CoulGCache>,
    ) -> Result<Eri, PbcDfError> {
        self.0.ao2mo_cached(mos, kidx, compact, cache)
    }
    fn get_ao_eri(&self, kidx: [usize; 4], compact: bool) -> Result<Eri, PbcDfError> {
        self.0.get_ao_eri(kidx, compact)
    }
    fn ao2mo_7d(&self, mos: MoKpts<'_>, factor: f64) -> Result<Eri7d, PbcDfError> {
        self.0.ao2mo_7d(mos, factor)
    }
    fn has_cderi(&self) -> bool {
        self.0.has_cderi()
    }
    fn sr_loop(&self, ki: usize, kj: usize, compact: bool) -> Result<Vec<SrBlock>, PbcDfError> {
        self.0.sr_loop(ki, kj, compact)
    }
    fn get_naoaux(&self) -> Result<usize, PbcDfError> {
        self.0.get_naoaux()
    }
    fn get_jk_e1(
        &self,
        dms: &[KMats],
        kpts: &[[f64; 3]],
        opts: JkOpts<'_>,
        mo: Option<&pyscf_pbc_df::fft_jk_grad::TaggedMo>,
    ) -> Result<pyscf_pbc_df::fft_jk_grad::GradJkResult, PbcDfError> {
        self.0.get_jk_e1(dms, kpts, opts, mo)
    }
    fn get_j_e1(
        &self,
        dms: &[KMats],
        kpts: &[[f64; 3]],
        kpts_band: Option<&[[f64; 3]]>,
    ) -> Result<pyscf_pbc_df::fft_jk_grad::GradMats, PbcDfError> {
        self.0.get_j_e1(dms, kpts, kpts_band)
    }
    fn get_k_e1(
        &self,
        dms: &[KMats],
        kpts: &[[f64; 3]],
        kpts_band: Option<&[[f64; 3]]>,
        exxdiv: Option<ExxDiv>,
        omega: Option<f64>,
        mo: Option<&pyscf_pbc_df::fft_jk_grad::TaggedMo>,
    ) -> Result<pyscf_pbc_df::fft_jk_grad::GradMats, PbcDfError> {
        self.0.get_k_e1(dms, kpts, kpts_band, exxdiv, omega, mo)
    }
}

/// The driver-side entry point: any `PeriodicDf` Python object (a native
/// builder or a Python subclass of one) → a `Box<dyn PeriodicDf>` over its
/// CURRENT builder. `TypeError` for anything else.
pub fn extract_df(obj: &Bound<'_, PyAny>) -> PyResult<Box<dyn PeriodicDf>> {
    let df = obj.cast::<PyPeriodicDf>().map_err(|_| {
        PyTypeError::new_err(
            "with_df must be a pyscf.pbc.df builder (FFTDF, AFTDF, GDF, MDF or RSDF)",
        )
    })?;
    Ok(df.borrow().boxed())
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration + the concrete builder
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum DfArc {
    Fftdf(Arc<Fftdf>),
    Aftdf(Arc<Aftdf>),
    Gdf(Arc<Gdf>),
    Mdf(Arc<Mdf>),
    Rsdf(Arc<Rsdf>),
    /// Produced by `pyscf_pbc_df::density_fit` (`density_fit.rs:73`); any
    /// configuration change rebuilds it as a concrete variant.
    Factory(Arc<dyn PeriodicDf>),
}

impl DfArc {
    fn dyn_arc(&self) -> Arc<dyn PeriodicDf> {
        match self {
            DfArc::Fftdf(a) => a.clone(),
            DfArc::Aftdf(a) => a.clone(),
            DfArc::Gdf(a) => a.clone(),
            DfArc::Mdf(a) => a.clone(),
            DfArc::Rsdf(a) => a.clone(),
            DfArc::Factory(a) => a.clone(),
        }
    }
}

#[derive(Debug, Clone)]
struct DfConfig {
    kind: DfKind,
    kpts: Vec<[f64; 3]>,
    mesh: Option<[usize; 3]>,
    auxbasis: Option<String>,
    prefer_ccdf: Option<bool>,
    j_only: bool,
    exp_to_discard: Option<f64>,
    cderi_to_save: Option<PathBuf>,
    cderi_file: Option<PathBuf>,
    max_memory: Option<f64>,
}

impl DfConfig {
    fn new(kind: DfKind, kpts: Vec<[f64; 3]>) -> Self {
        Self {
            kind,
            kpts,
            mesh: None,
            auxbasis: None,
            // Upstream `MDF._prefer_ccdf = False` (`pyscf/pbc/df/mdf.py:80`);
            // the port's `Mdf::new` defaults to `true` (`mdf/mod.rs:115`), so the
            // binding pins upstream's default. GDF already matches (`df.py:132`).
            prefer_ccdf: if kind == DfKind::Mdf {
                Some(false)
            } else {
                None
            },
            j_only: false,
            exp_to_discard: None,
            cderi_to_save: None,
            cderi_file: None,
            max_memory: None,
        }
    }
}

fn configure_gdf(g: &mut Gdf, cfg: &DfConfig) {
    g.auxbasis = cfg.auxbasis.clone();
    g.j_only = cfg.j_only;
    g.exp_to_discard = cfg.exp_to_discard;
    g.cderi_to_save = cfg.cderi_to_save.clone();
}

fn make_builder(cell: Cell, cfg: &DfConfig) -> PyResult<DfArc> {
    let k = &cfg.kpts;
    Ok(match cfg.kind {
        DfKind::Fftdf => {
            let mut d = match cfg.mesh {
                Some(m) => Fftdf::with_mesh(cell, k, m),
                None => Fftdf::new(cell, k),
            }
            .map_err(pbc_df_to_py)?;
            if let Some(mm) = cfg.max_memory {
                d.max_memory = mm;
            }
            DfArc::Fftdf(Arc::new(d))
        }
        DfKind::Aftdf => {
            let mut d = match cfg.mesh {
                Some(m) => Aftdf::with_mesh(cell, k, m),
                None => Aftdf::new(cell, k),
            }
            .map_err(pbc_df_to_py)?;
            if let Some(mm) = cfg.max_memory {
                d.max_memory = mm;
            }
            DfArc::Aftdf(Arc::new(d))
        }
        DfKind::Gdf => {
            let mut g = match &cfg.cderi_file {
                Some(p) => Gdf::load_cderi(cell, p).map_err(pbc_df_to_py)?,
                None => Gdf::new(cell, k),
            };
            configure_gdf(&mut g, cfg);
            if let Some(p) = cfg.prefer_ccdf {
                g.prefer_ccdf = p;
            }
            DfArc::Gdf(Arc::new(g))
        }
        DfKind::Rsdf => {
            let mut d = match &cfg.cderi_file {
                Some(p) => {
                    let mut g = Gdf::load_cderi(cell, p).map_err(pbc_df_to_py)?;
                    g.prefer_ccdf = false;
                    Rsdf { gdf: g }
                }
                None => Rsdf::new(cell, k),
            };
            configure_gdf(&mut d.gdf, cfg);
            DfArc::Rsdf(Arc::new(d))
        }
        DfKind::Mdf => {
            let mut d = Mdf::new(cell, k);
            d.auxbasis = cfg.auxbasis.clone();
            d.mesh = cfg.mesh;
            d.j_only = cfg.j_only;
            d.exp_to_discard = cfg.exp_to_discard;
            if let Some(p) = cfg.prefer_ccdf {
                d.prefer_ccdf = p;
            }
            DfArc::Mdf(Arc::new(d))
        }
    })
}

fn kind_label(kind: DfKind) -> &'static str {
    match kind {
        DfKind::Fftdf => "FFTDF",
        DfKind::Aftdf => "AFTDF",
        DfKind::Gdf => "GDF",
        DfKind::Mdf => "MDF",
        DfKind::Rsdf => "RSDF",
    }
}

fn parse_kind(s: &str) -> PyResult<DfKind> {
    match s.to_ascii_uppercase().as_str() {
        "FFTDF" => Ok(DfKind::Fftdf),
        "AFTDF" | "PWDF" => Ok(DfKind::Aftdf),
        "GDF" | "DF" => Ok(DfKind::Gdf),
        "MDF" => Ok(DfKind::Mdf),
        "RSDF" | "RSGDF" => Ok(DfKind::Rsdf),
        other => Err(PyValueError::new_err(format!(
            "unknown DF kind {other:?} (FFTDF, AFTDF, GDF, MDF, RSDF)"
        ))),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PyPeriodicDf
// ─────────────────────────────────────────────────────────────────────────────

/// Base class of the five periodic DF builders. Construct through `FFTDF`,
/// `AFTDF`, `GDF`, `MDF` or `RSDF`; every method lives here.
#[pyclass(
    subclass,
    name = "PeriodicDf",
    module = "pyscf._native.pbc.df",
    skip_from_py_object
)]
pub struct PyPeriodicDf {
    py_cell: Py<PyAny>,
    cfg: DfConfig,
    inner: DfArc,
}

impl PyPeriodicDf {
    fn construct(
        cell: &Bound<'_, PyAny>,
        kpts: Option<&Bound<'_, PyAny>>,
        kind: DfKind,
    ) -> PyResult<Self> {
        let rust_cell = crate::bridge::extract_cell_from_pyany(cell.py(), cell)?;
        let kpts = extract_kpts_opt(kpts)?.map_or_else(|| vec![[0.0; 3]], |(k, _)| k);
        let cfg = DfConfig::new(kind, kpts);
        let inner = make_builder(rust_cell, &cfg)?;
        Ok(Self {
            py_cell: cell.clone().unbind(),
            cfg,
            inner,
        })
    }

    /// The current builder, shared. Cheap (an `Arc` clone).
    pub fn shared(&self) -> Arc<dyn PeriodicDf> {
        self.inner.dyn_arc()
    }

    /// The current builder as the `Box<dyn PeriodicDf>` a driver stores.
    pub fn boxed(&self) -> Box<dyn PeriodicDf> {
        Box::new(SharedDf(self.shared()))
    }

    /// Rebuild the builder from `cfg` (after a configuration change).
    fn rebuild(&mut self) -> PyResult<()> {
        let cell = self.shared().cell().clone();
        self.inner = make_builder(cell, &self.cfg)?;
        Ok(())
    }

    fn kind_is(&self, kinds: &[DfKind], attr: &str) -> PyResult<()> {
        if kinds.contains(&self.cfg.kind) {
            Ok(())
        } else {
            Err(PyAttributeError::new_err(format!(
                "{} has no attribute {attr:?}",
                kind_label(self.cfg.kind)
            )))
        }
    }

    fn kidx(&self, k: &[f64; 3]) -> PyResult<usize> {
        let df = self.shared();
        df.kpts()
            .iter()
            .position(|q| (0..3).all(|i| (q[i] - k[i]).abs() < 1e-9))
            .ok_or_else(|| {
                PyValueError::new_err(format!("k-point {k:?} is not one of this DF object's kpts"))
            })
    }
}

fn exxdiv_arg(v: Option<&Bound<'_, PyAny>>) -> PyResult<Option<ExxDiv>> {
    match v {
        None => Ok(None),
        Some(o) if o.is_none() => Ok(None),
        Some(o) => {
            if let Ok(s) = o.extract::<String>() {
                let lower = s.trim().to_ascii_lowercase();
                return match ExxDiv::parse(&lower) {
                    Some(e) => Ok(Some(e)),
                    None if lower.is_empty() || lower == "none" || lower == "false" => Ok(None),
                    None => Err(PyValueError::new_err(format!(
                        "exxdiv = {s:?} is not one of None, 'ewald', 'vcut_sph', 'vcut_ws'"
                    ))),
                };
            }
            if matches!(o.extract::<bool>(), Ok(false)) {
                return Ok(None);
            }
            Err(PyTypeError::new_err("exxdiv must be None or a string"))
        }
    }
}

/// How the density matrix came in — decides the output nesting.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DmForm {
    /// `(nao, nao)` — one set, one k-point.
    Single,
    /// `(nkpts, nao, nao)` or a list of arrays — one set.
    OneSet,
    /// `(nset, nkpts, nao, nao)` or a list of lists.
    Nested,
}

/// Normalise any upstream-shaped density into the 20-07 `KDms` list form with
/// `complex128` blocks (a real input is widened exactly), and read it.
fn normalise_dms<'py>(py: Python<'py>, dm: &Bound<'py, PyAny>) -> PyResult<(Vec<KMats>, DmForm)> {
    let np = PyModule::import(py, "numpy")?;
    let c128 = np.getattr("complex128")?;
    let as_c = |x: &Bound<'py, PyAny>| -> PyResult<Bound<'py, PyAny>> {
        let kw = pyo3::types::PyDict::new(py);
        kw.set_item("dtype", &c128)?;
        np.call_method("asarray", (x,), Some(&kw))
    };
    let seq: Option<Vec<Bound<'_, PyAny>>> =
        if dm.cast::<PyList>().is_ok() || dm.cast::<PyTuple>().is_ok() {
            Some(dm.extract()?)
        } else {
            None
        };
    let (nested, form): (Vec<Vec<Bound<'_, PyAny>>>, DmForm) = match seq {
        Some(items) if items.is_empty() => {
            return Err(PyValueError::new_err("dm is an empty sequence"));
        }
        Some(items) => {
            if items[0].cast::<PyList>().is_ok() || items[0].cast::<PyTuple>().is_ok() {
                let sets = items
                    .iter()
                    .map(|s| {
                        let ks: Vec<Bound<'_, PyAny>> = s.extract()?;
                        ks.iter().map(|x| as_c(x)).collect::<PyResult<Vec<_>>>()
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                (sets, DmForm::Nested)
            } else {
                (
                    vec![
                        items
                            .iter()
                            .map(|x| as_c(x))
                            .collect::<PyResult<Vec<_>>>()?,
                    ],
                    DmForm::OneSet,
                )
            }
        }
        None => {
            let arr = as_c(dm)?;
            let ndim: usize = arr.getattr("ndim")?.extract()?;
            match ndim {
                2 => (vec![vec![arr]], DmForm::Single),
                3 => {
                    let ks: Vec<Bound<'_, PyAny>> = arr.try_iter()?.collect::<PyResult<_>>()?;
                    (vec![ks], DmForm::OneSet)
                }
                4 => {
                    let mut sets = Vec::new();
                    for s in arr.try_iter()? {
                        let s = s?;
                        sets.push(s.try_iter()?.collect::<PyResult<Vec<_>>>()?);
                    }
                    (sets, DmForm::Nested)
                }
                n => {
                    return Err(PyValueError::new_err(format!(
                        "dm must be 2-, 3- or 4-dimensional, got ndim = {n}"
                    )));
                }
            }
        }
    };
    let lists = nested
        .into_iter()
        .map(|ks| PyList::new(py, ks))
        .collect::<PyResult<Vec<_>>>()?;
    let pylist = PyList::new(py, lists)?;
    let (kdms, shapes) = to_kdms(pylist.as_any(), BufOrder::C)?;
    for (s, set) in shapes.iter().enumerate() {
        for (k, sh) in set.iter().enumerate() {
            if sh.len() != 2 || sh[0] != sh[1] {
                return Err(PyValueError::new_err(format!(
                    "dm[{s}][{k}] must be square (nao, nao), got {sh:?}"
                )));
            }
        }
    }
    Ok((kdms, form))
}

/// `Vec<CTensor>` (row-major `nao x nao`) → one array or a list.
fn kmats_out(py: Python<'_>, m: &[CTensor], nao: usize, single: bool) -> PyResult<Py<PyAny>> {
    let shapes = vec![vec![nao, nao]; m.len()];
    if single {
        let arr =
            ctensor_to_array(&m[0], &[nao, nao], BufOrder::C).map_err(PyValueError::new_err)?;
        return Ok(arr.into_pyarray(py).into_any().unbind());
    }
    Ok(kmats_to_pylist(py, m, &shapes, BufOrder::C)?
        .into_any()
        .unbind())
}

fn jk_half_out(
    py: Python<'_>,
    v: Option<Vec<KMats>>,
    nao: usize,
    nested: bool,
    single_k: bool,
) -> PyResult<Py<PyAny>> {
    let Some(v) = v else {
        return Ok(py.None());
    };
    if nested {
        if single_k {
            let items = v
                .iter()
                .map(|set| kmats_out(py, set, nao, true))
                .collect::<PyResult<Vec<_>>>()?;
            return Ok(PyList::new(py, items)?.into_any().unbind());
        }
        let shapes: Vec<Vec<Vec<usize>>> =
            v.iter().map(|s| vec![vec![nao, nao]; s.len()]).collect();
        return Ok(kdms_to_pylist(py, &v, &shapes, BufOrder::C)?
            .into_any()
            .unbind());
    }
    kmats_out(py, &v[0], nao, single_k)
}

fn mo_coeff_from(obj: &Bound<'_, PyAny>) -> PyResult<MoCoeff> {
    let py = obj.py();
    let np = PyModule::import(py, "numpy")?;
    let kw = pyo3::types::PyDict::new(py);
    kw.set_item("dtype", np.getattr("complex128")?)?;
    let arr = np.call_method("asarray", (obj,), Some(&kw))?;
    let ro: numpy::PyReadonlyArrayDyn<'_, Complex64> = arr.extract()?;
    let (t, shape) = to_ctensor(ro, BufOrder::C)?;
    if shape.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "an MO coefficient block must be (nao, nmo), got {shape:?}"
        )));
    }
    Ok(MoCoeff::new(shape[0], shape[1], t))
}

fn eri_out(py: Python<'_>, e: &Eri, real: bool) -> PyResult<Py<PyAny>> {
    let (r, c) = (e.row.len(), e.col.len());
    if real && e.data.im.iter().all(|v| *v == 0.0) {
        let arr = Array2::from_shape_vec((r, c), e.data.re.clone())
            .map_err(|x| PyValueError::new_err(x.to_string()))?;
        return Ok(arr.into_pyarray(py).into_any().unbind());
    }
    let arr = ctensor_to_array(&e.data, &[r, c], BufOrder::C).map_err(PyValueError::new_err)?;
    Ok(arr.into_pyarray(py).into_any().unbind())
}

#[pymethods]
impl PyPeriodicDf {
    fn __repr__(&self) -> String {
        let df = self.shared();
        format!(
            "<pyscf._native.pbc.df.{} name={} nkpts={}>",
            kind_label(self.cfg.kind),
            df.name(),
            df.kpts().len()
        )
    }

    /// The builder's own name (`traits.rs:132`): `"FFTDF"`, `"AFTDF"`, `"GDF"`,
    /// `"MDF"`, `"RSGDF"`.
    #[getter]
    fn name(&self) -> &'static str {
        self.shared().name()
    }

    /// The cell object this builder was constructed with (`mydf.cell is cell`).
    #[getter]
    fn cell(&self, py: Python<'_>) -> Py<PyAny> {
        self.py_cell.clone_ref(py)
    }

    /// Sampling k-points `(nkpts, 3)`.
    #[getter]
    fn kpts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        kpts_to_pyarray(py, self.shared().kpts())
    }
    #[setter]
    fn set_kpts(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        self.cfg.kpts = extract_kpts_opt(Some(v))?.map_or_else(|| vec![[0.0; 3]], |(k, _)| k);
        self.cfg.cderi_file = None;
        self.rebuild()
    }

    /// The mesh (`traits.rs` `mesh()`): FFT/AFT/MDF plane-wave mesh; for GDF/RSDF
    /// the builder's compensating/long-range mesh (read-only there).
    #[getter]
    fn mesh(&self) -> Vec<usize> {
        self.shared().mesh().to_vec()
    }
    #[setter]
    fn set_mesh(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        if matches!(self.cfg.kind, DfKind::Gdf | DfKind::Rsdf) {
            return Err(PyNotImplementedError::new_err(
                "GDF/RSDF mesh is chosen by the builder (_guess_eta/_guess_omega); setting it is not bound",
            ));
        }
        self.cfg.mesh = if v.is_none() {
            None
        } else {
            Some(extract_usize3(v, "mesh")?)
        };
        self.rebuild()
    }

    #[getter]
    fn max_memory(&self) -> PyResult<f64> {
        match &self.inner {
            DfArc::Fftdf(d) => Ok(d.max_memory),
            DfArc::Aftdf(d) => Ok(d.max_memory),
            _ => Err(PyAttributeError::new_err(
                "max_memory is bound on FFTDF/AFTDF only",
            )),
        }
    }
    #[setter]
    fn set_max_memory(&mut self, v: f64) -> PyResult<()> {
        self.kind_is(&[DfKind::Fftdf, DfKind::Aftdf], "max_memory")?;
        self.cfg.max_memory = Some(v);
        self.rebuild()
    }

    /// Auxiliary basis name (GDF/MDF/RSDF); `None` runs `make_auxbasis`.
    #[getter]
    fn auxbasis(&self) -> PyResult<Option<String>> {
        self.kind_is(&[DfKind::Gdf, DfKind::Mdf, DfKind::Rsdf], "auxbasis")?;
        Ok(self.cfg.auxbasis.clone())
    }
    #[setter]
    fn set_auxbasis(&mut self, v: Option<String>) -> PyResult<()> {
        self.kind_is(&[DfKind::Gdf, DfKind::Mdf, DfKind::Rsdf], "auxbasis")?;
        self.cfg.auxbasis = v;
        self.cfg.cderi_file = None;
        self.rebuild()
    }

    /// `exp_to_discard` (GDF/MDF/RSDF). Setting it is accepted; the builder
    /// REFUSES at build/first use (`gdf/mod.rs:264`, `mdf/mod.rs:128`).
    #[getter]
    fn exp_to_discard(&self) -> PyResult<Option<f64>> {
        self.kind_is(&[DfKind::Gdf, DfKind::Mdf, DfKind::Rsdf], "exp_to_discard")?;
        Ok(self.cfg.exp_to_discard)
    }
    #[setter]
    fn set_exp_to_discard(&mut self, v: Option<f64>) -> PyResult<()> {
        self.kind_is(&[DfKind::Gdf, DfKind::Mdf, DfKind::Rsdf], "exp_to_discard")?;
        self.cfg.exp_to_discard = v;
        self.rebuild()
    }

    /// `_j_only` (GDF/MDF/RSDF).
    #[getter(_j_only)]
    fn get_j_only_attr(&self) -> PyResult<bool> {
        self.kind_is(&[DfKind::Gdf, DfKind::Mdf, DfKind::Rsdf], "_j_only")?;
        Ok(self.cfg.j_only)
    }
    #[setter(_j_only)]
    fn set_j_only_attr(&mut self, v: bool) -> PyResult<()> {
        self.kind_is(&[DfKind::Gdf, DfKind::Mdf, DfKind::Rsdf], "_j_only")?;
        self.cfg.j_only = v;
        self.rebuild()
    }

    /// `_prefer_ccdf` (GDF/MDF). Both default to `False`, upstream's value; the
    /// binding pins it for MDF, whose crate default is `true` (`mdf/mod.rs:115`).
    #[getter(_prefer_ccdf)]
    fn get_prefer_ccdf_attr(&self) -> PyResult<bool> {
        match &self.inner {
            DfArc::Gdf(g) => Ok(g.prefer_ccdf),
            DfArc::Mdf(m) => Ok(m.prefer_ccdf),
            _ => Err(PyAttributeError::new_err(
                "_prefer_ccdf is bound on GDF/MDF only",
            )),
        }
    }
    #[setter(_prefer_ccdf)]
    fn set_prefer_ccdf_attr(&mut self, v: bool) -> PyResult<()> {
        self.kind_is(&[DfKind::Gdf, DfKind::Mdf], "_prefer_ccdf")?;
        self.cfg.prefer_ccdf = Some(v);
        self.rebuild()
    }

    /// `_cderi`: the HDF5 path the fitted tensor was written to or read from,
    /// else `None`. Assigning a path makes the builder READ that file
    /// (`Gdf::load_cderi`, `gdf/mod.rs:166`) instead of fitting — upstream's
    /// `mydf._cderi = 'f.h5'`. GDF/RSDF only.
    #[getter(_cderi)]
    fn get_cderi_attr(&self) -> PyResult<Option<String>> {
        self.kind_is(&[DfKind::Gdf, DfKind::Rsdf], "_cderi")?;
        let written = match &self.inner {
            DfArc::Gdf(g) => g.cderi_path(),
            DfArc::Rsdf(r) => r.gdf.cderi_path(),
            _ => None,
        };
        Ok(written
            .or_else(|| self.cfg.cderi_file.clone())
            .map(|p| p.to_string_lossy().into_owned()))
    }
    #[setter(_cderi)]
    fn set_cderi_attr(&mut self, v: Option<String>) -> PyResult<()> {
        self.kind_is(&[DfKind::Gdf, DfKind::Rsdf], "_cderi")?;
        self.cfg.cderi_file = v.map(PathBuf::from);
        self.rebuild()
    }

    /// `_cderi_to_save`: where `build()` writes the fitted tensor (kept after the
    /// object is dropped). GDF/RSDF only.
    #[getter(_cderi_to_save)]
    fn get_cderi_to_save_attr(&self) -> PyResult<Option<String>> {
        self.kind_is(&[DfKind::Gdf, DfKind::Rsdf], "_cderi_to_save")?;
        Ok(self
            .cfg
            .cderi_to_save
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned()))
    }
    #[setter(_cderi_to_save)]
    fn set_cderi_to_save_attr(&mut self, v: Option<String>) -> PyResult<()> {
        self.kind_is(&[DfKind::Gdf, DfKind::Rsdf], "_cderi_to_save")?;
        self.cfg.cderi_to_save = v.map(PathBuf::from);
        self.cfg.cderi_file = None;
        self.rebuild()
    }

    /// `isinstance(with_df, df.GDF)` route discriminator (`traits.rs`).
    fn has_cderi(&self) -> bool {
        self.shared().has_cderi()
    }

    /// Eagerly build what the builder caches (idempotent; every builder is
    /// otherwise lazy). Returns `self`.
    fn build<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        let inner = slf.borrow().inner.clone();
        let res: Result<(), PbcDfError> = py.detach(move || match inner {
            DfArc::Fftdf(d) => {
                let k = d.kpts.clone();
                d.ao_kpts(&k).map(|_| ())
            }
            DfArc::Aftdf(mut d) => match Arc::get_mut(&mut d) {
                Some(m) => PeriodicDf::build(m),
                None => Ok(()),
            },
            DfArc::Gdf(g) => g.cderi().map(|_| ()),
            DfArc::Rsdf(r) => r.gdf.cderi().map(|_| ()),
            DfArc::Mdf(m) => {
                m.gdf()?;
                m.aftdf()?;
                m.resolved_mesh().map(|_| ())
            }
            DfArc::Factory(mut f) => match Arc::get_mut(&mut f) {
                Some(m) => m.build(),
                None => Ok(()),
            },
        });
        res.map_err(pbc_df_to_py)?;
        Ok(slf.clone())
    }

    /// Nuclear attraction per k-point (`traits.rs` `get_nuc`). `kpts=None` or a
    /// single `(3,)` k-point → one `(nao, nao)` complex array; `(nkpts, 3)` → a
    /// list.
    #[pyo3(signature = (kpts = None))]
    fn get_nuc(&self, py: Python<'_>, kpts: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        self.one_body(py, kpts, |df, k| df.get_nuc(k))
    }

    /// GTH pseudopotential per k-point (`traits.rs` `get_pp`); shapes as `get_nuc`.
    #[pyo3(signature = (kpts = None))]
    fn get_pp(&self, py: Python<'_>, kpts: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        self.one_body(py, kpts, |df, k| df.get_pp(k))
    }

    /// `T + V_pp` (pseudopotential cell) or `T + V_ne` (all-electron) —
    /// `pyscf_pbc_df::get_hcore(&dyn PeriodicDf, kpts)` (`fftdf.rs:565`). This is
    /// where a periodic hcore lives; `Cell` has none. Shapes as `get_nuc`.
    #[pyo3(signature = (kpts = None))]
    fn get_hcore(&self, py: Python<'_>, kpts: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        self.one_body(py, kpts, |df, k| pyscf_pbc_df::get_hcore(df, k))
    }

    /// `get_jk(dm, hermi=1, kpts=None, kpts_band=None, with_j=True, with_k=True,
    /// omega=None, exxdiv=None, kk_symmetry=None)` — upstream's signature
    /// (`df.py:459`, `aft.py:707`), plus `kk_symmetry` (default
    /// `JkOpts::kk_symmetry_default()`, `traits.rs:82`).
    ///
    /// `dm`: `(nao, nao)`, `(nkpts, nao, nao)`, `(nset, nkpts, nao, nao)`, a list
    /// of per-k arrays, or a list of such lists; real input is widened to
    /// complex exactly. Returns `(vj, vk)` shaped like the input — one array for
    /// `(nao, nao)` (or a single `(3,)` `kpts_band`), a list of per-k complex
    /// arrays, or a list of per-set lists (the 20-07 `KMats`/`KDms` forms); a
    /// half not requested is `None`. `omega=0` means full Coulomb; a non-zero
    /// `omega` takes the builder's RSH route (GDF/MDF since 20-05: `omega > 0` is an
    /// AFTDF on an omega-derived mesh, `omega < 0` the range-separated builder,
    /// which refuses under `_prefer_ccdf = True`).
    #[pyo3(signature = (dm, hermi = 1, kpts = None, kpts_band = None, with_j = true, with_k = true,
                        omega = None, exxdiv = None, kk_symmetry = None))]
    #[allow(clippy::too_many_arguments)]
    fn get_jk(
        &self,
        py: Python<'_>,
        dm: &Bound<'_, PyAny>,
        hermi: i32,
        kpts: Option<&Bound<'_, PyAny>>,
        kpts_band: Option<&Bound<'_, PyAny>>,
        with_j: bool,
        with_k: bool,
        omega: Option<f64>,
        exxdiv: Option<&Bound<'_, PyAny>>,
        kk_symmetry: Option<bool>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let (dms, form) = normalise_dms(py, dm)?;
        let df = self.shared();
        let nao = df.cell().mol.nao_nr;
        let kpts = match extract_kpts_opt(kpts)? {
            Some((k, _)) => k,
            None => df.kpts().to_vec(),
        };
        for (s, set) in dms.iter().enumerate() {
            if set.len() != kpts.len() {
                return Err(PyValueError::new_err(format!(
                    "dm set {s} has {} k-blocks but {} kpts were given",
                    set.len(),
                    kpts.len()
                )));
            }
            if let Some(t) = set.iter().find(|t| t.re.len() != nao * nao) {
                return Err(PyValueError::new_err(format!(
                    "dm blocks must be ({nao}, {nao}); got {} elements",
                    t.re.len()
                )));
            }
        }
        let band = extract_kpts_opt(kpts_band)?;
        let single_k = match &band {
            Some((_, single)) => *single,
            None => form == DmForm::Single,
        };
        let exxdiv = exxdiv_arg(exxdiv)?;
        let omega = omega.filter(|w| *w != 0.0);
        let kk = kk_symmetry.unwrap_or_else(JkOpts::kk_symmetry_default);
        let res = py
            .detach(|| {
                let opts = JkOpts {
                    hermi,
                    kpts_band: band.as_ref().map(|(b, _)| b.as_slice()),
                    with_j,
                    with_k,
                    exxdiv,
                    omega,
                    kk_symmetry: kk,
                };
                df.get_jk(&dms, &kpts, opts)
            })
            .map_err(pbc_df_to_py)?;
        let nested = form == DmForm::Nested;
        let vj = jk_half_out(py, res.vj, nao, nested, single_k)?;
        let vk = jk_half_out(py, res.vk, nao, nested, single_k)?;
        Ok((vj, vk))
    }

    /// Auxiliary rank (`gdf/mod.rs:298`); raises on the exact builders.
    fn get_naoaux(&self) -> PyResult<usize> {
        self.shared().get_naoaux().map_err(pbc_df_to_py)
    }

    /// `sr_loop(kpti_kptj=None, compact=True)` (`gdf/mod.rs:290`): a list of
    /// `(LpqR, LpqI, sign)` with `LpqR`/`LpqI` shaped `(naux, ncol)`.
    /// `kpti_kptj=None` is the gamma pair.
    #[pyo3(signature = (kpti_kptj = None, compact = true))]
    fn sr_loop(
        &self,
        py: Python<'_>,
        kpti_kptj: Option<&Bound<'_, PyAny>>,
        compact: bool,
    ) -> PyResult<Py<PyList>> {
        let (ki, kj) = match extract_kpts_opt(kpti_kptj)? {
            None => (self.kidx(&[0.0; 3])?, self.kidx(&[0.0; 3])?),
            Some((k, _)) if k.len() == 2 => (self.kidx(&k[0])?, self.kidx(&k[1])?),
            Some(_) => return Err(PyValueError::new_err("kpti_kptj must have shape (2, 3)")),
        };
        let df = self.shared();
        let blocks = py
            .detach(|| df.sr_loop(ki, kj, compact))
            .map_err(pbc_df_to_py)?;
        let out = PyList::empty(py);
        for b in blocks {
            let re = Array2::from_shape_vec((b.naux, b.ncol), b.re)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let im = Array2::from_shape_vec((b.naux, b.ncol), b.im)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            out.append((re.into_pyarray(py), im.into_pyarray(py), b.sign))?;
        }
        Ok(out.unbind())
    }

    /// `get_eri(kpts=None, compact=True)` — the AO ERI block at one k-quadruple
    /// (`get_ao_eri`, `df_ao2mo.rs:496` for GDF). `kpts` is `(4, 3)` k-vectors
    /// from `self.kpts`, or one `(3,)` k-point broadcast to all four
    /// (upstream `_format_kpts`, fft_ao2mo.py:430); `None` is the gamma quadruple. Real when every k is
    /// gamma and the block has no imaginary part, else complex.
    #[pyo3(signature = (kpts = None, compact = true))]
    fn get_eri(
        &self,
        py: Python<'_>,
        kpts: Option<&Bound<'_, PyAny>>,
        compact: bool,
    ) -> PyResult<Py<PyAny>> {
        let (kidx, gamma) = self.quad(kpts)?;
        let df = self.shared();
        let e = py
            .detach(|| df.get_ao_eri(kidx, compact))
            .map_err(pbc_df_to_py)?;
        eri_out(py, &e, gamma)
    }

    /// `ao2mo(mo_coeffs, kpts=None, compact=True)` — `general`
    /// (`df_ao2mo.rs:615` for GDF). `mo_coeffs` is one `(nao, nmo)` block or
    /// four; `kpts` as in `get_eri`.
    #[pyo3(signature = (mo_coeffs, kpts = None, compact = true))]
    fn ao2mo(
        &self,
        py: Python<'_>,
        mo_coeffs: &Bound<'_, PyAny>,
        kpts: Option<&Bound<'_, PyAny>>,
        compact: bool,
    ) -> PyResult<Py<PyAny>> {
        let mos: Vec<MoCoeff> =
            if mo_coeffs.cast::<PyList>().is_ok() || mo_coeffs.cast::<PyTuple>().is_ok() {
                let items: Vec<Bound<'_, PyAny>> = mo_coeffs.extract()?;
                if items.len() != 4 {
                    return Err(PyValueError::new_err("mo_coeffs must be one block or four"));
                }
                items.iter().map(mo_coeff_from).collect::<PyResult<_>>()?
            } else {
                let m = mo_coeff_from(mo_coeffs)?;
                vec![m.clone(), m.clone(), m.clone(), m]
            };
        let (kidx, gamma) = self.quad(kpts)?;
        let df = self.shared();
        let e = py
            .detach(|| df.ao2mo([&mos[0], &mos[1], &mos[2], &mos[3]], kidx, compact))
            .map_err(pbc_df_to_py)?;
        eri_out(py, &e, gamma && mos.iter().all(MoCoeff::is_real))
    }

    /// `ao2mo_7d(mo_coeff_kpts, factor=1.0)` (`df_ao2mo.rs:738` for GDF):
    /// `mo_coeff_kpts` is a list of `nkpts` `(nao, nmo)` blocks (used for all
    /// four indices) or four such lists. Returns a complex
    /// `(nk, nk, nk, n0, n1, n2, n3)` array in upstream's index order.
    #[pyo3(signature = (mo_coeff_kpts, factor = 1.0))]
    fn ao2mo_7d(
        &self,
        py: Python<'_>,
        mo_coeff_kpts: &Bound<'_, PyAny>,
        factor: f64,
    ) -> PyResult<Py<PyAny>> {
        let outer: Vec<Bound<'_, PyAny>> = mo_coeff_kpts.extract()?;
        let lists: Vec<Vec<MoCoeff>> = if outer.len() == 4
            && (outer[0].cast::<PyList>().is_ok() || outer[0].cast::<PyTuple>().is_ok())
        {
            outer
                .iter()
                .map(|l| {
                    let items: Vec<Bound<'_, PyAny>> = l.extract()?;
                    items
                        .iter()
                        .map(mo_coeff_from)
                        .collect::<PyResult<Vec<_>>>()
                })
                .collect::<PyResult<_>>()?
        } else {
            let one = outer
                .iter()
                .map(mo_coeff_from)
                .collect::<PyResult<Vec<_>>>()?;
            vec![one.clone(), one.clone(), one.clone(), one]
        };
        let df = self.shared();
        let e = py
            .detach(|| df.ao2mo_7d([&lists[0], &lists[1], &lists[2], &lists[3]], factor))
            .map_err(pbc_df_to_py)?;
        let nk = e.nkpts;
        let shape = [nk, nk, nk, e.nmo[0], e.nmo[1], e.nmo[2], e.nmo[3]];
        let arr: ArrayD<Complex64> =
            ctensor_to_array(&e.data, &shape, BufOrder::C).map_err(PyValueError::new_err)?;
        Ok(arr.into_pyarray(py).into_any().unbind())
    }
}

impl PyPeriodicDf {
    fn one_body<F>(
        &self,
        py: Python<'_>,
        kpts: Option<&Bound<'_, PyAny>>,
        f: F,
    ) -> PyResult<Py<PyAny>>
    where
        F: FnOnce(&dyn PeriodicDf, &[[f64; 3]]) -> Result<Vec<CTensor>, PbcDfError> + Send,
    {
        let (k, single) = extract_kpts_opt(kpts)?.unwrap_or_else(|| (vec![[0.0; 3]], true));
        let df = self.shared();
        let nao = df.cell().mol.nao_nr;
        let out = py.detach(|| f(df.as_ref(), &k)).map_err(pbc_df_to_py)?;
        kmats_out(py, &out, nao, single)
    }

    fn quad(&self, kpts: Option<&Bound<'_, PyAny>>) -> PyResult<([usize; 4], bool)> {
        match extract_kpts_opt(kpts)? {
            None => {
                let g = self.kidx(&[0.0; 3])?;
                Ok(([g; 4], true))
            }
            // 20-18: upstream `_format_kpts` (pyscf/pbc/df/fft_ao2mo.py:430-439,
            // shared by aft_ao2mo.py:41/130 and df_ao2mo.py:39/114) broadcasts a
            // single k-point (`kpts.size == 3`, i.e. `(3,)` or `(1, 3)`) to all
            // four indices: `numpy.vstack([kpts]*4)`.
            Some((k, _)) if k.len() == 1 => {
                let gamma = k[0].iter().all(|x| x.abs() < 1e-9);
                Ok(([self.kidx(&k[0])?; 4], gamma))
            }
            Some((k, _)) if k.len() == 4 => {
                let gamma = k.iter().all(|q| q.iter().all(|x| x.abs() < 1e-9));
                Ok((
                    [
                        self.kidx(&k[0])?,
                        self.kidx(&k[1])?,
                        self.kidx(&k[2])?,
                        self.kidx(&k[3])?,
                    ],
                    gamma,
                ))
            }
            Some(_) => Err(PyValueError::new_err(
                "kpts must be one k-point, shape (3,), or four k-points, shape (4, 3)",
            )),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The five constructors
// ─────────────────────────────────────────────────────────────────────────────

macro_rules! df_subclass {
    ($ty:ident, $name:literal, $kind:expr, $doc:literal) => {
        #[doc = $doc]
        #[pyclass(extends = PyPeriodicDf, subclass, name = $name, module = "pyscf._native.pbc.df")]
        pub struct $ty {}

        #[pymethods]
        impl $ty {
            #[new]
            #[pyo3(signature = (cell, kpts = None))]
            fn new(
                cell: &Bound<'_, PyAny>,
                kpts: Option<&Bound<'_, PyAny>>,
            ) -> PyResult<PyClassInitializer<Self>> {
                let base = PyPeriodicDf::construct(cell, kpts, $kind)?;
                Ok(PyClassInitializer::from(base).add_subclass($ty {}))
            }
        }
    };
}

df_subclass!(
    PyFftdf,
    "FFTDF",
    DfKind::Fftdf,
    "`FFTDF(cell, kpts=None)` — `fftdf.rs:76`."
);
df_subclass!(
    PyAftdf,
    "AFTDF",
    DfKind::Aftdf,
    "`AFTDF(cell, kpts=None)` (alias `PWDF`) — `aftdf.rs:53`."
);
df_subclass!(
    PyGdf,
    "GDF",
    DfKind::Gdf,
    "`GDF(cell, kpts=None)` (alias `DF`) — `gdf/mod.rs:64`."
);
df_subclass!(
    PyMdf,
    "MDF",
    DfKind::Mdf,
    "`MDF(cell, kpts=None)` — `mdf/mod.rs:72`."
);
df_subclass!(
    PyRsdf,
    "RSDF",
    DfKind::Rsdf,
    "`RSDF(cell, kpts=None)` (alias `RSGDF`) — `rsdf.rs:56`."
);

/// `density_fit(cell, kpts=None, kind='GDF', auxbasis=None, mesh=None)` — one
/// builder from `pyscf_pbc_df::density_fit` (`density_fit.rs:73`), returned as
/// an instance of the matching class (`FFTDF`/`AFTDF`/`GDF`/`MDF`/`RSDF`).
/// This is the Rust half of `mf.density_fit()`; the driver graft is 20-12's.
#[pyfunction(signature = (cell, kpts = None, kind = "GDF", auxbasis = None, mesh = None))]
fn density_fit(
    py: Python<'_>,
    cell: &Bound<'_, PyAny>,
    kpts: Option<&Bound<'_, PyAny>>,
    kind: &str,
    auxbasis: Option<String>,
    mesh: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let kind = parse_kind(kind)?;
    let rust_cell = crate::bridge::extract_cell_from_pyany(py, cell)?;
    let klist = extract_kpts_opt(kpts)?.map_or_else(|| vec![[0.0; 3]], |(k, _)| k);
    let mesh = match mesh {
        Some(m) if !m.is_none() => Some(extract_usize3(m, "mesh")?),
        _ => None,
    };
    let mut cfg = DfConfig::new(kind, klist.clone());
    cfg.auxbasis = auxbasis.clone();
    // `density_fit` ignores `mesh` for GDF/RSDF (`density_fit.rs:66`).
    cfg.mesh = if matches!(kind, DfKind::Gdf | DfKind::Rsdf) {
        None
    } else {
        mesh
    };
    let inner = if kind == DfKind::Mdf {
        // The factory's `Mdf::new` carries the port's `prefer_ccdf = true`;
        // build MDF concretely so upstream's `_prefer_ccdf = False` holds.
        make_builder(rust_cell, &cfg)?
    } else {
        let boxed = pyscf_pbc_df::density_fit(rust_cell, &klist, kind, DfOpts { auxbasis, mesh })
            .map_err(pbc_df_to_py)?;
        DfArc::Factory(Arc::from(boxed))
    };
    let base = PyPeriodicDf {
        py_cell: cell.clone().unbind(),
        cfg,
        inner,
    };
    let init = PyClassInitializer::from(base);
    Ok(match kind {
        DfKind::Fftdf => Py::new(py, init.add_subclass(PyFftdf {}))?.into_any(),
        DfKind::Aftdf => Py::new(py, init.add_subclass(PyAftdf {}))?.into_any(),
        DfKind::Gdf => Py::new(py, init.add_subclass(PyGdf {}))?.into_any(),
        DfKind::Mdf => Py::new(py, init.add_subclass(PyMdf {}))?.into_any(),
        DfKind::Rsdf => Py::new(py, init.add_subclass(PyRsdf {}))?.into_any(),
    })
}

/// PRIVATE self-test hook for the driver ownership contract (plan 20-10): takes
/// ANY Python object, goes through [`extract_df`] exactly as a 20-12 driver
/// will, and runs `get_jk` (`hermi=1`, both halves, `exxdiv`) on the resulting
/// `Box<dyn PeriodicDf>` — the [`SharedDf`] adapter. Returns
/// `(name, mesh, vj, vk)` with `vj`/`vk` as per-k lists for one density set
/// (`dm` a list of `(nao, nao)` arrays at the builder's own k-points).
#[pyfunction(signature = (with_df, dm, exxdiv = None))]
fn _driver_handle_get_jk(
    py: Python<'_>,
    with_df: &Bound<'_, PyAny>,
    dm: &Bound<'_, PyAny>,
    exxdiv: Option<&Bound<'_, PyAny>>,
) -> PyResult<(String, Vec<usize>, Py<PyAny>, Py<PyAny>)> {
    let mut boxed = extract_df(with_df)?;
    boxed.build().map_err(pbc_df_to_py)?;
    let (dms, _) = normalise_dms(py, dm)?;
    let nao = boxed.cell().mol.nao_nr;
    let kpts = boxed.kpts().to_vec();
    let exxdiv = exxdiv_arg(exxdiv)?;
    let res = py
        .detach(|| {
            let mut opts = JkOpts::hermitian();
            opts.exxdiv = exxdiv;
            boxed
                .get_jk(&dms, &kpts, opts)
                .map(|r| (r, boxed.name(), boxed.mesh()))
        })
        .map_err(pbc_df_to_py)?;
    let (r, name, mesh) = res;
    let vj = jk_half_out(py, r.vj, nao, false, false)?;
    let vk = jk_half_out(py, r.vk, nao, false, false)?;
    Ok((name.to_string(), mesh.to_vec(), vj, vk))
}
