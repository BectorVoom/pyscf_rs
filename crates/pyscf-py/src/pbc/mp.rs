//! `pyscf._native.pbc.mp` — periodic MP2 (plan 20-15).
//!
//! # Classes
//!
//! | Python | Rust | notes |
//! |---|---|---|
//! | `KMP2(mf, frozen=None)` (`KRMP2`) | `Kmp2` (`kmp2.rs:37`) | a `KsymAdaptedKRHF` mean field selects `KsymAdaptedKmp2` |
//! | `KsymAdaptedKMP2(mf, frozen=None)` | `KsymAdaptedKmp2` (`kmp2_ksymm.rs:224`) | requires a k-symmetric mean field |
//! | `KUMP2(mf, frozen=None)` | `Kump2` (`kump2.rs:25`) | bookkeeping only; `kernel` raises (upstream parity, `kump2.py:38`) |
//! | `KMP2_stagger(mf, frozen=None, flag_submesh=False)` | `Kmp2Stagger` (`kmp2_stagger.rs:117`) | |
//!
//! Every driver takes an ALREADY-CONVERGED `pyscf.pbc.scf` driver (plan 20-12)
//! and reads its stored `KScfResult` plus its `with_df` builder
//! ([`PyKscf::post_scf_input`]); nothing here re-runs SCF.
//!
//! # The `KPoints` dispatch
//!
//! Upstream `pbc/mp/__init__.py:KRMP2` is a FUNCTION branching on
//! `isinstance(mf.kpts, KPoints)`. The Phase-20 identity gate requires
//! `pyscf.pbc.mp.KMP2 is pyscf._native.pbc.mp.KMP2`, so — as 20-12 did for
//! `KRHF` — the constructor applies the rule: `KMP2(ksymm_mf)` runs the
//! k-symmetric route. `type(...)` stays `KMP2`.
//!
//! # Frozen specifications (`FrozenK` / `FrozenU`)
//!
//! `None`, an `int` (lowest `n` at every k-point), a list of `int` (the same
//! indices at every k-point) or a list of per-k lists. `KUMP2` additionally
//! takes a 2-sequence of such specs, one per spin (`kump2.py:175-189`).
//! `'auto'`/window forms raise (`PbcMpError::UnsupportedFrozen`).

use numpy::{Complex64, PyArray1, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyNotImplementedError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use pyscf_algebra::CTensor;
use pyscf_mp2::Frozen;
use pyscf_pbc_df::PeriodicDf;
use pyscf_pbc_mp::{
    FrozenK, FrozenU, KCount, Kmp2, Kmp2Result, Kmp2Stagger, KsymAdaptedKmp2, Kump2, PartialT2,
    PbcMpError, Rdm2, RdmKind, T2, unfold_kscf_result,
};
use pyscf_pbc_scf::{KScfResult, Krhf};

use crate::errors::{PyscfRsRuntimeError, pyscf_to_py};
use crate::numpy_io::{BufOrder, ctensor_to_pyarray, to_ctensor};
use crate::pbc::df::{extract_df, pbc_df_to_py};
use crate::pbc::scf::{PostScfInput, PyKscf};

/// Register the MP2 drivers on `pyscf._native.pbc.mp`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKmp2>()?;
    m.add_class::<PyKsymAdaptedKmp2>()?;
    m.add_class::<PyKump2>()?;
    m.add_class::<PyKmp2Stagger>()?;
    m.add("KRMP2", m.getattr("KMP2")?)?;
    m.add_function(wrap_pyfunction!(_rust_reference_kmp2, m)?)?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared post-SCF plumbing (also used by `pbc/cc.rs` and `pbc/ci.rs`)
// ─────────────────────────────────────────────────────────────────────────────

/// The converged mean field a correlated driver was given.
///
/// # Errors
/// `TypeError` if `mf` is not a `pyscf.pbc.scf` K-point driver, `ValueError`
/// before its `kernel()`.
pub(crate) fn mean_field(py: Python<'_>, mf: &Bound<'_, PyAny>) -> PyResult<PostScfInput> {
    let scf = mf.cast::<PyKscf>().map_err(|_| {
        PyTypeError::new_err(
            "expected a converged pyscf.pbc.scf K-point driver (KRHF, KsymAdaptedKRHF, KUHF, \
             KROHF, KGHF); KS drivers from pyscf.pbc.dft are not accepted by the 20-15 bindings",
        )
    })?;
    scf.borrow().post_scf_input(py)
}

/// `PbcMpError` → Python. Upstream refusals become `NotImplementedError`.
pub(crate) fn pbc_mp_to_py(err: PbcMpError) -> PyErr {
    match err {
        PbcMpError::Core(e) => pyscf_to_py(e),
        PbcMpError::Df(e) => pbc_df_to_py(e),
        e @ (PbcMpError::Kump2NotImplemented | PbcMpError::UnsupportedFrozen) => {
            PyNotImplementedError::new_err(e.to_string())
        }
        PbcMpError::Shape { what } if what.contains("NotImplementedError") => {
            PyNotImplementedError::new_err(what)
        }
        other => PyscfRsRuntimeError::new_err((other.to_string(), "PbcMp", Vec::<String>::new())),
    }
}

fn is_int(obj: &Bound<'_, PyAny>) -> bool {
    !obj.is_instance_of::<pyo3::types::PyBool>() && obj.extract::<i64>().is_ok()
}

fn seq_items<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Option<Vec<Bound<'py, PyAny>>>> {
    if obj.is_instance_of::<pyo3::types::PyString>() {
        return Ok(None);
    }
    match obj.try_iter() {
        Ok(it) => Ok(Some(it.collect::<PyResult<Vec<_>>>()?)),
        Err(_) => Ok(None),
    }
}

fn usize_list(items: &[Bound<'_, PyAny>], what: &str) -> PyResult<Vec<usize>> {
    items
        .iter()
        .map(|x| {
            x.extract::<usize>()
                .map_err(|_| PyTypeError::new_err(format!("{what}: orbital indices must be ints")))
        })
        .collect()
}

/// One k-resolved frozen spec (`kmp2.py:401-458`).
///
/// # Errors
/// `TypeError` on an unsupported form; `NotImplementedError` for `'auto'`.
pub(crate) fn frozen_k_from_py(obj: Option<&Bound<'_, PyAny>>) -> PyResult<FrozenK> {
    let Some(obj) = obj.filter(|o| !o.is_none()) else {
        return Ok(FrozenK::default());
    };
    if is_int(obj) {
        return Ok(FrozenK::Uniform(Frozen::Count(obj.extract::<usize>()?)));
    }
    if obj.is_instance_of::<pyo3::types::PyString>() {
        return Err(PyNotImplementedError::new_err(
            "frozen='auto'/'window' at k-points is not implemented (PbcMpError::UnsupportedFrozen)",
        ));
    }
    let items = seq_items(obj)?.ok_or_else(|| {
        PyTypeError::new_err("frozen must be None, an int, a list of ints or a list of per-k lists")
    })?;
    if items.is_empty() || items.iter().all(is_int) {
        return Ok(FrozenK::Uniform(Frozen::List(usize_list(
            &items, "frozen",
        )?)));
    }
    let per_k = items
        .iter()
        .map(|k| {
            let inner = seq_items(k)?.ok_or_else(|| {
                PyTypeError::new_err("frozen: a per-k entry must be a list of ints")
            })?;
            usize_list(&inner, "frozen")
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(FrozenK::PerKpt(per_k))
}

/// `KUMP2`'s frozen spec (`kump2.py:175-189`): an `int` shared by both spins,
/// or a 2-sequence of per-spin k-resolved specs.
fn frozen_u_from_py(obj: Option<&Bound<'_, PyAny>>) -> PyResult<FrozenU> {
    let Some(obj) = obj.filter(|o| !o.is_none()) else {
        return Ok(FrozenU::default());
    };
    if is_int(obj) {
        return Ok(FrozenU::Both(frozen_k_from_py(Some(obj))?));
    }
    let items = seq_items(obj)?
        .ok_or_else(|| PyTypeError::new_err("KUMP2 frozen must be None, an int or a 2-sequence"))?;
    if items.len() == 2 && !items.iter().any(is_int) {
        return Ok(FrozenU::PerSpin(
            frozen_k_from_py(Some(&items[0]))?,
            frozen_k_from_py(Some(&items[1]))?,
        ));
    }
    Ok(FrozenU::Both(frozen_k_from_py(Some(obj))?))
}

/// Any array-like → `(CTensor row-major, shape)` via `numpy.asarray(x, complex128)`.
pub(crate) fn complex_from_py(obj: &Bound<'_, PyAny>) -> PyResult<(CTensor, Vec<usize>)> {
    let np = obj.py().import("numpy")?;
    let arr = np.call_method1("ascontiguousarray", (obj, "complex128"))?;
    let ro: PyReadonlyArrayDyn<'_, Complex64> = arr.extract()?;
    to_ctensor(ro, BufOrder::C)
}

/// A row-major `CTensor` → complex numpy array of `shape`.
pub(crate) fn complex_to_py<'py>(
    py: Python<'py>,
    t: &CTensor,
    shape: &[usize],
) -> PyResult<Bound<'py, PyArrayDyn<Complex64>>> {
    ctensor_to_pyarray(py, t, shape, BufOrder::C)
}

fn t2_to_py<'py>(py: Python<'py>, t2: &T2) -> PyResult<Bound<'py, PyAny>> {
    let n = t2.nocc * t2.nocc * t2.nvir * t2.nvir;
    let mut flat = CTensor::zeros(t2.blocks.len() * n);
    for (b, blk) in t2.blocks.iter().enumerate() {
        flat.re[b * n..(b + 1) * n].copy_from_slice(&blk.re);
        flat.im[b * n..(b + 1) * n].copy_from_slice(&blk.im);
    }
    let nk = t2.nkpts;
    Ok(complex_to_py(py, &flat, &[nk, nk, nk, t2.nocc, t2.nocc, t2.nvir, t2.nvir])?.into_any())
}

fn t2_from_py(obj: &Bound<'_, PyAny>) -> PyResult<T2> {
    let (flat, shape) = complex_from_py(obj)?;
    if shape.len() != 7 || shape[0] != shape[1] || shape[1] != shape[2] {
        return Err(PyValueError::new_err(format!(
            "t2 must be (nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir), got {shape:?}"
        )));
    }
    let (nk, no, nv) = (shape[0], shape[3], shape[5]);
    let n = no * no * nv * nv;
    let blocks = (0..nk * nk * nk)
        .map(|b| CTensor {
            re: flat.re[b * n..(b + 1) * n].to_vec(),
            im: flat.im[b * n..(b + 1) * n].to_vec(),
        })
        .collect();
    Ok(T2 {
        nkpts: nk,
        nocc: no,
        nvir: nv,
        blocks,
    })
}

fn rdm_kind(kind: &str) -> PyResult<RdmKind> {
    match kind {
        "compact" => Ok(RdmKind::Compact),
        "padded" => Ok(RdmKind::Padded),
        _ => Err(PyValueError::new_err(
            "The 'kind' argument should be either 'compact' or 'padded'",
        )),
    }
}

/// Per-k square blocks → a list of `(n, n)` complex arrays.
fn square_blocks_to_py<'py>(py: Python<'py>, blocks: &[CTensor]) -> PyResult<Bound<'py, PyList>> {
    let arrays = blocks
        .iter()
        .map(|b| {
            let n = (b.re.len() as f64).sqrt().round() as usize;
            complex_to_py(py, b, &[n, n])
        })
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, arrays)
}

// ─────────────────────────────────────────────────────────────────────────────
// KMP2 / KsymAdaptedKMP2
// ─────────────────────────────────────────────────────────────────────────────

/// `KMP2(mf, frozen=None, mo_coeff=None, mo_occ=None)` — restricted k-point
/// MP2 on a converged `KRHF` (`kmp2.py:692`). A `KsymAdaptedKRHF` mean field
/// runs the k-symmetric route (`kmp2_ksymm.py:226`).
///
/// `kernel(mo_energy=None, mo_coeff=None, with_t2=True)` → `(e_corr, t2)`.
/// Results: `e_hf`, `e_corr`, `e_corr_ss`, `e_corr_os`, `e_tot`, `t2`
/// (`(nk, nk, nk, nocc, nocc, nvir, nvir)` complex, or `None`).
#[pyclass(
    subclass,
    dict,
    name = "KMP2",
    module = "pyscf._native.pbc.mp",
    skip_from_py_object
)]
pub struct PyKmp2 {
    mf: Py<PyAny>,
    frozen: Py<PyAny>,
    max_memory: f64,
    res: Option<Kmp2Result>,
}

impl PyKmp2 {
    fn construct(
        py: Python<'_>,
        mf: &Bound<'_, PyAny>,
        frozen: Option<&Bound<'_, PyAny>>,
        mo_coeff: Option<&Bound<'_, PyAny>>,
        mo_occ: Option<&Bound<'_, PyAny>>,
        force_ksymm: bool,
    ) -> PyResult<Self> {
        if mo_coeff.is_some_and(|x| !x.is_none()) || mo_occ.is_some_and(|x| !x.is_none()) {
            return Err(PyNotImplementedError::new_err(
                "KMP2(mf, mo_coeff=..., mo_occ=...): only the mean field's own orbitals are bound",
            ));
        }
        let input = mean_field(py, mf)?;
        if input.kind != "KRHF" && input.kind != "KsymAdaptedKRHF" {
            return Err(PyTypeError::new_err(format!(
                "KMP2 needs a restricted KRHF mean field, got {}",
                input.kind
            )));
        }
        if force_ksymm && input.kpoints.is_none() {
            return Err(PyTypeError::new_err(
                "KsymAdaptedKMP2 needs a k-symmetric mean field (KRHF(cell, kpts=KPoints))",
            ));
        }
        frozen_k_from_py(frozen)?;
        Ok(Self {
            mf: mf.clone().unbind(),
            frozen: frozen.map_or_else(|| py.None(), |f| f.clone().unbind()),
            max_memory: 4_000.0,
            res: None,
        })
    }

    fn with_driver<R: Send>(
        &self,
        py: Python<'_>,
        body: impl FnOnce(Route<'_, '_>, &pyscf_pbc_gto::Cell) -> Result<R, PbcMpError> + Send,
    ) -> PyResult<R> {
        let input = mean_field(py, self.mf.bind(py))?;
        let df = extract_df(input.with_df.bind(py))?;
        let frozen = frozen_k_from_py(Some(self.frozen.bind(py)))?;
        let max_memory = self.max_memory;
        let (result, kpoints) = (input.result, input.kpoints);
        py.detach(move || {
            let df: &dyn PeriodicDf = df.as_ref();
            let cell = df.cell();
            match &kpoints {
                Some(kp) => {
                    let full = unfold_kscf_result(&result, kp, cell)?;
                    let mut mp = KsymAdaptedKmp2::new(&full, df, kp)?;
                    mp.set_frozen(frozen);
                    mp.mp.max_memory = max_memory;
                    body(Route::Ksymm(&mp), cell)
                }
                None => {
                    let mut mp = Kmp2::new(&result, df)?;
                    mp.frozen = frozen;
                    mp.max_memory = max_memory;
                    body(Route::Full(&mut mp), cell)
                }
            }
        })
        .map_err(pbc_mp_to_py)
    }
}

/// The Rust driver one call runs on.
enum Route<'a, 'b> {
    Full(&'b mut Kmp2<'a>),
    Ksymm(&'b KsymAdaptedKmp2<'a>),
}

#[pymethods]
impl PyKmp2 {
    #[new]
    #[pyo3(signature = (mf, frozen = None, mo_coeff = None, mo_occ = None))]
    fn new(
        py: Python<'_>,
        mf: &Bound<'_, PyAny>,
        frozen: Option<&Bound<'_, PyAny>>,
        mo_coeff: Option<&Bound<'_, PyAny>>,
        mo_occ: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Self::construct(py, mf, frozen, mo_coeff, mo_occ, false)
    }

    /// The mean field (`mp._scf`).
    #[getter]
    fn _scf(&self, py: Python<'_>) -> Py<PyAny> {
        self.mf.clone_ref(py)
    }
    #[getter]
    fn cell(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self.mf.bind(py).getattr("cell")?.unbind())
    }
    #[getter]
    fn kpts(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self.mf.bind(py).getattr("kpts")?.unbind())
    }
    #[getter]
    fn nkpts(&self, py: Python<'_>) -> PyResult<usize> {
        let input = mean_field(py, self.mf.bind(py))?;
        Ok(input
            .kpoints
            .as_ref()
            .map_or(input.result.nkpts, |k| k.nkpts()))
    }
    /// `True` when the builder carries `_cderi` (GDF/RSDF), `kmp2.py:707`.
    #[getter]
    fn with_df_ints(&self, py: Python<'_>) -> PyResult<bool> {
        let input = mean_field(py, self.mf.bind(py))?;
        Ok(extract_df(input.with_df.bind(py))?.has_cderi())
    }
    #[getter]
    fn frozen(&self, py: Python<'_>) -> Py<PyAny> {
        self.frozen.clone_ref(py)
    }
    #[setter]
    fn set_frozen(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        frozen_k_from_py(Some(v))?;
        self.frozen = v.clone().unbind();
        self.res = None;
        Ok(())
    }
    #[getter]
    fn max_memory(&self) -> f64 {
        self.max_memory
    }
    #[setter]
    fn set_max_memory(&mut self, v: f64) {
        self.max_memory = v;
    }

    /// `get_nocc(per_kpoint=False)` (`kmp2.py:401`).
    #[pyo3(signature = (per_kpoint = false))]
    fn get_nocc<'py>(&self, py: Python<'py>, per_kpoint: bool) -> PyResult<Bound<'py, PyAny>> {
        let c = self.with_driver(py, move |r, _| {
            let (mf, frozen) = match r {
                Route::Full(mp) => (mp.mf, mp.frozen.clone()),
                Route::Ksymm(mp) => (mp.mp.mf, mp.mp.frozen.clone()),
            };
            pyscf_pbc_mp::get_nocc(mf.mo_occ, &frozen, per_kpoint)
        })?;
        kcount_to_py(py, c)
    }

    /// `get_nmo(per_kpoint=False)` (`kmp2.py:461`).
    #[pyo3(signature = (per_kpoint = false))]
    fn get_nmo<'py>(&self, py: Python<'py>, per_kpoint: bool) -> PyResult<Bound<'py, PyAny>> {
        let c = self.with_driver(py, move |r, _| {
            let (mf, frozen) = match r {
                Route::Full(mp) => (mp.mf, mp.frozen.clone()),
                Route::Ksymm(mp) => (mp.mp.mf, mp.mp.frozen.clone()),
            };
            pyscf_pbc_mp::get_nmo(mf.mo_occ, &frozen, per_kpoint)
        })?;
        kcount_to_py(py, c)
    }

    #[getter(nocc)]
    fn nocc_attr<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.get_nocc(py, false)
    }
    #[getter(nmo)]
    fn nmo_attr<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.get_nmo(py, false)
    }

    /// `get_frozen_mask()` → per-k boolean arrays (`kmp2.py:517`).
    fn get_frozen_mask<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let masks = self.with_driver(py, |r, _| {
            let (mf, frozen) = match r {
                Route::Full(mp) => (mp.mf, mp.frozen.clone()),
                Route::Ksymm(mp) => (mp.mp.mf, mp.mp.frozen.clone()),
            };
            pyscf_pbc_mp::get_frozen_mask(mf.mo_occ, &frozen)
        })?;
        PyList::new(py, masks.iter().map(|m| PyArray1::from_slice(py, m)))
    }

    /// `kernel(mo_energy=None, mo_coeff=None, with_t2=None)` → `(e_corr, t2)`.
    ///
    /// `with_t2` defaults to upstream's per-route default: `True` for the
    /// full-BZ route (`kmp2.py:43`), `False` for the k-symmetric one
    /// (`kmp2_ksymm.py:28`), where `True` runs the full-BZ kernel exactly as
    /// upstream's `kernel_with_t2` does.
    #[pyo3(signature = (mo_energy = None, mo_coeff = None, with_t2 = None))]
    fn kernel<'py>(
        &mut self,
        py: Python<'py>,
        mo_energy: Option<&Bound<'py, PyAny>>,
        mo_coeff: Option<&Bound<'py, PyAny>>,
        with_t2: Option<bool>,
    ) -> PyResult<(f64, Bound<'py, PyAny>)> {
        if mo_energy.is_some_and(|x| !x.is_none()) || mo_coeff.is_some_and(|x| !x.is_none()) {
            return Err(PyNotImplementedError::new_err(
                "KMP2.kernel(mo_energy=..., mo_coeff=...): only the mean field's own orbitals \
                 are bound",
            ));
        }
        let res = self.with_driver(py, move |r, cell| match r {
            Route::Full(mp) => {
                mp.with_t2 = with_t2.unwrap_or(true);
                mp.kernel()
            }
            Route::Ksymm(mp) => {
                if with_t2.unwrap_or(false) {
                    mp.kernel_with_t2()
                } else {
                    mp.kernel(cell)
                }
            }
        })?;
        let e = res.e_corr;
        self.res = Some(res);
        let t2 = self.t2(py)?;
        Ok((e, t2))
    }

    #[getter]
    fn e_hf(&self) -> Option<f64> {
        self.res.as_ref().map(|r| r.e_hf)
    }
    #[getter]
    fn e_corr(&self) -> Option<f64> {
        self.res.as_ref().map(|r| r.e_corr)
    }
    #[getter]
    fn e_corr_ss(&self) -> Option<f64> {
        self.res.as_ref().map(|r| r.e_corr_ss)
    }
    #[getter]
    fn e_corr_os(&self) -> Option<f64> {
        self.res.as_ref().map(|r| r.e_corr_os)
    }
    #[getter]
    fn e_tot(&self) -> Option<f64> {
        self.res.as_ref().map(|r| r.e_tot)
    }
    #[getter]
    fn t2<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.res.as_ref().and_then(|r| r.t2.as_ref()) {
            Some(t2) => t2_to_py(py, t2),
            None => Ok(py.None().into_bound(py)),
        }
    }

    /// `make_rdm1(t2=None, kind='compact')` → per-k `(n, n)` complex arrays
    /// (`kmp2.py:562`; k-symmetric: `kmp2_ksymm.py:129`, IBZ-sized).
    #[pyo3(signature = (t2 = None, kind = "compact"))]
    fn make_rdm1<'py>(
        &self,
        py: Python<'py>,
        t2: Option<&Bound<'py, PyAny>>,
        kind: &str,
    ) -> PyResult<Bound<'py, PyList>> {
        let kind = rdm_kind(kind)?;
        let given = match t2.filter(|x| !x.is_none()) {
            Some(x) => Some(t2_from_py(x)?),
            None => None,
        };
        let stored = self.res.as_ref().and_then(|r| r.t2.clone());
        let blocks = self.with_driver(py, move |r, cell| match r {
            Route::Full(mp) => {
                let t2 = given.or(stored).ok_or_else(|| PbcMpError::Shape {
                    what: "KMP2.make_rdm1: no t2 — run kernel(with_t2=True) or pass t2".into(),
                })?;
                mp.make_rdm1(&t2, kind)
            }
            Route::Ksymm(mp) => {
                let partial = match given {
                    Some(t2) => PartialT2::from_full(&t2),
                    None => mp.make_t2_for_rdm1(cell)?,
                };
                mp.make_rdm1(&partial, kind)
            }
        })?;
        square_blocks_to_py(py, &blocks)
    }

    /// `make_rdm2(t2=None, kind='compact')` (`kmp2.py:600`) — full BZ only;
    /// the k-symmetric route raises (upstream `kmp2_ksymm.py:253`).
    /// `'padded'` → `(nk, nk, nk, nmo, nmo, nmo, nmo)`; `'compact'` → the
    /// same stacked array when every k-point has the same orbital count, else
    /// a flat list of `nk³` blocks `(kp, kq, kr)`.
    #[pyo3(signature = (t2 = None, kind = "compact"))]
    fn make_rdm2<'py>(
        &self,
        py: Python<'py>,
        t2: Option<&Bound<'py, PyAny>>,
        kind: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kind = rdm_kind(kind)?;
        let given = match t2.filter(|x| !x.is_none()) {
            Some(x) => Some(t2_from_py(x)?),
            None => None,
        };
        let stored = self.res.as_ref().and_then(|r| r.t2.clone());
        let (rdm, nk) = self.with_driver(py, move |r, _| match r {
            Route::Full(mp) => {
                let t2 = given.or(stored).ok_or_else(|| PbcMpError::Shape {
                    what: "KMP2.make_rdm2: no t2 — run kernel(with_t2=True) or pass t2".into(),
                })?;
                Ok((mp.make_rdm2(&t2, kind)?, t2.nkpts))
            }
            Route::Ksymm(mp) => mp.make_rdm2().map(|r| (r, 0)),
        })?;
        match rdm {
            Rdm2::Padded { nmo, data } => {
                Ok(complex_to_py(py, &data, &[nk, nk, nk, nmo, nmo, nmo, nmo])?.into_any())
            }
            Rdm2::Compact(blocks) => {
                let n4 = blocks.first().map_or(0, |b| b.re.len());
                let n = (n4 as f64).powf(0.25).round() as usize;
                if n.pow(4) == n4 && blocks.iter().all(|b| b.re.len() == n4) {
                    let mut flat = CTensor::zeros(blocks.len() * n4);
                    for (i, b) in blocks.iter().enumerate() {
                        flat.re[i * n4..(i + 1) * n4].copy_from_slice(&b.re);
                        flat.im[i * n4..(i + 1) * n4].copy_from_slice(&b.im);
                    }
                    Ok(complex_to_py(py, &flat, &[nk, nk, nk, n, n, n, n])?.into_any())
                } else {
                    let arrays = blocks
                        .iter()
                        .map(|b| complex_to_py(py, b, &[b.re.len()]))
                        .collect::<PyResult<Vec<_>>>()?;
                    Ok(PyList::new(py, arrays)?.into_any())
                }
            }
        }
    }

    /// `run(**attrs)` → self.
    #[pyo3(signature = (**kwargs))]
    fn run<'py>(
        slf: &Bound<'py, Self>,
        kwargs: Option<&Bound<'py, pyo3::types::PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        if let Some(kw) = kwargs {
            for (k, v) in kw.iter() {
                slf.setattr(k.cast::<pyo3::types::PyString>()?, v)?;
            }
        }
        slf.call_method0("kernel")?;
        Ok(slf.clone())
    }
}

fn kcount_to_py(py: Python<'_>, c: KCount) -> PyResult<Bound<'_, PyAny>> {
    Ok(match c {
        KCount::Dense(n) => n.into_pyobject(py)?.into_any(),
        KCount::PerKpoint(v) => PyList::new(py, v)?.into_any(),
    })
}

/// `KsymAdaptedKMP2(mf, frozen=None)` (`kmp2_ksymm.py:226`) — requires a
/// `KsymAdaptedKRHF` / `KRHF(cell, kpts=KPoints)` mean field. `kernel()`
/// defaults to `with_t2=False`; `make_rdm1` is IBZ-sized; `make_rdm2` raises.
#[pyclass(
    extends = PyKmp2,
    subclass,
    name = "KsymAdaptedKMP2",
    module = "pyscf._native.pbc.mp"
)]
pub struct PyKsymAdaptedKmp2 {}

#[pymethods]
impl PyKsymAdaptedKmp2 {
    #[new]
    #[pyo3(signature = (mf, frozen = None, mo_coeff = None, mo_occ = None))]
    fn new(
        py: Python<'_>,
        mf: &Bound<'_, PyAny>,
        frozen: Option<&Bound<'_, PyAny>>,
        mo_coeff: Option<&Bound<'_, PyAny>>,
        mo_occ: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        Ok(
            PyClassInitializer::from(PyKmp2::construct(py, mf, frozen, mo_coeff, mo_occ, true)?)
                .add_subclass(Self {}),
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KUMP2
// ─────────────────────────────────────────────────────────────────────────────

/// `KUMP2(mf, frozen=None)` (`kump2.py`) — the bookkeeping surface on a
/// converged `KUHF`. `kernel()` RAISES `NotImplementedError`, as upstream's
/// does (`kump2.py:38,384,402`).
#[pyclass(
    subclass,
    dict,
    name = "KUMP2",
    module = "pyscf._native.pbc.mp",
    skip_from_py_object
)]
pub struct PyKump2 {
    mf: Py<PyAny>,
    frozen: Py<PyAny>,
}

impl PyKump2 {
    fn with_kump2<R>(
        &self,
        py: Python<'_>,
        body: impl FnOnce(&Kump2<'_>) -> Result<R, PbcMpError>,
    ) -> PyResult<R> {
        let input = mean_field(py, self.mf.bind(py))?;
        let mut mp = Kump2::new(&input.result).map_err(pbc_mp_to_py)?;
        mp.frozen = frozen_u_from_py(Some(self.frozen.bind(py)))?;
        body(&mp).map_err(pbc_mp_to_py)
    }
}

#[pymethods]
impl PyKump2 {
    #[new]
    #[pyo3(signature = (mf, frozen = None, mo_coeff = None, mo_occ = None))]
    fn new(
        py: Python<'_>,
        mf: &Bound<'_, PyAny>,
        frozen: Option<&Bound<'_, PyAny>>,
        mo_coeff: Option<&Bound<'_, PyAny>>,
        mo_occ: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if mo_coeff.is_some_and(|x| !x.is_none()) || mo_occ.is_some_and(|x| !x.is_none()) {
            return Err(PyNotImplementedError::new_err(
                "KUMP2(mf, mo_coeff=..., mo_occ=...): only the mean field's own orbitals are bound",
            ));
        }
        let input = mean_field(py, mf)?;
        if input.result.nset != 2 {
            return Err(PyTypeError::new_err(format!(
                "KUMP2 needs a two-channel KUHF mean field, got {}",
                input.kind
            )));
        }
        frozen_u_from_py(frozen)?;
        Ok(Self {
            mf: mf.clone().unbind(),
            frozen: frozen.map_or_else(|| py.None(), |f| f.clone().unbind()),
        })
    }

    #[getter]
    fn _scf(&self, py: Python<'_>) -> Py<PyAny> {
        self.mf.clone_ref(py)
    }
    #[getter]
    fn frozen(&self, py: Python<'_>) -> Py<PyAny> {
        self.frozen.clone_ref(py)
    }
    #[setter]
    fn set_frozen(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        frozen_u_from_py(Some(v))?;
        self.frozen = v.clone().unbind();
        Ok(())
    }

    /// `get_nocc(per_kpoint=False)` → `(nocca, noccb)`.
    #[pyo3(signature = (per_kpoint = false))]
    fn get_nocc<'py>(&self, py: Python<'py>, per_kpoint: bool) -> PyResult<Bound<'py, PyTuple>> {
        let [a, b] = self.with_kump2(py, |mp| mp.get_nocc(per_kpoint))?;
        PyTuple::new(py, [kcount_to_py(py, a)?, kcount_to_py(py, b)?])
    }

    /// `get_nmo(per_kpoint=False)` → `(nmoa, nmob)`.
    #[pyo3(signature = (per_kpoint = false))]
    fn get_nmo<'py>(&self, py: Python<'py>, per_kpoint: bool) -> PyResult<Bound<'py, PyTuple>> {
        let [a, b] = self.with_kump2(py, |mp| mp.get_nmo(per_kpoint))?;
        PyTuple::new(py, [kcount_to_py(py, a)?, kcount_to_py(py, b)?])
    }

    /// `get_frozen_mask()` → `(masks_alpha, masks_beta)`, per-k boolean arrays.
    fn get_frozen_mask<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let [a, b] = self.with_kump2(py, |mp| mp.get_frozen_mask())?;
        let a = PyList::new(py, a.iter().map(|m| PyArray1::from_slice(py, m)))?;
        let b = PyList::new(py, b.iter().map(|m| PyArray1::from_slice(py, m)))?;
        PyTuple::new(py, [a, b])
    }

    /// Raises `NotImplementedError` — upstream `kump2.py` refuses its energy.
    #[pyo3(signature = (*_args, **_kwargs))]
    fn kernel(
        &self,
        py: Python<'_>,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<()> {
        self.with_kump2(py, |mp| mp.kernel())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KMP2_stagger
// ─────────────────────────────────────────────────────────────────────────────

/// `KMP2_stagger(mf, frozen=None, flag_submesh=False)` (`kmp2_stagger.py:215`)
/// on a converged full-BZ `KRHF`. `kernel()` → `e_corr`.
///
/// `flag_submesh=True` needs an even Monkhorst-Pack mesh and takes the two
/// staggered half-meshes from `mf.kpts`; `False` re-evaluates bands on a
/// half-shifted mesh (`Kmp2Stagger::new_full_mesh`), which supports no
/// frozen orbitals in the Rust port (`FrozenK::default()` there).
#[pyclass(
    subclass,
    dict,
    name = "KMP2_stagger",
    module = "pyscf._native.pbc.mp",
    skip_from_py_object
)]
pub struct PyKmp2Stagger {
    mf: Py<PyAny>,
    frozen: Py<PyAny>,
    flag_submesh: bool,
    e_corr: Option<f64>,
}

#[pymethods]
impl PyKmp2Stagger {
    #[new]
    #[pyo3(signature = (mf, frozen = None, flag_submesh = false))]
    fn new(
        py: Python<'_>,
        mf: &Bound<'_, PyAny>,
        frozen: Option<&Bound<'_, PyAny>>,
        flag_submesh: bool,
    ) -> PyResult<Self> {
        let input = mean_field(py, mf)?;
        if input.kind != "KRHF" {
            return Err(PyTypeError::new_err(format!(
                "KMP2_stagger needs a full-BZ KRHF mean field, got {}",
                input.kind
            )));
        }
        let fz = frozen_k_from_py(frozen)?;
        if !flag_submesh && fz != FrozenK::default() {
            return Err(PyNotImplementedError::new_err(
                "KMP2_stagger(flag_submesh=False) with frozen orbitals is not implemented \
                 (Kmp2Stagger::new_full_mesh pads with FrozenK::default())",
            ));
        }
        Ok(Self {
            mf: mf.clone().unbind(),
            frozen: frozen.map_or_else(|| py.None(), |f| f.clone().unbind()),
            flag_submesh,
            e_corr: None,
        })
    }

    #[getter]
    fn flag_submesh(&self) -> bool {
        self.flag_submesh
    }
    #[getter]
    fn e_corr(&self) -> Option<f64> {
        self.e_corr
    }
    #[getter]
    fn _scf(&self, py: Python<'_>) -> Py<PyAny> {
        self.mf.clone_ref(py)
    }

    /// `kernel()` → `e_corr`.
    fn kernel(&mut self, py: Python<'_>) -> PyResult<f64> {
        let input = mean_field(py, self.mf.bind(py))?;
        let df = extract_df(input.with_df.bind(py))?;
        let frozen = frozen_k_from_py(Some(self.frozen.bind(py)))?;
        let flag = self.flag_submesh;
        let (result, exxdiv) = (input.result, input.exxdiv);
        let e = py
            .detach(move || -> Result<f64, PbcMpError> {
                if flag {
                    let dfr: &dyn PeriodicDf = df.as_ref();
                    let kmesh =
                        pyscf_pbc_gto::get_monkhorst_pack_size_default(dfr.cell(), dfr.kpts())?;
                    let mut mp = Kmp2::new(&result, dfr)?;
                    mp.frozen = frozen;
                    Kmp2Stagger::new(mp, kmesh)?.kernel()
                } else {
                    let mut mf = Krhf::from_df(df);
                    mf.exxdiv = exxdiv;
                    Kmp2Stagger::new_full_mesh(&mf, &result)?.kernel()
                }
            })
            .map_err(pbc_mp_to_py)?;
        self.e_corr = Some(e);
        Ok(e)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The Rust reference path (tests)
// ─────────────────────────────────────────────────────────────────────────────

/// PRIVATE (test reference): an all-Rust `Krhf::kernel` → `Kmp2::kernel` on a
/// FRESH builder of the same kind as `with_df` (`FFTDF` at `with_df.mesh`, or
/// `GDF` over the same cell/k-points), with no Python object between the two.
/// Returns `(e_hf, e_corr)`. `test_pbc_post_scf.py` asserts the binding path
/// reproduces these BITWISE.
#[pyfunction]
#[pyo3(signature = (with_df, exxdiv, conv_tol, conv_tol_grad, max_cycle))]
fn _rust_reference_kmp2(
    py: Python<'_>,
    with_df: &Bound<'_, PyAny>,
    exxdiv: Option<&str>,
    conv_tol: f64,
    conv_tol_grad: Option<f64>,
    max_cycle: u32,
) -> PyResult<(f64, f64)> {
    let (scf, df2) = reference_scf(py, with_df, exxdiv, conv_tol, conv_tol_grad, max_cycle)?;
    py.detach(move || -> Result<(f64, f64), PbcMpError> {
        let mp = Kmp2::new(&scf, df2.as_ref())?;
        let r = mp.kernel()?;
        Ok((scf.e_tot, r.e_corr))
    })
    .map_err(pbc_mp_to_py)
}

/// A fresh `Krhf` SCF on a fresh copy of `with_df`'s builder kind, plus a
/// second fresh builder for the correlated step (the Rust tests' pattern,
/// `krccsd_smoke.rs`).
pub(crate) fn reference_scf(
    py: Python<'_>,
    with_df: &Bound<'_, PyAny>,
    exxdiv: Option<&str>,
    conv_tol: f64,
    conv_tol_grad: Option<f64>,
    max_cycle: u32,
) -> PyResult<(KScfResult, Box<dyn PeriodicDf>)> {
    let proto = extract_df(with_df)?;
    let fresh = |proto: &dyn PeriodicDf| -> PyResult<Box<dyn PeriodicDf>> {
        let cell = proto.cell().clone();
        let kpts = proto.kpts().to_vec();
        Ok(match proto.name() {
            "FFTDF" => Box::new(
                pyscf_pbc_df::Fftdf::with_mesh(cell, &kpts, proto.mesh()).map_err(pbc_df_to_py)?,
            ),
            "GDF" => Box::new(pyscf_pbc_df::Gdf::new(cell, &kpts)),
            other => {
                return Err(PyNotImplementedError::new_err(format!(
                    "_rust_reference: builder {other} is not wired (FFTDF, GDF)"
                )));
            }
        })
    };
    let df1 = fresh(proto.as_ref())?;
    let df2 = fresh(proto.as_ref())?;
    let exx = match exxdiv {
        None => None,
        Some("ewald") => Some(pyscf_pbc_gto::ExxDiv::Ewald),
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "_rust_reference: exxdiv {other:?} is not wired (None, 'ewald')"
            )));
        }
    };
    let scf = py
        .detach(move || {
            let mut mf = Krhf::from_df(df1);
            mf.exxdiv = exx;
            let mut cfg = pyscf_pbc_scf::KScfConfig::for_cell(mf.with_df.cell());
            cfg.conv_tol = conv_tol;
            cfg.conv_tol_grad = conv_tol_grad;
            cfg.max_cycle = max_cycle;
            mf.kernel(&cfg)
        })
        .map_err(pyscf_to_py)?;
    Ok((scf, df2))
}
