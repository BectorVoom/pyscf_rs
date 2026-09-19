//! `pyscf._native.pbc.ci` — k-point CIS (plan 20-15).
//!
//! | Python | Rust | notes |
//! |---|---|---|
//! | `KCIS(mf, frozen=None)` (`CIS`) | `kernel_at_kshift` (`kcis_rhf.rs:248`) with `KcisOpts` (`:50`) | full-BZ `KRHF` |
//!
//! `KCIS` reads the same `_ERIS` blocks as `KRCCSD` — upstream `_CIS_ERIS`
//! (`kcis_rhf.py:458-505`) builds the Fock matrix with `exxdiv` suppressed and
//! re-adds the Madelung shift exactly as `kccsd_rhf._ERIS` does, which is why
//! the Rust port drives both from `pyscf_pbc_cc::keris::KEris`.
//!
//! # `pbc/ci/cisd.py` is DEFERRED BY DESIGN
//!
//! `RCISD`/`UCISD`/`GCISD` are Γ-point shims over the MOLECULAR CISD, and this
//! port has no molecular CI crate (`pyscf-pbc-ci/src/lib.rs:3-22`). They are
//! not bound; the overlay leaves them to 20-17's fallthrough policy.

use numpy::PyArray1;
use pyo3::exceptions::{PyNotImplementedError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyscf_pbc_cc::kccsd_rhf::Krccsd;
use pyscf_pbc_cc::keris::KErisOpts;
use pyscf_pbc_ci::{KcisOpts, PbcCiError, kernel_at_kshift};
use pyscf_pbc_df::PeriodicDf;

use crate::errors::{PyscfRsRuntimeError, pyscf_to_py};
use crate::pbc::cc::pbc_cc_to_py;
use crate::pbc::df::extract_df;
use crate::pbc::mp::{frozen_k_from_py, mean_field};

/// Register `KCIS` (and its upstream alias `CIS`) on `pyscf._native.pbc.ci`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKcis>()?;
    m.add("CIS", m.getattr("KCIS")?)?;
    Ok(())
}

fn pbc_ci_to_py(err: PbcCiError) -> PyErr {
    match err {
        PbcCiError::Core(e) => pyscf_to_py(e),
        e @ PbcCiError::NotImplementedUpstream { .. } => {
            PyNotImplementedError::new_err(e.to_string())
        }
        other => PyscfRsRuntimeError::new_err((other.to_string(), "PbcCi", Vec::<String>::new())),
    }
}

enum Fail {
    Cc(pyscf_pbc_cc::PbcCcError),
    Ci(PbcCiError),
}

/// `KCIS(mf, frozen=None)` — k-point CIS on a converged full-BZ `KRHF`
/// (`kcis_rhf.py:321`).
///
/// Attributes: `max_space = 20`, `max_cycle = 50`, `conv_tol = 1e-7`,
/// `davidson = True`, `build_full_H = False`, `keep_exxdiv = False`.
/// `kernel(nroots=1, eris=None, kptlist=None)` → `(e, v)` where `e` is a
/// `(len(kptlist), nroots)` array and `v` is `None` — `kernel_at_kshift`
/// returns eigenvalues only.
#[pyclass(
    subclass,
    dict,
    name = "KCIS",
    module = "pyscf._native.pbc.ci",
    skip_from_py_object
)]
pub struct PyKcis {
    mf: Py<PyAny>,
    opts: KcisOpts,
    keep_exxdiv: bool,
    e: Option<Py<PyAny>>,
}

#[pymethods]
impl PyKcis {
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
                "KCIS(mf, mo_coeff=..., mo_occ=...): only the mean field's own orbitals are bound",
            ));
        }
        if frozen_k_from_py(frozen)? != pyscf_pbc_mp::FrozenK::default() {
            return Err(PyNotImplementedError::new_err(
                "KCIS(frozen=...) is not implemented in pyscf-rs: the ERIs come from \
                 Krccsd::new, which pads with FrozenK::default()",
            ));
        }
        let input = mean_field(py, mf)?;
        if input.kind != "KRHF" {
            return Err(PyTypeError::new_err(format!(
                "KCIS needs a converged full-BZ KRHF mean field, got {}",
                input.kind
            )));
        }
        Ok(Self {
            mf: mf.clone().unbind(),
            opts: KcisOpts::default(),
            keep_exxdiv: false,
            e: None,
        })
    }

    #[getter]
    fn _scf(&self, py: Python<'_>) -> Py<PyAny> {
        self.mf.clone_ref(py)
    }
    #[getter]
    fn max_space(&self) -> usize {
        self.opts.max_space
    }
    #[setter]
    fn set_max_space(&mut self, v: usize) {
        self.opts.max_space = v;
    }
    #[getter]
    fn max_cycle(&self) -> usize {
        self.opts.max_cycle
    }
    #[setter]
    fn set_max_cycle(&mut self, v: usize) {
        self.opts.max_cycle = v;
    }
    #[getter]
    fn conv_tol(&self) -> f64 {
        self.opts.conv_tol
    }
    #[setter]
    fn set_conv_tol(&mut self, v: f64) {
        self.opts.conv_tol = v;
    }
    #[getter]
    fn davidson(&self) -> bool {
        self.opts.davidson
    }
    #[setter]
    fn set_davidson(&mut self, v: bool) {
        self.opts.davidson = v;
    }
    #[getter(build_full_H)]
    fn build_full_h(&self) -> bool {
        self.opts.build_full_h
    }
    #[setter(build_full_H)]
    fn set_build_full_h(&mut self, v: bool) {
        self.opts.build_full_h = v;
    }
    #[getter]
    fn keep_exxdiv(&self) -> bool {
        self.keep_exxdiv
    }
    #[setter]
    fn set_keep_exxdiv(&mut self, v: bool) {
        self.keep_exxdiv = v;
    }
    #[getter]
    fn e(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.e.as_ref().map(|x| x.clone_ref(py))
    }

    /// `kernel(nroots=1, eris=None, kptlist=None)` → `(e, None)`.
    #[pyo3(signature = (nroots = 1, eris = None, kptlist = None))]
    fn kernel<'py>(
        &mut self,
        py: Python<'py>,
        nroots: usize,
        eris: Option<&Bound<'py, PyAny>>,
        kptlist: Option<Vec<usize>>,
    ) -> PyResult<(Bound<'py, PyAny>, Py<PyAny>)> {
        if eris.is_some_and(|x| !x.is_none()) {
            return Err(PyNotImplementedError::new_err(
                "KCIS.kernel(eris=...) is not bound; the integrals are rebuilt from the mean field",
            ));
        }
        let input = mean_field(py, self.mf.bind(py))?;
        let df = extract_df(input.with_df.bind(py))?;
        let (result, exxdiv) = (input.result, input.exxdiv);
        let opts = self.opts;
        let keep_exxdiv = self.keep_exxdiv;
        let nk = result.nkpts;
        let list = kptlist.unwrap_or_else(|| (0..nk).collect());
        if let Some(&bad) = list.iter().find(|&&k| k >= nk) {
            return Err(PyValueError::new_err(format!(
                "kptlist entry {bad} is out of range for {nk} k-points"
            )));
        }
        let roots = py
            .detach(move || -> Result<Vec<Vec<f64>>, Fail> {
                let dfr: &dyn PeriodicDf = df.as_ref();
                let mut cc = Krccsd::new(&result, dfr).map_err(Fail::Cc)?;
                cc.eris_opts = KErisOpts {
                    keep_exxdiv,
                    exxdiv,
                    ..KErisOpts::default()
                };
                let eris = cc.ao2mo().map_err(Fail::Cc)?;
                list.iter()
                    .map(|&k| {
                        kernel_at_kshift(&eris, &cc.khelper.kconserv, k, nroots, &opts)
                            .map_err(Fail::Ci)
                    })
                    .collect()
            })
            .map_err(|f| match f {
                Fail::Cc(e) => pbc_cc_to_py(e),
                Fail::Ci(e) => pbc_ci_to_py(e),
            })?;
        let rows = PyList::new(py, roots.iter().map(|r| PyArray1::from_slice(py, r)))?;
        let e = py.import("numpy")?.call_method1("asarray", (rows,))?;
        self.e = Some(e.clone().unbind());
        Ok((e, py.None()))
    }
}
