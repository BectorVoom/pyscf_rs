//! `pyscf._native.pbc.scf` — the periodic SCF drivers (plan 20-12).
//!
//! # Classes
//!
//! | Python | Rust | notes |
//! |---|---|---|
//! | `KSCF` | — | native base: state, the eleven hooks, `kernel`, results |
//! | `KRHF(KSCF)` | `Krhf` (`krhf.rs:27`) | a `KPoints` `kpts` selects `KsymAdaptedKrhf` |
//! | `KsymAdaptedKRHF(KRHF)` | `KsymAdaptedKrhf` (`khf_ksymm.rs:136`) | requires a built `KPoints` |
//! | `KROHF(KRHF)` | `Krohf` (`krohf.rs:29`) | |
//! | `KUHF(KSCF)` | `Kuhf` (`kuhf.rs:29`) | |
//! | `KGHF(KSCF)` | `Kghf` (`kghf.rs:31`) | |
//!
//! The hierarchy mirrors upstream (`KROHF(khf.KRHF)`, `KUHF(khf.KSCF)`,
//! `KsymAdaptedKRHF(khf.KRHF)`). Every hook and attribute lives on `KSCF`, so a
//! Python subclass that overrides a hook is detected by 20-11's
//! [`KPyOverrideBridge`] against ONE native base (`KSCF`), and `super().get_veff`
//! reaches the Rust default.
//!
//! # Ownership (20-10 contract)
//!
//! A driver stores the Python `with_df` object (`mf.with_df is mydf`; settable)
//! and calls [`extract_df`] at every entry, building a fresh Rust driver
//! ([`Driver`]) over the CURRENT builder. Nothing Rust-side outlives a call
//! except the last [`KScfResult`]. Construction without a DF object builds
//! upstream's default `FFTDF(cell, kpts)`.
//!
//! # Where the `KPoints` dispatch lives
//!
//! Upstream's `pbc/scf/__init__.py:40-106` makes `KRHF`/`KUHF`/`KGHF` FUNCTIONS
//! that branch on `isinstance(kpts, KPoints)`. The Phase-20 identity gate
//! requires `pyscf.pbc.scf.KRHF is pyscf._native.pbc.scf.KRHF`, so those three
//! names are these classes and their constructors apply the same rule: `KRHF`
//! with a `KPoints` runs `KsymAdaptedKrhf`; `KUHF`/`KGHF` with a `KPoints`
//! RAISE (this port has no `kuhf_ksymm`/`kghf_ksymm`) instead of falling
//! through to upstream Python. The function-shaped upstream names (`RHF`,
//! `UHF`, `ROHF`, `GHF`, `HF`, `KHF`) keep their Python dispatch in
//! `python/pyscf/pbc/scf/__init__.py`.
//!
//! # GIL
//!
//! Held for the whole kernel: `Krhf`/`Kuhf` carry a `Cell<Option<f64>>` (the
//! smearing entropy) and are not `Sync`, and overridden hooks re-enter Python
//! every cycle (20-11 SUMMARY).
//!
//! # The 20-11 harness
//!
//! `_KRHFBridgeSelftest(with_df)` is kept unchanged: `test_pbc_override_dispatch.py`
//! runs every case against it AND `KRHF`.

use pyo3::exceptions::{
    PyAttributeError, PyNotImplementedError, PyOSError, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyscf_algebra::CTensor;
use pyscf_core::PyscfRsError;
use pyscf_pbc_df::PeriodicDf;
use pyscf_pbc_gto::{Cell, ExxDiv};
use pyscf_pbc_scf::krhf::to_row_major;
use pyscf_pbc_scf::{
    KDms, KInitGuess, KMats, KOverrideHooks, KScfConfig, KScfResult, Kghf, Krhf, Krohf,
    KsymAdaptedKrhf, Kuhf, Smearing, SmearingMethod, dump_kscf_to_file, load_kscf_from_file,
};

use crate::bridge::extract_cell_from_pyany;
use crate::errors::pyscf_to_py;
use crate::pbc::convert::{extract_kpts, extract_kpts_opt, kpts_to_pyarray};
use crate::pbc::df::{PyFftdf, extract_df, pbc_df_to_py};
use crate::pbc::gto::PyCell;
use crate::pbc::kbridge::{
    KPyOverrideBridge, kdms_from_py, kdms_to_py, kmats_from_py, kmats_to_py, mo_coeff_from_py,
    mo_coeff_to_py, mo_values_from_py, mo_values_to_py,
};
use crate::pbc::symm::PyKPoints;

/// Register the drivers, the gamma shims, the addons/chkfile functions and the
/// 20-11 private harness on `pyscf._native.pbc.scf`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKscf>()?;
    m.add_class::<PyKrhf>()?;
    m.add_class::<PyKsymAdaptedKrhf>()?;
    m.add_class::<PyKrohf>()?;
    m.add_class::<PyKuhf>()?;
    m.add_class::<PyKghf>()?;
    m.add_function(wrap_pyfunction!(gamma_rhf, m)?)?;
    m.add_function(wrap_pyfunction!(gamma_uhf, m)?)?;
    m.add_function(wrap_pyfunction!(gamma_rohf, m)?)?;
    m.add_function(wrap_pyfunction!(gamma_ghf, m)?)?;
    m.add_function(wrap_pyfunction!(smearing_, m)?)?;
    m.add_function(wrap_pyfunction!(project_mo_nr2nr, m)?)?;
    m.add_function(wrap_pyfunction!(load_scf, m)?)?;
    m.add_class::<PyKrhfBridgeSelftest>()?;
    m.add_function(wrap_pyfunction!(_kbridge_probe_count, m)?)?;
    Ok(())
}

/// `_kbridge_probe_count()` — k-point override probes run so far.
#[pyfunction]
fn _kbridge_probe_count() -> usize {
    crate::caches::k_override_probes_run()
}

/// PRIVATE (plan 20-11): the minimal KRHF-shaped driver that exercises
/// `KPyOverrideBridge`. Not a public API; see the module docs.
#[pyclass(
    subclass,
    dict,
    name = "_KRHFBridgeSelftest",
    module = "pyscf._native.pbc.scf",
    skip_from_py_object
)]
pub struct PyKrhfBridgeSelftest {
    with_df: Py<PyAny>,
    py_cell: Py<PyAny>,
}

impl PyKrhfBridgeSelftest {
    /// A `Krhf` over the CURRENT builder of `with_df` (20-10 ownership
    /// contract: `extract_df` at every entry, sharing the builder's caches).
    fn driver(&self, py: Python<'_>) -> PyResult<Krhf> {
        Ok(Krhf::from_df(extract_df(self.with_df.bind(py))?))
    }
}

#[pymethods]
impl PyKrhfBridgeSelftest {
    #[new]
    fn new(with_df: &Bound<'_, PyAny>) -> PyResult<Self> {
        extract_df(with_df)?;
        Ok(Self {
            with_df: with_df.clone().unbind(),
            py_cell: with_df.getattr("cell")?.unbind(),
        })
    }

    #[getter]
    fn cell(&self, py: Python<'_>) -> Py<PyAny> {
        self.py_cell.clone_ref(py)
    }

    #[getter]
    fn with_df(&self, py: Python<'_>) -> Py<PyAny> {
        self.with_df.clone_ref(py)
    }

    /// `get_ovlp(cell=None)` — per-k list. `cell` is accepted and ignored.
    #[pyo3(signature = (cell = None))]
    fn get_ovlp<'py>(
        &self,
        py: Python<'py>,
        cell: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = cell;
        let d = self.driver(py)?;
        let s = d.get_ovlp().map_err(pyscf_to_py)?;
        kmats_to_py(py, &s, d.nao())
    }

    /// `get_hcore(cell=None)` — per-k list. `cell` is accepted and ignored.
    #[pyo3(signature = (cell = None))]
    fn get_hcore<'py>(
        &self,
        py: Python<'py>,
        cell: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = cell;
        let d = self.driver(py)?;
        let h = d.get_hcore().map_err(pyscf_to_py)?;
        kmats_to_py(py, &h, d.nao())
    }

    /// `get_init_guess(cell=None, key='minao', s1e=None)`.
    #[pyo3(signature = (cell = None, key = "minao", s1e = None))]
    fn get_init_guess<'py>(
        &self,
        py: Python<'py>,
        cell: Option<&Bound<'py, PyAny>>,
        key: &str,
        s1e: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = cell;
        let d = self.driver(py)?;
        let (nk, nao) = (d.kpts().len(), d.nao());
        let mode = parse_init_guess(key)?;
        let s1e = match s1e {
            Some(s) if !s.is_none() => kmats_from_py(s, nk, nao, "s1e")?,
            _ => d.get_ovlp().map_err(pyscf_to_py)?,
        };
        let dm = d.get_init_guess(&mode, &s1e).map_err(pyscf_to_py)?;
        kdms_to_py(py, &dm, nao)
    }

    /// `get_veff(cell=None, dm_kpts=None)` — `vj - vk/2`.
    #[pyo3(signature = (cell = None, dm_kpts = None))]
    fn get_veff<'py>(
        &self,
        py: Python<'py>,
        cell: Option<&Bound<'py, PyAny>>,
        dm_kpts: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = cell;
        let d = self.driver(py)?;
        let (nk, nao) = (d.kpts().len(), d.nao());
        let dm = dm_kpts.filter(|x| !x.is_none()).ok_or_else(|| {
            pyo3::exceptions::PyTypeError::new_err("get_veff: dm_kpts is required")
        })?;
        let dms = kdms_from_py(dm, d.nset(), nk, nao, "dm_kpts")?;
        let v = d.get_veff(&dms).map_err(pyscf_to_py)?;
        kdms_to_py(py, &v, nao)
    }

    /// `get_fock(h1e, s1e=None, vhf, dm=None, cycle=-1)` — bare `h1e + vhf`
    /// (damping, DIIS and level shift are the kernel's, as at `cycle=-1`).
    #[pyo3(signature = (h1e = None, s1e = None, vhf = None, dm = None, cycle = -1))]
    fn get_fock<'py>(
        &self,
        py: Python<'py>,
        h1e: Option<&Bound<'py, PyAny>>,
        s1e: Option<&Bound<'py, PyAny>>,
        vhf: Option<&Bound<'py, PyAny>>,
        dm: Option<&Bound<'py, PyAny>>,
        cycle: i64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = s1e;
        if cycle >= 0 {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "_KRHFBridgeSelftest.get_fock: only the bare Fock (cycle=-1) is bound",
            ));
        }
        let d = self.driver(py)?;
        let (nk, nao) = (d.kpts().len(), d.nao());
        fn required<'a, 'py>(
            x: Option<&'a Bound<'py, PyAny>>,
            name: &str,
        ) -> PyResult<&'a Bound<'py, PyAny>> {
            x.filter(|v| !v.is_none()).ok_or_else(|| {
                pyo3::exceptions::PyTypeError::new_err(format!("get_fock: {name} is required"))
            })
        }
        let h1e = kmats_from_py(required(h1e, "h1e")?, nk, nao, "h1e")?;
        let vhf = kdms_from_py(required(vhf, "vhf")?, d.nset(), nk, nao, "vhf")?;
        let dms = match dm {
            Some(x) if !x.is_none() => kdms_from_py(x, d.nset(), nk, nao, "dm")?,
            _ => Vec::new(),
        };
        let f = d.get_fock(&h1e, &vhf, &dms).map_err(pyscf_to_py)?;
        kdms_to_py(py, &f, nao)
    }

    /// `eig(h_kpts, s_kpts)` → `(mo_energy, mo_coeff)`.
    fn eig<'py>(
        &self,
        py: Python<'py>,
        h_kpts: &Bound<'py, PyAny>,
        s_kpts: &Bound<'py, PyAny>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let d = self.driver(py)?;
        let (nk, nao, nfock) = (d.kpts().len(), d.nao(), d.nfock());
        let fock = kdms_from_py(h_kpts, nfock, nk, nao, "h_kpts")?;
        let s1e = kmats_from_py(s_kpts, nk, nao, "s_kpts")?;
        let (e, c) = d.eig(&fock, &s1e).map_err(pyscf_to_py)?;
        Ok((
            mo_values_to_py(py, &e, nfock)?,
            mo_coeff_to_py(py, &c, nfock, nao)?,
        ))
    }

    /// `get_occ(mo_energy_kpts, mo_coeff_kpts=None)` → `mo_occ`.
    #[pyo3(signature = (mo_energy_kpts, mo_coeff_kpts = None))]
    fn get_occ<'py>(
        &self,
        py: Python<'py>,
        mo_energy_kpts: &Bound<'py, PyAny>,
        mo_coeff_kpts: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = mo_coeff_kpts;
        let d = self.driver(py)?;
        let nfock = d.nfock();
        let e = mo_values_from_py(mo_energy_kpts, nfock, d.kpts().len(), "mo_energy_kpts")?;
        let (occ, _fermi) = d.get_occ(&e).map_err(pyscf_to_py)?;
        mo_values_to_py(py, &occ, nfock)
    }

    /// `make_rdm1(mo_coeff_kpts, mo_occ_kpts)`.
    fn make_rdm1<'py>(
        &self,
        py: Python<'py>,
        mo_coeff_kpts: &Bound<'py, PyAny>,
        mo_occ_kpts: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let d = self.driver(py)?;
        let (nk, nao, nfock) = (d.kpts().len(), d.nao(), d.nfock());
        let c = mo_coeff_from_py(mo_coeff_kpts, nfock, nk, nao, "mo_coeff_kpts")?;
        let occ = mo_values_from_py(mo_occ_kpts, nfock, nk, "mo_occ_kpts")?;
        let dm = d.make_rdm1(&c, &occ).map_err(pyscf_to_py)?;
        kdms_to_py(py, &dm, nao)
    }

    /// `energy_elec(dm_kpts, h1e_kpts, vhf_kpts)` → `(e_elec, e_coul)`.
    fn energy_elec(
        &self,
        py: Python<'_>,
        dm_kpts: &Bound<'_, PyAny>,
        h1e_kpts: &Bound<'_, PyAny>,
        vhf_kpts: &Bound<'_, PyAny>,
    ) -> PyResult<(f64, f64)> {
        let d = self.driver(py)?;
        let (nk, nao, nset) = (d.kpts().len(), d.nao(), d.nset());
        let dm = kdms_from_py(dm_kpts, nset, nk, nao, "dm_kpts")?;
        let h1e = kmats_from_py(h1e_kpts, nk, nao, "h1e_kpts")?;
        let vhf = kdms_from_py(vhf_kpts, nset, nk, nao, "vhf_kpts")?;
        d.energy_elec(&dm, &h1e, &vhf).map_err(pyscf_to_py)
    }

    /// `energy_nuc()` — the Ewald nuclear repulsion.
    fn energy_nuc(&self, py: Python<'_>) -> PyResult<f64> {
        self.driver(py)?.energy_nuc().map_err(pyscf_to_py)
    }

    /// `get_grad(mo_coeff_kpts, mo_occ_kpts, fock)` — 1-D orbital gradient.
    fn get_grad<'py>(
        &self,
        py: Python<'py>,
        mo_coeff_kpts: &Bound<'py, PyAny>,
        mo_occ_kpts: &Bound<'py, PyAny>,
        fock: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let d = self.driver(py)?;
        let (nk, nao, nfock, nset) = (d.kpts().len(), d.nao(), d.nfock(), d.nset());
        let c = mo_coeff_from_py(mo_coeff_kpts, nfock, nk, nao, "mo_coeff_kpts")?;
        let occ = mo_values_from_py(mo_occ_kpts, nfock, nk, "mo_occ_kpts")?;
        let fock = kdms_from_py(fock, nset, nk, nao, "fock")?;
        // The trait takes (h1e, vhf) and builds h1e + vhf; hand it the Fock as
        // `vhf` over a zero `h1e` (x + 0.0 == x).
        let zero: Vec<pyscf_algebra::CTensor> = (0..nk)
            .map(|_| pyscf_algebra::CTensor {
                re: vec![0.0; nao * nao],
                im: vec![0.0; nao * nao],
            })
            .collect();
        let g = d.get_grad(&c, &occ, &zero, &fock);
        Ok(numpy::PyArray1::from_vec(py, g))
    }

    /// `kernel(dm0=None, conv_tol=1e-10, max_cycle=50, init_guess='minao',
    /// use_bridge=True)` → a dict (`e_tot`, `e_elec`, `e_coul`, `e_nuc`,
    /// `converged`, `cycles`, `mo_energy`, `mo_occ`, `overridden`).
    ///
    /// `use_bridge=False` runs `Krhf::kernel` with no bridge at all — the
    /// reference the negative tests compare against bitwise.
    #[pyo3(signature = (dm0 = None, conv_tol = 1e-10, max_cycle = 50, init_guess = "minao", use_bridge = true))]
    fn kernel<'py>(
        slf: &Bound<'py, Self>,
        dm0: Option<&Bound<'py, PyAny>>,
        conv_tol: f64,
        max_cycle: u32,
        init_guess: &str,
        use_bridge: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let py = slf.py();
        let (d, py_cell) = {
            let me = slf.borrow();
            (me.driver(py)?, me.py_cell.clone_ref(py))
        };
        let (nk, nao) = (d.kpts().len(), d.nao());
        let init_guess = match dm0 {
            Some(x) if !x.is_none() => {
                KInitGuess::UserDm(kdms_from_py(x, d.nset(), nk, nao, "dm0")?)
            }
            _ => parse_init_guess(init_guess)?,
        };
        let cfg = KScfConfig {
            conv_tol,
            max_cycle,
            init_guess,
            ..KScfConfig::for_cell(d.cell())
        };
        let base = py.get_type::<Self>();
        let (res, overridden) = if use_bridge {
            // The GIL is held: `Krhf` is not `Sync`, and overridden hooks
            // re-enter Python every cycle (as `PyRHF`'s subclass path does).
            let bridge =
                KPyOverrideBridge::new(py, slf.clone().into_any().unbind(), py_cell, &base, &d)?;
            let res = bridge.finish(pyscf_pbc_scf::kernel(&bridge, &cfg))?;
            (res, bridge.overridden_hooks())
        } else {
            (d.kernel(&cfg).map_err(pyscf_to_py)?, Vec::new())
        };
        let out = PyDict::new(py);
        out.set_item("e_tot", res.e_tot)?;
        out.set_item("e_elec", res.e_elec)?;
        out.set_item("e_coul", res.e_coul)?;
        out.set_item("e_nuc", res.e_nuc)?;
        out.set_item("converged", res.converged)?;
        out.set_item("cycles", res.cycles)?;
        out.set_item("mo_energy", mo_values_to_py(py, &res.mo_energy, d.nfock())?)?;
        out.set_item("mo_occ", mo_values_to_py(py, &res.mo_occ, d.nfock())?)?;
        out.set_item("overridden", overridden)?;
        Ok(out)
    }
}

fn parse_init_guess(key: &str) -> PyResult<KInitGuess> {
    match key.to_ascii_lowercase().as_str() {
        "minao" => Ok(KInitGuess::Minao),
        "atom" => Ok(KInitGuess::Atom),
        "1e" | "hcore" => Ok(KInitGuess::OneElectron),
        other => Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "init guess {other:?} is not bound (minao, atom, 1e, chkfile)"
        ))),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The Rust driver behind one call
// ─────────────────────────────────────────────────────────────────────────────

/// Which periodic SCF method an instance runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Rhf,
    KsymRhf,
    Uhf,
    Rohf,
    Ghf,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Rhf => "KRHF",
            Kind::KsymRhf => "KsymAdaptedKRHF",
            Kind::Uhf => "KUHF",
            Kind::Rohf => "KROHF",
            Kind::Ghf => "KGHF",
        }
    }
}

/// A freshly built Rust driver over the current `with_df` builder. Short-lived
/// (one per call), so the variant size spread is irrelevant.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
enum Driver {
    Rhf(Krhf),
    KsymRhf(KsymAdaptedKrhf),
    Uhf(Kuhf),
    Rohf(Krohf),
    Ghf(Kghf),
}

/// Run `$body` with `$x` bound to the concrete driver.
macro_rules! each_driver {
    ($d:expr, $x:ident => $body:expr) => {
        match $d {
            Driver::Rhf($x) => $body,
            Driver::KsymRhf($x) => $body,
            Driver::Uhf($x) => $body,
            Driver::Rohf($x) => $body,
            Driver::Ghf($x) => $body,
        }
    };
}

impl Driver {
    /// The concrete driver's own `kernel` — no bridge, no Python.
    fn kernel_direct(&self, cfg: &KScfConfig) -> Result<KScfResult, PyscfRsError> {
        each_driver!(self, d => d.kernel(cfg))
    }

    fn df(&self) -> &dyn PeriodicDf {
        each_driver!(self, d => d.with_df.as_ref())
    }
}

/// Pure delegation, so a kernel over `Driver` is op-for-op the concrete
/// driver's kernel.
impl KOverrideHooks for Driver {
    fn cell(&self) -> &Cell {
        each_driver!(self, d => KOverrideHooks::cell(d))
    }
    fn kpts(&self) -> &[[f64; 3]] {
        each_driver!(self, d => KOverrideHooks::kpts(d))
    }
    fn nset(&self) -> usize {
        each_driver!(self, d => d.nset())
    }
    fn nfock(&self) -> usize {
        each_driver!(self, d => d.nfock())
    }
    fn nao(&self) -> usize {
        each_driver!(self, d => KOverrideHooks::nao(d))
    }
    fn get_ovlp(&self) -> Result<KMats, PyscfRsError> {
        each_driver!(self, d => d.get_ovlp())
    }
    fn get_hcore(&self) -> Result<KMats, PyscfRsError> {
        each_driver!(self, d => d.get_hcore())
    }
    fn get_init_guess(&self, mode: &KInitGuess, s1e: &KMats) -> Result<KDms, PyscfRsError> {
        each_driver!(self, d => d.get_init_guess(mode, s1e))
    }
    fn get_veff(&self, dms: &KDms) -> Result<KDms, PyscfRsError> {
        each_driver!(self, d => d.get_veff(dms))
    }
    fn get_fock(&self, h1e: &KMats, vhf: &KDms, dms: &KDms) -> Result<KDms, PyscfRsError> {
        each_driver!(self, d => d.get_fock(h1e, vhf, dms))
    }
    fn diis_dms(&self, dms: &KDms) -> KDms {
        each_driver!(self, d => d.diis_dms(dms))
    }
    fn eig(&self, fock: &KDms, s1e: &KMats) -> Result<(Vec<Vec<f64>>, Vec<CTensor>), PyscfRsError> {
        each_driver!(self, d => d.eig(fock, s1e))
    }
    fn get_occ(&self, mo_energy: &[Vec<f64>]) -> Result<(Vec<Vec<f64>>, Vec<f64>), PyscfRsError> {
        each_driver!(self, d => d.get_occ(mo_energy))
    }
    fn make_rdm1(&self, mo_coeff: &[CTensor], mo_occ: &[Vec<f64>]) -> Result<KDms, PyscfRsError> {
        each_driver!(self, d => d.make_rdm1(mo_coeff, mo_occ))
    }
    fn energy_elec(&self, dms: &KDms, h1e: &KMats, vhf: &KDms) -> Result<(f64, f64), PyscfRsError> {
        each_driver!(self, d => d.energy_elec(dms, h1e, vhf))
    }
    fn energy_nuc(&self) -> Result<f64, PyscfRsError> {
        each_driver!(self, d => d.energy_nuc())
    }
    fn get_grad(
        &self,
        mo_coeff: &[CTensor],
        mo_occ: &[Vec<f64>],
        h1e: &KMats,
        vhf: &KDms,
    ) -> Vec<f64> {
        each_driver!(self, d => d.get_grad(mo_coeff, mo_occ, h1e, vhf))
    }
    fn free_energy(&self) -> Option<f64> {
        each_driver!(self, d => d.free_energy())
    }
}

/// The last kernel's result plus the layout needed to hand it to Python.
#[derive(Debug, Clone)]
struct Solved {
    res: KScfResult,
    nao: usize,
    nfock: usize,
    kpts: Vec<[f64; 3]>,
    sigma: Option<f64>,
}

fn not_impl(msg: impl Into<String>) -> PyErr {
    PyNotImplementedError::new_err(msg.into())
}

fn chk_err(e: pyscf_chkfile::ChkfileError) -> PyErr {
    PyOSError::new_err(format!("periodic chkfile: {e}"))
}

fn some<'a, 'py>(x: Option<&'a Bound<'py, PyAny>>) -> Option<&'a Bound<'py, PyAny>> {
    x.filter(|v| !v.is_none())
}

// ─────────────────────────────────────────────────────────────────────────────
// KSCF — the native base class
// ─────────────────────────────────────────────────────────────────────────────

/// `KSCF` — the native base of the periodic SCF drivers (`khf.py:KSCF`).
///
/// Not constructed directly: use `KRHF`, `KUHF`, `KROHF`, `KGHF` or
/// `KsymAdaptedKRHF`. Holds the configuration (`conv_tol`, `max_cycle`, …,
/// upstream's attribute idiom), the Python `cell` / `with_df` objects, and the
/// last result (`e_tot`, `mo_energy`, `mo_coeff`, `mo_occ`, `converged`, …).
#[pyclass(
    subclass,
    dict,
    name = "KSCF",
    module = "pyscf._native.pbc.scf",
    skip_from_py_object
)]
pub struct PyKscf {
    kind: Kind,
    py_cell: Py<PyAny>,
    with_df: Py<PyAny>,
    /// The `KPoints` object (ksymm only).
    kpoints: Option<Py<PyAny>>,
    exxdiv: Option<ExxDiv>,
    conv_tol: f64,
    conv_tol_grad: Option<f64>,
    max_cycle: u32,
    diis: bool,
    diis_space: usize,
    diis_start_cycle: u32,
    damp: f64,
    level_shift: f64,
    init_guess: String,
    chkfile: Option<String>,
    verbose: i64,
    smearing: Option<Smearing>,
    nelec: Option<(usize, usize)>,
    init_guess_breaksym: i32,
    use_ao_symmetry: bool,
    solved: Option<Solved>,
    overridden: Vec<&'static str>,
}

impl PyKscf {
    /// Shared constructor body. `kind` is the family the Python class names;
    /// a `KPoints` argument switches `Rhf` to `KsymRhf` and is refused for the
    /// others (Task 6).
    fn construct(
        cell: &Bound<'_, PyAny>,
        kpts: Option<&Bound<'_, PyAny>>,
        exxdiv: Option<&str>,
        kind: Kind,
    ) -> PyResult<Self> {
        let py = cell.py();
        let kpoints = some(kpts).filter(|k| k.cast::<PyKPoints>().is_ok());
        let kind = match (kind, kpoints.is_some()) {
            (Kind::Rhf | Kind::KsymRhf, true) => Kind::KsymRhf,
            (Kind::KsymRhf, false) => {
                return Err(PyTypeError::new_err(
                    "KsymAdaptedKRHF requires a built pyscf.pbc.symm.KPoints as kpts",
                ));
            }
            (Kind::Uhf, true) => {
                return Err(not_impl(
                    "KUHF with a KPoints object (k-point symmetry) is not implemented in \
                     pyscf-rs; upstream has pbc/scf/kuhf_ksymm.py. Pass kpts.kpts (the full \
                     Brillouin zone) for a plain KUHF.",
                ));
            }
            (Kind::Ghf, true) => {
                return Err(not_impl(
                    "KGHF with a KPoints object (k-point symmetry) is not implemented in \
                     pyscf-rs; upstream has pbc/scf/kghf_ksymm.py. Pass kpts.kpts (the full \
                     Brillouin zone) for a plain KGHF.",
                ));
            }
            (Kind::Rohf, true) => {
                return Err(not_impl(
                    "KROHF with a KPoints object (k-point symmetry) is not implemented in \
                     pyscf-rs (upstream has no k-point-symmetry KROHF either).",
                ));
            }
            (k, false) => k,
        };
        let mut df_cell = cell.clone();
        // 20-18: `ksymm_scf_common_init` (khf_ksymm.py:142-149) sets
        // `use_ao_symmetry = cell.dimension == 3 and use_ao_symmetry and not
        // kpts.time_reversal and kpts.symmorphic and len(kpts.little_cogroup_ops) > 0`
        // and builds `cell.symm_orb` only when that holds. Without the
        // time-reversal clause the binding forced the AO-symmetry route (and its
        // D-17-07-01 refusal) on `examples/pbc/22-k_points_mp2_ksymm.py`.
        let mut use_ao_symmetry = true;
        let df_kpts: Bound<'_, PyAny> = match kpoints {
            Some(kp) => {
                let kp = kp.cast::<PyKPoints>()?.borrow();
                let inner = kp.kpoints();
                if inner.kpts_ibz.is_empty() {
                    return Err(PyValueError::new_err(
                        "KsymAdaptedKRHF: the KPoints object is not built (call build())",
                    ));
                }
                // `ksymm_scf_common_init` (khf_ksymm.py:142): `use_ao_symmetry`
                // defaults to True and needs `cell.build_symmetry(kpts)`. Upstream
                // mutates the user's cell; here the default FFTDF is built over a
                // COPY carrying `symm_orb` (the DF owns the cell the driver reads).
                // A failure (e.g. time-reversal `little_cogroup_ops`, 17-07) is
                // left to the kernel, which names it, so `use_ao_symmetry = False`
                // remains reachable.
                let mut sym = extract_cell_from_pyany(py, cell)?;
                use_ao_symmetry = sym.dimension == 3
                    && !inner.time_reversal
                    && inner.symmetry.symmorphic
                    && !inner.little_cogroup_ops.is_empty();
                if use_ao_symmetry {
                    let input = pyscf_pbc_symm::basis::SymmAdaptedBasisInput {
                        kpts_scaled_ibz: inner.kpts_scaled_ibz.clone(),
                        little_cogroup_ops: inner.little_cogroup_ops.clone(),
                        ops: inner.symmetry.ops.clone(),
                        dmats: inner.symmetry.dmats.clone(),
                    };
                    if pyscf_pbc_symm::basis::build_symmetry(&mut sym, &input).is_ok() {
                        df_cell = Py::new(py, PyCell::from_cell(sym))?
                            .into_bound(py)
                            .into_any();
                    }
                }
                kpts_to_pyarray(py, &inner.kpts)?.into_any()
            }
            None => match some(kpts) {
                Some(k) => k.clone(),
                None => kpts_to_pyarray(py, &[[0.0; 3]])?.into_any(),
            },
        };
        let with_df = py.get_type::<PyFftdf>().call1((&df_cell, df_kpts))?;
        let conv_tol = {
            let df = extract_df(&with_df)?;
            KScfConfig::for_cell(df.cell()).conv_tol
        };
        let d = KScfConfig::default();
        Ok(Self {
            kind,
            py_cell: cell.clone().unbind(),
            with_df: with_df.unbind(),
            kpoints: kpoints.map(|k| k.clone().unbind()),
            exxdiv: match exxdiv {
                None => None,
                Some(s) => parse_exxdiv(s)?,
            },
            conv_tol,
            conv_tol_grad: d.conv_tol_grad,
            max_cycle: d.max_cycle,
            diis: d.diis,
            diis_space: d.diis_space,
            diis_start_cycle: d.diis_start_cycle,
            damp: d.damp,
            level_shift: d.level_shift,
            init_guess: "minao".into(),
            chkfile: None,
            verbose: 3,
            smearing: None,
            nelec: None,
            init_guess_breaksym: 1,
            use_ao_symmetry,
            solved: None,
            overridden: Vec::new(),
        })
    }

    /// A Rust driver over the CURRENT `with_df` (20-10 contract).
    fn driver(&self, py: Python<'_>) -> PyResult<Driver> {
        let df = extract_df(self.with_df.bind(py))?;
        Ok(match self.kind {
            Kind::Rhf => {
                let mut d = Krhf::from_df(df);
                d.exxdiv = self.exxdiv;
                d.smearing = self.smearing.clone();
                Driver::Rhf(d)
            }
            Kind::KsymRhf => {
                let kp_obj = self
                    .kpoints
                    .as_ref()
                    .ok_or_else(|| PyValueError::new_err("KsymAdaptedKRHF has no KPoints"))?;
                let kp = kp_obj
                    .bind(py)
                    .cast::<PyKPoints>()?
                    .borrow()
                    .kpoints()
                    .clone();
                if df.kpts().len() != kp.nkpts() {
                    return Err(PyValueError::new_err(format!(
                        "KsymAdaptedKRHF: with_df samples {} k-points but the KPoints full \
                         Brillouin zone has {}; with_df must be built over kpts.kpts",
                        df.kpts().len(),
                        kp.nkpts()
                    )));
                }
                let mut d = KsymAdaptedKrhf::from_df(df, kp);
                d.exxdiv = self.exxdiv;
                d.use_ao_symmetry = self.use_ao_symmetry;
                Driver::KsymRhf(d)
            }
            Kind::Uhf => {
                let mut d = Kuhf::from_df(df);
                d.exxdiv = self.exxdiv;
                d.smearing = self.smearing.clone();
                d.nelec = self.nelec;
                d.init_guess_breaksym = self.init_guess_breaksym;
                Driver::Uhf(d)
            }
            Kind::Rohf => {
                let mut d = Krohf::from_df(df);
                d.exxdiv = self.exxdiv;
                d.nelec = self.nelec;
                Driver::Rohf(d)
            }
            Kind::Ghf => {
                let mut d = Kghf::from_df(df);
                d.exxdiv = self.exxdiv;
                Driver::Ghf(d)
            }
        })
    }

    fn solved(&self) -> PyResult<&Solved> {
        self.solved.as_ref().ok_or_else(|| {
            PyValueError::new_err(format!(
                "{}: no SCF result yet — call kernel() first",
                self.kind.name()
            ))
        })
    }

    fn cfg(&self, init_guess: KInitGuess) -> KScfConfig {
        KScfConfig {
            conv_tol: self.conv_tol,
            conv_tol_grad: self.conv_tol_grad,
            max_cycle: self.max_cycle,
            diis: self.diis,
            diis_space: self.diis_space,
            diis_start_cycle: self.diis_start_cycle,
            damp: self.damp,
            level_shift: self.level_shift,
            init_guess,
            chkfile: None,
            verbose: self.verbose >= 5,
        }
    }

    /// The density of the stored result, through the driver's `make_rdm1`.
    fn stored_dm(&self, d: &Driver) -> PyResult<KDms> {
        let s = self.solved()?;
        d.make_rdm1(&s.res.mo_coeff, &s.res.mo_occ)
            .map_err(pyscf_to_py)
    }

    fn dms_arg(&self, d: &Driver, x: Option<&Bound<'_, PyAny>>, what: &str) -> PyResult<KDms> {
        match some(x) {
            Some(v) => kdms_from_py(v, d.nset(), d.kpts().len(), KOverrideHooks::nao(d), what),
            None => self.stored_dm(d),
        }
    }

    /// `k`-points given explicitly to a one-electron hook: `(kpts, single)`.
    fn explicit_kpts(
        &self,
        kpts: Option<&Bound<'_, PyAny>>,
        what: &str,
    ) -> PyResult<Option<(Vec<[f64; 3]>, bool)>> {
        let Some(k) = extract_kpts_opt(kpts)? else {
            return Ok(None);
        };
        if self.kind == Kind::Ghf {
            return Err(not_impl(format!(
                "KGHF.{what}(kpts=...): only the driver's own k-points are bound"
            )));
        }
        Ok(Some(k))
    }

    /// Run the SCF, through the bridge (`use_bridge`) or the concrete
    /// driver's own `kernel`, and store the result.
    fn run_kernel(
        slf: &Bound<'_, Self>,
        dm0: Option<&Bound<'_, PyAny>>,
        use_bridge: bool,
    ) -> PyResult<f64> {
        let py = slf.py();
        let (d, py_cell, cfg_guess, chkfile) = {
            let me = slf.borrow();
            let d = me.driver(py)?;
            let (nk, nao, nset) = (d.kpts().len(), KOverrideHooks::nao(&d), d.nset());
            let guess = match some(dm0) {
                Some(x) => KInitGuess::UserDm(kdms_from_py(x, nset, nk, nao, "dm0")?),
                None => match me.solved.as_ref() {
                    // hf.SCF.scf: restart from the existing wavefunction.
                    Some(_) => KInitGuess::UserDm(me.stored_dm(&d)?),
                    None if me.init_guess.eq_ignore_ascii_case("chkfile") => {
                        let path = me.chkfile.clone().ok_or_else(|| {
                            PyValueError::new_err("init_guess='chkfile' needs mf.chkfile")
                        })?;
                        KInitGuess::UserDm(chkfile_dm(&d, &path, None)?)
                    }
                    None => parse_init_guess(&me.init_guess)?,
                },
            };
            (
                d,
                me.py_cell.clone_ref(py),
                me.cfg(guess),
                me.chkfile.clone(),
            )
        };
        let (res, overridden) = if use_bridge {
            let base = py.get_type::<PyKscf>();
            let bridge =
                KPyOverrideBridge::new(py, slf.clone().into_any().unbind(), py_cell, &base, &d)?;
            let res = bridge.finish(pyscf_pbc_scf::kernel(&bridge, &cfg_guess))?;
            (res, bridge.overridden_hooks())
        } else {
            (
                d.kernel_direct(&cfg_guess).map_err(pyscf_to_py)?,
                Vec::new(),
            )
        };
        let nao = KOverrideHooks::nao(&d);
        if let Some(path) = chkfile.as_deref() {
            let json = pyscf_pbc_gto::dumps(d.cell()).map_err(pyscf_to_py)?;
            dump_kscf_to_file(std::path::Path::new(path), &res, d.kpts(), nao, &json)
                .map_err(chk_err)?;
        }
        let e_tot = res.e_tot;
        let mut me = slf.borrow_mut();
        me.solved = Some(Solved {
            nao,
            nfock: d.nfock(),
            kpts: d.kpts().to_vec(),
            sigma: me.smearing.as_ref().map(|s| s.sigma),
            res,
        });
        me.overridden = overridden;
        Ok(e_tot)
    }
}

/// The density a periodic chkfile implies for `d` —
/// `khf.init_guess_by_chkfile` / `kuhf.init_guess_by_chkfile`.
///
/// The stored k-points must be `d`'s (upstream's k-point remapping is not
/// ported). When the stored AO count differs from `d`'s, or `project` is
/// `true`, the orbitals are projected onto `d.cell()`'s basis with
/// `addons::project_mo_nr2nr` using the cell stored under `/mol`.
fn chkfile_dm(d: &Driver, path: &str, project: Option<bool>) -> PyResult<KDms> {
    let ck = load_kscf_from_file(std::path::Path::new(path)).map_err(chk_err)?;
    let kpts = d.kpts();
    let nk = kpts.len();
    let same_kpts = ck.kpts.len() == nk
        && ck
            .kpts
            .iter()
            .zip(kpts)
            .all(|(a, b)| (0..3).all(|i| (a[i] - b[i]).abs() < 1e-9));
    if !same_kpts || nk == 0 || ck.mo_coeff.len() % nk != 0 {
        return Err(not_impl(format!(
            "init_guess_by_chkfile: the chkfile holds {} k-points / {} MO blocks, the driver \
             samples {nk}; upstream's k-point remapping is not ported",
            ck.kpts.len(),
            ck.mo_coeff.len()
        )));
    }
    let nch = ck.mo_coeff.len() / nk;
    let nao = KOverrideHooks::nao(d);
    let project = project.unwrap_or(ck.nao != nao);
    let mut mo = ck.mo_coeff;
    if project {
        if matches!(d, Driver::Ghf(_)) {
            return Err(not_impl(
                "KGHF.init_guess_by_chkfile(project=True) is not bound",
            ));
        }
        let file = pyscf_chkfile::primitives::open_for_read(path).map_err(chk_err)?;
        let json = pyscf_chkfile::primitives::read_mol(&file).map_err(chk_err)?;
        let cell1 = pyscf_pbc_gto::loads(&json).map_err(pyscf_to_py)?;
        let mut projected = Vec::with_capacity(mo.len());
        for ch in 0..nch {
            projected.extend(
                pyscf_pbc_scf::addons::project_mo_nr2nr(
                    &cell1,
                    &mo[ch * nk..(ch + 1) * nk],
                    d.cell(),
                    kpts,
                )
                .map_err(pyscf_to_py)?,
            );
        }
        mo = projected;
    }
    let chans: Vec<KMats> = (0..nch)
        .map(|ch| {
            pyscf_pbc_scf::krdm::make_rdm1(
                &mo[ch * nk..(ch + 1) * nk],
                &ck.mo_occ[ch * nk..(ch + 1) * nk],
                nao,
            )
        })
        .collect();
    let scale = |m: &CTensor, f: f64| {
        CTensor::from_planes(
            m.re.iter().map(|v| v * f).collect(),
            m.im.iter().map(|v| v * f).collect(),
        )
    };
    Ok(match (d.nset(), nch) {
        (1, 1) | (2, 2) => chans,
        (1, 2) => vec![
            chans[0]
                .iter()
                .zip(&chans[1])
                .map(|(a, b)| {
                    let mut m = a.clone();
                    for i in 0..m.len() {
                        m.re[i] += b.re[i];
                        m.im[i] += b.im[i];
                    }
                    m
                })
                .collect(),
        ],
        (2, 1) => {
            let half: KMats = chans[0].iter().map(|m| scale(m, 0.5)).collect();
            vec![half.clone(), half]
        }
        (nset, nch) => {
            return Err(not_impl(format!(
                "init_guess_by_chkfile: {nch} stored channel(s) for a {nset}-channel driver"
            )));
        }
    })
}

fn parse_exxdiv(s: &str) -> PyResult<Option<ExxDiv>> {
    match s.to_ascii_lowercase().as_str() {
        "" | "none" | "false" => Ok(None),
        "ewald" => Ok(Some(ExxDiv::Ewald)),
        "vcut_sph" => Ok(Some(ExxDiv::VcutSph)),
        "vcut_ws" => Ok(Some(ExxDiv::VcutWs)),
        other => Err(PyValueError::new_err(format!(
            "exxdiv must be None, 'ewald', 'vcut_sph' or 'vcut_ws', got {other:?}"
        ))),
    }
}

fn parse_smearing_method(s: &str) -> PyResult<SmearingMethod> {
    match s.to_ascii_lowercase().as_str() {
        "fermi" | "fermi-dirac" | "fd" => Ok(SmearingMethod::Fermi),
        "gauss" | "gaussian" => Ok(SmearingMethod::Gaussian),
        other => Err(not_impl(format!(
            "smearing method {other:?} is not ported (fermi, gaussian)"
        ))),
    }
}

/// A per-k list, or its single element when the caller passed one `(3,)` k-point.
fn maybe_single<'py>(list: Bound<'py, PyAny>, single: bool) -> PyResult<Bound<'py, PyAny>> {
    if single { list.get_item(0) } else { Ok(list) }
}

#[pymethods]
impl PyKscf {
    // ── configuration (upstream's `mf.conv_tol = ...` idiom) ────────────────

    #[getter]
    fn cell(&self, py: Python<'_>) -> Py<PyAny> {
        self.py_cell.clone_ref(py)
    }

    /// The density-fitting object (`mf.with_df is mydf`). Assigning validates
    /// it is a `pyscf.pbc.df` builder; the next call uses it.
    #[getter]
    fn with_df(&self, py: Python<'_>) -> Py<PyAny> {
        self.with_df.clone_ref(py)
    }
    #[setter]
    fn set_with_df(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        extract_df(v)?;
        self.with_df = v.clone().unbind();
        Ok(())
    }

    /// `(nkpts, 3)` sampling k-points (the `with_df`'s); the `KPoints` object
    /// for `KsymAdaptedKRHF`. Assigning an array re-targets `with_df.kpts`.
    #[getter]
    fn kpts(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if let Some(kp) = &self.kpoints {
            return Ok(kp.clone_ref(py));
        }
        Ok(self.with_df.bind(py).getattr("kpts")?.unbind())
    }
    #[setter]
    fn set_kpts(&mut self, py: Python<'_>, v: &Bound<'_, PyAny>) -> PyResult<()> {
        if v.cast::<PyKPoints>().is_ok() || self.kind == Kind::KsymRhf {
            return Err(not_impl(
                "switching between a k-point array and a KPoints object after construction is \
                 not bound; construct a new driver",
            ));
        }
        self.with_df.bind(py).setattr("kpts", v)?;
        self.solved = None;
        Ok(())
    }

    #[getter]
    fn exxdiv(&self) -> Option<&'static str> {
        self.exxdiv.map(ExxDiv::as_str)
    }
    #[setter]
    fn set_exxdiv(&mut self, v: Option<&str>) -> PyResult<()> {
        self.exxdiv = match v {
            None => None,
            Some(s) => parse_exxdiv(s)?,
        };
        Ok(())
    }

    #[getter]
    fn conv_tol(&self) -> f64 {
        self.conv_tol
    }
    #[setter]
    fn set_conv_tol(&mut self, v: f64) {
        self.conv_tol = v;
    }
    /// `None` means `sqrt(conv_tol)` (upstream's rule).
    #[getter]
    fn conv_tol_grad(&self) -> Option<f64> {
        self.conv_tol_grad
    }
    #[setter]
    fn set_conv_tol_grad(&mut self, v: Option<f64>) {
        self.conv_tol_grad = v;
    }
    #[getter]
    fn max_cycle(&self) -> u32 {
        self.max_cycle
    }
    #[setter]
    fn set_max_cycle(&mut self, v: u32) {
        self.max_cycle = v;
    }
    /// `True`/`False` (Pulay C-DIIS on/off). A DIIS object is not accepted.
    #[getter]
    fn diis(&self) -> bool {
        self.diis
    }
    #[setter]
    fn set_diis(&mut self, v: Option<bool>) {
        self.diis = v.unwrap_or(false);
    }
    #[getter]
    fn diis_space(&self) -> usize {
        self.diis_space
    }
    #[setter]
    fn set_diis_space(&mut self, v: usize) {
        self.diis_space = v;
    }
    #[getter]
    fn diis_start_cycle(&self) -> u32 {
        self.diis_start_cycle
    }
    #[setter]
    fn set_diis_start_cycle(&mut self, v: u32) {
        self.diis_start_cycle = v;
    }
    #[getter]
    fn damp(&self) -> f64 {
        self.damp
    }
    #[setter]
    fn set_damp(&mut self, v: f64) {
        self.damp = v;
    }
    #[getter]
    fn level_shift(&self) -> f64 {
        self.level_shift
    }
    #[setter]
    fn set_level_shift(&mut self, v: f64) {
        self.level_shift = v;
    }
    /// `'minao'`, `'atom'`, `'1e'`/`'hcore'` or `'chkfile'` (reads `mf.chkfile`).
    #[getter(init_guess)]
    fn init_guess_key(&self) -> &str {
        &self.init_guess
    }
    #[setter(init_guess)]
    fn set_init_guess_key(&mut self, v: String) -> PyResult<()> {
        if !v.eq_ignore_ascii_case("chkfile") {
            parse_init_guess(&v)?;
        }
        self.init_guess = v;
        Ok(())
    }
    /// HDF5 checkpoint path; when set, `kernel()` writes the result there.
    #[getter]
    fn chkfile(&self) -> Option<String> {
        self.chkfile.clone()
    }
    #[setter]
    fn set_chkfile(&mut self, v: Option<String>) {
        self.chkfile = v;
    }
    #[getter]
    fn verbose(&self) -> i64 {
        self.verbose
    }
    #[setter]
    fn set_verbose(&mut self, v: i64) {
        self.verbose = v;
    }

    /// `(nalpha, nbeta)` override (KUHF/KROHF); `None` derives it from `cell.spin`.
    #[getter]
    fn nelec(&self, py: Python<'_>) -> PyResult<Option<(usize, usize)>> {
        match self.kind {
            Kind::Uhf | Kind::Rohf => match self.nelec {
                Some(n) => Ok(Some(n)),
                None => Ok(Some(match self.driver(py)? {
                    Driver::Uhf(d) => d.nelec().map_err(pyscf_to_py)?,
                    Driver::Rohf(d) => d.nelec().map_err(pyscf_to_py)?,
                    _ => unreachable!("kind checked above"),
                })),
            },
            _ => Err(PyAttributeError::new_err(format!(
                "{} has no attribute 'nelec'",
                self.kind.name()
            ))),
        }
    }
    #[setter]
    fn set_nelec(&mut self, v: Option<(usize, usize)>) -> PyResult<()> {
        if !matches!(self.kind, Kind::Uhf | Kind::Rohf) {
            return Err(PyAttributeError::new_err(format!(
                "{} has no attribute 'nelec'",
                self.kind.name()
            )));
        }
        self.nelec = v;
        Ok(())
    }

    /// `init_guess_breaksym` (KUHF only; upstream default 1).
    #[getter]
    fn init_guess_breaksym(&self) -> PyResult<i32> {
        if self.kind != Kind::Uhf {
            return Err(PyAttributeError::new_err(
                "only KUHF has init_guess_breaksym",
            ));
        }
        Ok(self.init_guess_breaksym)
    }
    #[setter]
    fn set_init_guess_breaksym(&mut self, v: i32) -> PyResult<()> {
        if self.kind != Kind::Uhf {
            return Err(PyAttributeError::new_err(
                "only KUHF has init_guess_breaksym",
            ));
        }
        self.init_guess_breaksym = v;
        Ok(())
    }

    /// `use_ao_symmetry` (KsymAdaptedKRHF only; upstream default `True`).
    #[getter]
    fn use_ao_symmetry(&self) -> PyResult<bool> {
        if self.kind != Kind::KsymRhf {
            return Err(PyAttributeError::new_err(
                "only KsymAdaptedKRHF has use_ao_symmetry",
            ));
        }
        Ok(self.use_ao_symmetry)
    }
    #[setter]
    fn set_use_ao_symmetry(&mut self, v: bool) -> PyResult<()> {
        if self.kind != Kind::KsymRhf {
            return Err(PyAttributeError::new_err(
                "only KsymAdaptedKRHF has use_ao_symmetry",
            ));
        }
        self.use_ao_symmetry = v;
        Ok(())
    }

    // ── smearing (`addons.smearing_`, `smearing.rs:34`) ─────────────────────

    /// `mf.smearing_(sigma=None, method='fermi', mu0=None)` — attach smearing IN
    /// PLACE and return `self` (KRHF/KUHF only; `sigma=None` or `0` detaches).
    #[pyo3(signature = (sigma = None, method = "fermi", mu0 = None))]
    fn smearing_<'py>(
        slf: &Bound<'py, Self>,
        sigma: Option<f64>,
        method: &str,
        mu0: Option<f64>,
    ) -> PyResult<Bound<'py, Self>> {
        let mut me = slf.borrow_mut();
        if !matches!(me.kind, Kind::Rhf | Kind::Uhf) {
            return Err(not_impl(format!(
                "{}.smearing_: smearing is ported for KRHF and KUHF only (addons.rs:26,35)",
                me.kind.name()
            )));
        }
        let method = parse_smearing_method(method)?;
        me.smearing = match sigma {
            Some(s) if s != 0.0 => Some(Smearing {
                sigma: s,
                method,
                mu0,
            }),
            _ => None,
        };
        drop(me);
        Ok(slf.clone())
    }

    /// Alias of `smearing_` (upstream's `mf.smearing(...)` returns a new
    /// object; this port mutates and returns `self`).
    #[pyo3(signature = (sigma = None, method = "fermi", mu0 = None))]
    fn smearing<'py>(
        slf: &Bound<'py, Self>,
        sigma: Option<f64>,
        method: &str,
        mu0: Option<f64>,
    ) -> PyResult<Bound<'py, Self>> {
        Self::smearing_(slf, sigma, method, mu0)
    }

    #[getter]
    fn sigma(&self) -> Option<f64> {
        self.smearing.as_ref().map(|s| s.sigma)
    }
    /// 20-18: `mf.sigma = x` after `smearing(...)` — upstream `_SmearingSCF.sigma`
    /// is a plain attribute read by `get_occ`/`energy_tot` on the next kernel
    /// (pyscf/scf/smearing.py:130,151,258; `examples/pbc/23-smearing.py:47-51`).
    /// `0` (or `None`) disables smearing, as upstream's `get_occ` falls back to
    /// integer occupations at `sigma == 0` (smearing.py:151-153); here that drops
    /// the smearing configuration, exactly like `smearing_(sigma=0)`. A non-zero
    /// value on an object without smearing raises `AttributeError` (upstream
    /// would store an attribute nothing reads).
    #[setter]
    fn set_sigma(&mut self, v: Option<f64>) -> PyResult<()> {
        match (v, self.smearing.as_mut()) {
            (None | Some(0.0), _) => {
                self.smearing = None;
                Ok(())
            }
            (Some(x), Some(sm)) => {
                sm.sigma = x;
                Ok(())
            }
            (Some(_), None) => Err(PyAttributeError::new_err(
                "sigma: this object has no smearing; call mf.smearing_(sigma, method) first",
            )),
        }
    }
    #[getter]
    fn smearing_method(&self) -> Option<&'static str> {
        self.smearing.as_ref().map(|s| match s.method {
            SmearingMethod::Fermi => "fermi",
            SmearingMethod::Gaussian => "gaussian",
        })
    }

    // ── results (`KScfResult`, `types.rs:108`) ──────────────────────────────

    /// Total energy; `0.0` before `kernel()` (upstream's initial value).
    #[getter]
    fn e_tot(&self) -> f64 {
        self.solved.as_ref().map_or(0.0, |s| s.res.e_tot)
    }
    #[getter]
    fn e_elec(&self) -> Option<f64> {
        self.solved.as_ref().map(|s| s.res.e_elec)
    }
    #[getter]
    fn e_coul(&self) -> Option<f64> {
        self.solved.as_ref().map(|s| s.res.e_coul)
    }
    #[getter]
    fn e_nuc(&self) -> Option<f64> {
        self.solved.as_ref().map(|s| s.res.e_nuc)
    }
    #[getter]
    fn converged(&self) -> bool {
        self.solved.as_ref().is_some_and(|s| s.res.converged)
    }
    #[getter]
    fn cycles(&self) -> Option<u32> {
        self.solved.as_ref().map(|s| s.res.cycles)
    }
    /// Fermi level per channel from the last `get_occ`.
    #[getter]
    fn fermi(&self) -> Option<Vec<f64>> {
        self.solved.as_ref().map(|s| s.res.fermi.clone())
    }
    /// Smearing chemical potential (upstream `mf.mu`).
    #[getter]
    fn mu(&self) -> Option<f64> {
        self.solved
            .as_ref()
            .filter(|s| s.sigma.is_some())
            .and_then(|s| s.res.fermi.first().copied())
    }
    #[getter]
    fn e_free(&self) -> Option<f64> {
        self.solved.as_ref().and_then(|s| s.res.e_free)
    }
    #[getter]
    fn e_zero(&self) -> Option<f64> {
        self.solved.as_ref().and_then(|s| s.res.e_zero)
    }
    /// Smearing entropy `S` (`e_free = e_tot - sigma * S`).
    #[getter]
    fn entropy(&self) -> Option<f64> {
        let s = self.solved.as_ref()?;
        Some((s.res.e_tot - s.res.e_free?) / s.sigma?)
    }
    /// Per-k `mo_energy` (nested per spin for KUHF).
    #[getter]
    fn mo_energy<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.solved
            .as_ref()
            .map(|s| mo_values_to_py(py, &s.res.mo_energy, s.nfock))
            .transpose()
    }
    #[getter]
    fn mo_occ<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.solved
            .as_ref()
            .map(|s| mo_values_to_py(py, &s.res.mo_occ, s.nfock))
            .transpose()
    }
    /// Per-k complex `(nao, nmo)` arrays (20-07 boundary, column-major source).
    #[getter]
    fn mo_coeff<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.solved
            .as_ref()
            .map(|s| mo_coeff_to_py(py, &s.res.mo_coeff, s.nfock, s.nao))
            .transpose()
    }
    /// PRIVATE (20-11 dispatch tests): hooks the last bridged kernel sent to Python.
    #[getter]
    fn _overridden_hooks(&self) -> Vec<&'static str> {
        self.overridden.clone()
    }

    // ── the eleven hooks (their Rust defaults) ──────────────────────────────

    /// `get_ovlp(cell=None, kpts=None)` — per-k list (one array for a `(3,)` kpt).
    #[pyo3(signature = (cell = None, kpts = None))]
    fn get_ovlp<'py>(
        &self,
        py: Python<'py>,
        cell: Option<&Bound<'py, PyAny>>,
        kpts: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = cell;
        let d = self.driver(py)?;
        let nao = KOverrideHooks::nao(&d);
        match self.explicit_kpts(kpts, "get_ovlp")? {
            None => kmats_to_py(py, &d.get_ovlp().map_err(pyscf_to_py)?, nao),
            Some((k, single)) => {
                let s = pyscf_pbc_gto::get_ovlp_scf(d.cell(), &k).map_err(pyscf_to_py)?;
                maybe_single(kmats_to_py(py, &to_row_major(s, nao), nao)?, single)
            }
        }
    }

    /// `get_hcore(cell=None, kpts=None)` — per-k list (one array for a `(3,)` kpt).
    #[pyo3(signature = (cell = None, kpts = None))]
    fn get_hcore<'py>(
        &self,
        py: Python<'py>,
        cell: Option<&Bound<'py, PyAny>>,
        kpts: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = cell;
        let d = self.driver(py)?;
        let nao = KOverrideHooks::nao(&d);
        match self.explicit_kpts(kpts, "get_hcore")? {
            None => kmats_to_py(py, &d.get_hcore().map_err(pyscf_to_py)?, nao),
            Some((k, single)) => {
                let h = pyscf_pbc_df::get_hcore(d.df(), &k).map_err(pbc_df_to_py)?;
                maybe_single(kmats_to_py(py, &h, nao)?, single)
            }
        }
    }

    /// `get_init_guess(cell=None, key='minao', s1e=None)`.
    #[pyo3(signature = (cell = None, key = "minao", s1e = None))]
    fn get_init_guess<'py>(
        &self,
        py: Python<'py>,
        cell: Option<&Bound<'py, PyAny>>,
        key: &str,
        s1e: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = cell;
        let d = self.driver(py)?;
        let (nk, nao) = (d.kpts().len(), KOverrideHooks::nao(&d));
        if key.eq_ignore_ascii_case("chkfile") {
            let path = self.chkfile.clone().ok_or_else(|| {
                PyValueError::new_err("get_init_guess('chkfile') needs mf.chkfile")
            })?;
            return kdms_to_py(py, &chkfile_dm(&d, &path, None)?, nao);
        }
        let mode = parse_init_guess(key)?;
        let s1e = match some(s1e) {
            Some(s) => kmats_from_py(s, nk, nao, "s1e")?,
            None => d.get_ovlp().map_err(pyscf_to_py)?,
        };
        let dm = d.get_init_guess(&mode, &s1e).map_err(pyscf_to_py)?;
        kdms_to_py(py, &dm, nao)
    }

    /// `get_veff(cell=None, dm_kpts=None, dm_last=0, vhf_last=0, hermi=1,
    /// kpts=None, kpts_band=None)`. `dm_kpts=None` uses the stored result.
    #[pyo3(signature = (cell = None, dm_kpts = None, dm_last = None, vhf_last = None,
                        hermi = 1, kpts = None, kpts_band = None))]
    #[allow(clippy::too_many_arguments)]
    fn get_veff<'py>(
        &self,
        py: Python<'py>,
        cell: Option<&Bound<'py, PyAny>>,
        dm_kpts: Option<&Bound<'py, PyAny>>,
        dm_last: Option<&Bound<'py, PyAny>>,
        vhf_last: Option<&Bound<'py, PyAny>>,
        hermi: i32,
        kpts: Option<&Bound<'py, PyAny>>,
        kpts_band: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = (cell, dm_last, vhf_last);
        if hermi != 1 || some(kpts).is_some() || some(kpts_band).is_some() {
            return Err(not_impl(format!(
                "{}.get_veff: only hermi=1 at the driver's own k-points is bound \
                 (use get_bands for band k-points)",
                self.kind.name()
            )));
        }
        let d = self.driver(py)?;
        let dms = self.dms_arg(&d, dm_kpts, "dm_kpts")?;
        let v = d.get_veff(&dms).map_err(pyscf_to_py)?;
        kdms_to_py(py, &v, KOverrideHooks::nao(&d))
    }

    /// `get_fock(h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, ...)` — the
    /// bare Fock (KROHF: the Roothaan effective Fock). Damping, DIIS and level
    /// shift belong to the kernel, as at upstream's `cycle=-1`.
    #[pyo3(signature = (h1e = None, s1e = None, vhf = None, dm = None, cycle = -1, diis = None,
                        diis_start_cycle = None, level_shift_factor = None, damp_factor = None,
                        fock_last = None))]
    #[allow(clippy::too_many_arguments)]
    fn get_fock<'py>(
        &self,
        py: Python<'py>,
        h1e: Option<&Bound<'py, PyAny>>,
        s1e: Option<&Bound<'py, PyAny>>,
        vhf: Option<&Bound<'py, PyAny>>,
        dm: Option<&Bound<'py, PyAny>>,
        cycle: i64,
        diis: Option<&Bound<'py, PyAny>>,
        diis_start_cycle: Option<&Bound<'py, PyAny>>,
        level_shift_factor: Option<&Bound<'py, PyAny>>,
        damp_factor: Option<&Bound<'py, PyAny>>,
        fock_last: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = (
            s1e,
            diis,
            diis_start_cycle,
            level_shift_factor,
            damp_factor,
            fock_last,
        );
        if cycle >= 0 {
            return Err(not_impl(format!(
                "{}.get_fock: only the bare Fock (cycle=-1) is bound",
                self.kind.name()
            )));
        }
        let d = self.driver(py)?;
        let (nk, nao, nset) = (d.kpts().len(), KOverrideHooks::nao(&d), d.nset());
        let h1e = match some(h1e) {
            Some(x) => kmats_from_py(x, nk, nao, "h1e")?,
            None => d.get_hcore().map_err(pyscf_to_py)?,
        };
        let dms = match some(dm) {
            Some(x) => kdms_from_py(x, nset, nk, nao, "dm")?,
            None if some(vhf).is_some() && self.solved.is_none() => Vec::new(),
            None => self.stored_dm(&d)?,
        };
        let vhf = match some(vhf) {
            Some(x) => kdms_from_py(x, nset, nk, nao, "vhf")?,
            None => d.get_veff(&dms).map_err(pyscf_to_py)?,
        };
        let f = d.get_fock(&h1e, &vhf, &dms).map_err(pyscf_to_py)?;
        kdms_to_py(py, &f, nao)
    }

    /// `eig(h_kpts, s_kpts)` → `(mo_energy, mo_coeff)`.
    fn eig<'py>(
        &self,
        py: Python<'py>,
        h_kpts: &Bound<'py, PyAny>,
        s_kpts: &Bound<'py, PyAny>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let d = self.driver(py)?;
        let (nk, nao, nfock) = (d.kpts().len(), KOverrideHooks::nao(&d), d.nfock());
        let fock = kdms_from_py(h_kpts, nfock, nk, nao, "h_kpts")?;
        let s1e = kmats_from_py(s_kpts, nk, nao, "s_kpts")?;
        let (e, c) = d.eig(&fock, &s1e).map_err(pyscf_to_py)?;
        Ok((
            mo_values_to_py(py, &e, nfock)?,
            mo_coeff_to_py(py, &c, nfock, nao)?,
        ))
    }

    /// `get_occ(mo_energy_kpts=None, mo_coeff_kpts=None)` → `mo_occ`.
    #[pyo3(signature = (mo_energy_kpts = None, mo_coeff_kpts = None))]
    fn get_occ<'py>(
        &self,
        py: Python<'py>,
        mo_energy_kpts: Option<&Bound<'py, PyAny>>,
        mo_coeff_kpts: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = mo_coeff_kpts;
        let d = self.driver(py)?;
        let nfock = d.nfock();
        let e = match some(mo_energy_kpts) {
            Some(x) => mo_values_from_py(x, nfock, d.kpts().len(), "mo_energy_kpts")?,
            None => self.solved()?.res.mo_energy.clone(),
        };
        let (occ, _fermi) = d.get_occ(&e).map_err(pyscf_to_py)?;
        mo_values_to_py(py, &occ, nfock)
    }

    /// `make_rdm1(mo_coeff_kpts=None, mo_occ_kpts=None)`.
    #[pyo3(signature = (mo_coeff_kpts = None, mo_occ_kpts = None))]
    fn make_rdm1<'py>(
        &self,
        py: Python<'py>,
        mo_coeff_kpts: Option<&Bound<'py, PyAny>>,
        mo_occ_kpts: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let d = self.driver(py)?;
        let (nk, nao, nfock) = (d.kpts().len(), KOverrideHooks::nao(&d), d.nfock());
        let c = match some(mo_coeff_kpts) {
            Some(x) => mo_coeff_from_py(x, nfock, nk, nao, "mo_coeff_kpts")?,
            None => self.solved()?.res.mo_coeff.clone(),
        };
        let occ = match some(mo_occ_kpts) {
            Some(x) => mo_values_from_py(x, nfock, nk, "mo_occ_kpts")?,
            None => self.solved()?.res.mo_occ.clone(),
        };
        let dm = d.make_rdm1(&c, &occ).map_err(pyscf_to_py)?;
        kdms_to_py(py, &dm, nao)
    }

    /// `energy_elec(dm_kpts=None, h1e_kpts=None, vhf_kpts=None)` → `(e_elec, e_coul)`.
    #[pyo3(signature = (dm_kpts = None, h1e_kpts = None, vhf_kpts = None))]
    fn energy_elec(
        &self,
        py: Python<'_>,
        dm_kpts: Option<&Bound<'_, PyAny>>,
        h1e_kpts: Option<&Bound<'_, PyAny>>,
        vhf_kpts: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(f64, f64)> {
        let d = self.driver(py)?;
        let (nk, nao, nset) = (d.kpts().len(), KOverrideHooks::nao(&d), d.nset());
        let dm = self.dms_arg(&d, dm_kpts, "dm_kpts")?;
        let h1e = match some(h1e_kpts) {
            Some(x) => kmats_from_py(x, nk, nao, "h1e_kpts")?,
            None => d.get_hcore().map_err(pyscf_to_py)?,
        };
        let vhf = match some(vhf_kpts) {
            Some(x) => kdms_from_py(x, nset, nk, nao, "vhf_kpts")?,
            None => d.get_veff(&dm).map_err(pyscf_to_py)?,
        };
        d.energy_elec(&dm, &h1e, &vhf).map_err(pyscf_to_py)
    }

    /// `energy_nuc()` — the Ewald nuclear repulsion.
    fn energy_nuc(&self, py: Python<'_>) -> PyResult<f64> {
        self.driver(py)?.energy_nuc().map_err(pyscf_to_py)
    }

    /// `energy_tot(dm_kpts=None, h1e_kpts=None, vhf_kpts=None)`.
    #[pyo3(signature = (dm_kpts = None, h1e_kpts = None, vhf_kpts = None))]
    fn energy_tot(
        &self,
        py: Python<'_>,
        dm_kpts: Option<&Bound<'_, PyAny>>,
        h1e_kpts: Option<&Bound<'_, PyAny>>,
        vhf_kpts: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<f64> {
        let (e, _) = self.energy_elec(py, dm_kpts, h1e_kpts, vhf_kpts)?;
        Ok(e + self.energy_nuc(py)?)
    }

    /// `get_grad(mo_coeff_kpts, mo_occ_kpts, fock=None)` — 1-D orbital gradient.
    #[pyo3(signature = (mo_coeff_kpts, mo_occ_kpts, fock = None))]
    fn get_grad<'py>(
        &self,
        py: Python<'py>,
        mo_coeff_kpts: &Bound<'py, PyAny>,
        mo_occ_kpts: &Bound<'py, PyAny>,
        fock: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let d = self.driver(py)?;
        let (nk, nao, nfock, nset) = (d.kpts().len(), KOverrideHooks::nao(&d), d.nfock(), d.nset());
        let c = mo_coeff_from_py(mo_coeff_kpts, nfock, nk, nao, "mo_coeff_kpts")?;
        let occ = mo_values_from_py(mo_occ_kpts, nfock, nk, "mo_occ_kpts")?;
        let g = match some(fock) {
            Some(f) => {
                let fock = kdms_from_py(f, nset, nk, nao, "fock")?;
                // The trait takes (h1e, vhf) and builds h1e + vhf; hand it the
                // Fock as `vhf` over a zero `h1e` (x + 0.0 == x).
                let zero: KMats = (0..nk).map(|_| CTensor::zeros(nao * nao)).collect();
                d.get_grad(&c, &occ, &zero, &fock)
            }
            None => {
                let h1e = d.get_hcore().map_err(pyscf_to_py)?;
                let dm = d.make_rdm1(&c, &occ).map_err(pyscf_to_py)?;
                let vhf = d.get_veff(&dm).map_err(pyscf_to_py)?;
                d.get_grad(&c, &occ, &h1e, &vhf)
            }
        };
        Ok(numpy::PyArray1::from_vec(py, g))
    }

    // ── kernel ──────────────────────────────────────────────────────────────

    /// `kernel(dm0=None)` → `e_tot`. Drives `pyscf_pbc_scf::kernel` through
    /// `KPyOverrideBridge`, so Python subclass overrides of the eleven hooks
    /// are honoured. A second call restarts from the stored wavefunction
    /// (`hf.SCF.scf`). Writes `mf.chkfile` when set.
    #[pyo3(signature = (dm0 = None, **kwargs))]
    fn kernel(
        slf: &Bound<'_, Self>,
        dm0: Option<&Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<f64> {
        if let Some(kw) = kwargs
            && !kw.is_empty()
        {
            return Err(PyTypeError::new_err(format!(
                "kernel(): keyword arguments {:?} are not bound (set attributes instead)",
                kw.keys().to_string()
            )));
        }
        Self::run_kernel(slf, dm0, true)
    }

    /// Alias of `kernel` (`hf.SCF.scf`).
    #[pyo3(signature = (dm0 = None))]
    fn scf(slf: &Bound<'_, Self>, dm0: Option<&Bound<'_, PyAny>>) -> PyResult<f64> {
        Self::run_kernel(slf, dm0, true)
    }

    /// PRIVATE: the concrete Rust driver's own `kernel()` with no bridge — the
    /// bitwise reference the dispatch tests compare against.
    #[pyo3(signature = (dm0 = None))]
    fn _kernel_without_bridge(
        slf: &Bound<'_, Self>,
        dm0: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<f64> {
        Self::run_kernel(slf, dm0, false)
    }

    /// `run(*args, **kwargs)` — set attributes from `kwargs`, run `kernel(*args)`,
    /// return `self` (`lib.StreamObject.run`).
    #[pyo3(signature = (*args, **kwargs))]
    fn run<'py>(
        slf: &Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        if let Some(kw) = kwargs {
            for (k, v) in kw.iter() {
                slf.setattr(k.cast::<pyo3::types::PyString>()?, v)?;
            }
        }
        slf.call_method1("kernel", args)?;
        Ok(slf.clone())
    }

    // ── bands, chkfile, analysis ────────────────────────────────────────────

    /// `get_bands(kpts_band, cell=None, dm_kpts=None, kpts=None)` →
    /// `(mo_energy, mo_coeff)` at arbitrary k-points (`Krhf::get_bands:106`,
    /// `Kuhf::get_bands:138`). A `(3,)` `kpts_band` returns single arrays
    /// (per spin for KUHF); `(n, 3)` returns per-k lists.
    #[pyo3(signature = (kpts_band, cell = None, dm_kpts = None, kpts = None))]
    fn get_bands<'py>(
        &self,
        py: Python<'py>,
        kpts_band: &Bound<'py, PyAny>,
        cell: Option<&Bound<'py, PyAny>>,
        dm_kpts: Option<&Bound<'py, PyAny>>,
        kpts: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let _ = cell;
        if some(kpts).is_some() {
            return Err(not_impl(
                "get_bands(kpts=...): the density's k-points are the driver's",
            ));
        }
        let (kband, single) = extract_kpts(kpts_band)?;
        let d = self.driver(py)?;
        let nao = KOverrideHooks::nao(&d);
        let dms = self.dms_arg(&d, dm_kpts, "dm_kpts")?;
        let (e, c, nch) = match &d {
            Driver::Rhf(x) => {
                let (e, c) = x.get_bands(&kband, &dms).map_err(pyscf_to_py)?;
                (e, c, 1)
            }
            Driver::Uhf(x) => {
                let (e, c) = x.get_bands(&kband, &dms).map_err(pyscf_to_py)?;
                (e, c, 2)
            }
            _ => {
                return Err(not_impl(format!(
                    "{}.get_bands is not ported (Krhf::get_bands and Kuhf::get_bands only)",
                    self.kind.name()
                )));
            }
        };
        let e_py = mo_values_to_py(py, &e, nch)?;
        let c_py = mo_coeff_to_py(py, &c, nch, nao)?;
        if !single {
            return Ok((e_py, c_py));
        }
        if nch == 1 {
            return Ok((e_py.get_item(0)?, c_py.get_item(0)?));
        }
        let pick = |x: &Bound<'py, PyAny>| -> PyResult<Bound<'py, PyAny>> {
            Ok(PyList::new(
                py,
                [x.get_item(0)?.get_item(0)?, x.get_item(1)?.get_item(0)?],
            )?
            .into_any())
        };
        Ok((pick(&e_py)?, pick(&c_py)?))
    }

    /// `dump_chk(envs_or_file=None)` — write the last result to a path (or to
    /// `mf.chkfile`), schema of `chkfile.rs:37`. Returns `self`.
    #[pyo3(signature = (envs_or_file = None))]
    fn dump_chk<'py>(
        slf: &Bound<'py, Self>,
        envs_or_file: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        let me = slf.borrow();
        let path = match some(envs_or_file).and_then(|v| v.extract::<String>().ok()) {
            Some(p) => p,
            None => me
                .chkfile
                .clone()
                .ok_or_else(|| PyValueError::new_err("dump_chk: no path and mf.chkfile is None"))?,
        };
        let s = me.solved()?;
        let d = me.driver(py)?;
        let json = pyscf_pbc_gto::dumps(d.cell()).map_err(pyscf_to_py)?;
        dump_kscf_to_file(std::path::Path::new(&path), &s.res, &s.kpts, s.nao, &json)
            .map_err(chk_err)?;
        drop(me);
        Ok(slf.clone())
    }

    /// `init_guess_by_chkfile(chk=None, project=None, kpts=None)` — the density
    /// from a periodic chkfile; projects across a basis change with
    /// `addons.project_mo_nr2nr`.
    #[pyo3(signature = (chk = None, project = None, kpts = None))]
    fn init_guess_by_chkfile<'py>(
        &self,
        py: Python<'py>,
        chk: Option<String>,
        project: Option<bool>,
        kpts: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if some(kpts).is_some() {
            return Err(not_impl("init_guess_by_chkfile(kpts=...) is not bound"));
        }
        let path = chk
            .or_else(|| self.chkfile.clone())
            .ok_or_else(|| PyValueError::new_err("init_guess_by_chkfile: no chkfile"))?;
        let d = self.driver(py)?;
        kdms_to_py(
            py,
            &chkfile_dm(&d, &path, project)?,
            KOverrideHooks::nao(&d),
        )
    }

    /// Alias of `init_guess_by_chkfile` (upstream's name).
    #[pyo3(signature = (chk = None, project = None, kpts = None))]
    #[allow(clippy::wrong_self_convention)]
    fn from_chk<'py>(
        &self,
        py: Python<'py>,
        chk: Option<String>,
        project: Option<bool>,
        kpts: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.init_guess_by_chkfile(py, chk, project, kpts)
    }

    /// `spin_square(mo_coeff=None, s=None)` → `(<S^2>, 2S+1)` (KUHF only,
    /// `KScfResult::spin_square`).
    fn spin_square(&self, py: Python<'_>) -> PyResult<(f64, f64)> {
        if self.kind != Kind::Uhf {
            return Err(not_impl(format!(
                "{}.spin_square is bound for KUHF only",
                self.kind.name()
            )));
        }
        let s = self.solved()?;
        let d = self.driver(py)?;
        let s1e = d.get_ovlp().map_err(pyscf_to_py)?;
        s.res
            .spin_square(&s1e, s.nao)
            .ok_or_else(|| PyValueError::new_err("spin_square: not an unrestricted result"))
    }

    /// `density_fit(auxbasis=None, with_df=None)` — switch `with_df` to a
    /// `GDF` over the same cell and k-points (or to `with_df`), IN PLACE, and
    /// return `self` (upstream returns a new object).
    #[pyo3(signature = (auxbasis = None, with_df = None))]
    fn density_fit<'py>(
        slf: &Bound<'py, Self>,
        auxbasis: Option<String>,
        with_df: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        let new_df = match some(with_df) {
            Some(df) => df.clone(),
            None => {
                let me = slf.borrow();
                let kpts = me.with_df.bind(py).getattr("kpts")?;
                let gdf = py
                    .get_type::<crate::pbc::df::PyGdf>()
                    .call1((me.py_cell.bind(py), kpts))?;
                if let Some(a) = auxbasis {
                    gdf.setattr("auxbasis", a)?;
                }
                gdf
            }
        };
        slf.borrow_mut().set_with_df(&new_df)?;
        Ok(slf.clone())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The public classes (constructors only; everything else is on KSCF)
// ─────────────────────────────────────────────────────────────────────────────

/// `KRHF(cell, kpts=np.zeros((1,3)), exxdiv='ewald')` — `khf.py:KRHF`. A built
/// `pyscf.pbc.symm.KPoints` as `kpts` runs the k-point-symmetry adapted driver
/// (`KsymAdaptedKrhf`), as upstream's `pbc.scf.KRHF` dispatch does.
#[pyclass(extends = PyKscf, subclass, name = "KRHF", module = "pyscf._native.pbc.scf")]
pub struct PyKrhf {}

#[pymethods]
impl PyKrhf {
    #[new]
    #[pyo3(signature = (cell, kpts = None, exxdiv = Some("ewald")))]
    fn new(
        cell: &Bound<'_, PyAny>,
        kpts: Option<&Bound<'_, PyAny>>,
        exxdiv: Option<&str>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let base = PyKscf::construct(cell, kpts, exxdiv, Kind::Rhf)?;
        Ok(PyClassInitializer::from(base).add_subclass(PyKrhf {}))
    }
}

/// `KsymAdaptedKRHF(cell, kpts, exxdiv='ewald')` — `khf_ksymm.py:410`; `kpts`
/// must be a built `KPoints`, and `with_df` samples its full Brillouin zone.
#[pyclass(
    extends = PyKrhf,
    subclass,
    name = "KsymAdaptedKRHF",
    module = "pyscf._native.pbc.scf"
)]
pub struct PyKsymAdaptedKrhf {}

#[pymethods]
impl PyKsymAdaptedKrhf {
    #[new]
    #[pyo3(signature = (cell, kpts, exxdiv = Some("ewald")))]
    fn new(
        cell: &Bound<'_, PyAny>,
        kpts: &Bound<'_, PyAny>,
        exxdiv: Option<&str>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let base = PyKscf::construct(cell, Some(kpts), exxdiv, Kind::KsymRhf)?;
        Ok(PyClassInitializer::from(base)
            .add_subclass(PyKrhf {})
            .add_subclass(PyKsymAdaptedKrhf {}))
    }
}

/// `KROHF(cell, kpts=np.zeros((1,3)), exxdiv='ewald')` — `krohf.py`.
#[pyclass(extends = PyKrhf, subclass, name = "KROHF", module = "pyscf._native.pbc.scf")]
pub struct PyKrohf {}

#[pymethods]
impl PyKrohf {
    #[new]
    #[pyo3(signature = (cell, kpts = None, exxdiv = Some("ewald")))]
    fn new(
        cell: &Bound<'_, PyAny>,
        kpts: Option<&Bound<'_, PyAny>>,
        exxdiv: Option<&str>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let base = PyKscf::construct(cell, kpts, exxdiv, Kind::Rohf)?;
        Ok(PyClassInitializer::from(base)
            .add_subclass(PyKrhf {})
            .add_subclass(PyKrohf {}))
    }
}

/// `KUHF(cell, kpts=np.zeros((1,3)), exxdiv='ewald')` — `kuhf.py`. A `KPoints`
/// argument RAISES (no `kuhf_ksymm` in this port).
#[pyclass(extends = PyKscf, subclass, name = "KUHF", module = "pyscf._native.pbc.scf")]
pub struct PyKuhf {}

#[pymethods]
impl PyKuhf {
    #[new]
    #[pyo3(signature = (cell, kpts = None, exxdiv = Some("ewald")))]
    fn new(
        cell: &Bound<'_, PyAny>,
        kpts: Option<&Bound<'_, PyAny>>,
        exxdiv: Option<&str>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let base = PyKscf::construct(cell, kpts, exxdiv, Kind::Uhf)?;
        Ok(PyClassInitializer::from(base).add_subclass(PyKuhf {}))
    }
}

/// `KGHF(cell, kpts=np.zeros((1,3)), exxdiv='ewald')` — `kghf.py`. A `KPoints`
/// argument RAISES (no `kghf_ksymm` in this port).
#[pyclass(extends = PyKscf, subclass, name = "KGHF", module = "pyscf._native.pbc.scf")]
pub struct PyKghf {}

#[pymethods]
impl PyKghf {
    #[new]
    #[pyo3(signature = (cell, kpts = None, exxdiv = Some("ewald")))]
    fn new(
        cell: &Bound<'_, PyAny>,
        kpts: Option<&Bound<'_, PyAny>>,
        exxdiv: Option<&str>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let base = PyKscf::construct(cell, kpts, exxdiv, Kind::Ghf)?;
        Ok(PyClassInitializer::from(base).add_subclass(PyKghf {}))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Gamma-point shims and module functions
// ─────────────────────────────────────────────────────────────────────────────

/// The `(1, 3)` k-list a gamma shim samples.
fn one_kpt<'py>(py: Python<'py>, kpt: Option<&Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    let k = match extract_kpts_opt(kpt)? {
        None => [0.0; 3],
        Some((v, true)) => v[0],
        Some((_, false)) => {
            return Err(PyValueError::new_err("kpt must be ONE (3,) k-point"));
        }
    };
    Ok(kpts_to_pyarray(py, &[k])?.into_any())
}

/// `RHF(cell, kpt=np.zeros(3), exxdiv='ewald')` — `gamma.rs:27`: a `KRHF` at
/// one k-point. Results are per-k lists of length 1 (upstream's gamma class
/// returns bare arrays).
#[pyfunction(name = "RHF", signature = (cell, kpt = None, exxdiv = Some("ewald")))]
fn gamma_rhf<'py>(
    cell: &Bound<'py, PyAny>,
    kpt: Option<&Bound<'py, PyAny>>,
    exxdiv: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = cell.py();
    py.get_type::<PyKrhf>()
        .call1((cell, one_kpt(py, kpt)?, exxdiv))
}

/// `UHF(cell, kpt=np.zeros(3), exxdiv='ewald')` — `gamma.rs:43`.
#[pyfunction(name = "UHF", signature = (cell, kpt = None, exxdiv = Some("ewald")))]
fn gamma_uhf<'py>(
    cell: &Bound<'py, PyAny>,
    kpt: Option<&Bound<'py, PyAny>>,
    exxdiv: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = cell.py();
    py.get_type::<PyKuhf>()
        .call1((cell, one_kpt(py, kpt)?, exxdiv))
}

/// `ROHF(cell, kpt=np.zeros(3), exxdiv='ewald')` — `gamma.rs:51`.
#[pyfunction(name = "ROHF", signature = (cell, kpt = None, exxdiv = Some("ewald")))]
fn gamma_rohf<'py>(
    cell: &Bound<'py, PyAny>,
    kpt: Option<&Bound<'py, PyAny>>,
    exxdiv: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = cell.py();
    py.get_type::<PyKrohf>()
        .call1((cell, one_kpt(py, kpt)?, exxdiv))
}

/// `GHF(cell, kpt=np.zeros(3), exxdiv='ewald')` — `gamma.rs:59`.
#[pyfunction(name = "GHF", signature = (cell, kpt = None, exxdiv = Some("ewald")))]
fn gamma_ghf<'py>(
    cell: &Bound<'py, PyAny>,
    kpt: Option<&Bound<'py, PyAny>>,
    exxdiv: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = cell.py();
    py.get_type::<PyKghf>()
        .call1((cell, one_kpt(py, kpt)?, exxdiv))
}

/// `smearing_(mf, sigma=None, method='fermi', mu0=None)` — `addons.rs:26,35`.
#[pyfunction(signature = (mf, sigma = None, method = "fermi", mu0 = None))]
fn smearing_<'py>(
    mf: &Bound<'py, PyAny>,
    sigma: Option<f64>,
    method: &str,
    mu0: Option<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    mf.call_method1("smearing_", (sigma, method, mu0))
}

/// `project_mo_nr2nr(cell1, mo1, cell2, kpts=None)` — `addons.py:39-57`.
/// `kpts=None` or `(3,)`: `mo1` is one `(nao1, nmo)` array and one array
/// returns; `(n, 3)`: per-k lists.
#[pyfunction(signature = (cell1, mo1, cell2, kpts = None))]
fn project_mo_nr2nr<'py>(
    cell1: &Bound<'py, PyAny>,
    mo1: &Bound<'py, PyAny>,
    cell2: &Bound<'py, PyAny>,
    kpts: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = cell1.py();
    let c1 = extract_cell_from_pyany(py, cell1)?;
    let c2 = extract_cell_from_pyany(py, cell2)?;
    let (k, single) = extract_kpts_opt(kpts)?.unwrap_or_else(|| (vec![[0.0; 3]], true));
    let (nao1, nao2) = (c1.mol.nao_nr, c2.mol.nao_nr);
    let mo = if single {
        let one = PyList::new(py, [mo1])?;
        mo_coeff_from_py(&one, 1, 1, nao1, "mo1")?
    } else {
        mo_coeff_from_py(mo1, 1, k.len(), nao1, "mo1")?
    };
    let out = pyscf_pbc_scf::addons::project_mo_nr2nr(&c1, &mo, &c2, &k).map_err(pyscf_to_py)?;
    maybe_single(mo_coeff_to_py(py, &out, 1, nao2)?, single)
}

/// `load_scf(chkfile)` → `(cell, scf_rec)` — `pbc/scf/chkfile.py:load_scf` over
/// `chkfile.rs:122`. `cell` is a native `Cell` from `/mol` (`None` if absent);
/// `scf_rec` holds `e_tot`, `kpts`, `mo_energy`, `mo_occ`, `mo_coeff`, nested
/// per spin when the file holds two channels.
#[pyfunction]
fn load_scf<'py>(py: Python<'py>, chkfile: &str) -> PyResult<(Py<PyAny>, Bound<'py, PyDict>)> {
    let ck = load_kscf_from_file(std::path::Path::new(chkfile)).map_err(chk_err)?;
    let file = pyscf_chkfile::primitives::open_for_read(chkfile).map_err(chk_err)?;
    let cell: Py<PyAny> = match pyscf_chkfile::primitives::read_mol(&file) {
        Ok(json) => Py::new(
            py,
            PyCell::from_cell(pyscf_pbc_gto::loads(&json).map_err(pyscf_to_py)?),
        )?
        .into_any(),
        Err(_) => py.None(),
    };
    let nk = ck.kpts.len().max(1);
    let nch = (ck.mo_coeff.len() / nk).max(1);
    let rec = PyDict::new(py);
    rec.set_item("e_tot", ck.e_tot)?;
    rec.set_item("kpts", kpts_to_pyarray(py, &ck.kpts)?)?;
    rec.set_item("mo_energy", mo_values_to_py(py, &ck.mo_energy, nch)?)?;
    rec.set_item("mo_occ", mo_values_to_py(py, &ck.mo_occ, nch)?)?;
    rec.set_item("mo_coeff", mo_coeff_to_py(py, &ck.mo_coeff, nch, ck.nao)?)?;
    Ok((cell, rec))
}

// ─────────────────────────────────────────────────────────────────────────────
// Plan 20-15 — read-only access for the correlated drivers (additive)
// ─────────────────────────────────────────────────────────────────────────────

/// What a post-SCF driver (`pbc.mp` / `pbc.cc` / `pbc.ci`, plan 20-15) reads
/// from a converged periodic SCF object: the stored [`KScfResult`] (IBZ-sized
/// for `KsymAdaptedKRHF`), the Python `with_df`, the Rust `KPoints`
/// (ksymm only) and the SCF's `exxdiv`. The correlated drivers never re-run SCF.
pub(crate) struct PostScfInput {
    /// `"KRHF"`, `"KsymAdaptedKRHF"`, `"KUHF"`, `"KROHF"` or `"KGHF"`.
    pub kind: &'static str,
    pub result: KScfResult,
    pub with_df: Py<PyAny>,
    pub kpoints: Option<pyscf_pbc_symm::kpts::KPoints>,
    pub exxdiv: Option<ExxDiv>,
}

impl PyKscf {
    /// Snapshot of the last kernel's result plus the objects a correlated
    /// driver needs. `ValueError` before `kernel()`.
    pub(crate) fn post_scf_input(&self, py: Python<'_>) -> PyResult<PostScfInput> {
        let s = self.solved()?;
        let kpoints = match &self.kpoints {
            Some(kp) => Some(kp.bind(py).cast::<PyKPoints>()?.borrow().kpoints().clone()),
            None => None,
        };
        Ok(PostScfInput {
            kind: self.kind.name(),
            result: s.res.clone(),
            with_df: self.with_df.clone_ref(py),
            kpoints,
            exxdiv: self.exxdiv,
        })
    }
}
