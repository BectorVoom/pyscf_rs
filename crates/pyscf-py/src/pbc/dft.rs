//! `pyscf._native.pbc.dft` — periodic Kohn-Sham DFT (plan 20-13).
//!
//! # Classes
//!
//! | Python | Rust | notes |
//! |---|---|---|
//! | `KohnShamDFT` | — | native base: state, the eleven hooks, `kernel`, results |
//! | `KRKS(KohnShamDFT)` | `krks::Krks` | a `KPoints` `kpts` selects `KsymAdaptedKrks` |
//! | `KsymAdaptedKRKS(KRKS)` | `krks_ksymm::KsymAdaptedKrks` | requires a built `KPoints` |
//! | `KUKS(KohnShamDFT)` | `kuks::Kuks` | a `KPoints` selects `KsymAdaptedKuks` |
//! | `KsymAdaptedKUKS(KUKS)` | `krks_ksymm::KsymAdaptedKuks` | |
//! | `KROKS(KohnShamDFT)` | `kroks::Kroks` | a `KPoints` RAISES (no upstream ksymm) |
//! | `KGKS(KohnShamDFT)` | `kgks::Kgks` | a `KPoints` RAISES; hybrid RAISES |
//! | `KRKSpU(KRKS)` / `KUKSpU(KUKS)` | `Krks`/`Kuks` + `kspu::add_vhubbard` | see "DFT+U" |
//! | `KsymAdaptedKRKSpU` / `KsymAdaptedKUKSpU` | `krks_ksymm::KsymAdaptedKr/ukspu` | |
//! | `UniformGrids` / `BeckeGrids` | `gen_grid::PeriodicGrids::uniform` / `::becke` | `mf.grids` |
//! | `KNumInt` / `MultiGridNumInt` / `MultiGridNumInt2` | `numint::KsNumInt::grid` / `multigrid` / `multigrid2` | `mf._numint` |
//!
//! `pyscf-pbc-dft` has NO root re-exports (`lib.rs:8-21`), so every Rust type
//! is named through its module.
//!
//! # Pattern (20-12 SUMMARY, "for 20-13 to copy")
//!
//! One native base holds every attribute, hook and the kernel; the public
//! classes are `extends` subclasses with a `#[new]` only. `KPyOverrideBridge`
//! probes against the base (`KohnShamDFT`), so an unsubclassed `KRKS` reports
//! no override and a Python subclass's override is dispatched. Each call
//! builds a fresh Rust driver from `extract_df(with_df)` plus the stored
//! configuration behind [`Driver`], whose `KOverrideHooks` impl is pure
//! delegation — so the bridged kernel is op-for-op the concrete kernel.
//!
//! Unlike upstream, `KRKS` is NOT a subclass of `pyscf.pbc.scf.KRHF`: the scf
//! base (`KSCF`) is 20-12's and its hook bodies know only the HF drivers.
//!
//! # The `KPoints` dispatch
//!
//! Upstream's `pbc/dft/__init__.py:37-73` makes `KRKS`/`KUKS`/`KRKSpU`/`KUKSpU`
//! functions branching on `isinstance(kpts, KPoints)`. The identity gate
//! requires `pyscf.pbc.dft.KRKS is pyscf._native.pbc.dft.KRKS`, so the rule is
//! applied in these constructors (20-12 D1): with a `KPoints` they run the
//! ksymm driver and `type(mf)` stays the called class.
//!
//! # DFT+U
//!
//! `kspu::Krkspu`/`Kukspu` carry no `KOverrideHooks` impl, so [`PuDriver`]
//! supplies one here: `get_veff` is the KS driver's own `get_veff` followed
//! by `kspu::add_vhubbard` (exactly `Krkspu::get_veff_tagged`'s two calls),
//! and `energy_elec` adds `E_U` to the KS driver's (`krkspu.py:139-160`). The
//! ksymm variants use the Rust `KsymAdaptedKrkspu`/`KsymAdaptedKukspu` hooks.
//! Sites are named upstream-style (`U_idx=['Ni 3d']`); the principal quantum
//! number maps to the Rust contraction index as `n - l - 1` (core first).
//!
//! # GIL
//!
//! Held for the whole kernel: the drivers carry `Cell` energy tags and are not
//! `Sync`, and overridden hooks re-enter Python every cycle.

use std::cell::Cell as StdCell;

use pyo3::exceptions::{
    PyAttributeError, PyNotImplementedError, PyOSError, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};
use pyscf_algebra::CTensor;
use pyscf_core::PyscfRsError;
use pyscf_pbc_df::PeriodicDf;
use pyscf_pbc_dft::PbcDftError;
use pyscf_pbc_dft::gen_grid::PeriodicGrids;
use pyscf_pbc_dft::kgks::Kgks;
use pyscf_pbc_dft::krks::{Krks, KsEnergyTags};
use pyscf_pbc_dft::krks_ksymm::{
    KsymAdaptedKrks, KsymAdaptedKrkspu, KsymAdaptedKuks, KsymAdaptedKukspu,
};
use pyscf_pbc_dft::kroks::Kroks;
use pyscf_pbc_dft::kspu::{HubbardU, USite, add_vhubbard, add_vhubbard_weighted};
use pyscf_pbc_dft::kuks::Kuks;
use pyscf_pbc_dft::numint::KNumIntCache;
use pyscf_pbc_dft::numint::KsNumInt;
use pyscf_pbc_dft::numint2c::Collinear;
use pyscf_pbc_gto::{Cell, ExxDiv};
use pyscf_pbc_scf::krhf::to_row_major;
use pyscf_pbc_scf::{
    KDms, KInitGuess, KMats, KOverrideHooks, KScfConfig, KScfResult, Smearing, SmearingMethod,
    dump_kscf_to_file,
};

use crate::bridge::extract_cell_from_pyany;
use crate::errors::pyscf_to_py;
use crate::pbc::convert::{extract_kpts, extract_kpts_opt, extract_usize3, kpts_to_pyarray};
use crate::pbc::df::{PyFftdf, PyGdf, extract_df, pbc_df_to_py};
use crate::pbc::gto::PyCell;
use crate::pbc::kbridge::{
    KPyOverrideBridge, kdms_from_py, kdms_to_py, kmats_from_py, kmats_to_py, mo_coeff_from_py,
    mo_coeff_to_py, mo_values_from_py, mo_values_to_py,
};
use crate::pbc::symm::PyKPoints;

/// Register the KS drivers, grids, numint selectors and gamma shims on
/// `pyscf._native.pbc.dft`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.setattr(
        "__doc__",
        "pyscf._native.pbc.dft — periodic Kohn-Sham bindings (plan 20-13).",
    )?;
    m.add_class::<PyKohnShamDft>()?;
    m.add_class::<PyKrks>()?;
    m.add_class::<PyKsymAdaptedKrks>()?;
    m.add_class::<PyKuks>()?;
    m.add_class::<PyKsymAdaptedKuks>()?;
    m.add_class::<PyKroks>()?;
    m.add_class::<PyKgks>()?;
    m.add_class::<PyKrkspu>()?;
    m.add_class::<PyKukspu>()?;
    m.add_class::<PyKsymAdaptedKrkspu>()?;
    m.add_class::<PyKsymAdaptedKukspu>()?;
    m.add_class::<PyUniformGrids>()?;
    m.add_class::<PyBeckeGrids>()?;
    m.add_class::<PyKNumInt>()?;
    m.add_class::<PyMultiGridNumInt>()?;
    m.add_class::<PyMultiGridNumInt2>()?;
    m.add_function(wrap_pyfunction!(gamma_rks, m)?)?;
    m.add_function(wrap_pyfunction!(gamma_uks, m)?)?;
    m.add_function(wrap_pyfunction!(gamma_roks, m)?)?;
    m.add_function(wrap_pyfunction!(gamma_gks, m)?)?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Error plumbing and small parsers
// ─────────────────────────────────────────────────────────────────────────────

/// `krks::unwrap_err` is `pub(crate)` in `pyscf-pbc-dft`; this is the same map.
fn dft_err(e: PbcDftError) -> PyscfRsError {
    match e {
        PbcDftError::Core(c) => c,
        other => PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(other.to_string())),
    }
}

fn dft_to_py(e: PbcDftError) -> PyErr {
    pyscf_to_py(dft_err(e))
}

fn not_impl(msg: impl Into<String>) -> PyErr {
    PyNotImplementedError::new_err(msg.into())
}

fn some<'a, 'py>(x: Option<&'a Bound<'py, PyAny>>) -> Option<&'a Bound<'py, PyAny>> {
    x.filter(|v| !v.is_none())
}

fn parse_init_guess(key: &str) -> PyResult<KInitGuess> {
    match key.to_ascii_lowercase().as_str() {
        "minao" => Ok(KInitGuess::Minao),
        "atom" => Ok(KInitGuess::Atom),
        "1e" | "hcore" => Ok(KInitGuess::OneElectron),
        other => Err(not_impl(format!(
            "init guess {other:?} is not bound for the periodic KS drivers (minao, atom, 1e)"
        ))),
    }
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

fn parse_collinear(s: &str) -> PyResult<Collinear> {
    match s.to_ascii_lowercase().as_str() {
        "col" | "c" => Ok(Collinear::Col),
        "ncol" | "n" => Ok(Collinear::Ncol),
        "mcol" | "m" => Ok(Collinear::Mcol),
        other => Err(PyValueError::new_err(format!(
            "collinear must be 'col', 'ncol' or 'mcol', got {other:?}"
        ))),
    }
}

fn collinear_str(c: Collinear) -> &'static str {
    match c {
        Collinear::Col => "col",
        Collinear::Ncol => "ncol",
        Collinear::Mcol => "mcol",
    }
}

/// One upstream `U_idx` string — `'He 1s'`, `'Ni 3d'` — as a Rust site.
///
/// Upstream resolves it with `minao_mol.search_ao_label` (`rkspu.py:151`);
/// `pyscf-core` has no AO labels, so `kspu::USite::Shell` names the shell by
/// element, `l` and contraction index. In a core-first minimal reference
/// basis the contraction index of shell `n l` is `n - l - 1`. Anything else
/// (an atom index prefix, an `m` component like `2pz`) is refused.
fn parse_u_site(label: &str) -> PyResult<USite> {
    let refuse = || {
        not_impl(format!(
            "U_idx label {label:?}: only '<Element> <n><l>' (e.g. 'Ni 3d') is ported; atom-index \
             prefixes and m-components need upstream's AO-label search"
        ))
    };
    let parts: Vec<&str> = label.split_whitespace().collect();
    let [element, shell] = parts.as_slice() else {
        return Err(refuse());
    };
    if !element.chars().all(|c| c.is_ascii_alphabetic()) {
        return Err(refuse());
    }
    let digits: String = shell.chars().take_while(char::is_ascii_digit).collect();
    let rest = &shell[digits.len()..];
    let n: usize = digits.parse().map_err(|_| refuse())?;
    let l: u32 = match rest.to_ascii_lowercase().as_str() {
        "s" => 0,
        "p" => 1,
        "d" => 2,
        "f" => 3,
        "g" => 4,
        _ => return Err(refuse()),
    };
    if n < l as usize + 1 {
        return Err(PyValueError::new_err(format!(
            "U_idx label {label:?}: no {n}{rest} shell"
        )));
    }
    Ok(USite::Shell {
        element: (*element).to_string(),
        l,
        contraction: Some(n - l as usize - 1),
    })
}

fn extract_u_idx(obj: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let mut out = Vec::new();
    for item in obj.try_iter()? {
        let item = item?;
        match item.extract::<String>() {
            Ok(s) => {
                parse_u_site(&s)?;
                out.push(s);
            }
            Err(_) => {
                return Err(not_impl(
                    "U_idx as AO index lists is not ported: upstream maps large-basis indices \
                     through AO labels (rkspu.py:156-158), which pyscf-core lacks; use labels \
                     such as 'Ni 3d'",
                ));
            }
        }
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Grids and numint selectors (the Python objects `mf.grids` / `mf._numint`)
// ─────────────────────────────────────────────────────────────────────────────

/// `UniformGrids(cell)` — `pbc/dft/gen_grid.py:63`. `mesh = None` (the
/// default) resolves to `cell.mesh` at use, exactly `PeriodicGrids::uniform(cell,
/// None)` — the grid every Rust `from_df` constructor builds.
#[pyclass(
    subclass,
    name = "UniformGrids",
    module = "pyscf._native.pbc.dft",
    skip_from_py_object
)]
pub struct PyUniformGrids {
    py_cell: Py<PyAny>,
    mesh: Option<[usize; 3]>,
}

impl PyUniformGrids {
    fn built(&self, py: Python<'_>) -> PyResult<PeriodicGrids> {
        let cell = extract_cell_from_pyany(py, self.py_cell.bind(py))?;
        PeriodicGrids::uniform(&cell, self.mesh).map_err(dft_to_py)
    }
}

#[pymethods]
impl PyUniformGrids {
    #[new]
    fn new(cell: &Bound<'_, PyAny>) -> Self {
        Self {
            py_cell: cell.clone().unbind(),
            mesh: None,
        }
    }

    #[getter]
    fn cell(&self, py: Python<'_>) -> Py<PyAny> {
        self.py_cell.clone_ref(py)
    }

    /// The FFT mesh; `cell.mesh` until assigned.
    #[getter]
    fn mesh(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self.mesh {
            Some(m) => Ok(PyList::new(py, m)?.into_any().unbind()),
            None => Ok(self.py_cell.bind(py).getattr("mesh")?.unbind()),
        }
    }
    #[setter]
    fn set_mesh(&mut self, v: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        self.mesh = match some(v) {
            Some(x) => Some(extract_usize3(x, "mesh")?),
            None => None,
        };
        Ok(())
    }

    #[getter]
    fn size(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(self.built(py)?.size())
    }

    #[getter]
    fn coords<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let g = self.built(py)?;
        Ok(kpts_to_pyarray(py, g.coords().map_err(dft_to_py)?)?.into_any())
    }

    #[getter]
    fn weights<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let g = self.built(py)?;
        Ok(numpy::PyArray1::from_slice(
            py,
            g.weights().map_err(dft_to_py)?,
        ))
    }

    /// `build(cell=None, with_non0tab=False)` → self.
    #[pyo3(signature = (cell = None, with_non0tab = false))]
    fn build<'py>(
        slf: &Bound<'py, Self>,
        cell: Option<&Bound<'py, PyAny>>,
        with_non0tab: bool,
    ) -> PyResult<Bound<'py, Self>> {
        let _ = with_non0tab;
        if let Some(c) = some(cell) {
            slf.borrow_mut().py_cell = c.clone().unbind();
        }
        Ok(slf.clone())
    }

    fn __repr__(&self) -> String {
        format!("UniformGrids(mesh={:?})", self.mesh)
    }
}

/// `BeckeGrids(cell)` — `pbc/dft/gen_grid.py:240`; `level` (class default 3)
/// is the one bound control. Built lazily against its own `cell` and cached
/// until `level` or `cell` changes.
#[pyclass(
    subclass,
    name = "BeckeGrids",
    module = "pyscf._native.pbc.dft",
    skip_from_py_object
)]
pub struct PyBeckeGrids {
    py_cell: Py<PyAny>,
    level: usize,
    built: Option<PeriodicGrids>,
}

impl PyBeckeGrids {
    fn ensure(&mut self, py: Python<'_>) -> PyResult<&PeriodicGrids> {
        if self.built.is_none() {
            let cell = extract_cell_from_pyany(py, self.py_cell.bind(py))?;
            let config = pyscf_grids::Grids {
                level: self.level,
                ..pyscf_grids::Grids::new()
            };
            self.built = Some(PeriodicGrids::becke(&cell, config).map_err(dft_to_py)?);
        }
        self.built
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("BeckeGrids: build failed"))
    }
}

#[pymethods]
impl PyBeckeGrids {
    #[new]
    fn new(cell: &Bound<'_, PyAny>) -> Self {
        Self {
            py_cell: cell.clone().unbind(),
            level: pyscf_grids::Grids::new().level,
            built: None,
        }
    }

    #[getter]
    fn cell(&self, py: Python<'_>) -> Py<PyAny> {
        self.py_cell.clone_ref(py)
    }

    #[getter]
    fn level(&self) -> usize {
        self.level
    }
    #[setter]
    fn set_level(&mut self, v: usize) {
        self.level = v;
        self.built = None;
    }

    /// `build(cell=None)` → self (eager).
    #[pyo3(signature = (cell = None, with_non0tab = false))]
    fn build<'py>(
        slf: &Bound<'py, Self>,
        cell: Option<&Bound<'py, PyAny>>,
        with_non0tab: bool,
    ) -> PyResult<Bound<'py, Self>> {
        let _ = with_non0tab;
        let py = slf.py();
        let mut me = slf.borrow_mut();
        if let Some(c) = some(cell) {
            me.py_cell = c.clone().unbind();
            me.built = None;
        }
        me.ensure(py)?;
        drop(me);
        Ok(slf.clone())
    }

    #[getter]
    fn size(&mut self, py: Python<'_>) -> PyResult<usize> {
        Ok(self.ensure(py)?.size())
    }

    #[getter]
    fn coords<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let g = self.ensure(py)?;
        Ok(kpts_to_pyarray(py, g.coords().map_err(dft_to_py)?)?.into_any())
    }

    #[getter]
    fn weights<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let g = self.ensure(py)?;
        Ok(numpy::PyArray1::from_slice(
            py,
            g.weights().map_err(dft_to_py)?,
        ))
    }
}

/// `KNumInt(kpts=None)` — the default grid quadrature (`numint.py:KNumInt`).
/// A selector: the driver's own k-points are always used.
#[pyclass(
    subclass,
    name = "KNumInt",
    module = "pyscf._native.pbc.dft",
    skip_from_py_object
)]
pub struct PyKNumInt {}

#[pymethods]
impl PyKNumInt {
    #[new]
    #[pyo3(signature = (kpts = None))]
    fn new(kpts: Option<&Bound<'_, PyAny>>) -> Self {
        let _ = kpts;
        Self {}
    }
}

/// `MultiGridNumInt(cell=None)` — multigrid v1 (`multigrid/multigrid.py`),
/// gamma-point only. `mesh` other than `cell.mesh` is refused at use.
#[pyclass(
    subclass,
    name = "MultiGridNumInt",
    module = "pyscf._native.pbc.dft",
    skip_from_py_object
)]
pub struct PyMultiGridNumInt {
    mesh: Option<[usize; 3]>,
}

/// `MultiGridNumInt2(cell=None)` — the pair multigrid v2
/// (`multigrid/multigrid_pair.py`). Carries a DEFINITIONAL mesh-independent
/// ~2e-8 (diamond) … 2e-7 (si) floor against FFTDF (17-01 Gate E) — never
/// held to v1's 1e-12.
#[pyclass(
    subclass,
    name = "MultiGridNumInt2",
    module = "pyscf._native.pbc.dft",
    skip_from_py_object
)]
pub struct PyMultiGridNumInt2 {
    mesh: Option<[usize; 3]>,
}

macro_rules! mg_selector_methods {
    ($ty:ty) => {
        #[pymethods]
        impl $ty {
            #[new]
            #[pyo3(signature = (cell = None))]
            fn new(cell: Option<&Bound<'_, PyAny>>) -> Self {
                let _ = cell;
                Self { mesh: None }
            }
            /// `None` = `cell.mesh`, the only mesh the Rust multigrid engines use.
            #[getter]
            fn mesh(&self) -> Option<[usize; 3]> {
                self.mesh
            }
            #[setter]
            fn set_mesh(&mut self, v: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
                self.mesh = match some(v) {
                    Some(x) => Some(extract_usize3(x, "mesh")?),
                    None => None,
                };
                Ok(())
            }
        }
    };
}
mg_selector_methods!(PyMultiGridNumInt);
mg_selector_methods!(PyMultiGridNumInt2);

// ─────────────────────────────────────────────────────────────────────────────
// DFT+U driver (the KOverrideHooks the Rust `Krkspu`/`Kukspu` do not carry)
// ─────────────────────────────────────────────────────────────────────────────

/// `KRKSpU`/`KUKSpU` over a full-BZ KS driver `K`.
#[derive(Debug)]
struct PuDriver<K> {
    ks: K,
    u: HubbardU,
    e_u: StdCell<f64>,
}

impl<K: KOverrideHooks> KOverrideHooks for PuDriver<K> {
    fn cell(&self) -> &Cell {
        self.ks.cell()
    }
    fn kpts(&self) -> &[[f64; 3]] {
        self.ks.kpts()
    }
    fn nset(&self) -> usize {
        self.ks.nset()
    }
    fn nfock(&self) -> usize {
        self.ks.nfock()
    }
    fn nao(&self) -> usize {
        self.ks.nao()
    }
    fn get_ovlp(&self) -> Result<KMats, PyscfRsError> {
        self.ks.get_ovlp()
    }
    fn get_hcore(&self) -> Result<KMats, PyscfRsError> {
        self.ks.get_hcore()
    }
    fn get_init_guess(&self, mode: &KInitGuess, s1e: &KMats) -> Result<KDms, PyscfRsError> {
        self.ks.get_init_guess(mode, s1e)
    }
    /// `krkspu.py:37-65` / `kukspu.py:37-63`: the KS potential (which leaves
    /// the energy tags on `ks`), then the Hubbard term — the same two calls as
    /// `Krkspu::get_veff_tagged`.
    fn get_veff(&self, dms: &KDms) -> Result<KDms, PyscfRsError> {
        let mut v = self.ks.get_veff(dms)?;
        let e_u =
            add_vhubbard(&mut v, self.ks.cell(), self.ks.kpts(), dms, &self.u).map_err(dft_err)?;
        self.e_u.set(e_u);
        Ok(v)
    }
    fn get_fock(&self, h1e: &KMats, vhf: &KDms, dms: &KDms) -> Result<KDms, PyscfRsError> {
        self.ks.get_fock(h1e, vhf, dms)
    }
    fn diis_dms(&self, dms: &KDms) -> KDms {
        self.ks.diis_dms(dms)
    }
    fn eig(&self, fock: &KDms, s1e: &KMats) -> Result<(Vec<Vec<f64>>, Vec<CTensor>), PyscfRsError> {
        self.ks.eig(fock, s1e)
    }
    fn get_occ(&self, mo_energy: &[Vec<f64>]) -> Result<(Vec<Vec<f64>>, Vec<f64>), PyscfRsError> {
        self.ks.get_occ(mo_energy)
    }
    fn make_rdm1(&self, mo_coeff: &[CTensor], mo_occ: &[Vec<f64>]) -> Result<KDms, PyscfRsError> {
        self.ks.make_rdm1(mo_coeff, mo_occ)
    }
    /// `krkspu.py:139-160`: `(e1 + ecoul + exc + E_U, ecoul + exc + E_U)`.
    fn energy_elec(&self, dms: &KDms, h1e: &KMats, vhf: &KDms) -> Result<(f64, f64), PyscfRsError> {
        let (e, e2) = self.ks.energy_elec(dms, h1e, vhf)?;
        let e_u = self.e_u.get();
        Ok((e + e_u, e2 + e_u))
    }
    fn energy_nuc(&self) -> Result<f64, PyscfRsError> {
        self.ks.energy_nuc()
    }
    fn get_grad(
        &self,
        mo_coeff: &[CTensor],
        mo_occ: &[Vec<f64>],
        h1e: &KMats,
        vhf: &KDms,
    ) -> Vec<f64> {
        self.ks.get_grad(mo_coeff, mo_occ, h1e, vhf)
    }
    fn free_energy(&self) -> Option<f64> {
        self.ks.free_energy()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The Rust driver behind one call
// ─────────────────────────────────────────────────────────────────────────────

/// Which periodic KS method an instance runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Rks,
    KsymRks,
    Uks,
    KsymUks,
    Roks,
    Gks,
    Rkspu,
    KsymRkspu,
    Ukspu,
    KsymUkspu,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Rks => "KRKS",
            Kind::KsymRks => "KsymAdaptedKRKS",
            Kind::Uks => "KUKS",
            Kind::KsymUks => "KsymAdaptedKUKS",
            Kind::Roks => "KROKS",
            Kind::Gks => "KGKS",
            Kind::Rkspu => "KRKSpU",
            Kind::KsymRkspu => "KsymAdaptedKRKSpU",
            Kind::Ukspu => "KUKSpU",
            Kind::KsymUkspu => "KsymAdaptedKUKSpU",
        }
    }
    fn is_ksymm(self) -> bool {
        matches!(
            self,
            Kind::KsymRks | Kind::KsymUks | Kind::KsymRkspu | Kind::KsymUkspu
        )
    }
    fn is_pu(self) -> bool {
        matches!(
            self,
            Kind::Rkspu | Kind::KsymRkspu | Kind::Ukspu | Kind::KsymUkspu
        )
    }
    fn unrestricted(self) -> bool {
        matches!(
            self,
            Kind::Uks | Kind::KsymUks | Kind::Ukspu | Kind::KsymUkspu
        )
    }
    /// The ksymm twin a `KPoints` argument selects, or `None` when refused.
    fn with_kpoints(self) -> Option<Kind> {
        match self {
            Kind::Rks | Kind::KsymRks => Some(Kind::KsymRks),
            Kind::Uks | Kind::KsymUks => Some(Kind::KsymUks),
            Kind::Rkspu | Kind::KsymRkspu => Some(Kind::KsymRkspu),
            Kind::Ukspu | Kind::KsymUkspu => Some(Kind::KsymUkspu),
            Kind::Roks | Kind::Gks => None,
        }
    }
}

/// A freshly built Rust driver over the current `with_df` builder.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
enum Driver {
    Rks(Krks),
    KsymRks(KsymAdaptedKrks),
    Uks(Kuks),
    KsymUks(KsymAdaptedKuks),
    Roks(Kroks),
    Gks(Kgks),
    Rkspu(PuDriver<Krks>),
    KsymRkspu(KsymAdaptedKrkspu),
    Ukspu(PuDriver<Kuks>),
    KsymUkspu(KsymAdaptedKukspu),
}

/// Run `$body` with `$x` bound to the concrete driver.
macro_rules! each_driver {
    ($d:expr, $x:ident => $body:expr) => {
        match $d {
            Driver::Rks($x) => $body,
            Driver::KsymRks($x) => $body,
            Driver::Uks($x) => $body,
            Driver::KsymUks($x) => $body,
            Driver::Roks($x) => $body,
            Driver::Gks($x) => $body,
            Driver::Rkspu($x) => $body,
            Driver::KsymRkspu($x) => $body,
            Driver::Ukspu($x) => $body,
            Driver::KsymUkspu($x) => $body,
        }
    };
}

impl Driver {
    fn kernel_direct(&self, cfg: &KScfConfig) -> Result<KScfResult, PyscfRsError> {
        match self {
            Driver::Rks(d) => d.kernel(cfg),
            Driver::KsymRks(d) => d.kernel(cfg),
            Driver::Uks(d) => d.kernel(cfg),
            Driver::KsymUks(d) => d.kernel(cfg),
            Driver::Roks(d) => d.kernel(cfg),
            Driver::Gks(d) => d.kernel(cfg),
            Driver::KsymRkspu(d) => d.kernel(cfg),
            Driver::KsymUkspu(d) => d.kernel(cfg),
            // No Rust `Krkspu::kernel` exists; the SCF driver over the
            // delegation impl above is the kernel.
            Driver::Rkspu(d) => pyscf_pbc_scf::kernel(d, cfg),
            Driver::Ukspu(d) => pyscf_pbc_scf::kernel(d, cfg),
        }
    }

    fn df(&self) -> &dyn PeriodicDf {
        match self {
            Driver::Rks(d) => d.with_df.as_ref(),
            Driver::KsymRks(d) => d.with_df.as_ref(),
            Driver::Uks(d) => d.with_df.as_ref(),
            Driver::KsymUks(d) => d.with_df.as_ref(),
            Driver::Roks(d) => d.hf.with_df.as_ref(),
            Driver::Gks(d) => d.hf.with_df.as_ref(),
            Driver::Rkspu(d) => d.ks.with_df.as_ref(),
            Driver::KsymRkspu(d) => d.ks.with_df.as_ref(),
            Driver::Ukspu(d) => d.ks.with_df.as_ref(),
            Driver::KsymUkspu(d) => d.ks.with_df.as_ref(),
        }
    }

    /// `E_U` of the last `get_veff` (DFT+U only).
    fn e_u(&self) -> Option<f64> {
        match self {
            Driver::Rkspu(d) => Some(d.e_u.get()),
            Driver::Ukspu(d) => Some(d.e_u.get()),
            Driver::KsymRkspu(d) => Some(d.e_u()),
            Driver::KsymUkspu(d) => Some(d.e_u()),
            _ => None,
        }
    }

    /// The KS potential with upstream's `tag_array` attributes (`ecoul`,
    /// `exc`, `E_U`) — the driver's own `get_veff_tagged`, plus the Hubbard
    /// term for DFT+U exactly as the corresponding `get_veff` hook adds it.
    fn veff_components(
        &self,
        dms: &KDms,
        band: Option<&[[f64; 3]]>,
    ) -> Result<(KDms, KsEnergyTags, Option<f64>), PyErr> {
        let pu_band = || {
            not_impl(
                "DFT+U get_veff(kpts_band=...) is not ported (the Hubbard term is ground-state)",
            )
        };
        Ok(match self {
            Driver::Rks(d) => {
                let (v, t) = d.get_veff_tagged(dms, band).map_err(dft_to_py)?;
                (v, t, None)
            }
            Driver::KsymRks(d) => {
                let (v, t) = d.get_veff_tagged(dms, band).map_err(dft_to_py)?;
                (v, t, None)
            }
            Driver::Uks(d) => {
                let (v, t) = d.get_veff_tagged(dms, band).map_err(dft_to_py)?;
                (v, t, None)
            }
            Driver::KsymUks(d) => {
                let (v, t) = d.get_veff_tagged(dms, band).map_err(dft_to_py)?;
                (v, t, None)
            }
            Driver::Roks(d) => {
                let (v, t) = d.get_veff_tagged(dms, band).map_err(dft_to_py)?;
                (v, t, None)
            }
            Driver::Gks(d) => {
                let (v, t) = d.get_veff_tagged(dms, band).map_err(dft_to_py)?;
                (v, t, None)
            }
            Driver::Rkspu(p) => {
                if band.is_some() {
                    return Err(pu_band());
                }
                let (mut v, t) = p.ks.get_veff_tagged(dms, None).map_err(dft_to_py)?;
                let e =
                    add_vhubbard(&mut v, p.ks.cell(), p.ks.kpts(), dms, &p.u).map_err(dft_to_py)?;
                (v, t, Some(e))
            }
            Driver::Ukspu(p) => {
                if band.is_some() {
                    return Err(pu_band());
                }
                let (mut v, t) = p.ks.get_veff_tagged(dms, None).map_err(dft_to_py)?;
                let e =
                    add_vhubbard(&mut v, p.ks.cell(), p.ks.kpts(), dms, &p.u).map_err(dft_to_py)?;
                (v, t, Some(e))
            }
            // `KsymAdaptedKrkspu::get_veff` (krks_ksymm.rs:481-497), op for op.
            Driver::KsymRkspu(p) => {
                if band.is_some() {
                    return Err(pu_band());
                }
                let (mut v, t) = p.ks.get_veff_tagged(dms, None).map_err(dft_to_py)?;
                let e = add_vhubbard_weighted(
                    &mut v,
                    p.ks.cell(),
                    KOverrideHooks::kpts(&p.ks),
                    dms,
                    &p.u,
                    &p.ks.kpts.weights_ibz,
                )
                .map_err(dft_to_py)?;
                (v, t, Some(e))
            }
            Driver::KsymUkspu(p) => {
                if band.is_some() {
                    return Err(pu_band());
                }
                let (mut v, t) = p.ks.get_veff_tagged(dms, None).map_err(dft_to_py)?;
                let e = add_vhubbard_weighted(
                    &mut v,
                    p.ks.cell(),
                    KOverrideHooks::kpts(&p.ks),
                    dms,
                    &p.u,
                    &p.ks.kpts.weights_ibz,
                )
                .map_err(dft_to_py)?;
                (v, t, Some(e))
            }
        })
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

/// The KS energy-tag contract across the Python bridge.
///
/// Every Rust KS driver's `energy_elec` reads the `ecoul`/`exc` (and `E_U`)
/// its own `get_veff` left behind — upstream's `lib.tag_array` on `vhf`. When
/// a Python subclass overrides `get_veff`, the bridge never calls the Rust
/// `get_veff`, so those tags are never refreshed: `Krks` recomputes them once
/// and then serves the cycle-1 values forever (measured: 6.2e-3 Ha on He),
/// and the ksymm drivers refuse outright. This guard sits between the bridge
/// and the driver: `get_veff` through it marks the tags fresh; an
/// `energy_elec` that follows a `get_veff` it did not see first refreshes
/// them on the SAME density — upstream's "untagged `vhf` → recompute" rule
/// (`krks.py:118-119`). With no override the tags are always fresh, no
/// extra call is made, and the kernel stays op-for-op the concrete one.
struct TagGuard<'a> {
    d: &'a Driver,
    fresh: StdCell<bool>,
}

impl KOverrideHooks for TagGuard<'_> {
    fn cell(&self) -> &Cell {
        KOverrideHooks::cell(self.d)
    }
    fn kpts(&self) -> &[[f64; 3]] {
        KOverrideHooks::kpts(self.d)
    }
    fn nset(&self) -> usize {
        self.d.nset()
    }
    fn nfock(&self) -> usize {
        self.d.nfock()
    }
    fn nao(&self) -> usize {
        KOverrideHooks::nao(self.d)
    }
    fn get_ovlp(&self) -> Result<KMats, PyscfRsError> {
        self.d.get_ovlp()
    }
    fn get_hcore(&self) -> Result<KMats, PyscfRsError> {
        self.d.get_hcore()
    }
    fn get_init_guess(&self, mode: &KInitGuess, s1e: &KMats) -> Result<KDms, PyscfRsError> {
        self.d.get_init_guess(mode, s1e)
    }
    fn get_veff(&self, dms: &KDms) -> Result<KDms, PyscfRsError> {
        let v = self.d.get_veff(dms)?;
        self.fresh.set(true);
        Ok(v)
    }
    fn get_fock(&self, h1e: &KMats, vhf: &KDms, dms: &KDms) -> Result<KDms, PyscfRsError> {
        self.d.get_fock(h1e, vhf, dms)
    }
    fn diis_dms(&self, dms: &KDms) -> KDms {
        self.d.diis_dms(dms)
    }
    fn eig(&self, fock: &KDms, s1e: &KMats) -> Result<(Vec<Vec<f64>>, Vec<CTensor>), PyscfRsError> {
        self.d.eig(fock, s1e)
    }
    fn get_occ(&self, mo_energy: &[Vec<f64>]) -> Result<(Vec<Vec<f64>>, Vec<f64>), PyscfRsError> {
        self.d.get_occ(mo_energy)
    }
    fn make_rdm1(&self, mo_coeff: &[CTensor], mo_occ: &[Vec<f64>]) -> Result<KDms, PyscfRsError> {
        self.d.make_rdm1(mo_coeff, mo_occ)
    }
    fn energy_elec(&self, dms: &KDms, h1e: &KMats, vhf: &KDms) -> Result<(f64, f64), PyscfRsError> {
        if !self.fresh.get() {
            self.d.get_veff(dms)?;
        }
        self.fresh.set(false);
        self.d.energy_elec(dms, h1e, vhf)
    }
    fn energy_nuc(&self) -> Result<f64, PyscfRsError> {
        self.d.energy_nuc()
    }
    fn get_grad(
        &self,
        mo_coeff: &[CTensor],
        mo_occ: &[Vec<f64>],
        h1e: &KMats,
        vhf: &KDms,
    ) -> Vec<f64> {
        self.d.get_grad(mo_coeff, mo_occ, h1e, vhf)
    }
    fn free_energy(&self) -> Option<f64> {
        self.d.free_energy()
    }
}

/// The last kernel's result plus the layout needed to hand it to Python.
#[derive(Debug, Clone)]
struct Solved {
    res: KScfResult,
    nao: usize,
    nfock: usize,
    sigma: Option<f64>,
    e_u: Option<f64>,
}

/// The numint backend `mf._numint` selects.
enum NumIntSel {
    Grid,
    MultiGrid(Option<[usize; 3]>),
    MultiGrid2(Option<[usize; 3]>),
}

// ─────────────────────────────────────────────────────────────────────────────
// KohnShamDFT — the native base class
// ─────────────────────────────────────────────────────────────────────────────

/// `KohnShamDFT` — the native base of the periodic KS drivers
/// (`pbc/dft/rks.py:KohnShamDFT` + `khf.KSCF`).
///
/// Not constructed directly: use `KRKS`, `KUKS`, `KROKS`, `KGKS`, `KRKSpU`,
/// `KUKSpU` or a `KsymAdapted*` class.
#[pyclass(
    subclass,
    dict,
    name = "KohnShamDFT",
    module = "pyscf._native.pbc.dft",
    skip_from_py_object
)]
pub struct PyKohnShamDft {
    kind: Kind,
    py_cell: Py<PyAny>,
    with_df: Py<PyAny>,
    kpoints: Option<Py<PyAny>>,
    xc: String,
    grids: Py<PyAny>,
    numint: Py<PyAny>,
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
    collinear: Collinear,
    u_idx: Vec<String>,
    u_val: Vec<f64>,
    alpha: Option<f64>,
    minao_ref: String,
    solved: Option<Solved>,
    overridden: Vec<&'static str>,
    /// BAND-04 — the grid numint's AO cache, kept across the per-call driver
    /// rebuilds and tagged with the cell/k-point fingerprint it was filled for.
    ao_cache: std::sync::Mutex<Option<(u64, KNumIntCache)>>,
}

/// DFT+U constructor inputs (`krkspu.py:188-190`).
struct PuArgs<'a, 'py> {
    u_idx: Option<&'a Bound<'py, PyAny>>,
    u_val: Option<Vec<f64>>,
    c_ao_lo: Option<&'a Bound<'py, PyAny>>,
    minao_ref: &'a str,
}

impl PyKohnShamDft {
    fn construct(
        cell: &Bound<'_, PyAny>,
        kpts: Option<&Bound<'_, PyAny>>,
        xc: &str,
        exxdiv: Option<&str>,
        kind: Kind,
        pu: Option<PuArgs<'_, '_>>,
    ) -> PyResult<Self> {
        let py = cell.py();
        let kpoints = some(kpts).filter(|k| k.cast::<PyKPoints>().is_ok());
        let kind = match (kind.is_ksymm(), kpoints.is_some()) {
            (_, true) => kind.with_kpoints().ok_or_else(|| {
                not_impl(format!(
                    "{} with a KPoints object (k-point symmetry) is not implemented in pyscf-rs \
                     (upstream has no {} k-point-symmetry class either); pass kpts.kpts",
                    kind.name(),
                    kind.name()
                ))
            })?,
            (true, false) => {
                return Err(PyTypeError::new_err(format!(
                    "{} requires a built pyscf.pbc.symm.KPoints as kpts",
                    kind.name()
                )));
            }
            (false, false) => kind,
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
                    return Err(PyValueError::new_err(format!(
                        "{}: the KPoints object is not built (call build())",
                        kind.name()
                    )));
                }
                // `ksymm_scf_common_init` (khf_ksymm.py:142): `use_ao_symmetry`
                // defaults to True and needs `cell.build_symmetry(kpts)`. As in
                // 20-12 (D2), the default FFTDF gets a symmetry-built COPY of the
                // cell; a failure is left to the kernel, which names it.
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
        let grids = Py::new(py, PyUniformGrids::new(cell))?.into_any();
        let numint = Py::new(py, PyKNumInt {})?.into_any();
        let d = KScfConfig::default();
        let mut me = Self {
            kind,
            py_cell: cell.clone().unbind(),
            with_df: with_df.unbind(),
            kpoints: kpoints.map(|k| k.clone().unbind()),
            xc: xc.to_string(),
            grids,
            numint,
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
            collinear: Collinear::default(),
            u_idx: Vec::new(),
            u_val: Vec::new(),
            alpha: None,
            minao_ref: "MINAO".into(),
            solved: None,
            overridden: Vec::new(),
            ao_cache: std::sync::Mutex::new(None),
        };
        if let Some(pu) = pu {
            if let Some(idx) = some(pu.u_idx) {
                me.u_idx = extract_u_idx(idx)?;
            }
            me.u_val = pu.u_val.unwrap_or_default();
            if me.u_idx.len() != me.u_val.len() {
                return Err(PyValueError::new_err(format!(
                    "{}: {} U_idx entries for {} U_val values",
                    kind.name(),
                    me.u_idx.len(),
                    me.u_val.len()
                )));
            }
            match some(pu.c_ao_lo) {
                None => {}
                Some(c)
                    if c.extract::<String>()
                        .is_ok_and(|s| s.eq_ignore_ascii_case("minao")) => {}
                Some(_) => {
                    return Err(not_impl(
                        "C_ao_lo as explicit local-orbital arrays is not bound; the Löwdin \
                         MINAO local orbitals (C_ao_lo=None / 'minao') are",
                    ));
                }
            }
            me.minao_ref = pu.minao_ref.to_string();
        }
        Ok(me)
    }

    fn hubbard(&self) -> PyResult<HubbardU> {
        Ok(HubbardU {
            sites: self
                .u_idx
                .iter()
                .map(|s| parse_u_site(s))
                .collect::<PyResult<Vec<_>>>()?,
            u_val: self.u_val.clone(),
            alpha: self.alpha.map_or_else(Vec::new, |a| vec![a]),
            minao_ref: self.minao_ref.to_ascii_lowercase(),
            c_ao_lo: None,
        })
    }

    /// The grid `mf.grids` describes, over the driver's cell. `None` means
    /// "the `from_df` default" (a `UniformGrids` with no explicit mesh), so the
    /// default path builds exactly what the Rust constructor builds.
    fn grids_override(&self, py: Python<'_>, cell: &Cell) -> PyResult<Option<PeriodicGrids>> {
        let g = self.grids.bind(py);
        if let Ok(u) = g.cast::<PyUniformGrids>() {
            return match u.borrow().mesh {
                None => Ok(None),
                Some(m) => PeriodicGrids::uniform(cell, Some(m))
                    .map(Some)
                    .map_err(dft_to_py),
            };
        }
        if let Ok(b) = g.cast::<PyBeckeGrids>() {
            return Ok(Some(b.borrow_mut().ensure(py)?.clone()));
        }
        Err(PyTypeError::new_err(
            "mf.grids must be a pyscf.pbc.dft UniformGrids or BeckeGrids",
        ))
    }

    fn numint_sel(&self, py: Python<'_>) -> PyResult<NumIntSel> {
        let n = self.numint.bind(py);
        if n.cast::<PyKNumInt>().is_ok() {
            Ok(NumIntSel::Grid)
        } else if let Ok(m) = n.cast::<PyMultiGridNumInt>() {
            Ok(NumIntSel::MultiGrid(m.borrow().mesh))
        } else if let Ok(m) = n.cast::<PyMultiGridNumInt2>() {
            Ok(NumIntSel::MultiGrid2(m.borrow().mesh))
        } else {
            Err(PyTypeError::new_err(
                "mf._numint must be a pyscf.pbc.dft KNumInt, MultiGridNumInt or MultiGridNumInt2",
            ))
        }
    }

    fn kpoints_inner(&self, py: Python<'_>) -> PyResult<pyscf_pbc_symm::kpts::KPoints> {
        let kp_obj = self
            .kpoints
            .as_ref()
            .ok_or_else(|| PyValueError::new_err(format!("{} has no KPoints", self.kind.name())))?;
        Ok(kp_obj
            .bind(py)
            .cast::<PyKPoints>()?
            .borrow()
            .kpoints()
            .clone())
    }

    /// A Rust driver over the CURRENT `with_df`, `grids`, `_numint` and
    /// configuration (20-10 contract).
    fn driver(&self, py: Python<'_>) -> PyResult<Driver> {
        let df = extract_df(self.with_df.bind(py))?;
        let grids = self.grids_override(py, df.cell())?;
        let sel = self.numint_sel(py)?;
        let cell_mesh = df.cell().try_mesh().map_err(pyscf_to_py)?;
        let ni = match sel {
            NumIntSel::Grid => None,
            NumIntSel::MultiGrid(m) | NumIntSel::MultiGrid2(m)
                if m.is_some_and(|m| m != cell_mesh) =>
            {
                return Err(not_impl(format!(
                    "multigrid_numint(mesh={:?}): the Rust multigrid engines integrate on \
                     cell.mesh = {cell_mesh:?} only",
                    m.unwrap_or_default()
                )));
            }
            NumIntSel::MultiGrid(_) => Some(KsNumInt::multigrid()),
            NumIntSel::MultiGrid2(_) => Some(KsNumInt::multigrid2()),
        };
        if ni.is_some() && self.kind == Kind::Gks {
            return Err(not_impl(
                "KGKS with a multigrid numint is not ported (KNumInt2C has no multigrid arm)",
            ));
        }
        let xc = self.xc.as_str();
        let kp_needed = self.kind.is_ksymm();
        let kp = if kp_needed {
            let kp = self.kpoints_inner(py)?;
            if df.kpts().len() != kp.nkpts() {
                return Err(PyValueError::new_err(format!(
                    "{}: with_df samples {} k-points but the KPoints full Brillouin zone has {}; \
                     with_df must be built over kpts.kpts",
                    self.kind.name(),
                    df.kpts().len(),
                    kp.nkpts()
                )));
            }
            Some(kp)
        } else {
            None
        };
        macro_rules! krks {
            ($df:expr) => {{
                let mut d = Krks::from_df($df, xc).map_err(dft_to_py)?;
                d.exxdiv = self.exxdiv;
                d.smearing = self.smearing.clone();
                if let Some(g) = grids.clone() {
                    d.grids = g;
                }
                if let Some(n) = ni {
                    d.ni = n;
                }
                d
            }};
        }
        macro_rules! kuks {
            ($df:expr) => {{
                let mut d = Kuks::from_df($df, xc).map_err(dft_to_py)?;
                d.exxdiv = self.exxdiv;
                d.smearing = self.smearing.clone();
                d.nelec = self.nelec;
                d.init_guess_breaksym = self.init_guess_breaksym;
                if let Some(g) = grids.clone() {
                    d.grids = g;
                }
                if let Some(n) = ni {
                    d.ni = n;
                }
                d
            }};
        }
        macro_rules! ksym_rks {
            ($df:expr, $kp:expr) => {{
                let g = match grids.clone() {
                    Some(g) => g,
                    None => PeriodicGrids::uniform($df.cell(), None).map_err(dft_to_py)?,
                };
                let mut d = KsymAdaptedKrks::from_df($df, $kp, xc, g);
                d.exxdiv = self.exxdiv;
                d.use_ao_symmetry = self.use_ao_symmetry;
                if let Some(n) = ni {
                    d.ni = n;
                }
                d
            }};
        }
        macro_rules! ksym_uks {
            ($df:expr, $kp:expr) => {{
                // `KsymAdaptedKuks` has no `from_df`: build with `new` over the
                // same cell and swap the pub `with_df`/`grids` in.
                let mut d = KsymAdaptedKuks::new($df.cell().clone(), $kp, xc).map_err(dft_to_py)?;
                d.grids = match grids.clone() {
                    Some(g) => g,
                    None => PeriodicGrids::uniform($df.cell(), None).map_err(dft_to_py)?,
                };
                d.with_df = $df;
                d.exxdiv = self.exxdiv;
                d.nelec = self.nelec;
                d.use_ao_symmetry = self.use_ao_symmetry;
                d.init_guess_breaksym = self.init_guess_breaksym;
                if let Some(n) = ni {
                    d.ni = n;
                }
                d
            }};
        }
        let kp = || {
            kp.clone()
                .ok_or_else(|| PyValueError::new_err("ksymm driver without KPoints"))
        };
        Ok(match self.kind {
            Kind::Rks => {
                let mut d = krks!(df);
                self.share_ao_cache(&mut d.ni, d.with_df.cell(), d.with_df.kpts());
                Driver::Rks(d)
            }
            Kind::Uks => {
                let mut d = kuks!(df);
                self.share_ao_cache(&mut d.ni, d.with_df.cell(), d.with_df.kpts());
                Driver::Uks(d)
            }
            Kind::KsymRks => Driver::KsymRks(ksym_rks!(df, kp()?)),
            Kind::KsymUks => Driver::KsymUks(ksym_uks!(df, kp()?)),
            Kind::Roks => {
                let mut d = Kroks::from_df(df, xc).map_err(dft_to_py)?;
                d.hf.exxdiv = self.exxdiv;
                d.hf.nelec = self.nelec;
                if let Some(g) = grids {
                    d.grids = g;
                }
                if let Some(n) = ni {
                    d.ni = n;
                }
                Driver::Roks(d)
            }
            Kind::Gks => {
                let mut d = Kgks::from_df(df, xc).map_err(dft_to_py)?;
                d.hf.exxdiv = self.exxdiv;
                d.ni.collinear = self.collinear;
                if let Some(g) = grids {
                    d.grids = g;
                }
                Driver::Gks(d)
            }
            Kind::Rkspu => Driver::Rkspu(PuDriver {
                ks: krks!(df),
                u: self.hubbard()?,
                e_u: StdCell::new(0.0),
            }),
            Kind::Ukspu => Driver::Ukspu(PuDriver {
                ks: kuks!(df),
                u: self.hubbard()?,
                e_u: StdCell::new(0.0),
            }),
            Kind::KsymRkspu => Driver::KsymRkspu(KsymAdaptedKrkspu::new(
                ksym_rks!(df, kp()?),
                self.hubbard()?,
            )),
            Kind::KsymUkspu => Driver::KsymUkspu(KsymAdaptedKukspu::new(
                ksym_uks!(df, kp()?),
                self.hubbard()?,
            )),
        })
    }

    /// BAND-04: hand the grid numint the AO cache an earlier call on this
    /// object filled — the SCF's tables, so `get_bands` after `kernel()` does
    /// not re-evaluate them — or keep its fresh cache for the next call. The
    /// cache is keyed by grid and k-points but not by the cell, so the held
    /// handle is dropped whenever the cell or k-point fingerprint changes.
    /// `PYSCF_PBC_KEEP_AO_CACHE=0` turns this off.
    fn share_ao_cache(&self, ni: &mut KsNumInt, cell: &Cell, kpts: &[[f64; 3]]) {
        use std::hash::{Hash, Hasher};
        let KsNumInt::Grid(knum) = ni else { return };
        if std::env::var("PYSCF_PBC_KEEP_AO_CACHE").is_ok_and(|v| v == "0") {
            return;
        }
        let mut h = std::collections::hash_map::DefaultHasher::new();
        format!("{:?}", cell.mol._atom).hash(&mut h);
        let mut basis: Vec<_> = cell.mol._basis.iter().collect();
        basis.sort_by(|a, b| a.0.cmp(b.0));
        format!("{basis:?}").hash(&mut h);
        cell.mol.cart.hash(&mut h);
        for x in cell.lattice_vectors().iter().chain(kpts) {
            for v in x {
                v.to_bits().hash(&mut h);
            }
        }
        let key = h.finish();
        let Ok(mut held) = self.ao_cache.lock() else { return };
        match held.as_ref() {
            Some((k, cache)) if *k == key => knum.adopt_ao_cache(cache),
            _ => *held = Some((key, knum.ao_cache_handle())),
        }
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

    fn explicit_kpts(
        &self,
        kpts: Option<&Bound<'_, PyAny>>,
        what: &str,
    ) -> PyResult<Option<(Vec<[f64; 3]>, bool)>> {
        let Some(k) = extract_kpts_opt(kpts)? else {
            return Ok(None);
        };
        if self.kind == Kind::Gks {
            return Err(not_impl(format!(
                "KGKS.{what}(kpts=...): only the driver's own k-points are bound"
            )));
        }
        Ok(Some(k))
    }

    fn attr_err(&self, attr: &str) -> PyErr {
        PyAttributeError::new_err(format!(
            "'{}' object has no attribute '{attr}'",
            self.kind.name()
        ))
    }

    fn run_kernel(
        slf: &Bound<'_, Self>,
        dm0: Option<&Bound<'_, PyAny>>,
        use_bridge: bool,
    ) -> PyResult<f64> {
        let py = slf.py();
        let (d, py_cell, cfg, chkfile) = {
            let me = slf.borrow();
            let d = me.driver(py)?;
            let (nk, nao, nset) = (d.kpts().len(), KOverrideHooks::nao(&d), d.nset());
            let guess = match some(dm0) {
                Some(x) => KInitGuess::UserDm(kdms_from_py(x, nset, nk, nao, "dm0")?),
                None => match me.solved.as_ref() {
                    Some(_) => KInitGuess::UserDm(me.stored_dm(&d)?),
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
            let base = py.get_type::<PyKohnShamDft>();
            let guard = TagGuard {
                d: &d,
                fresh: StdCell::new(false),
            };
            let bridge = KPyOverrideBridge::new(
                py,
                slf.clone().into_any().unbind(),
                py_cell,
                &base,
                &guard,
            )?;
            let res = bridge.finish(pyscf_pbc_scf::kernel(&bridge, &cfg))?;
            (res, bridge.overridden_hooks())
        } else {
            (d.kernel_direct(&cfg).map_err(pyscf_to_py)?, Vec::new())
        };
        let nao = KOverrideHooks::nao(&d);
        if let Some(path) = chkfile.as_deref() {
            let json = pyscf_pbc_gto::dumps(d.cell()).map_err(pyscf_to_py)?;
            dump_kscf_to_file(std::path::Path::new(path), &res, d.kpts(), nao, &json)
                .map_err(|e| PyOSError::new_err(format!("periodic chkfile: {e}")))?;
        }
        let e_tot = res.e_tot;
        let mut me = slf.borrow_mut();
        me.solved = Some(Solved {
            nao,
            nfock: d.nfock(),
            sigma: me.smearing.as_ref().map(|s| s.sigma),
            e_u: d.e_u(),
            res,
        });
        me.overridden = overridden;
        Ok(e_tot)
    }
}

fn maybe_single<'py>(list: Bound<'py, PyAny>, single: bool) -> PyResult<Bound<'py, PyAny>> {
    if single { list.get_item(0) } else { Ok(list) }
}

#[pymethods]
impl PyKohnShamDft {
    // ── configuration ───────────────────────────────────────────────────────

    #[getter]
    fn cell(&self, py: Python<'_>) -> Py<PyAny> {
        self.py_cell.clone_ref(py)
    }

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
    /// for the ksymm drivers.
    #[getter]
    fn kpts(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if let Some(kp) = &self.kpoints {
            return Ok(kp.clone_ref(py));
        }
        Ok(self.with_df.bind(py).getattr("kpts")?.unbind())
    }
    #[setter]
    fn set_kpts(&mut self, py: Python<'_>, v: &Bound<'_, PyAny>) -> PyResult<()> {
        if v.cast::<PyKPoints>().is_ok() || self.kind.is_ksymm() {
            return Err(not_impl(
                "switching between a k-point array and a KPoints object after construction is \
                 not bound; construct a new driver",
            ));
        }
        self.with_df.bind(py).setattr("kpts", v)?;
        self.solved = None;
        Ok(())
    }

    /// PRIVATE: whether the k-point-symmetry driver runs.
    #[getter]
    fn _is_ksymm(&self) -> bool {
        self.kind.is_ksymm()
    }

    /// The XC functional string (upstream default `'LDA,VWN'`).
    #[getter]
    fn xc(&self) -> String {
        self.xc.clone()
    }
    #[setter]
    fn set_xc(&mut self, v: String) {
        self.xc = v;
    }

    /// `nlc` — non-local correlation. Only `''` is ported.
    #[getter]
    fn nlc(&self) -> &'static str {
        ""
    }
    #[setter]
    fn set_nlc(&mut self, v: Option<String>) -> PyResult<()> {
        match v.as_deref() {
            None | Some("") => Ok(()),
            Some(other) => Err(not_impl(format!(
                "nlc = {other:?}: non-local correlation (VV10) is not ported for PBC"
            ))),
        }
    }

    /// The XC quadrature (`UniformGrids(cell)` by default). The object is
    /// shared: `mf.grids.mesh = X` is seen by the next call.
    #[getter]
    fn grids(&self, py: Python<'_>) -> Py<PyAny> {
        self.grids.clone_ref(py)
    }
    #[setter]
    fn set_grids(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        if v.cast::<PyUniformGrids>().is_err() && v.cast::<PyBeckeGrids>().is_err() {
            return Err(PyTypeError::new_err(
                "mf.grids must be a pyscf.pbc.dft UniformGrids or BeckeGrids",
            ));
        }
        self.grids = v.clone().unbind();
        Ok(())
    }

    /// The numint backend selector: `KNumInt` (grid), `MultiGridNumInt` (v1)
    /// or `MultiGridNumInt2` (v2).
    #[getter]
    fn _numint(&self, py: Python<'_>) -> Py<PyAny> {
        self.numint.clone_ref(py)
    }
    #[setter(_numint)]
    fn set_numint(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        if v.cast::<PyKNumInt>().is_err()
            && v.cast::<PyMultiGridNumInt>().is_err()
            && v.cast::<PyMultiGridNumInt2>().is_err()
        {
            return Err(PyTypeError::new_err(
                "mf._numint must be a pyscf.pbc.dft KNumInt, MultiGridNumInt or MultiGridNumInt2",
            ));
        }
        self.numint = v.clone().unbind();
        Ok(())
    }

    /// `multigrid_numint(mesh=None)` — `krks.py:284`: select multigrid v1.
    /// Mutates IN PLACE and returns `self` (upstream returns a copy).
    #[pyo3(signature = (mesh = None))]
    fn multigrid_numint<'py>(
        slf: &Bound<'py, Self>,
        mesh: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        let mesh = match some(mesh) {
            Some(m) => Some(extract_usize3(m, "mesh")?),
            None => None,
        };
        let sel = Py::new(py, PyMultiGridNumInt { mesh })?.into_any();
        slf.borrow_mut().numint = sel;
        Ok(slf.clone())
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
    #[getter(init_guess)]
    fn init_guess_key(&self) -> &str {
        &self.init_guess
    }
    #[setter(init_guess)]
    fn set_init_guess_key(&mut self, v: String) -> PyResult<()> {
        parse_init_guess(&v)?;
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

    /// `(nalpha, nbeta)` (KUKS/KROKS and their variants).
    #[getter]
    fn nelec(&self, py: Python<'_>) -> PyResult<(usize, usize)> {
        if !(self.kind.unrestricted() || self.kind == Kind::Roks) {
            return Err(self.attr_err("nelec"));
        }
        if let Some(n) = self.nelec {
            return Ok(n);
        }
        match self.driver(py)? {
            Driver::Uks(d) => d.nelec(),
            Driver::KsymUks(d) => d.nelec(),
            Driver::Ukspu(d) => d.ks.nelec(),
            Driver::KsymUkspu(d) => d.ks.nelec(),
            Driver::Roks(d) => d.hf.nelec(),
            _ => unreachable!("kind checked above"),
        }
        .map_err(pyscf_to_py)
    }
    #[setter]
    fn set_nelec(&mut self, v: Option<(usize, usize)>) -> PyResult<()> {
        if !(self.kind.unrestricted() || self.kind == Kind::Roks) {
            return Err(self.attr_err("nelec"));
        }
        self.nelec = v;
        Ok(())
    }

    #[getter]
    fn init_guess_breaksym(&self) -> PyResult<i32> {
        if !self.kind.unrestricted() {
            return Err(self.attr_err("init_guess_breaksym"));
        }
        Ok(self.init_guess_breaksym)
    }
    #[setter]
    fn set_init_guess_breaksym(&mut self, v: i32) -> PyResult<()> {
        if !self.kind.unrestricted() {
            return Err(self.attr_err("init_guess_breaksym"));
        }
        self.init_guess_breaksym = v;
        Ok(())
    }

    #[getter]
    fn use_ao_symmetry(&self) -> PyResult<bool> {
        if !self.kind.is_ksymm() {
            return Err(self.attr_err("use_ao_symmetry"));
        }
        Ok(self.use_ao_symmetry)
    }
    #[setter]
    fn set_use_ao_symmetry(&mut self, v: bool) -> PyResult<()> {
        if !self.kind.is_ksymm() {
            return Err(self.attr_err("use_ao_symmetry"));
        }
        self.use_ao_symmetry = v;
        Ok(())
    }

    /// `collinear` (KGKS only; `'col'` default, `'ncol'` LDA only, `'mcol'`
    /// refused at use — `numint2c.rs:104,152`).
    #[getter]
    fn collinear(&self) -> PyResult<&'static str> {
        if self.kind != Kind::Gks {
            return Err(self.attr_err("collinear"));
        }
        Ok(collinear_str(self.collinear))
    }
    #[setter]
    fn set_collinear(&mut self, v: &str) -> PyResult<()> {
        if self.kind != Kind::Gks {
            return Err(self.attr_err("collinear"));
        }
        self.collinear = parse_collinear(v)?;
        Ok(())
    }

    // ── DFT+U attributes (`krkspu.py:180-205`) ──────────────────────────────

    #[getter(U_idx)]
    fn u_idx(&self) -> PyResult<Vec<String>> {
        if !self.kind.is_pu() {
            return Err(self.attr_err("U_idx"));
        }
        Ok(self.u_idx.clone())
    }
    #[setter(U_idx)]
    fn set_u_idx(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        if !self.kind.is_pu() {
            return Err(self.attr_err("U_idx"));
        }
        self.u_idx = extract_u_idx(v)?;
        Ok(())
    }
    /// Effective `U - J` per `U_idx` entry, in eV.
    #[getter(U_val)]
    fn u_val(&self) -> PyResult<Vec<f64>> {
        if !self.kind.is_pu() {
            return Err(self.attr_err("U_val"));
        }
        Ok(self.u_val.clone())
    }
    #[setter(U_val)]
    fn set_u_val(&mut self, v: Vec<f64>) -> PyResult<()> {
        if !self.kind.is_pu() {
            return Err(self.attr_err("U_val"));
        }
        self.u_val = v;
        Ok(())
    }
    #[getter]
    fn minao_ref(&self) -> PyResult<String> {
        if !self.kind.is_pu() {
            return Err(self.attr_err("minao_ref"));
        }
        Ok(self.minao_ref.clone())
    }
    #[setter]
    fn set_minao_ref(&mut self, v: String) -> PyResult<()> {
        if !self.kind.is_pu() {
            return Err(self.attr_err("minao_ref"));
        }
        self.minao_ref = v;
        Ok(())
    }
    /// The LR-cDFT perturbation, added raw to the local potential as upstream's
    /// code does (`krkspu.py:114-118`). Only `None` or one scalar is bound.
    #[getter]
    fn alpha(&self) -> PyResult<Option<f64>> {
        if !self.kind.is_pu() {
            return Err(self.attr_err("alpha"));
        }
        Ok(self.alpha)
    }
    #[setter]
    fn set_alpha(&mut self, v: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        if !self.kind.is_pu() {
            return Err(self.attr_err("alpha"));
        }
        self.alpha = match some(v) {
            None => None,
            Some(x) => Some(x.extract::<f64>().map_err(|_| {
                not_impl("alpha as a per-site list is not bound (None or one scalar)")
            })?),
        };
        Ok(())
    }
    /// `C_ao_lo` — always `None` here (the Löwdin MINAO orbitals).
    #[getter(C_ao_lo)]
    fn c_ao_lo(&self) -> PyResult<Option<()>> {
        if !self.kind.is_pu() {
            return Err(self.attr_err("C_ao_lo"));
        }
        Ok(None)
    }

    /// `E_U` of the last kernel (DFT+U), `None` otherwise.
    #[getter]
    fn e_u(&self) -> Option<f64> {
        self.solved.as_ref().and_then(|s| s.e_u)
    }

    // ── smearing ────────────────────────────────────────────────────────────

    /// `mf.smearing_(sigma=None, method='fermi', mu0=None)` — IN PLACE, returns
    /// `self` (KRKS/KUKS and their DFT+U forms; `Krks`/`Kuks` carry smearing).
    #[pyo3(signature = (sigma = None, method = "fermi", mu0 = None))]
    fn smearing_<'py>(
        slf: &Bound<'py, Self>,
        sigma: Option<f64>,
        method: &str,
        mu0: Option<f64>,
    ) -> PyResult<Bound<'py, Self>> {
        let mut me = slf.borrow_mut();
        if !matches!(me.kind, Kind::Rks | Kind::Uks | Kind::Rkspu | Kind::Ukspu) {
            return Err(not_impl(format!(
                "{}.smearing_: smearing is ported for KRKS/KUKS (krks.rs:92, kuks.rs:49) only",
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
    /// `smearing_method` (upstream `_SmearingSCF.smearing_method`,
    /// pyscf/scf/smearing.py:131), as `KSCF.smearing_method` in `pbc/scf.rs`.
    #[getter]
    fn smearing_method(&self) -> Option<&'static str> {
        self.smearing.as_ref().map(|s| match s.method {
            SmearingMethod::Fermi => "fermi",
            SmearingMethod::Gaussian => "gaussian",
        })
    }
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

    // ── results ─────────────────────────────────────────────────────────────

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
    #[getter]
    fn fermi(&self) -> Option<Vec<f64>> {
        self.solved.as_ref().map(|s| s.res.fermi.clone())
    }
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
    /// Smearing entropy `S` (`e_free = e_tot - sigma * S`), the same formula as
    /// `pbc/scf.rs` `KSCF.entropy`. Upstream sets `mf.entropy` in
    /// `pyscf/pbc/scf/smearing.py:87-89,123-126` (per-k average, ×2 restricted);
    /// `None` without smearing, as upstream's `entropy = None` (scf/smearing.py:133).
    #[getter]
    fn entropy(&self) -> Option<f64> {
        let s = self.solved.as_ref()?;
        Some((s.res.e_tot - s.res.e_free?) / s.sigma?)
    }
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
    #[getter]
    fn mo_coeff<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.solved
            .as_ref()
            .map(|s| mo_coeff_to_py(py, &s.res.mo_coeff, s.nfock, s.nao))
            .transpose()
    }
    #[getter]
    fn _overridden_hooks(&self) -> Vec<&'static str> {
        self.overridden.clone()
    }

    // ── the eleven hooks (their Rust defaults) ──────────────────────────────

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
        let mode = parse_init_guess(key)?;
        let s1e = match some(s1e) {
            Some(s) => kmats_from_py(s, nk, nao, "s1e")?,
            None => d.get_ovlp().map_err(pyscf_to_py)?,
        };
        let dm = d.get_init_guess(&mode, &s1e).map_err(pyscf_to_py)?;
        kdms_to_py(py, &dm, nao)
    }

    /// `get_veff(cell=None, dm_kpts=None, dm_last=0, vhf_last=0, hermi=1,
    /// kpts=None, kpts_band=None)`. `dm_kpts=None` uses the stored result;
    /// `kpts_band` returns the potential at those k-points.
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
        if hermi != 1 || some(kpts).is_some() {
            return Err(not_impl(format!(
                "{}.get_veff: only hermi=1 at the driver's own k-points is bound",
                self.kind.name()
            )));
        }
        let d = self.driver(py)?;
        let dms = self.dms_arg(&d, dm_kpts, "dm_kpts")?;
        let nao = KOverrideHooks::nao(&d);
        match extract_kpts_opt(kpts_band)? {
            None => kdms_to_py(py, &d.get_veff(&dms).map_err(pyscf_to_py)?, nao),
            Some((band, single)) => {
                let (v, _, _) = d.veff_components(&dms, Some(&band))?;
                let out = kdms_to_py(py, &v, nao)?;
                if single && v.len() == 1 {
                    out.get_item(0)
                } else {
                    Ok(out)
                }
            }
        }
    }

    /// PRIVATE: `get_veff` plus upstream's `tag_array` attributes, as a dict
    /// `{vxc, ecoul, exc, nelec, E_U}` (`E_U` is `None` without DFT+U).
    #[pyo3(signature = (dm_kpts = None))]
    fn _veff_components<'py>(
        &self,
        py: Python<'py>,
        dm_kpts: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let d = self.driver(py)?;
        let dms = self.dms_arg(&d, dm_kpts, "dm_kpts")?;
        let (v, tags, e_u) = d.veff_components(&dms, None)?;
        let out = PyDict::new(py);
        out.set_item("vxc", kdms_to_py(py, &v, KOverrideHooks::nao(&d))?)?;
        out.set_item("ecoul", tags.ecoul)?;
        out.set_item("exc", tags.exc)?;
        out.set_item("nelec", tags.nelec)?;
        out.set_item("E_U", e_u)?;
        Ok(out)
    }

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

    /// `energy_elec(dm_kpts=None, h1e_kpts=None, vhf_kpts=None)`. The KS
    /// energy components come from a `get_veff` on `dm_kpts` (upstream reads
    /// them off the tagged `vhf`; an untagged `vhf` makes it recompute too).
    #[pyo3(signature = (dm_kpts = None, h1e_kpts = None, vhf_kpts = None))]
    fn energy_elec(
        &self,
        py: Python<'_>,
        dm_kpts: Option<&Bound<'_, PyAny>>,
        h1e_kpts: Option<&Bound<'_, PyAny>>,
        vhf_kpts: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(f64, f64)> {
        let d = self.driver(py)?;
        let (nk, nao) = (d.kpts().len(), KOverrideHooks::nao(&d));
        let dm = self.dms_arg(&d, dm_kpts, "dm_kpts")?;
        let h1e = match some(h1e_kpts) {
            Some(x) => kmats_from_py(x, nk, nao, "h1e_kpts")?,
            None => d.get_hcore().map_err(pyscf_to_py)?,
        };
        let _ = vhf_kpts;
        let vhf = d.get_veff(&dm).map_err(pyscf_to_py)?;
        d.energy_elec(&dm, &h1e, &vhf).map_err(pyscf_to_py)
    }

    fn energy_nuc(&self, py: Python<'_>) -> PyResult<f64> {
        self.driver(py)?.energy_nuc().map_err(pyscf_to_py)
    }

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

    /// `get_rho(dm=None, grids=None, kpts=None)` — the real-space density on
    /// `mf.grids` (`krks.py:105-112`; spin-summed for KUKS/KROKS). Grid numint
    /// only.
    #[pyo3(signature = (dm = None, grids = None, kpts = None))]
    fn get_rho<'py>(
        &self,
        py: Python<'py>,
        dm: Option<&Bound<'py, PyAny>>,
        grids: Option<&Bound<'py, PyAny>>,
        kpts: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        if some(grids).is_some() || some(kpts).is_some() {
            return Err(not_impl(
                "get_rho(grids=..., kpts=...): set mf.grids instead",
            ));
        }
        let d = self.driver(py)?;
        let dms = self.dms_arg(&d, dm, "dm")?;
        let sum = |dms: &KDms| -> KMats {
            dms[0]
                .iter()
                .zip(&dms[1])
                .map(|(a, b)| {
                    let mut m = a.clone();
                    for i in 0..m.len() {
                        m.re[i] += b.re[i];
                        m.im[i] += b.im[i];
                    }
                    m
                })
                .collect()
        };
        let rho = match &d {
            Driver::Rks(x) => x.get_rho(&dms[0]),
            Driver::Rkspu(x) => x.ks.get_rho(&dms[0]),
            Driver::Uks(x) => x.get_rho(&dms),
            Driver::Ukspu(x) => x.ks.get_rho(&dms),
            Driver::KsymRks(x) => x.ni.get_rho(x.cell(), &dms[0], &x.grids),
            Driver::KsymRkspu(x) => x.ks.ni.get_rho(x.ks.cell(), &dms[0], &x.ks.grids),
            Driver::KsymUks(x) => x.ni.get_rho(x.cell(), &sum(&dms), &x.grids),
            Driver::KsymUkspu(x) => x.ks.ni.get_rho(x.ks.cell(), &sum(&dms), &x.ks.grids),
            Driver::Roks(x) => x.ni.get_rho(x.cell(), &sum(&dms), &x.grids),
            Driver::Gks(_) => return Err(not_impl("KGKS.get_rho is not bound")),
        }
        .map_err(dft_to_py)?;
        Ok(numpy::PyArray1::from_vec(py, rho))
    }

    /// `nr_fxc(dms, dm0=None, hermi=1)` — the 2-component XC response
    /// (`numint2c.rs:246`, KGKS only); a non-collinear treatment refuses.
    #[pyo3(signature = (dms, dm0 = None, hermi = 1))]
    fn nr_fxc<'py>(
        &self,
        py: Python<'py>,
        dms: &Bound<'py, PyAny>,
        dm0: Option<&Bound<'py, PyAny>>,
        hermi: i32,
    ) -> PyResult<Bound<'py, PyAny>> {
        let Driver::Gks(d) = self.driver(py)? else {
            return Err(self.attr_err("nr_fxc"));
        };
        let (nk, n2) = (d.kpts().len(), KOverrideHooks::nao(&d));
        let dms = kmats_from_py(dms, nk, n2, "dms")?;
        let dm0 = match some(dm0) {
            Some(x) => Some(kmats_from_py(x, nk, n2, "dm0")?),
            None => None,
        };
        let v =
            d.ni.nr_fxc(d.cell(), &d.grids, &d.xc, dm0.as_ref(), &dms, hermi == 1)
                .map_err(dft_to_py)?;
        kmats_to_py(py, &v, n2)
    }

    // ── kernel ──────────────────────────────────────────────────────────────

    /// `kernel(dm0=None)` → `e_tot`, through `KPyOverrideBridge`.
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

    #[pyo3(signature = (dm0 = None))]
    fn scf(slf: &Bound<'_, Self>, dm0: Option<&Bound<'_, PyAny>>) -> PyResult<f64> {
        Self::run_kernel(slf, dm0, true)
    }

    /// PRIVATE: the concrete Rust driver's own `kernel()` with no bridge — the
    /// bitwise reference.
    #[pyo3(signature = (dm0 = None))]
    fn _kernel_without_bridge(
        slf: &Bound<'_, Self>,
        dm0: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<f64> {
        Self::run_kernel(slf, dm0, false)
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn run<'py>(
        slf: &Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        if let Some(kw) = kwargs {
            for (k, v) in kw.iter() {
                slf.setattr(k.cast::<PyString>()?, v)?;
            }
        }
        slf.call_method1("kernel", args)?;
        Ok(slf.clone())
    }

    /// `get_bands(kpts_band, cell=None, dm_kpts=None, kpts=None)` — KRKS/KUKS
    /// (`krks.rs:271`, `kuks.rs:591`).
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
        if !matches!(self.kind, Kind::Rks | Kind::Uks) {
            return Err(not_impl(format!(
                "{}.get_bands is not ported (Krks::get_bands and Kuks::get_bands only)",
                self.kind.name()
            )));
        }
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
            Driver::Rks(x) => {
                let (e, c) = x.get_bands(&kband, &dms).map_err(pyscf_to_py)?;
                (e, c, 1)
            }
            Driver::Uks(x) => {
                let (e, c) = x.get_bands(&kband, &dms).map_err(pyscf_to_py)?;
                (e, c, 2)
            }
            _ => unreachable!("kind checked above"),
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

    /// `density_fit(auxbasis=None, with_df=None)` — `rks.py:_patch_df_beckegrids`:
    /// switch `with_df` to `GDF` (J-only for a pure functional) AND `grids` to
    /// `BeckeGrids(cell)`, IN PLACE; returns `self`.
    #[pyo3(signature = (auxbasis = None, with_df = None))]
    fn density_fit<'py>(
        slf: &Bound<'py, Self>,
        auxbasis: Option<String>,
        with_df: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        let (new_df, cell, xc) = {
            let me = slf.borrow();
            let df = match some(with_df) {
                Some(df) => df.clone(),
                None => {
                    let kpts = me.with_df.bind(py).getattr("kpts")?;
                    let cell = me.with_df.bind(py).getattr("cell")?;
                    let gdf = py.get_type::<PyGdf>().call1((cell, kpts))?;
                    if let Some(a) = auxbasis {
                        gdf.setattr("auxbasis", a)?;
                    }
                    gdf
                }
            };
            (df, me.py_cell.clone_ref(py), me.xc.clone())
        };
        let hybrid = pyscf_pbc_dft::xc::is_hybrid_xc(&xc).map_err(dft_to_py)?;
        if !hybrid && some(with_df).is_none() {
            new_df.setattr("_j_only", true)?;
        }
        let becke = Py::new(py, PyBeckeGrids::new(cell.bind(py)))?.into_any();
        let mut me = slf.borrow_mut();
        me.set_with_df(&new_df)?;
        me.grids = becke;
        drop(me);
        Ok(slf.clone())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The public classes (constructors only; everything else is on KohnShamDFT)
// ─────────────────────────────────────────────────────────────────────────────

/// `KRKS(cell, kpts=None, xc='LDA,VWN', exxdiv='ewald')` — `krks.py:250`. A
/// built `KPoints` runs `KsymAdaptedKrks` (upstream's `pbc.dft.KRKS` dispatch).
#[pyclass(extends = PyKohnShamDft, subclass, name = "KRKS", module = "pyscf._native.pbc.dft")]
pub struct PyKrks {}

#[pymethods]
impl PyKrks {
    #[new]
    #[pyo3(signature = (cell, kpts = None, xc = "LDA,VWN", exxdiv = Some("ewald")))]
    fn new(
        cell: &Bound<'_, PyAny>,
        kpts: Option<&Bound<'_, PyAny>>,
        xc: &str,
        exxdiv: Option<&str>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let base = PyKohnShamDft::construct(cell, kpts, xc, exxdiv, Kind::Rks, None)?;
        Ok(PyClassInitializer::from(base).add_subclass(PyKrks {}))
    }
}

/// `KsymAdaptedKRKS(cell, kpts, xc='LDA,VWN', exxdiv='ewald', use_ao_symmetry=True)`
/// — `krks_ksymm.py:88`; `kpts` must be a built `KPoints`.
#[pyclass(
    extends = PyKrks,
    subclass,
    name = "KsymAdaptedKRKS",
    module = "pyscf._native.pbc.dft"
)]
pub struct PyKsymAdaptedKrks {}

#[pymethods]
impl PyKsymAdaptedKrks {
    #[new]
    #[pyo3(signature = (cell, kpts, xc = "LDA,VWN", exxdiv = Some("ewald"), use_ao_symmetry = true))]
    fn new(
        cell: &Bound<'_, PyAny>,
        kpts: &Bound<'_, PyAny>,
        xc: &str,
        exxdiv: Option<&str>,
        use_ao_symmetry: bool,
    ) -> PyResult<PyClassInitializer<Self>> {
        let mut base = PyKohnShamDft::construct(cell, Some(kpts), xc, exxdiv, Kind::KsymRks, None)?;
        // khf_ksymm.py:143-147: the argument is AND-ed with the cell/KPoints conditions.
        base.use_ao_symmetry = base.use_ao_symmetry && use_ao_symmetry;
        Ok(PyClassInitializer::from(base)
            .add_subclass(PyKrks {})
            .add_subclass(PyKsymAdaptedKrks {}))
    }
}

/// `KUKS(cell, kpts=None, xc='LDA,VWN', exxdiv='ewald')` — `kuks.py:153`. A
/// built `KPoints` runs `KsymAdaptedKuks`.
#[pyclass(extends = PyKohnShamDft, subclass, name = "KUKS", module = "pyscf._native.pbc.dft")]
pub struct PyKuks {}

#[pymethods]
impl PyKuks {
    #[new]
    #[pyo3(signature = (cell, kpts = None, xc = "LDA,VWN", exxdiv = Some("ewald")))]
    fn new(
        cell: &Bound<'_, PyAny>,
        kpts: Option<&Bound<'_, PyAny>>,
        xc: &str,
        exxdiv: Option<&str>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let base = PyKohnShamDft::construct(cell, kpts, xc, exxdiv, Kind::Uks, None)?;
        Ok(PyClassInitializer::from(base).add_subclass(PyKuks {}))
    }
}

/// `KsymAdaptedKUKS(cell, kpts, xc='LDA,VWN', exxdiv='ewald', use_ao_symmetry=True)`
/// — `kuks_ksymm.py:88`.
#[pyclass(
    extends = PyKuks,
    subclass,
    name = "KsymAdaptedKUKS",
    module = "pyscf._native.pbc.dft"
)]
pub struct PyKsymAdaptedKuks {}

#[pymethods]
impl PyKsymAdaptedKuks {
    #[new]
    #[pyo3(signature = (cell, kpts, xc = "LDA,VWN", exxdiv = Some("ewald"), use_ao_symmetry = true))]
    fn new(
        cell: &Bound<'_, PyAny>,
        kpts: &Bound<'_, PyAny>,
        xc: &str,
        exxdiv: Option<&str>,
        use_ao_symmetry: bool,
    ) -> PyResult<PyClassInitializer<Self>> {
        let mut base = PyKohnShamDft::construct(cell, Some(kpts), xc, exxdiv, Kind::KsymUks, None)?;
        // khf_ksymm.py:143-147: the argument is AND-ed with the cell/KPoints conditions.
        base.use_ao_symmetry = base.use_ao_symmetry && use_ao_symmetry;
        Ok(PyClassInitializer::from(base)
            .add_subclass(PyKuks {})
            .add_subclass(PyKsymAdaptedKuks {}))
    }
}

/// `KROKS(cell, kpts=None, xc='LDA,VWN', exxdiv='ewald')` — `kroks.py:44`. A
/// `KPoints` RAISES.
#[pyclass(extends = PyKohnShamDft, subclass, name = "KROKS", module = "pyscf._native.pbc.dft")]
pub struct PyKroks {}

#[pymethods]
impl PyKroks {
    #[new]
    #[pyo3(signature = (cell, kpts = None, xc = "LDA,VWN", exxdiv = Some("ewald")))]
    fn new(
        cell: &Bound<'_, PyAny>,
        kpts: Option<&Bound<'_, PyAny>>,
        xc: &str,
        exxdiv: Option<&str>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let base = PyKohnShamDft::construct(cell, kpts, xc, exxdiv, Kind::Roks, None)?;
        Ok(PyClassInitializer::from(base).add_subclass(PyKroks {}))
    }
}

/// `KGKS(cell, kpts=None, xc='LDA,VWN', exxdiv='ewald')` — `kgks.py:128`. A
/// hybrid functional RAISES at `get_veff` (`kgks.rs:124`, as upstream).
#[pyclass(extends = PyKohnShamDft, subclass, name = "KGKS", module = "pyscf._native.pbc.dft")]
pub struct PyKgks {}

#[pymethods]
impl PyKgks {
    #[new]
    #[pyo3(signature = (cell, kpts = None, xc = "LDA,VWN", exxdiv = Some("ewald")))]
    fn new(
        cell: &Bound<'_, PyAny>,
        kpts: Option<&Bound<'_, PyAny>>,
        xc: &str,
        exxdiv: Option<&str>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let base = PyKohnShamDft::construct(cell, kpts, xc, exxdiv, Kind::Gks, None)?;
        Ok(PyClassInitializer::from(base).add_subclass(PyKgks {}))
    }
}

/// `KRKSpU(cell, kpts=None, xc='LDA,VWN', exxdiv='ewald', U_idx=[], U_val=[],
/// C_ao_lo=None, minao_ref='MINAO')` — `krkspu.py:180`. A `KPoints` runs
/// `KsymAdaptedKrkspu`.
#[pyclass(extends = PyKrks, subclass, name = "KRKSpU", module = "pyscf._native.pbc.dft")]
pub struct PyKrkspu {}

#[pymethods]
impl PyKrkspu {
    #[new]
    #[pyo3(signature = (cell, kpts = None, xc = "LDA,VWN", exxdiv = Some("ewald"), U_idx = None,
                        U_val = None, C_ao_lo = None, minao_ref = "MINAO"))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn new(
        cell: &Bound<'_, PyAny>,
        kpts: Option<&Bound<'_, PyAny>>,
        xc: &str,
        exxdiv: Option<&str>,
        U_idx: Option<&Bound<'_, PyAny>>,
        U_val: Option<Vec<f64>>,
        C_ao_lo: Option<&Bound<'_, PyAny>>,
        minao_ref: &str,
    ) -> PyResult<PyClassInitializer<Self>> {
        let pu = PuArgs {
            u_idx: U_idx,
            u_val: U_val,
            c_ao_lo: C_ao_lo,
            minao_ref,
        };
        let base = PyKohnShamDft::construct(cell, kpts, xc, exxdiv, Kind::Rkspu, Some(pu))?;
        Ok(PyClassInitializer::from(base)
            .add_subclass(PyKrks {})
            .add_subclass(PyKrkspu {}))
    }
}

/// `KUKSpU(...)` — `kukspu.py`; same arguments as `KRKSpU`.
#[pyclass(extends = PyKuks, subclass, name = "KUKSpU", module = "pyscf._native.pbc.dft")]
pub struct PyKukspu {}

#[pymethods]
impl PyKukspu {
    #[new]
    #[pyo3(signature = (cell, kpts = None, xc = "LDA,VWN", exxdiv = Some("ewald"), U_idx = None,
                        U_val = None, C_ao_lo = None, minao_ref = "MINAO"))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn new(
        cell: &Bound<'_, PyAny>,
        kpts: Option<&Bound<'_, PyAny>>,
        xc: &str,
        exxdiv: Option<&str>,
        U_idx: Option<&Bound<'_, PyAny>>,
        U_val: Option<Vec<f64>>,
        C_ao_lo: Option<&Bound<'_, PyAny>>,
        minao_ref: &str,
    ) -> PyResult<PyClassInitializer<Self>> {
        let pu = PuArgs {
            u_idx: U_idx,
            u_val: U_val,
            c_ao_lo: C_ao_lo,
            minao_ref,
        };
        let base = PyKohnShamDft::construct(cell, kpts, xc, exxdiv, Kind::Ukspu, Some(pu))?;
        Ok(PyClassInitializer::from(base)
            .add_subclass(PyKuks {})
            .add_subclass(PyKukspu {}))
    }
}

/// `KsymAdaptedKRKSpU(cell, kpts, ...)` — `krkspu_ksymm.py:56`.
#[pyclass(
    extends = PyKsymAdaptedKrks,
    subclass,
    name = "KsymAdaptedKRKSpU",
    module = "pyscf._native.pbc.dft"
)]
pub struct PyKsymAdaptedKrkspu {}

#[pymethods]
impl PyKsymAdaptedKrkspu {
    #[new]
    #[pyo3(signature = (cell, kpts, xc = "LDA,VWN", exxdiv = Some("ewald"), U_idx = None,
                        U_val = None, C_ao_lo = None, minao_ref = "MINAO"))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn new(
        cell: &Bound<'_, PyAny>,
        kpts: &Bound<'_, PyAny>,
        xc: &str,
        exxdiv: Option<&str>,
        U_idx: Option<&Bound<'_, PyAny>>,
        U_val: Option<Vec<f64>>,
        C_ao_lo: Option<&Bound<'_, PyAny>>,
        minao_ref: &str,
    ) -> PyResult<PyClassInitializer<Self>> {
        let pu = PuArgs {
            u_idx: U_idx,
            u_val: U_val,
            c_ao_lo: C_ao_lo,
            minao_ref,
        };
        let base =
            PyKohnShamDft::construct(cell, Some(kpts), xc, exxdiv, Kind::KsymRkspu, Some(pu))?;
        Ok(PyClassInitializer::from(base)
            .add_subclass(PyKrks {})
            .add_subclass(PyKsymAdaptedKrks {})
            .add_subclass(PyKsymAdaptedKrkspu {}))
    }
}

/// `KsymAdaptedKUKSpU(cell, kpts, ...)` — `kukspu_ksymm.py:41`.
#[pyclass(
    extends = PyKsymAdaptedKuks,
    subclass,
    name = "KsymAdaptedKUKSpU",
    module = "pyscf._native.pbc.dft"
)]
pub struct PyKsymAdaptedKukspu {}

#[pymethods]
impl PyKsymAdaptedKukspu {
    #[new]
    #[pyo3(signature = (cell, kpts, xc = "LDA,VWN", exxdiv = Some("ewald"), U_idx = None,
                        U_val = None, C_ao_lo = None, minao_ref = "MINAO"))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn new(
        cell: &Bound<'_, PyAny>,
        kpts: &Bound<'_, PyAny>,
        xc: &str,
        exxdiv: Option<&str>,
        U_idx: Option<&Bound<'_, PyAny>>,
        U_val: Option<Vec<f64>>,
        C_ao_lo: Option<&Bound<'_, PyAny>>,
        minao_ref: &str,
    ) -> PyResult<PyClassInitializer<Self>> {
        let pu = PuArgs {
            u_idx: U_idx,
            u_val: U_val,
            c_ao_lo: C_ao_lo,
            minao_ref,
        };
        let base =
            PyKohnShamDft::construct(cell, Some(kpts), xc, exxdiv, Kind::KsymUkspu, Some(pu))?;
        Ok(PyClassInitializer::from(base)
            .add_subclass(PyKuks {})
            .add_subclass(PyKsymAdaptedKuks {})
            .add_subclass(PyKsymAdaptedKukspu {}))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Gamma-point shims (`pyscf-pbc-dft/src/gamma.rs`: a K-driver at one k-point)
// ─────────────────────────────────────────────────────────────────────────────

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

/// `RKS(cell, kpt=np.zeros(3), xc='LDA,VWN', exxdiv='ewald')` — `rks.py:RKS`
/// as `gamma.rs:24`: a `KRKS` at one k-point (results are per-k lists of 1).
#[pyfunction(name = "RKS", signature = (cell, kpt = None, xc = "LDA,VWN", exxdiv = Some("ewald")))]
fn gamma_rks<'py>(
    cell: &Bound<'py, PyAny>,
    kpt: Option<&Bound<'py, PyAny>>,
    xc: &str,
    exxdiv: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = cell.py();
    py.get_type::<PyKrks>()
        .call1((cell, one_kpt(py, kpt)?, xc, exxdiv))
}

/// `UKS(cell, kpt=np.zeros(3), xc='LDA,VWN', exxdiv='ewald')` — `gamma.rs:40`.
#[pyfunction(name = "UKS", signature = (cell, kpt = None, xc = "LDA,VWN", exxdiv = Some("ewald")))]
fn gamma_uks<'py>(
    cell: &Bound<'py, PyAny>,
    kpt: Option<&Bound<'py, PyAny>>,
    xc: &str,
    exxdiv: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = cell.py();
    py.get_type::<PyKuks>()
        .call1((cell, one_kpt(py, kpt)?, xc, exxdiv))
}

/// `ROKS(cell, kpt=np.zeros(3), xc='LDA,VWN', exxdiv='ewald')` — `gamma.rs:56`.
#[pyfunction(name = "ROKS", signature = (cell, kpt = None, xc = "LDA,VWN", exxdiv = Some("ewald")))]
fn gamma_roks<'py>(
    cell: &Bound<'py, PyAny>,
    kpt: Option<&Bound<'py, PyAny>>,
    xc: &str,
    exxdiv: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = cell.py();
    py.get_type::<PyKroks>()
        .call1((cell, one_kpt(py, kpt)?, xc, exxdiv))
}

/// `GKS(cell, kpt=np.zeros(3), xc='LDA,VWN', exxdiv='ewald')` — `gamma.rs:64`.
#[pyfunction(name = "GKS", signature = (cell, kpt = None, xc = "LDA,VWN", exxdiv = Some("ewald")))]
fn gamma_gks<'py>(
    cell: &Bound<'py, PyAny>,
    kpt: Option<&Bound<'py, PyAny>>,
    xc: &str,
    exxdiv: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = cell.py();
    py.get_type::<PyKgks>()
        .call1((cell, one_kpt(py, kpt)?, xc, exxdiv))
}
