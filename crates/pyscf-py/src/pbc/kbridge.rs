//! `KPyOverrideBridge` — the k-point subclass-override contract (plan 20-11).
//!
//! The periodic analogue of `crate::bridge::PyOverrideBridge`. It implements
//! `pyscf_pbc_scf::KOverrideHooks`, so `pyscf_pbc_scf::kernel` can drive a
//! Python object: every one of the eleven hooks in [`K_HOOKS`] that the Python
//! object OVERRIDES is dispatched through `Bound::call_method*` on `slf`, which
//! resolves the Python MRO (D-PBC-14, the Phase-3 BIND-07 contract). A hook it
//! does NOT override runs the Rust default of the wrapped driver directly — the
//! `bridge.rs:100-104` fallback, applied to every hook.
//!
//! # What "overrides" means, and the cached probe
//!
//! A hook is overridden when the object resolved for it on `type(slf)` (a raw
//! `__mro__` / `__dict__` walk, no descriptor invocation) is not the object the
//! native `base` class resolves, or when the instance `__dict__` shadows it
//! (`mf.get_hcore = lambda *a: h1`, the common PySCF idiom). The type half is
//! probed once per `(type, base)` for the life of the interpreter
//! (`caches::k_override_mask`, a `PyOnceLock`); the instance half is re-read
//! once per bridge, i.e. once per `kernel()`. Inside the SCF loop a hook check is
//! a bit test. (`KOverrideHooks` hooks are batched over k — one call per cycle
//! carries every k-point — so an uncached probe would cost one `hasattr` per hook
//! per cycle, not per k-point as 20-11-PLAN states.)
//!
//! # Python payload layout (the hook protocol 20-12's default methods follow)
//!
//! * `nao x nao` k-matrices (`KMats`): a `list` of per-k `complex128` arrays,
//!   ROW-MAJOR (20-07 `kmats_to_pylist`, `BufOrder::C`).
//! * density-shaped channels (`KDms`: dm, vhf, fock): ONE channel is passed as
//!   that per-k list, exactly upstream `KRHF`'s `(nkpts, nao, nao)`; more than
//!   one is a list of per-channel lists (upstream `KUHF`'s `(2, nkpts, …)`,
//!   20-07 `kdms_to_pylist`).
//! * `mo_coeff`: per-k `(nao, nmo)` complex arrays laid out COLUMN-MAJOR
//!   (`BufOrder::F`); `mo_energy` / `mo_occ`: per-k float64 1-D arrays. Same
//!   one-channel-drops-the-axis rule.
//! * Returned values are read tolerantly: a `list`/`tuple` OR a stacked
//!   ndarray, complex or real (real is widened exactly by numpy), any strides.
//!   Shapes, k counts and channel counts are validated; a mismatch raises
//!   `TypeError`/`ValueError`, never truncates.
//!
//! Hook call signatures (positional, upstream-compatible):
//!
//! | hook | Python call | returns |
//! |---|---|---|
//! | `get_ovlp` | `get_ovlp(cell)` | per-k list |
//! | `get_hcore` | `get_hcore(cell)` | per-k list |
//! | `get_init_guess` | `get_init_guess(cell, key, s1e)` | dm (`nset` channels) |
//! | `get_veff` | `get_veff(cell, dm_kpts)` | vhf (`nset`) |
//! | `get_fock` | `get_fock(h1e, None, vhf, dm)` (upstream `cycle=-1`: bare `h1e + vhf`) | fock (`nfock`) |
//! | `eig` | `eig(fock, s1e)` | `(mo_energy, mo_coeff)` (`nfock`) |
//! | `get_occ` | `get_occ(mo_energy)` | `mo_occ` |
//! | `make_rdm1` | `make_rdm1(mo_coeff, mo_occ)` | dm (`nset`) |
//! | `energy_elec` | `energy_elec(dm, h1e, vhf)` | `(e_elec, e_coul)` |
//! | `energy_nuc` | `energy_nuc()` | float |
//! | `get_grad` | `get_grad(mo_coeff, mo_occ, fock)` with `fock = h1e + vhf` | 1-D float array |
//!
//! `get_init_guess` with a user density (`KInitGuess::UserDm`) returns that
//! density without a Python call, as `bridge.rs` does; `Chkfile(path)` passes the
//! key `"chkfile"` (the path is not forwarded). A Python `get_occ` returns
//! occupations only (upstream's signature); the Fermi level reported for that
//! cycle is then the highest occupied orbital energy per channel, which is what
//! the Rust aufbau default computes.
//!
//! # Exceptions
//!
//! A Python exception inside an override becomes a `PyscfRsError` so the kernel
//! `?`-propagates it (the `call_hook` pattern, `bridge.rs:57-72`), AND the
//! original `PyErr` is stashed. [`KPyOverrideBridge::finish`] re-raises that
//! original exception, so `ValueError` stays `ValueError`. `get_grad` is
//! infallible in the trait: a failure there returns `[NaN]` and POISONS the
//! bridge, so the next hook call — overridden or not — fails and the kernel
//! stops instead of running on to `max_cycle`.
//!
//! # GIL
//!
//! Only the dispatch of an overridden hook attaches to the interpreter
//! (`Python::attach`). The bridge is `Sync` whenever the wrapped driver is, so a
//! driver that is `Sync` may run `kernel` under `py.detach`; `Krhf` is not (its
//! smearing entropy is a `Cell`), so its caller keeps the GIL, as `PyRHF`'s
//! subclass path does.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use numpy::{Complex64, PyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple, PyType};
use pyscf_algebra::CTensor;
use pyscf_core::{CoreError, PyscfRsError};
use pyscf_pbc_gto::Cell;
use pyscf_pbc_scf::{KDms, KInitGuess, KMats, KOverrideHooks};

use crate::errors::{py_to_pyscf, pyscf_to_py};
use crate::numpy_io::{BufOrder, ctensor_to_pyarray, kdms_to_pylist, kmats_to_pylist, to_ctensor};

/// The eleven dispatched hooks, in `KOverrideHooks` declaration order. Bit `i`
/// of an override mask is `K_HOOKS[i]`. The trait's accessors (`cell`, `kpts`,
/// `nset`, `nfock`, `nao`) and its two driver-internal seams (`diis_dms`,
/// `free_energy`) have no upstream Python counterpart and always run the Rust
/// driver's.
pub const K_HOOKS: [&str; 11] = [
    "get_ovlp",
    "get_hcore",
    "get_init_guess",
    "get_veff",
    "get_fock",
    "eig",
    "get_occ",
    "make_rdm1",
    "energy_elec",
    "energy_nuc",
    "get_grad",
];

const GET_OVLP: usize = 0;
const GET_HCORE: usize = 1;
const GET_INIT_GUESS: usize = 2;
const GET_VEFF: usize = 3;
const GET_FOCK: usize = 4;
const EIG: usize = 5;
const GET_OCC: usize = 6;
const MAKE_RDM1: usize = 7;
const ENERGY_ELEC: usize = 8;
const ENERGY_NUC: usize = 9;
const GET_GRAD: usize = 10;

/// Bridge from `KOverrideHooks` to a Python driver object.
///
/// Construct inside a driver's `kernel` pymethod, pass `&bridge` to
/// `pyscf_pbc_scf::kernel`, then hand the result to [`Self::finish`].
pub struct KPyOverrideBridge<'a, D: KOverrideHooks + ?Sized> {
    /// Python `self` (the native driver or a Python subclass of it).
    pub slf: Py<PyAny>,
    /// The Python cell object, passed as the `cell` argument of the hooks
    /// that take one (so an override sees the user's object, not a copy).
    pub py_cell: Py<PyAny>,
    inner: &'a D,
    mask: u16,
    poisoned: AtomicBool,
    py_err: Mutex<Option<PyErr>>,
}

impl<'a, D: KOverrideHooks + ?Sized> KPyOverrideBridge<'a, D> {
    /// Wrap `slf` over the Rust driver `inner` (whose hooks are the defaults).
    ///
    /// `base` is the NATIVE driver class whose methods are the defaults (e.g.
    /// `py.get_type::<PyKrhf>()`); a hook counts as overridden when `type(slf)`
    /// or the instance `__dict__` resolves it to something else.
    ///
    /// # Errors
    /// Propagates a failing `__mro__` / `__dict__` read.
    pub fn new(
        py: Python<'_>,
        slf: Py<PyAny>,
        py_cell: Py<PyAny>,
        base: &Bound<'_, PyType>,
        inner: &'a D,
    ) -> PyResult<Self> {
        let bound = slf.bind(py);
        let ty = bound.get_type();
        let mut mask =
            crate::caches::k_override_mask(py, &ty, base, || type_override_mask(&ty, base))?;
        mask |= instance_override_mask(bound)?;
        Ok(Self {
            slf,
            py_cell,
            inner,
            mask,
            poisoned: AtomicBool::new(false),
            py_err: Mutex::new(None),
        })
    }

    /// The wrapped Rust driver.
    pub fn inner(&self) -> &D {
        self.inner
    }

    /// Whether `hook` (a [`K_HOOKS`] name) is dispatched to Python.
    pub fn overrides(&self, hook: &str) -> bool {
        K_HOOKS
            .iter()
            .position(|h| *h == hook)
            .is_some_and(|i| self.is_overridden(i))
    }

    /// The overridden hook names, in [`K_HOOKS`] order.
    pub fn overridden_hooks(&self) -> Vec<&'static str> {
        (0..K_HOOKS.len())
            .filter(|&i| self.is_overridden(i))
            .map(|i| K_HOOKS[i])
            .collect()
    }

    /// Convert the kernel's result, re-raising the ORIGINAL Python exception
    /// if an override raised one (even when the kernel itself returned `Ok`
    /// after an infallible hook failed).
    ///
    /// # Errors
    /// The stashed Python exception, else the kernel's error via `pyscf_to_py`.
    pub fn finish<T>(&self, res: Result<T, PyscfRsError>) -> PyResult<T> {
        if let Some(err) = self.take_py_err() {
            return Err(err);
        }
        res.map_err(pyscf_to_py)
    }

    /// Take the stashed Python exception, if any.
    pub fn take_py_err(&self) -> Option<PyErr> {
        self.py_err.lock().unwrap_or_else(|p| p.into_inner()).take()
    }

    fn is_overridden(&self, i: usize) -> bool {
        self.mask & (1 << i) != 0
    }

    fn guard(&self) -> Result<(), PyscfRsError> {
        if self.poisoned.load(Ordering::Relaxed) {
            return Err(PyscfRsError::Core(CoreError::InvalidMolecule(
                "periodic SCF: aborted after a Python override raised".into(),
            )));
        }
        Ok(())
    }

    /// Stash the first Python error and convert it for `?` propagation.
    fn fail(&self, py: Python<'_>, err: PyErr) -> PyscfRsError {
        let converted = py_to_pyscf(err.clone_ref(py));
        let mut slot = self.py_err.lock().unwrap_or_else(|p| p.into_inner());
        if slot.is_none() {
            *slot = Some(err);
        }
        self.poisoned.store(true, Ordering::Relaxed);
        converted
    }

    fn nkpts(&self) -> usize {
        self.inner.kpts().len()
    }

    fn py_or_fail<T>(
        &self,
        f: impl for<'py> FnOnce(Python<'py>) -> PyResult<T>,
    ) -> Result<T, PyscfRsError> {
        Python::attach(|py| f(py).map_err(|e| self.fail(py, e)))
    }
}

impl<D: KOverrideHooks + ?Sized> KOverrideHooks for KPyOverrideBridge<'_, D> {
    fn cell(&self) -> &Cell {
        self.inner.cell()
    }
    fn kpts(&self) -> &[[f64; 3]] {
        self.inner.kpts()
    }
    fn nset(&self) -> usize {
        self.inner.nset()
    }
    fn nfock(&self) -> usize {
        self.inner.nfock()
    }
    fn nao(&self) -> usize {
        self.inner.nao()
    }
    fn diis_dms(&self, dms: &KDms) -> KDms {
        self.inner.diis_dms(dms)
    }
    fn free_energy(&self) -> Option<f64> {
        self.inner.free_energy()
    }

    fn get_ovlp(&self) -> Result<KMats, PyscfRsError> {
        self.guard()?;
        if !self.is_overridden(GET_OVLP) {
            return self.inner.get_ovlp();
        }
        let (nk, nao) = (self.nkpts(), self.inner.nao());
        self.py_or_fail(|py| {
            let out = self
                .slf
                .bind(py)
                .call_method1("get_ovlp", (self.py_cell.bind(py),))?;
            kmats_from_py(&out, nk, nao, "get_ovlp")
        })
    }

    fn get_hcore(&self) -> Result<KMats, PyscfRsError> {
        self.guard()?;
        if !self.is_overridden(GET_HCORE) {
            return self.inner.get_hcore();
        }
        let (nk, nao) = (self.nkpts(), self.inner.nao());
        self.py_or_fail(|py| {
            let out = self
                .slf
                .bind(py)
                .call_method1("get_hcore", (self.py_cell.bind(py),))?;
            kmats_from_py(&out, nk, nao, "get_hcore")
        })
    }

    fn get_init_guess(&self, mode: &KInitGuess, s1e: &KMats) -> Result<KDms, PyscfRsError> {
        self.guard()?;
        if !self.is_overridden(GET_INIT_GUESS) {
            return self.inner.get_init_guess(mode, s1e);
        }
        let key = match mode {
            KInitGuess::Minao => "minao",
            KInitGuess::Atom => "atom",
            KInitGuess::OneElectron => "1e",
            KInitGuess::Chkfile(_) => "chkfile",
            // A user density is already a density — no Python round trip.
            KInitGuess::UserDm(d) => return Ok(d.clone()),
        };
        let (nk, nao, nset) = (self.nkpts(), self.inner.nao(), self.inner.nset());
        self.py_or_fail(|py| {
            let s1e_py = kmats_to_py(py, s1e, nao)?;
            let out = self
                .slf
                .bind(py)
                .call_method1("get_init_guess", (self.py_cell.bind(py), key, s1e_py))?;
            kdms_from_py(&out, nset, nk, nao, "get_init_guess")
        })
    }

    fn get_veff(&self, dms: &KDms) -> Result<KDms, PyscfRsError> {
        self.guard()?;
        if !self.is_overridden(GET_VEFF) {
            return self.inner.get_veff(dms);
        }
        let (nk, nao, nset) = (self.nkpts(), self.inner.nao(), self.inner.nset());
        self.py_or_fail(|py| {
            let dm_py = kdms_to_py(py, dms, nao)?;
            let out = self
                .slf
                .bind(py)
                .call_method1("get_veff", (self.py_cell.bind(py), dm_py))?;
            kdms_from_py(&out, nset, nk, nao, "get_veff")
        })
    }

    fn get_fock(&self, h1e: &KMats, vhf: &KDms, dms: &KDms) -> Result<KDms, PyscfRsError> {
        self.guard()?;
        if !self.is_overridden(GET_FOCK) {
            return self.inner.get_fock(h1e, vhf, dms);
        }
        let (nk, nao, nfock) = (self.nkpts(), self.inner.nao(), self.inner.nfock());
        self.py_or_fail(|py| {
            let args = (
                kmats_to_py(py, h1e, nao)?,
                py.None(),
                kdms_to_py(py, vhf, nao)?,
                kdms_to_py(py, dms, nao)?,
            );
            let out = self.slf.bind(py).call_method1("get_fock", args)?;
            kdms_from_py(&out, nfock, nk, nao, "get_fock")
        })
    }

    fn eig(&self, fock: &KDms, s1e: &KMats) -> Result<(Vec<Vec<f64>>, Vec<CTensor>), PyscfRsError> {
        self.guard()?;
        if !self.is_overridden(EIG) {
            return self.inner.eig(fock, s1e);
        }
        let (nk, nao, nfock) = (self.nkpts(), self.inner.nao(), self.inner.nfock());
        self.py_or_fail(|py| {
            let args = (kdms_to_py(py, fock, nao)?, kmats_to_py(py, s1e, nao)?);
            let out = self.slf.bind(py).call_method1("eig", args)?;
            let (e_any, c_any): (Bound<'_, PyAny>, Bound<'_, PyAny>) =
                out.extract().map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err(
                        "eig override must return a (mo_energy, mo_coeff) pair",
                    )
                })?;
            let mo_energy = mo_values_from_py(&e_any, nfock, nk, "eig: mo_energy")?;
            let mo_coeff = mo_coeff_from_py(&c_any, nfock, nk, nao, "eig: mo_coeff")?;
            check_nmo(&mo_energy, &mo_coeff, nao, "eig")?;
            Ok((mo_energy, mo_coeff))
        })
    }

    fn get_occ(&self, mo_energy: &[Vec<f64>]) -> Result<(Vec<Vec<f64>>, Vec<f64>), PyscfRsError> {
        self.guard()?;
        if !self.is_overridden(GET_OCC) {
            return self.inner.get_occ(mo_energy);
        }
        let nk = self.nkpts();
        let nch = channels_of(mo_energy.len(), nk);
        self.py_or_fail(|py| {
            let e_py = mo_values_to_py(py, mo_energy, nch)?;
            let out = self.slf.bind(py).call_method1("get_occ", (e_py,))?;
            let occ = mo_values_from_py(&out, nch, nk, "get_occ")?;
            for (i, (o, e)) in occ.iter().zip(mo_energy).enumerate() {
                if o.len() != e.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "get_occ: block {i} has {} occupations for {} orbitals",
                        o.len(),
                        e.len()
                    )));
                }
            }
            let fermi = (0..nch)
                .map(|s| {
                    (0..nk)
                        .flat_map(|k| {
                            let i = s * nk + k;
                            occ[i]
                                .iter()
                                .zip(&mo_energy[i])
                                .filter(|(o, _)| **o > 0.0)
                                .map(|(_, e)| *e)
                        })
                        .fold(f64::NEG_INFINITY, f64::max)
                })
                .collect();
            Ok((occ, fermi))
        })
    }

    fn make_rdm1(&self, mo_coeff: &[CTensor], mo_occ: &[Vec<f64>]) -> Result<KDms, PyscfRsError> {
        self.guard()?;
        if !self.is_overridden(MAKE_RDM1) {
            return self.inner.make_rdm1(mo_coeff, mo_occ);
        }
        let (nk, nao, nset) = (self.nkpts(), self.inner.nao(), self.inner.nset());
        let nch = channels_of(mo_coeff.len(), nk);
        self.py_or_fail(|py| {
            let args = (
                mo_coeff_to_py(py, mo_coeff, nch, nao)?,
                mo_values_to_py(py, mo_occ, nch)?,
            );
            let out = self.slf.bind(py).call_method1("make_rdm1", args)?;
            kdms_from_py(&out, nset, nk, nao, "make_rdm1")
        })
    }

    fn energy_elec(&self, dms: &KDms, h1e: &KMats, vhf: &KDms) -> Result<(f64, f64), PyscfRsError> {
        self.guard()?;
        if !self.is_overridden(ENERGY_ELEC) {
            return self.inner.energy_elec(dms, h1e, vhf);
        }
        let nao = self.inner.nao();
        self.py_or_fail(|py| {
            let args = (
                kdms_to_py(py, dms, nao)?,
                kmats_to_py(py, h1e, nao)?,
                kdms_to_py(py, vhf, nao)?,
            );
            let out = self.slf.bind(py).call_method1("energy_elec", args)?;
            out.extract::<(f64, f64)>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "energy_elec override must return an (e_elec, e_coul) pair of floats",
                )
            })
        })
    }

    fn energy_nuc(&self) -> Result<f64, PyscfRsError> {
        self.guard()?;
        if !self.is_overridden(ENERGY_NUC) {
            return self.inner.energy_nuc();
        }
        self.py_or_fail(|py| {
            let out = self.slf.bind(py).call_method1("energy_nuc", ())?;
            out.extract::<f64>()
        })
    }

    fn get_grad(
        &self,
        mo_coeff: &[CTensor],
        mo_occ: &[Vec<f64>],
        h1e: &KMats,
        vhf: &KDms,
    ) -> Vec<f64> {
        if self.guard().is_err() {
            return vec![f64::NAN];
        }
        if !self.is_overridden(GET_GRAD) {
            return self.inner.get_grad(mo_coeff, mo_occ, h1e, vhf);
        }
        let (nk, nao) = (self.nkpts(), self.inner.nao());
        let nch = channels_of(mo_coeff.len(), nk);
        let fock = pyscf_pbc_scf::kscf::bare_fock(h1e, vhf);
        self.py_or_fail(|py| {
            let args = (
                mo_coeff_to_py(py, mo_coeff, nch, nao)?,
                mo_values_to_py(py, mo_occ, nch)?,
                kdms_to_py(py, &fock, nao)?,
            );
            let out = self.slf.bind(py).call_method1("get_grad", args)?;
            float_vec(&out, "get_grad")
        })
        .unwrap_or_else(|_| vec![f64::NAN])
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The override probe
// ─────────────────────────────────────────────────────────────────────────────

/// Bitmask of the [`K_HOOKS`] that `ty` resolves differently from `base`.
fn type_override_mask(ty: &Bound<'_, PyType>, base: &Bound<'_, PyType>) -> PyResult<u16> {
    let mut mask = 0u16;
    for (i, name) in K_HOOKS.iter().enumerate() {
        let overridden = match (mro_lookup(ty, name)?, mro_lookup(base, name)?) {
            (None, _) => false,
            (Some(_), None) => true,
            (Some(a), Some(b)) => !a.is(&b),
        };
        if overridden {
            mask |= 1 << i;
        }
    }
    Ok(mask)
}

/// The raw class-dict entry `name` resolves to along `ty.__mro__` — what
/// `inspect.getattr_static` finds, without invoking any descriptor (so a
/// method descriptor compares by identity).
fn mro_lookup<'py>(ty: &Bound<'py, PyType>, name: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    for cls in ty.getattr("__mro__")?.try_iter()? {
        let dict = cls?.getattr("__dict__")?;
        if dict.contains(name)? {
            return Ok(Some(dict.get_item(name)?));
        }
    }
    Ok(None)
}

/// Hooks shadowed in the instance `__dict__` (not cached: per object).
fn instance_override_mask(obj: &Bound<'_, PyAny>) -> PyResult<u16> {
    let Ok(dict) = obj.getattr("__dict__") else {
        return Ok(0);
    };
    let Ok(dict) = dict.cast::<PyDict>() else {
        return Ok(0);
    };
    let mut mask = 0u16;
    for (i, name) in K_HOOKS.iter().enumerate() {
        if dict.contains(name)? {
            mask |= 1 << i;
        }
    }
    Ok(mask)
}

// ─────────────────────────────────────────────────────────────────────────────
// Payload codec — public so 20-12's default hook methods use the SAME layout
// ─────────────────────────────────────────────────────────────────────────────

/// One channel of `nao x nao` k-matrices → a per-k `list` of row-major
/// `complex128` arrays (20-07 `kmats_to_pylist`, `BufOrder::C`).
///
/// # Errors
/// `ValueError` if a block is not `nao * nao` long.
pub fn kmats_to_py<'py>(
    py: Python<'py>,
    mats: &[CTensor],
    nao: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let shapes = vec![vec![nao, nao]; mats.len()];
    Ok(kmats_to_pylist(py, mats, &shapes, BufOrder::C)?.into_any())
}

/// Density-shaped channels → the per-k list for ONE channel, else a list of
/// per-channel lists (20-07 `kdms_to_pylist`).
///
/// # Errors
/// `ValueError` if a block is not `nao * nao` long.
pub fn kdms_to_py<'py>(py: Python<'py>, dms: &[KMats], nao: usize) -> PyResult<Bound<'py, PyAny>> {
    if dms.len() == 1 {
        return kmats_to_py(py, &dms[0], nao);
    }
    let shapes: Vec<Vec<Vec<usize>>> = dms.iter().map(|s| vec![vec![nao, nao]; s.len()]).collect();
    Ok(kdms_to_pylist(py, dms, &shapes, BufOrder::C)?.into_any())
}

/// Flat `mo_coeff[set * nkpts + k]` (column-major `nao x nmo`) → per-k list of
/// `(nao, nmo)` arrays, nested per channel when `nch > 1`.
///
/// # Errors
/// `ValueError` if a block is not a multiple of `nao` long or `nch` does not
/// divide the block count.
pub fn mo_coeff_to_py<'py>(
    py: Python<'py>,
    mo_coeff: &[CTensor],
    nch: usize,
    nao: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let blocks = split_flat(mo_coeff, nch, "mo_coeff")?;
    let channel = |set: &[CTensor]| -> PyResult<Bound<'py, PyList>> {
        let arrays = set
            .iter()
            .map(|c| {
                if nao == 0 || c.re.len() % nao != 0 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "mo_coeff block of {} elements is not (nao={nao}, nmo)",
                        c.re.len()
                    )));
                }
                ctensor_to_pyarray(py, c, &[nao, c.re.len() / nao], BufOrder::F)
            })
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, arrays)
    };
    nest(py, blocks, channel)
}

/// Flat `values[set * nkpts + k]` (mo_energy / mo_occ) → per-k list of
/// float64 1-D arrays, nested per channel when `nch > 1`.
///
/// # Errors
/// `ValueError` if `nch` does not divide the block count.
pub fn mo_values_to_py<'py>(
    py: Python<'py>,
    values: &[Vec<f64>],
    nch: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let blocks = split_flat(values, nch, "mo values")?;
    let channel = |set: &[Vec<f64>]| -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, set.iter().map(|v| PyArray1::from_slice(py, v)))
    };
    nest(py, blocks, channel)
}

/// Read `nk` `nao x nao` matrices (list/tuple or stacked `(nk, nao, nao)`
/// ndarray, complex or real, any strides) into row-major planes.
///
/// # Errors
/// `TypeError` / `ValueError` naming `what` on a count, shape or dtype mismatch.
pub fn kmats_from_py(obj: &Bound<'_, PyAny>, nk: usize, nao: usize, what: &str) -> PyResult<KMats> {
    let mut chans = kdms_from_py(obj, 1, nk, nao, what)?;
    Ok(chans.remove(0))
}

/// Read `nch` channels of `nk` `nao x nao` matrices. One channel may be given
/// with or without its channel axis.
///
/// # Errors
/// As [`kmats_from_py`].
pub fn kdms_from_py(
    obj: &Bound<'_, PyAny>,
    nch: usize,
    nk: usize,
    nao: usize,
    what: &str,
) -> PyResult<KDms> {
    let np = obj.py().import("numpy")?;
    split_channels(obj, nch, nk, 2, what)?
        .iter()
        .enumerate()
        .map(|(s, set)| {
            set.iter()
                .enumerate()
                .map(|(k, b)| {
                    let (t, shape) = complex_block(&np, b, BufOrder::C)?;
                    if shape != [nao, nao] {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "{what}: block [{s}][{k}] has shape {shape:?}, expected ({nao}, {nao})"
                        )));
                    }
                    Ok(t)
                })
                .collect()
        })
        .collect()
}

/// Read `nch` channels of `nk` `(nao, nmo)` MO coefficient arrays into flat,
/// column-major `mo_coeff[set * nk + k]`.
///
/// # Errors
/// As [`kmats_from_py`].
pub fn mo_coeff_from_py(
    obj: &Bound<'_, PyAny>,
    nch: usize,
    nk: usize,
    nao: usize,
    what: &str,
) -> PyResult<Vec<CTensor>> {
    let np = obj.py().import("numpy")?;
    let mut out = Vec::with_capacity(nch * nk);
    for (s, set) in split_channels(obj, nch, nk, 2, what)?.iter().enumerate() {
        for (k, b) in set.iter().enumerate() {
            let (t, shape) = complex_block(&np, b, BufOrder::F)?;
            if shape.len() != 2 || shape[0] != nao {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "{what}: block [{s}][{k}] has shape {shape:?}, expected ({nao}, nmo)"
                )));
            }
            out.push(t);
        }
    }
    Ok(out)
}

/// Read `nch` channels of `nk` float64 1-D arrays into flat
/// `values[set * nk + k]`.
///
/// # Errors
/// As [`kmats_from_py`].
pub fn mo_values_from_py(
    obj: &Bound<'_, PyAny>,
    nch: usize,
    nk: usize,
    what: &str,
) -> PyResult<Vec<Vec<f64>>> {
    let mut out = Vec::with_capacity(nch * nk);
    for (s, set) in split_channels(obj, nch, nk, 1, what)?.iter().enumerate() {
        for (k, b) in set.iter().enumerate() {
            out.push(float_vec(b, &format!("{what}[{s}][{k}]"))?);
        }
    }
    Ok(out)
}

// ── codec internals ─────────────────────────────────────────────────────────

fn channels_of(blocks: usize, nk: usize) -> usize {
    if nk == 0 { 1 } else { (blocks / nk).max(1) }
}

fn check_nmo(e: &[Vec<f64>], c: &[CTensor], nao: usize, what: &str) -> PyResult<()> {
    for (i, (e, c)) in e.iter().zip(c).enumerate() {
        if e.len() * nao != c.re.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{what}: block {i} has {} energies but {} MO columns",
                e.len(),
                c.re.len() / nao.max(1)
            )));
        }
    }
    Ok(())
}

fn split_flat<'s, T>(flat: &'s [T], nch: usize, what: &str) -> PyResult<Vec<&'s [T]>> {
    if nch == 0 || flat.len() % nch != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{what}: {} blocks do not split into {nch} channels",
            flat.len()
        )));
    }
    let per = flat.len() / nch;
    Ok((0..nch).map(|s| &flat[s * per..(s + 1) * per]).collect())
}

fn nest<'py, 's, T>(
    py: Python<'py>,
    blocks: Vec<&'s [T]>,
    channel: impl Fn(&'s [T]) -> PyResult<Bound<'py, PyList>>,
) -> PyResult<Bound<'py, PyAny>> {
    if blocks.len() == 1 {
        return Ok(channel(blocks[0])?.into_any());
    }
    let sets = blocks
        .into_iter()
        .map(channel)
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyList::new(py, sets)?.into_any())
}

/// Items of a list/tuple, or the leading-axis slices of an array-like.
fn items<'py>(obj: &Bound<'py, PyAny>, what: &str) -> PyResult<Vec<Bound<'py, PyAny>>> {
    if let Ok(l) = obj.cast::<PyList>() {
        return Ok(l.iter().collect());
    }
    if let Ok(t) = obj.cast::<PyTuple>() {
        return Ok(t.iter().collect());
    }
    if obj.hasattr("ndim")? && obj.getattr("ndim")?.extract::<usize>()? >= 1 {
        return (0..obj.len()?).map(|i| obj.get_item(i)).collect();
    }
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "{what}: expected a list/tuple or an ndarray of per-k blocks, got {}",
        obj.get_type()
            .name()
            .map(|n| n.to_string())
            .unwrap_or_else(|_| "<unknown>".into())
    )))
}

/// `numpy.ndim` of a block; `None` for a ragged or non-array object.
fn ndim_of(obj: &Bound<'_, PyAny>) -> Option<usize> {
    if let Ok(arr) = obj.cast::<numpy::PyUntypedArray>() {
        return Some(arr.ndim());
    }
    if obj.cast::<PyList>().is_ok() || obj.cast::<PyTuple>().is_ok() {
        let np = obj.py().import("numpy").ok()?;
        return np.getattr("ndim").ok()?.call1((obj,)).ok()?.extract().ok();
    }
    None
}

/// Split a returned payload into `nch` channels of `nk` blocks of
/// `block_ndim` dimensions. One channel may omit its channel axis.
fn split_channels<'py>(
    obj: &Bound<'py, PyAny>,
    nch: usize,
    nk: usize,
    block_ndim: usize,
    what: &str,
) -> PyResult<Vec<Vec<Bound<'py, PyAny>>>> {
    let top = items(obj, what)?;
    let is_blocks = |bs: &[Bound<'py, PyAny>]| bs.iter().all(|b| ndim_of(b) == Some(block_ndim));
    if nch == 1 && top.len() == nk && is_blocks(&top) {
        return Ok(vec![top]);
    }
    if top.len() != nch {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{what}: expected {nch} channel(s) of {nk} k-point block(s) of {block_ndim}-D arrays, \
             got a sequence of {} item(s)",
            top.len()
        )));
    }
    top.iter()
        .enumerate()
        .map(|(s, ch)| {
            let ks = items(ch, what)?;
            if ks.len() != nk || !is_blocks(&ks) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "{what}: channel {s} is not {nk} k-point block(s) of {block_ndim}-D arrays"
                )));
            }
            Ok(ks)
        })
        .collect()
}

/// One block → planar `CTensor` in `order` plus its shape. A `complex128`
/// array is read directly (20-07 `to_ctensor`); anything else goes through
/// `numpy.asarray(b, dtype=complex128)` first (exact for real input).
fn complex_block(
    np: &Bound<'_, PyModule>,
    b: &Bound<'_, PyAny>,
    order: BufOrder,
) -> PyResult<(CTensor, Vec<usize>)> {
    if let Ok(arr) = b.extract::<PyReadonlyArrayDyn<'_, Complex64>>() {
        return to_ctensor(arr, order);
    }
    let kw = PyDict::new(b.py());
    kw.set_item("dtype", np.getattr("complex128")?)?;
    let widened = np.getattr("asarray")?.call((b,), Some(&kw))?;
    to_ctensor(
        widened.extract::<PyReadonlyArrayDyn<'_, Complex64>>()?,
        order,
    )
}

/// A 1-D float64 vector from any array-like (`numpy.asarray(.., float64)`,
/// raveled only if it is already 1-D).
fn float_vec(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<f64>> {
    let np = obj.py().import("numpy")?;
    let kw = PyDict::new(obj.py());
    kw.set_item("dtype", np.getattr("float64")?)?;
    let arr = np.getattr("asarray")?.call((obj,), Some(&kw))?;
    let ro = arr.extract::<PyReadonlyArrayDyn<'_, f64>>()?;
    if ro.ndim() != 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{what}: expected a 1-D float array, got shape {:?}",
            ro.shape()
        )));
    }
    Ok(ro.as_array().iter().copied().collect())
}
