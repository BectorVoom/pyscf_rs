//! `pyscf._native.pbc.cc` — periodic coupled cluster (plan 20-15).
//!
//! # Classes
//!
//! | Python | Rust | notes |
//! |---|---|---|
//! | `_KCCSD` | — | native base: options, results, `kernel`, `ccsd_t`, `ipccsd`/`eaccsd` |
//! | `KRCCSD(mf)` (`KCCSD`) | `Krccsd` (`kccsd_rhf.rs:774`) | full-BZ `KRHF` |
//! | `KUCCSD(mf)` | `Kuccsd` (`kccsd_uhf.rs:1735`) | `KUHF` |
//! | `KGCCSD(mf)` | `Kgccsd` (`kccsd.rs:1052`) | `KGHF` |
//! | `KsymAdaptedRCCSD(mf)` | `run_ksym_rccsd` (`kccsd_rhf_ksymm.rs:1176`) | `KRHF(cell, kpts=KPoints)` |
//! | `EOMIP(cc)` / `EOMEA(cc)` | `eom_kernel` (`eom_kccsd_{rhf,uhf,ghf}.rs`) | `Excitation::Ip/Ea` |
//! | `EOMEESinglet(cc)` | `eom_kccsd_rhf::eom_kernel(Ee)` | KRCCSD only |
//! | `EOMEE(cc)` | `eom_kccsd_ghf::eom_kernel(Ee)` | KGCCSD only |
//!
//! Every driver takes an ALREADY-CONVERGED `pyscf.pbc.scf` driver (20-12) and
//! its `with_df` builder; nothing re-runs SCF. `kernel()` builds the MO
//! integrals once and KEEPS them with the amplitudes (upstream's `cc.eris`),
//! so `ccsd_t()` and the EOM solvers reuse them.
//!
//! # Route discipline
//!
//! The DF route is the mean field's `with_df` class (`kccsd_rhf.py:824-832`).
//! 16 measurements §1 gates `e_corr` at `1e-7` PER ROUTE because upstream's
//! FFTDF/MDF and GDF/RSDF pairs are `9.22e-4 Ha` apart on diamond.
//!
//! # EOM `partition`
//!
//! `'mp'` and `'full'` RAISE `NotImplementedError` with the Rust refusal text
//! (`eom_kccsd_ghf.rs:2385` / `eom_kccsd_uhf.rs:2690`), which is upstream's own
//! behaviour at `eom_kccsd_ghf.py:618`. The check is the solver's first
//! statement, as upstream's is.
//!
//! # Options
//!
//! Defaults are the Rust `KrccsdOpts::default()` — `conv_tol = 1e-9`,
//! `conv_tol_normt = 1e-7` — the setting every Phase-16 gate was measured at
//! (16 measurements §3), NOT upstream's `1e-7`/`1e-5`.

use std::sync::Arc;

use numpy::PyArray1;
use pyo3::exceptions::{PyNotImplementedError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyscf_pbc_cc::eom_kccsd_ghf::{self as eomg, EomOpts, EomRoots, Excitation, Partition};
use pyscf_pbc_cc::eom_kccsd_rhf as eomr;
use pyscf_pbc_cc::eom_kccsd_uhf as eomu;
use pyscf_pbc_cc::kccsd::{KgEris, Kgccsd};
use pyscf_pbc_cc::kccsd_rhf::{Krccsd, KrccsdOpts};
use pyscf_pbc_cc::kccsd_uhf::Kuccsd;
use pyscf_pbc_cc::keris::{KEris, KErisOpts};
use pyscf_pbc_cc::kintermediates_uhf::{UT1, UT2};
use pyscf_pbc_cc::kueris::KuEris;
use pyscf_pbc_cc::{KsymRccsdInputs, KsymRccsdOpts, PbcCcError, ZArr, run_ksym_rccsd};
use pyscf_pbc_df::PeriodicDf;
use pyscf_pbc_lib::KptsHelper;
use pyscf_pbc_mp::PaddedMos;
use pyscf_pbc_scf::Kghf;
use pyscf_runtime::ZWorkspacePool;

use crate::errors::{PyscfRsRuntimeError, pyscf_to_py};
use crate::pbc::df::extract_df;
use crate::pbc::mp::{
    complex_from_py, complex_to_py, frozen_k_from_py, mean_field, pbc_mp_to_py, reference_scf,
};

/// Register the CC drivers and the EOM solvers on `pyscf._native.pbc.cc`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKccsd>()?;
    m.add_class::<PyKrccsd>()?;
    m.add_class::<PyKuccsd>()?;
    m.add_class::<PyKgccsd>()?;
    m.add_class::<PyKsymAdaptedRccsd>()?;
    m.add_class::<PyEom>()?;
    m.add_class::<PyEomIp>()?;
    m.add_class::<PyEomEa>()?;
    m.add_class::<PyEomEeSinglet>()?;
    m.add_class::<PyEomEe>()?;
    m.add("KCCSD", m.getattr("KRCCSD")?)?;
    m.add_function(wrap_pyfunction!(_rust_reference_krccsd, m)?)?;
    Ok(())
}

/// `PbcCcError` → Python. Upstream refusals become `NotImplementedError`.
pub(crate) fn pbc_cc_to_py(err: PbcCcError) -> PyErr {
    match err {
        PbcCcError::Core(e) => pyscf_to_py(e),
        e @ PbcCcError::NotImplementedUpstream { .. } => {
            PyNotImplementedError::new_err(e.to_string())
        }
        other => PyscfRsRuntimeError::new_err((other.to_string(), "PbcCc", Vec::<String>::new())),
    }
}

fn zarr_to_py<'py>(py: Python<'py>, z: &ZArr) -> PyResult<Bound<'py, PyAny>> {
    Ok(complex_to_py(py, z.data(), z.shape())?.into_any())
}

fn zarr_from_py(obj: &Bound<'_, PyAny>, want: &[usize], what: &str) -> PyResult<ZArr> {
    let (ct, shape) = complex_from_py(obj)?;
    if shape != want {
        return Err(PyValueError::new_err(format!(
            "{what} must have shape {want:?}, got {shape:?}"
        )));
    }
    ZArr::from_ctensor(&shape, ct).map_err(pbc_cc_to_py)
}

// ─────────────────────────────────────────────────────────────────────────────
// State kept after `kernel()`
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    R,
    U,
    G,
    KsymR,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::R => "KRCCSD",
            Kind::U => "KUCCSD",
            Kind::G => "KGCCSD",
            Kind::KsymR => "KsymAdaptedRCCSD",
        }
    }
    fn scf_kind(self) -> &'static str {
        match self {
            Kind::R => "KRHF",
            Kind::U => "KUHF",
            Kind::G => "KGHF",
            Kind::KsymR => "KsymAdaptedKRHF",
        }
    }
}

/// The lattice data (T) needs.
#[derive(Debug, Clone)]
struct Lattice {
    a: [[f64; 3]; 3],
    kpts: Vec<[f64; 3]>,
}

/// Long-lived (one per solved driver); the variant size spread is irrelevant.
#[allow(clippy::large_enum_variant)]
enum Amps {
    R {
        eris: KEris,
        padded: PaddedMos,
        khelper: KptsHelper,
        t1: ZArr,
        t2: ZArr,
        lat: Lattice,
    },
    U {
        eris: KuEris,
        padded: (PaddedMos, PaddedMos),
        khelper: KptsHelper,
        t1: UT1,
        t2: UT2,
        max_memory: f64,
    },
    G {
        eris: KgEris,
        padded: PaddedMos,
        khelper: KptsHelper,
        t1: ZArr,
        t2: ZArr,
        lat: Lattice,
    },
    /// Full-BZ unfolded amplitudes of the k-symmetric run.
    KsymR { t1: ZArr, t2: ZArr, n_ao2mo: usize },
}

struct Solved {
    e_hf: f64,
    e_corr: f64,
    emp2: f64,
    converged: bool,
    cycles: usize,
    amps: Amps,
}

// ─────────────────────────────────────────────────────────────────────────────
// _KCCSD — the native base
// ─────────────────────────────────────────────────────────────────────────────

/// `_KCCSD` — native base of `KRCCSD`, `KUCCSD`, `KGCCSD`, `KsymAdaptedRCCSD`.
///
/// Options (upstream attribute idiom): `conv_tol`, `conv_tol_normt`,
/// `max_cycle`, `level_shift`, `diis` (bool), `diis_space`,
/// `diis_start_cycle`, `iterative_damping`, `max_memory` (MB), `max_space`
/// (EOM), `keep_exxdiv`. Results: `e_hf`, `e_corr`, `emp2`, `e_tot`,
/// `converged`, `cycles`, `t1`, `t2` (tuples per spin for KUCCSD).
#[pyclass(
    subclass,
    dict,
    name = "_KCCSD",
    module = "pyscf._native.pbc.cc",
    skip_from_py_object
)]
pub struct PyKccsd {
    kind: Kind,
    mf: Py<PyAny>,
    opts: KrccsdOpts,
    keep_exxdiv: bool,
    max_space: usize,
    solved: Option<Solved>,
}

impl PyKccsd {
    fn construct(
        py: Python<'_>,
        mf: &Bound<'_, PyAny>,
        frozen: Option<&Bound<'_, PyAny>>,
        mo_coeff: Option<&Bound<'_, PyAny>>,
        mo_occ: Option<&Bound<'_, PyAny>>,
        kind: Kind,
    ) -> PyResult<Self> {
        if mo_coeff.is_some_and(|x| !x.is_none()) || mo_occ.is_some_and(|x| !x.is_none()) {
            return Err(PyNotImplementedError::new_err(format!(
                "{}(mf, mo_coeff=..., mo_occ=...): only the mean field's own orbitals are bound",
                kind.name()
            )));
        }
        if frozen_k_from_py(frozen)? != pyscf_pbc_mp::FrozenK::default() {
            return Err(PyNotImplementedError::new_err(format!(
                "{}(frozen=...) is not implemented in pyscf-rs: the Rust drivers pad with \
                 FrozenK::default() (kccsd_rhf.rs Krccsd::new, kccsd_uhf.rs Kuccsd::new, \
                 kccsd.rs Kgccsd::new, kccsd_rhf_ksymm.rs KsymRccsdInputs::build)",
                kind.name()
            )));
        }
        let input = mean_field(py, mf)?;
        if input.kind != kind.scf_kind() {
            let hint = match (kind, input.kind) {
                (Kind::R, "KsymAdaptedKRHF") => {
                    "; a k-symmetric mean field needs KsymAdaptedRCCSD (pyscf.pbc.cc)"
                }
                _ => "",
            };
            return Err(PyTypeError::new_err(format!(
                "{} needs a converged {} mean field, got {}{hint}",
                kind.name(),
                kind.scf_kind(),
                input.kind
            )));
        }
        Ok(Self {
            kind,
            mf: mf.clone().unbind(),
            opts: KrccsdOpts::default(),
            keep_exxdiv: false,
            max_space: 20,
            solved: None,
        })
    }

    fn solved(&self) -> PyResult<&Solved> {
        self.solved.as_ref().ok_or_else(|| {
            PyValueError::new_err(format!(
                "{}: no amplitudes yet — call kernel() first",
                self.kind.name()
            ))
        })
    }

    fn solve(&mut self, py: Python<'_>, mbpt2: bool) -> PyResult<()> {
        let input = mean_field(py, self.mf.bind(py))?;
        if !input.result.converged {
            return Err(pbc_cc_to_py(PbcCcError::NotConverged {
                what: "the reference k-point SCF",
                detail: format!("{} refuses an unconverged mean field", self.kind.name()),
            }));
        }
        let df = extract_df(input.with_df.bind(py))?;
        let (result, kpoints, exxdiv) = (input.result, input.kpoints, input.exxdiv);
        let opts = self.opts;
        let eris_opts = KErisOpts {
            keep_exxdiv: self.keep_exxdiv,
            exxdiv,
            max_memory: opts.max_memory,
            ..KErisOpts::default()
        };
        let kind = self.kind;
        let solved = py.detach(move || -> Result<Solved, PyErrLite> {
            let e_hf = result.e_tot;
            match kind {
                Kind::R => {
                    let dfr: &dyn PeriodicDf = df.as_ref();
                    let mut cc = Krccsd::new(&result, dfr)?;
                    cc.opts = opts;
                    cc.eris_opts = eris_opts;
                    let eris = cc.ao2mo()?;
                    let (e_corr, emp2, converged, cycles, t1, t2) = if mbpt2 {
                        let (emp2, t1, t2) = pyscf_pbc_cc::kccsd_rhf::init_amps(
                            &eris,
                            &cc.padded,
                            &cc.khelper.kconserv,
                        )?;
                        (emp2, emp2, false, 0, t1, t2)
                    } else {
                        let r = cc.kernel_with(&eris)?;
                        (r.e_corr, r.emp2, r.converged, r.cycles, r.t1, r.t2)
                    };
                    let lat = Lattice {
                        a: dfr.cell().a,
                        kpts: dfr.kpts().to_vec(),
                    };
                    Ok(Solved {
                        e_hf,
                        e_corr,
                        emp2,
                        converged,
                        cycles,
                        amps: Amps::R {
                            eris,
                            padded: cc.padded,
                            khelper: cc.khelper,
                            t1,
                            t2,
                            lat,
                        },
                    })
                }
                Kind::U => {
                    if mbpt2 {
                        return Err(PyErrLite::NotImpl("KUCCSD.kernel(mbpt2=True) is not bound"));
                    }
                    let dfr: &dyn PeriodicDf = df.as_ref();
                    let mut cc = Kuccsd::new(&result, dfr)?;
                    cc.opts = opts;
                    cc.eris_opts = eris_opts;
                    let eris = cc.ao2mo()?;
                    let r = cc.kernel_with(&eris)?;
                    Ok(Solved {
                        e_hf,
                        e_corr: r.e_corr,
                        emp2: r.emp2,
                        converged: r.converged,
                        cycles: r.cycles,
                        amps: Amps::U {
                            eris,
                            padded: cc.padded,
                            khelper: cc.khelper,
                            t1: r.t1,
                            t2: r.t2,
                            max_memory: opts.max_memory,
                        },
                    })
                }
                Kind::G => {
                    if mbpt2 {
                        return Err(PyErrLite::NotImpl("KGCCSD.kernel(mbpt2=True) is not bound"));
                    }
                    if eris_opts.keep_exxdiv {
                        return Err(PyErrLite::NotImpl(
                            "KGCCSD.keep_exxdiv = True is not implemented (kccsd.rs Kgccsd::new \
                             hard-codes keep_exxdiv = false)",
                        ));
                    }
                    let mut mf = Kghf::from_df(df);
                    mf.exxdiv = exxdiv;
                    let mut cc = Kgccsd::new(&result, &mut mf)?;
                    cc.opts = opts;
                    let cell = mf.cell();
                    let kpts = mf.kpts().to_vec();
                    let khelper = KptsHelper::without_symm_map(&cell.a, &kpts);
                    let eris = cc.ao2mo(mf.with_df.as_ref(), &khelper)?;
                    let r = cc.kernel(&eris, &khelper.kconserv)?;
                    let lat = Lattice { a: cell.a, kpts };
                    Ok(Solved {
                        e_hf,
                        e_corr: r.e_corr,
                        emp2: r.emp2,
                        converged: r.converged,
                        cycles: r.cycles,
                        amps: Amps::G {
                            eris,
                            padded: cc.padded,
                            khelper,
                            t1: r.t1,
                            t2: r.t2,
                            lat,
                        },
                    })
                }
                Kind::KsymR => {
                    if mbpt2 {
                        return Err(PyErrLite::NotImpl(
                            "KsymAdaptedRCCSD.kernel(mbpt2=True) is not bound",
                        ));
                    }
                    let kp = kpoints.as_ref().ok_or(PyErrLite::NotImpl(
                        "KsymAdaptedRCCSD: mean field has no KPoints",
                    ))?;
                    let dfr: &dyn PeriodicDf = df.as_ref();
                    let full = pyscf_pbc_mp::unfold_kscf_result(&result, kp, dfr.cell())
                        .map_err(PyErrLite::Mp)?;
                    let inputs = KsymRccsdInputs::build(&full, dfr, kp, eris_opts)?;
                    let (r, n_ao2mo) = run_ksym_rccsd(
                        &inputs,
                        dfr,
                        kp,
                        &KsymRccsdOpts {
                            base: opts,
                            ktensor_direct: false,
                        },
                    )?;
                    let ctx = inputs.ctx(kp);
                    let (no, nv) = (inputs.nocc, inputs.nvir);
                    let t1 = ctx.dense2(&r.t1, [no, nv], &ctx.labels.ov, &ctx.labels.nc)?;
                    let t2 =
                        ctx.dense4(&r.t2, [no, no, nv, nv], &ctx.labels.oovv, &ctx.labels.nncc)?;
                    Ok(Solved {
                        e_hf: inputs.e_hf,
                        e_corr: r.e_corr,
                        emp2: r.emp2,
                        converged: r.converged,
                        cycles: r.cycles,
                        amps: Amps::KsymR { t1, t2, n_ao2mo },
                    })
                }
            }
        });
        self.solved = Some(solved.map_err(PyErrLite::into_py)?);
        Ok(())
    }

    /// `(T)` on the stored amplitudes (or `t1`/`t2` given).
    fn ccsd_t_impl(
        &self,
        py: Python<'_>,
        t1: Option<&Bound<'_, PyAny>>,
        t2: Option<&Bound<'_, PyAny>>,
        slow: bool,
    ) -> PyResult<f64> {
        let s = self.solved()?;
        let pick = |given: Option<&Bound<'_, PyAny>>, stored: &ZArr, what| match given
            .filter(|x| !x.is_none())
        {
            Some(x) => zarr_from_py(x, stored.shape(), what),
            None => Ok(stored.clone()),
        };
        match &s.amps {
            Amps::R {
                eris,
                padded,
                khelper,
                t1: s1,
                t2: s2,
                lat,
            } => {
                let (t1, t2) = (pick(t1, s1, "t1")?, pick(t2, s2, "t2")?);
                py.detach(|| {
                    let f = if slow {
                        pyscf_pbc_cc::kccsd_t_rhf_slow::kernel
                    } else {
                        pyscf_pbc_cc::kccsd_t_rhf::kernel
                    };
                    f(
                        eris,
                        padded,
                        &t1,
                        &t2,
                        &khelper.kconserv,
                        &lat.a,
                        &lat.kpts,
                        None,
                    )
                })
                .map_err(pbc_cc_to_py)
            }
            Amps::G {
                eris,
                padded,
                khelper,
                t1: s1,
                t2: s2,
                lat,
            } => {
                if slow {
                    return Err(PyNotImplementedError::new_err(
                        "KGCCSD has one (T) implementation (kccsd_t.rs); _ccsd_t_slow is KRCCSD's",
                    ));
                }
                let (t1, t2) = (pick(t1, s1, "t1")?, pick(t2, s2, "t2")?);
                py.detach(|| {
                    pyscf_pbc_cc::kccsd_t::kernel(
                        eris,
                        padded,
                        &t1,
                        &t2,
                        &khelper.kconserv,
                        &lat.a,
                        &lat.kpts,
                    )
                })
                .map_err(pbc_cc_to_py)
            }
            Amps::U { .. } => Err(PyNotImplementedError::new_err(
                "KUCCSD.ccsd_t: there is no unrestricted k-point (T) in pyscf-pbc-cc \
                 (upstream kccsd_uhf.py has none either)",
            )),
            Amps::KsymR { .. } => Err(PyNotImplementedError::new_err(
                "KsymAdaptedRCCSD.ccsd_t is not implemented in pyscf-rs; run KRCCSD on the \
                 full-BZ mean field for (T)",
            )),
        }
    }

    /// One EOM solve, for every `kshift` in `kptlist`.
    #[allow(clippy::too_many_arguments)]
    fn eom_impl(
        &self,
        kind: Excitation,
        nroots: usize,
        koopmans: bool,
        left: bool,
        partition: Option<&str>,
        kptlist: Option<Vec<usize>>,
        conv_tol: f64,
        max_cycle: usize,
        max_space: usize,
    ) -> PyResult<Vec<EomRoots>> {
        let partition = match partition {
            None => Partition::None,
            Some("mp") => Partition::Mp,
            Some("full") => Partition::Full,
            Some(other) => {
                return Err(PyValueError::new_err(format!(
                    "partition must be None, 'mp' or 'full', got {other:?}"
                )));
            }
        };
        // Upstream's refusal is the solver's FIRST statement (eom_kccsd_ghf.py:618).
        if partition != Partition::None {
            return Err(pbc_cc_to_py(match self.kind {
                Kind::U => eomu::partition_refusal(),
                _ => eomg::partition_refusal(),
            }));
        }
        let s = self.solved()?;
        let opts = EomOpts {
            conv_tol,
            max_cycle,
            max_space,
            nroots,
            koopmans,
            left,
            partition,
        };
        let run = |nk: usize,
                   solve: &dyn Fn(usize) -> Result<EomRoots, PbcCcError>|
         -> PyResult<Vec<EomRoots>> {
            let list = kptlist.clone().unwrap_or_else(|| (0..nk).collect());
            if let Some(&bad) = list.iter().find(|&&k| k >= nk) {
                return Err(PyValueError::new_err(format!(
                    "kptlist entry {bad} is out of range for {nk} k-points"
                )));
            }
            list.into_iter()
                .map(|k| solve(k).map_err(pbc_cc_to_py))
                .collect()
        };
        match (&s.amps, kind) {
            (
                Amps::R {
                    eris,
                    padded,
                    khelper,
                    t1,
                    t2,
                    ..
                },
                _,
            ) => {
                let kc = &khelper.kconserv;
                let imds = eomr::RhfEomImds::make_shared(t1, t2, eris, kc).map_err(pbc_cc_to_py)?;
                let imds = match kind {
                    Excitation::Ip => imds.make_ip(kc),
                    Excitation::Ea => imds.make_ea(kc),
                    Excitation::Ee => imds.make_ee(kc),
                }
                .map_err(pbc_cc_to_py)?;
                let padding = eomg::padding_from(padded).map_err(pbc_cc_to_py)?;
                run(eris.nkpts, &|k| {
                    eomr::eom_kernel(kind, k, &imds, &padding, kc, &opts)
                })
            }
            (
                Amps::G {
                    eris,
                    padded,
                    khelper,
                    t1,
                    t2,
                    ..
                },
                _,
            ) => {
                let kc = &khelper.kconserv;
                let imds = eomg::EomImds::make_shared(t1, t2, eris, kc).map_err(pbc_cc_to_py)?;
                let imds = match kind {
                    Excitation::Ip => imds.make_ip(kc),
                    Excitation::Ea => imds.make_ea(kc),
                    Excitation::Ee => imds.make_ip(kc).and_then(|i| i.make_ea(kc)),
                }
                .map_err(pbc_cc_to_py)?;
                let padding = eomg::padding_from(padded).map_err(pbc_cc_to_py)?;
                run(eris.nkpts, &|k| {
                    eomg::eom_kernel(kind, k, &imds, &padding, kc, &opts)
                })
            }
            (
                Amps::U {
                    eris,
                    padded,
                    khelper,
                    t1,
                    t2,
                    max_memory,
                },
                _,
            ) => {
                let kc = &khelper.kconserv;
                let budget = (max_memory * 1e6).max(0.0) as usize;
                let pool = Arc::new(ZWorkspacePool::new(budget));
                let imds = eomu::UhfEomImds::make_shared(&pool, budget, t1, t2, eris, kc)
                    .map_err(pbc_cc_to_py)?;
                let imds = match kind {
                    Excitation::Ip => imds.make_ip(kc),
                    Excitation::Ea => imds.make_ea(&pool, budget, kc),
                    // `eom_kernel` refuses Ee itself (eom_kccsd_uhf.py has no EOMEE).
                    Excitation::Ee => Ok(imds),
                }
                .map_err(pbc_cc_to_py)?;
                let padding =
                    eomu::UPadding::from_padded(&padded.0, &padded.1).map_err(pbc_cc_to_py)?;
                run(eris.nkpts, &|k| {
                    eomu::eom_kernel(kind, k, &imds, &padding, kc, &opts)
                })
            }
            (Amps::KsymR { .. }, _) => Err(PyNotImplementedError::new_err(
                "EOM on KsymAdaptedRCCSD is not implemented in pyscf-rs (no k-symmetric EOM in \
                 pyscf-pbc-cc); run KRCCSD on the full-BZ mean field",
            )),
        }
    }
}

/// Error carrier across `py.detach` (a `PyErr` must be built with the GIL).
enum PyErrLite {
    Cc(PbcCcError),
    Mp(pyscf_pbc_mp::PbcMpError),
    NotImpl(&'static str),
}

impl From<PbcCcError> for PyErrLite {
    fn from(e: PbcCcError) -> Self {
        Self::Cc(e)
    }
}

impl PyErrLite {
    fn into_py(self) -> PyErr {
        match self {
            Self::Cc(e) => pbc_cc_to_py(e),
            Self::Mp(e) => pbc_mp_to_py(e),
            Self::NotImpl(m) => PyNotImplementedError::new_err(m),
        }
    }
}

/// `(e, v)` in upstream's EOM return layout: `e` a `(len(kptlist), nroots)`
/// float array when every k-shift returned the same root count (a list of
/// arrays otherwise), `v` a per-kshift list of complex `(nroots, size)`
/// arrays. Also returns the per-root convergence flags.
fn eom_to_py<'py>(
    py: Python<'py>,
    roots: &[EomRoots],
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyList>, Bound<'py, PyList>)> {
    let np = py.import("numpy")?;
    let e_rows = PyList::new(py, roots.iter().map(|r| PyArray1::from_slice(py, &r.e)))?;
    let same = roots.windows(2).all(|w| w[0].e.len() == w[1].e.len());
    let e = if same && !roots.is_empty() {
        np.call_method1("asarray", (e_rows,))?
    } else {
        e_rows.into_any()
    };
    let v = PyList::new(
        py,
        roots
            .iter()
            .map(|r| {
                let size = r.v.first().map_or(0, ZArr::len);
                let mut flat = pyscf_algebra::CTensor::zeros(r.v.len() * size);
                for (i, x) in r.v.iter().enumerate() {
                    flat.re[i * size..(i + 1) * size].copy_from_slice(&x.data().re);
                    flat.im[i * size..(i + 1) * size].copy_from_slice(&x.data().im);
                }
                complex_to_py(py, &flat, &[r.v.len(), size])
            })
            .collect::<PyResult<Vec<_>>>()?,
    )?;
    let conv = PyList::new(py, roots.iter().map(|r| r.conv.clone()))?;
    Ok((e, v, conv))
}

fn reject_given(x: Option<&Bound<'_, PyAny>>, what: &str) -> PyResult<()> {
    if x.is_some_and(|v| !v.is_none()) {
        return Err(PyNotImplementedError::new_err(format!(
            "{what}=... is not bound; the stored integrals/amplitudes are used"
        )));
    }
    Ok(())
}

#[pymethods]
impl PyKccsd {
    // ── options ─────────────────────────────────────────────────────────────
    #[getter]
    fn _scf(&self, py: Python<'_>) -> Py<PyAny> {
        self.mf.clone_ref(py)
    }
    #[getter]
    fn frozen(&self, py: Python<'_>) -> Py<PyAny> {
        py.None()
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
    fn conv_tol_normt(&self) -> f64 {
        self.opts.conv_tol_normt
    }
    #[setter]
    fn set_conv_tol_normt(&mut self, v: f64) {
        self.opts.conv_tol_normt = v;
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
    fn level_shift(&self) -> f64 {
        self.opts.level_shift
    }
    #[setter]
    fn set_level_shift(&mut self, v: f64) {
        self.opts.level_shift = v;
    }
    #[getter]
    fn diis(&self) -> bool {
        self.opts.diis
    }
    #[setter]
    fn set_diis(&mut self, v: Option<bool>) {
        self.opts.diis = v.unwrap_or(false);
    }
    #[getter]
    fn diis_space(&self) -> usize {
        self.opts.diis_space
    }
    #[setter]
    fn set_diis_space(&mut self, v: usize) {
        self.opts.diis_space = v;
    }
    #[getter]
    fn diis_start_cycle(&self) -> usize {
        self.opts.diis_start_cycle
    }
    #[setter]
    fn set_diis_start_cycle(&mut self, v: usize) {
        self.opts.diis_start_cycle = v;
    }
    #[getter]
    fn iterative_damping(&self) -> f64 {
        self.opts.iterative_damping
    }
    #[setter]
    fn set_iterative_damping(&mut self, v: f64) {
        self.opts.iterative_damping = v;
    }
    #[getter]
    fn max_memory(&self) -> f64 {
        self.opts.max_memory
    }
    #[setter]
    fn set_max_memory(&mut self, v: f64) {
        self.opts.max_memory = v;
    }
    #[getter]
    fn max_space(&self) -> usize {
        self.max_space
    }
    #[setter]
    fn set_max_space(&mut self, v: usize) {
        self.max_space = v;
    }
    #[getter]
    fn keep_exxdiv(&self) -> bool {
        self.keep_exxdiv
    }
    #[setter]
    fn set_keep_exxdiv(&mut self, v: bool) {
        self.keep_exxdiv = v;
    }

    // ── results ─────────────────────────────────────────────────────────────
    #[getter]
    fn e_hf(&self) -> Option<f64> {
        self.solved.as_ref().map(|s| s.e_hf)
    }
    #[getter]
    fn e_corr(&self) -> Option<f64> {
        self.solved.as_ref().map(|s| s.e_corr)
    }
    /// Upstream alias `ecc` (pyscf/cc/ccsd.py:990-992, `return self.e_corr`),
    /// inherited by KRCCSD/KGCCSD/KUCCSD through `ccsd.CCSD`.
    #[getter]
    fn ecc(&self) -> Option<f64> {
        self.e_corr()
    }
    #[getter]
    fn emp2(&self) -> Option<f64> {
        self.solved.as_ref().map(|s| s.emp2)
    }
    #[getter]
    fn e_tot(&self) -> Option<f64> {
        self.solved.as_ref().map(|s| s.e_hf + s.e_corr)
    }
    #[getter]
    fn converged(&self) -> bool {
        self.solved.as_ref().is_some_and(|s| s.converged)
    }
    #[getter]
    fn cycles(&self) -> Option<usize> {
        self.solved.as_ref().map(|s| s.cycles)
    }
    /// `(nkpts, nocc, nvir)` complex; `(t1a, t1b)` for KUCCSD.
    #[getter]
    fn t1<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Some(s) = &self.solved else {
            return Ok(py.None().into_bound(py));
        };
        match &s.amps {
            Amps::R { t1, .. } | Amps::G { t1, .. } | Amps::KsymR { t1, .. } => zarr_to_py(py, t1),
            Amps::U { t1, .. } => {
                Ok(PyTuple::new(py, [zarr_to_py(py, &t1.0)?, zarr_to_py(py, &t1.1)?])?.into_any())
            }
        }
    }
    /// `(nk, nk, nk, nocc, nocc, nvir, nvir)` complex; `(t2aa, t2ab, t2bb)` for KUCCSD.
    #[getter]
    fn t2<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Some(s) = &self.solved else {
            return Ok(py.None().into_bound(py));
        };
        match &s.amps {
            Amps::R { t2, .. } | Amps::G { t2, .. } | Amps::KsymR { t2, .. } => zarr_to_py(py, t2),
            Amps::U { t2, .. } => Ok(PyTuple::new(
                py,
                [
                    zarr_to_py(py, &t2.0)?,
                    zarr_to_py(py, &t2.1)?,
                    zarr_to_py(py, &t2.2)?,
                ],
            )?
            .into_any()),
        }
    }
    /// PRIVATE: how many `ao2mo` transforms the k-symmetric IBZ loop ran.
    #[getter]
    fn _n_ao2mo(&self) -> Option<usize> {
        match self.solved.as_ref().map(|s| &s.amps) {
            Some(Amps::KsymR { n_ao2mo, .. }) => Some(*n_ao2mo),
            _ => None,
        }
    }
    #[getter]
    fn nkpts(&self, py: Python<'_>) -> PyResult<usize> {
        let input = mean_field(py, self.mf.bind(py))?;
        Ok(input
            .kpoints
            .as_ref()
            .map_or(input.result.nkpts, |k| k.nkpts()))
    }

    // ── kernel ──────────────────────────────────────────────────────────────

    /// `kernel(t1=None, t2=None, eris=None, mbpt2=False)` → `(e_corr, t1, t2)`.
    /// `mbpt2=True` (KRCCSD only) stops at `init_amps`.
    #[pyo3(signature = (t1 = None, t2 = None, eris = None, mbpt2 = false))]
    fn kernel<'py>(
        &mut self,
        py: Python<'py>,
        t1: Option<&Bound<'py, PyAny>>,
        t2: Option<&Bound<'py, PyAny>>,
        eris: Option<&Bound<'py, PyAny>>,
        mbpt2: bool,
    ) -> PyResult<(f64, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        reject_given(t1, "t1")?;
        reject_given(t2, "t2")?;
        reject_given(eris, "eris")?;
        self.solve(py, mbpt2)?;
        let e = self.solved()?.e_corr;
        Ok((e, self.t1(py)?, self.t2(py)?))
    }

    /// Alias of `kernel` (`ccsd.CCSD.ccsd`).
    #[pyo3(signature = (t1 = None, t2 = None, eris = None, mbpt2 = false))]
    fn ccsd<'py>(
        &mut self,
        py: Python<'py>,
        t1: Option<&Bound<'py, PyAny>>,
        t2: Option<&Bound<'py, PyAny>>,
        eris: Option<&Bound<'py, PyAny>>,
        mbpt2: bool,
    ) -> PyResult<(f64, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        self.kernel(py, t1, t2, eris, mbpt2)
    }

    /// `ccsd_t(t1=None, t2=None, eris=None)` — the (T) correction. KRCCSD:
    /// the blocked `kccsd_t_rhf` (`kccsd_t_rhf.rs:365`); KGCCSD: the
    /// spin-orbital `kccsd_t` (`kccsd_t.rs:72`).
    #[pyo3(signature = (t1 = None, t2 = None, eris = None))]
    fn ccsd_t(
        &self,
        py: Python<'_>,
        t1: Option<&Bound<'_, PyAny>>,
        t2: Option<&Bound<'_, PyAny>>,
        eris: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<f64> {
        reject_given(eris, "eris")?;
        self.ccsd_t_impl(py, t1, t2, false)
    }

    /// PRIVATE: KRCCSD's loop-explicit reference (T) (`kccsd_t_rhf_slow.rs:319`)
    /// on the same amplitudes — G4's second implementation.
    #[pyo3(signature = (t1 = None, t2 = None))]
    fn _ccsd_t_slow(
        &self,
        py: Python<'_>,
        t1: Option<&Bound<'_, PyAny>>,
        t2: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<f64> {
        self.ccsd_t_impl(py, t1, t2, true)
    }

    /// `ipccsd(nroots=1, left=False, koopmans=False, guess=None,
    /// partition=None, eris=None, kptlist=None)` → `(e, v)`.
    #[pyo3(signature = (nroots = 1, left = false, koopmans = false, guess = None, partition = None, eris = None, kptlist = None))]
    #[allow(clippy::too_many_arguments)]
    fn ipccsd<'py>(
        &self,
        py: Python<'py>,
        nroots: usize,
        left: bool,
        koopmans: bool,
        guess: Option<&Bound<'py, PyAny>>,
        partition: Option<&str>,
        eris: Option<&Bound<'py, PyAny>>,
        kptlist: Option<Vec<usize>>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyList>)> {
        reject_given(guess, "guess")?;
        reject_given(eris, "eris")?;
        let roots = self.eom_impl(
            Excitation::Ip,
            nroots,
            koopmans,
            left,
            partition,
            kptlist,
            self.opts.conv_tol,
            self.opts.max_cycle,
            self.max_space,
        )?;
        let (e, v, _) = eom_to_py(py, &roots)?;
        Ok((e, v))
    }

    /// `eaccsd(...)` → `(e, v)`; arguments as `ipccsd`.
    #[pyo3(signature = (nroots = 1, left = false, koopmans = false, guess = None, partition = None, eris = None, kptlist = None))]
    #[allow(clippy::too_many_arguments)]
    fn eaccsd<'py>(
        &self,
        py: Python<'py>,
        nroots: usize,
        left: bool,
        koopmans: bool,
        guess: Option<&Bound<'py, PyAny>>,
        partition: Option<&str>,
        eris: Option<&Bound<'py, PyAny>>,
        kptlist: Option<Vec<usize>>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyList>)> {
        reject_given(guess, "guess")?;
        reject_given(eris, "eris")?;
        let roots = self.eom_impl(
            Excitation::Ea,
            nroots,
            koopmans,
            left,
            partition,
            kptlist,
            self.opts.conv_tol,
            self.opts.max_cycle,
            self.max_space,
        )?;
        let (e, v, _) = eom_to_py(py, &roots)?;
        Ok((e, v))
    }

    /// `run(**attrs)` → self.
    #[pyo3(signature = (**kwargs))]
    fn run<'py>(
        slf: &Bound<'py, Self>,
        kwargs: Option<&Bound<'py, PyDict>>,
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

macro_rules! cc_subclass {
    ($ty:ident, $name:literal, $kind:expr, $doc:literal) => {
        #[doc = $doc]
        #[pyclass(extends = PyKccsd, subclass, name = $name, module = "pyscf._native.pbc.cc")]
        pub struct $ty {}

        #[pymethods]
        impl $ty {
            #[new]
            #[pyo3(signature = (mf, frozen = None, mo_coeff = None, mo_occ = None))]
            fn new(
                py: Python<'_>,
                mf: &Bound<'_, PyAny>,
                frozen: Option<&Bound<'_, PyAny>>,
                mo_coeff: Option<&Bound<'_, PyAny>>,
                mo_occ: Option<&Bound<'_, PyAny>>,
            ) -> PyResult<PyClassInitializer<Self>> {
                Ok(PyClassInitializer::from(PyKccsd::construct(
                    py, mf, frozen, mo_coeff, mo_occ, $kind,
                )?)
                .add_subclass(Self {}))
            }
        }
    };
}

cc_subclass!(
    PyKrccsd,
    "KRCCSD",
    Kind::R,
    "`KRCCSD(mf)` — restricted k-point CCSD on a converged full-BZ `KRHF` (`kccsd_rhf.py:498`)."
);
cc_subclass!(
    PyKuccsd,
    "KUCCSD",
    Kind::U,
    "`KUCCSD(mf)` — unrestricted k-point CCSD on a converged `KUHF` (`kccsd_uhf.py:638`)."
);
cc_subclass!(
    PyKgccsd,
    "KGCCSD",
    Kind::G,
    "`KGCCSD(mf)` — spin-orbital k-point CCSD on a converged `KGHF` (`kccsd.py`)."
);
cc_subclass!(
    PyKsymAdaptedRccsd,
    "KsymAdaptedRCCSD",
    Kind::KsymR,
    "`KsymAdaptedRCCSD(mf)` — k-symmetric restricted CCSD on `KRHF(cell, kpts=KPoints)` \
     (`kccsd_rhf_ksymm.py`). `t1`/`t2` are the amplitudes unfolded to the full BZ."
);

// ─────────────────────────────────────────────────────────────────────────────
// EOM solvers
// ─────────────────────────────────────────────────────────────────────────────

/// `_EOM` — native base of `EOMIP`, `EOMEA`, `EOMEESinglet`, `EOMEE`.
///
/// `EOM(cc)` takes a solved `KRCCSD`/`KUCCSD`/`KGCCSD`. Attributes (upstream
/// `eom_rccsd.EOM.__init__`): `max_space = 20`, `max_cycle = cc.max_cycle`,
/// `conv_tol = cc.conv_tol`, `partition = None`. `kernel(nroots=1,
/// koopmans=False, guess=None, left=False, eris=None, imds=None,
/// partition=None, kptlist=None, dtype=None)` → `(e, v)`; results `e`, `v`,
/// `converged`.
#[pyclass(
    subclass,
    dict,
    name = "_EOM",
    module = "pyscf._native.pbc.cc",
    skip_from_py_object
)]
pub struct PyEom {
    cc: Py<PyAny>,
    kind: Excitation,
    /// `EOMEE` / `EOMEESinglet` — which CC family may run it.
    ee_family: Option<Kind>,
    max_space: usize,
    max_cycle: usize,
    conv_tol: f64,
    partition: Option<String>,
    e: Option<Py<PyAny>>,
    v: Option<Py<PyAny>>,
    converged: Option<Py<PyAny>>,
}

impl PyEom {
    fn construct(
        cc: &Bound<'_, PyAny>,
        kind: Excitation,
        ee_family: Option<Kind>,
    ) -> PyResult<Self> {
        let c = cc.cast::<PyKccsd>().map_err(|_| {
            PyTypeError::new_err("EOM needs a pyscf.pbc.cc KRCCSD / KUCCSD / KGCCSD object")
        })?;
        let me = c.borrow();
        Ok(Self {
            cc: cc.clone().unbind(),
            kind,
            ee_family,
            max_space: 20,
            max_cycle: me.opts.max_cycle,
            conv_tol: me.opts.conv_tol,
            partition: None,
            e: None,
            v: None,
            converged: None,
        })
    }
}

#[pymethods]
impl PyEom {
    #[getter]
    fn _cc(&self, py: Python<'_>) -> Py<PyAny> {
        self.cc.clone_ref(py)
    }
    #[getter]
    fn max_space(&self) -> usize {
        self.max_space
    }
    #[setter]
    fn set_max_space(&mut self, v: usize) {
        self.max_space = v;
    }
    #[getter]
    fn max_cycle(&self) -> usize {
        self.max_cycle
    }
    #[setter]
    fn set_max_cycle(&mut self, v: usize) {
        self.max_cycle = v;
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
    fn partition(&self) -> Option<String> {
        self.partition.clone()
    }
    #[setter]
    fn set_partition(&mut self, v: Option<String>) {
        self.partition = v;
    }
    #[getter]
    fn e(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.e.as_ref().map(|x| x.clone_ref(py))
    }
    #[getter]
    fn v(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.v.as_ref().map(|x| x.clone_ref(py))
    }
    #[getter]
    fn converged(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.converged.as_ref().map(|x| x.clone_ref(py))
    }

    #[pyo3(signature = (nroots = 1, koopmans = false, guess = None, left = false, eris = None, imds = None, partition = None, kptlist = None, dtype = None))]
    #[allow(clippy::too_many_arguments)]
    fn kernel<'py>(
        &mut self,
        py: Python<'py>,
        nroots: usize,
        koopmans: bool,
        guess: Option<&Bound<'py, PyAny>>,
        left: bool,
        eris: Option<&Bound<'py, PyAny>>,
        imds: Option<&Bound<'py, PyAny>>,
        partition: Option<String>,
        kptlist: Option<Vec<usize>>,
        dtype: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyList>)> {
        reject_given(guess, "guess")?;
        reject_given(eris, "eris")?;
        reject_given(imds, "imds")?;
        let _ = dtype;
        let partition = partition.or_else(|| self.partition.clone());
        let cc = self.cc.bind(py).cast::<PyKccsd>()?.borrow();
        if let Some(family) = self.ee_family
            && partition.is_none()
            && cc.kind != family
        {
            return Err(PyNotImplementedError::new_err(match family {
                Kind::R => "EOMEESinglet is the KRCCSD (eom_kccsd_rhf.py:1425) solver",
                _ => {
                    "EOMEE over KRCCSD raises upstream (eom_kccsd_rhf.py:1417, use \
                     EOMEESinglet); KUCCSD has no EOMEE (eom_kccsd_uhf.py:1120); EOMEE here \
                     is KGCCSD's (eom_kccsd_ghf.py:1691)"
                }
            }));
        }
        let roots = cc.eom_impl(
            self.kind,
            nroots,
            koopmans,
            left,
            partition.as_deref(),
            kptlist,
            self.conv_tol,
            self.max_cycle,
            self.max_space,
        )?;
        drop(cc);
        let (e, v, conv) = eom_to_py(py, &roots)?;
        self.e = Some(e.clone().unbind());
        self.v = Some(v.clone().into_any().unbind());
        self.converged = Some(conv.into_any().unbind());
        Ok((e, v))
    }
}

macro_rules! eom_subclass {
    ($ty:ident, $name:literal, $kind:expr, $family:expr, $doc:literal) => {
        #[doc = $doc]
        #[pyclass(extends = PyEom, subclass, name = $name, module = "pyscf._native.pbc.cc")]
        pub struct $ty {}

        #[pymethods]
        impl $ty {
            #[new]
            fn new(cc: &Bound<'_, PyAny>) -> PyResult<PyClassInitializer<Self>> {
                Ok(
                    PyClassInitializer::from(PyEom::construct(cc, $kind, $family)?)
                        .add_subclass(Self {}),
                )
            }
        }
    };
}

eom_subclass!(
    PyEomIp,
    "EOMIP",
    Excitation::Ip,
    None,
    "`EOMIP(cc)` — EOM-IP-CCSD at k-points (RHF/UHF/GHF by the CC object)."
);
eom_subclass!(
    PyEomEa,
    "EOMEA",
    Excitation::Ea,
    None,
    "`EOMEA(cc)` — EOM-EA-CCSD at k-points (RHF/UHF/GHF by the CC object)."
);
eom_subclass!(
    PyEomEeSinglet,
    "EOMEESinglet",
    Excitation::Ee,
    Some(Kind::R),
    "`EOMEESinglet(cc)` — spin-adapted singlet EOM-EE on a KRCCSD (`eom_kccsd_rhf.py:1425`)."
);
eom_subclass!(
    PyEomEe,
    "EOMEE",
    Excitation::Ee,
    Some(Kind::G),
    "`EOMEE(cc)` — spin-orbital EOM-EE on a KGCCSD (`eom_kccsd_ghf.py:1691`)."
);

// ─────────────────────────────────────────────────────────────────────────────
// The Rust reference path (tests)
// ─────────────────────────────────────────────────────────────────────────────

/// PRIVATE (test reference): an all-Rust `Krhf::kernel` → `Krccsd::ao2mo` →
/// `kernel_with` on FRESH builders of `with_df`'s kind — `krccsd_smoke.rs`'s
/// exact sequence, with default `KrccsdOpts`. Returns `(e_hf, e_corr, e_t)`
/// where `e_t` is the blocked (T) on those amplitudes.
#[pyfunction]
#[pyo3(signature = (with_df, exxdiv, conv_tol, conv_tol_grad, max_cycle))]
fn _rust_reference_krccsd(
    py: Python<'_>,
    with_df: &Bound<'_, PyAny>,
    exxdiv: Option<&str>,
    conv_tol: f64,
    conv_tol_grad: Option<f64>,
    max_cycle: u32,
) -> PyResult<(f64, f64, f64)> {
    let (scf, df2) = reference_scf(py, with_df, exxdiv, conv_tol, conv_tol_grad, max_cycle)?;
    let exx = scf_exxdiv(exxdiv);
    py.detach(move || -> Result<(f64, f64, f64), PbcCcError> {
        let mut cc = Krccsd::new(&scf, df2.as_ref())?;
        cc.eris_opts.exxdiv = exx;
        let eris = cc.ao2mo()?;
        let r = cc.kernel_with(&eris)?;
        let cell = df2.cell();
        let e_t = pyscf_pbc_cc::kccsd_t_rhf::kernel(
            &eris,
            &cc.padded,
            &r.t1,
            &r.t2,
            &cc.khelper.kconserv,
            &cell.a,
            df2.kpts(),
            None,
        )?;
        Ok((scf.e_tot, r.e_corr, e_t))
    })
    .map_err(pbc_cc_to_py)
}

fn scf_exxdiv(s: Option<&str>) -> Option<pyscf_pbc_gto::ExxDiv> {
    match s {
        Some("ewald") => Some(pyscf_pbc_gto::ExxDiv::Ewald),
        _ => None,
    }
}
