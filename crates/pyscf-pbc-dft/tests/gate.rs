//! **The Phase-12 gate.** `KRKS(Si, 2x2x2, PBE)` against live upstream PySCF
//! **2.12.1** (the vendored tree — see `tests/common/mod.rs`), plus the
//! controls that say what any residual is made of.
//!
//! ```bash
//! PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-dft --release --test gate -- --ignored --nocapture
//! ```
//!
//! # Which XC library upstream is driven with
//!
//! This port evaluates functionals through `pyscf_dft::XcBackend`, whose
//! default is the native-Rust **xcfun** port. Upstream PySCF's default is
//! **libxc**. The two are independent implementations of the same functional
//! forms and are not bit-compatible with each other, so a comparison against
//! upstream-with-libxc measures the libxc/xcfun parameterisation gap and not
//! this port's fidelity. Every gate here therefore sets
//! `mf._numint.libxc = pyscf.dft.xcfun` upstream.
//! [`krks_si_222_pbe_against_libxc_default`] records the size of the gap
//! against upstream's DEFAULT so it is on the record rather than hidden.

mod common;

use common::{GATE, cell_args, he_all_electron, oracle_python, run_python, silicon};
use pyscf_pbc_df::Fftdf;
use pyscf_pbc_dft::krks::Krks;
use pyscf_pbc_dft::kuks::Kuks;
use pyscf_pbc_gto::{Cell, make_kpts_default};
use pyscf_pbc_scf::{KScfConfig, KScfResult, Krhf};

/// The mesh at which upstream's `ft_ao` non-local pseudopotential expansion has
/// converged against this port's exact real-space `get_pp_nl` — the same mesh
/// the Phase-11 gate uses.
const MESH_GATE: [usize; 3] = [31, 31, 31];

fn tight() -> KScfConfig {
    KScfConfig {
        conv_tol: 1e-12,
        conv_tol_grad: Some(1e-8),
        max_cycle: 60,
        ..KScfConfig::default()
    }
}

fn krks(cell: Cell, nk: [usize; 3], mesh: [usize; 3], xc: &str) -> Krks {
    let kpts = make_kpts_default(&cell, nk).expect("k-mesh");
    let df = Fftdf::with_mesh(cell, &kpts, mesh).expect("FFTDF");
    Krks::from_df(Box::new(df), xc).expect("KRKS")
}

// ---------------------------------------------------------------------------
// The oracle
// ---------------------------------------------------------------------------

const ORACLE_PY: &str = r#"
import json, sys
import numpy as np
from pyscf.pbc import gto, dft, scf

(a_json, xyz_json, sym_json, spin, charge, basis, pseudo,
 nk_json, mesh_json, method, xc, xclib) = sys.argv[1:13]

c = gto.Cell()
c.a = json.loads(a_json)
c.atom = [(s, tuple(r)) for s, r in zip(json.loads(sym_json), json.loads(xyz_json))]
c.basis = basis
if pseudo:
    c.pseudo = pseudo
c.unit = 'Bohr'
# U-00 step 1: spin and charge are set BEFORE build(), the only point at which
# `tot_electrons` / `nelec` are derived from them.
c.spin = int(spin)
c.charge = int(charge)
c.verbose = 0
c.build()

kpts = c.make_kpts(json.loads(nk_json))
mod = scf if method.endswith('HF') else dft
mf = getattr(mod, method)(c, kpts)
if xc:
    mf.xc = xc
# Drive upstream with the SAME functional library this port evaluates. See the
# module docstring of tests/gate.rs.
if xclib == 'xcfun':
    from pyscf.dft import xcfun
    mf._numint.libxc = xcfun
elif xclib == 'libxc':
    from pyscf.dft import libxc
    mf._numint.libxc = libxc
mesh = json.loads(mesh_json)
mf.with_df.mesh = mesh
# The XC quadrature grid is a SEPARATE object from the density-fitting grid:
# `UniformGrids.__init__` seeds `self.mesh = cell.mesh` (pbc/dft/gen_grid.py:72),
# which `with_df.mesh` does not touch. This port derives both from the one
# FFTDF mesh, so upstream must be pinned to the same mesh on BOTH or the two
# sides integrate the exchange-correlation energy on different quadratures --
# worth ~1e-9 Ha, which swamps everything the gate is trying to measure.
if hasattr(mf, 'grids'):
    mf.grids.mesh = mesh
mf.conv_tol = 1e-12
mf.conv_tol_grad = 1e-8
mf.max_cycle = 60
e = mf.kernel()
print(json.dumps({'version': __import__('pyscf').__version__,
                  'xclib': (getattr(mf, '_numint', None) is not None
                            and mf._numint.libxc.__name__.rsplit('.', 1)[-1] or ''),
                  'e_tot': float(e), 'e_nuc': float(c.energy_nuc()),
                  'converged': bool(mf.converged), 'nao': int(c.nao_nr())}))
"#;

struct Oracle {
    nk: [usize; 3],
    mesh: [usize; 3],
    method: &'static str,
    xc: &'static str,
    xclib: &'static str,
}

fn upstream(cell: &Cell, basis: &str, pseudo: &str, o: &Oracle) -> Option<serde_json::Value> {
    let py = oracle_python()?;
    let args = cell_args(
        cell,
        &[
            basis.to_string(),
            pseudo.to_string(),
            serde_json::to_string(&o.nk.to_vec()).expect("json"),
            serde_json::to_string(&o.mesh.to_vec()).expect("json"),
            o.method.to_string(),
            o.xc.to_string(),
            o.xclib.to_string(),
        ],
    );
    let v = run_python(&py, ORACLE_PY, &args);
    assert_eq!(
        v["version"].as_str().expect("version"),
        "2.12.1",
        "the oracle must be the VENDORED PySCF 2.12.1 — see tests/common/mod.rs"
    );
    if !o.xclib.is_empty() && !o.xc.is_empty() {
        assert_eq!(
            v["xclib"].as_str().expect("xclib"),
            o.xclib,
            "the upstream XC library switch did not take effect"
        );
    }
    assert!(
        v["converged"].as_bool().unwrap_or(false),
        "upstream did not converge"
    );
    Some(v)
}

/// Compare, asserting `e_nuc` FIRST so a pass can never come from two runs that
/// are quietly describing different cells.
fn assert_matches(got: &KScfResult, want: &serde_json::Value, tol: f64, label: &str) -> f64 {
    let e_ref = want["e_tot"].as_f64().expect("e_tot");
    let n_ref = want["e_nuc"].as_f64().expect("e_nuc");
    assert!(
        (got.e_nuc - n_ref).abs() < 1e-12,
        "{label}: e_nuc {} != {n_ref} — the two runs are not the same cell",
        got.e_nuc
    );
    let d = got.e_tot - e_ref;
    println!(
        "{label:<34} rust {:.15}  upstream {:.15}  delta {:.3e}  (tol {tol:.0e})",
        got.e_tot, e_ref, d
    );
    assert!(
        d.abs() < tol,
        "{label}: |delta| = {:e} exceeds {tol:e}",
        d.abs()
    );
    d.abs()
}

// ---------------------------------------------------------------------------
// THE GATE
// ---------------------------------------------------------------------------

/// **THE PHASE-12 GATE.** `KRKS(Si, 2x2x2, PBE)` against upstream-with-xcfun.
#[test]
#[ignore = "needs PYSCF_ORACLE_VENV + the vendored upstream PySCF"]
fn krks_si_222_pbe_matches_upstream() {
    let cell = silicon();
    let Some(want) = upstream(
        &cell,
        "gth-szv",
        "gth-pade",
        &Oracle {
            nk: [2, 2, 2],
            mesh: MESH_GATE,
            method: "KRKS",
            xc: "pbe",
            xclib: "libxc",
        },
    ) else {
        eprintln!("SKIP: {GATE} is not set");
        return;
    };
    let got = krks(cell, [2, 2, 2], MESH_GATE, "pbe")
        .kernel(&tight())
        .expect("KRKS");
    assert!(got.converged);
    assert_matches(&got, &want, 1e-11, "KRKS Si 2x2x2 PBE");
}

/// The LDA gate on the same cell — no `sigma` anywhere, so it isolates the
/// density/potential machinery from the GGA chain rule.
#[test]
#[ignore = "needs PYSCF_ORACLE_VENV + the vendored upstream PySCF"]
fn krks_si_222_lda_matches_upstream() {
    let cell = silicon();
    let Some(want) = upstream(
        &cell,
        "gth-szv",
        "gth-pade",
        &Oracle {
            nk: [2, 2, 2],
            mesh: MESH_GATE,
            method: "KRKS",
            xc: "lda,vwn",
            xclib: "libxc",
        },
    ) else {
        eprintln!("SKIP: {GATE} is not set");
        return;
    };
    let got = krks(cell, [2, 2, 2], MESH_GATE, "lda,vwn")
        .kernel(&tight())
        .expect("KRKS");
    assert!(got.converged);
    assert_matches(&got, &want, 1e-11, "KRKS Si 2x2x2 LDA,VWN");
}

/// **The all-electron control.** He-fcc / `sto-3g` has no pseudopotential, so
/// `get_nuc` carries no planewave-sum component and the comparison is not
/// sitting on the Phase-10/11 `get_pp` floor. The IDENTICAL Phase-12 code path
/// runs — AO block loop, complex `eval_rho`, `eval_xc_eff` chain rule,
/// `_vxc_mat` back-contraction, `ecoul`/`exc` book-keeping.
#[test]
#[ignore = "needs PYSCF_ORACLE_VENV + the vendored upstream PySCF"]
fn krks_he_all_electron_222_pbe_matches_upstream() {
    let cell = he_all_electron();
    let Some(want) = upstream(
        &cell,
        "sto-3g",
        "",
        &Oracle {
            nk: [2, 2, 2],
            mesh: MESH_GATE,
            method: "KRKS",
            xc: "pbe",
            xclib: "libxc",
        },
    ) else {
        eprintln!("SKIP: {GATE} is not set");
        return;
    };
    let got = krks(cell, [2, 2, 2], MESH_GATE, "pbe")
        .kernel(&tight())
        .expect("KRKS");
    assert!(got.converged);
    assert_matches(&got, &want, 1e-12, "KRKS He-fcc 2x2x2 PBE (AE)");
}

/// **The no-XC control.** `KRHF` on the SAME Si cell, the same mesh, the same
/// k-mesh, with no exchange-correlation functional anywhere. Whatever this
/// deviates by is a floor Phase 12 inherits rather than creates.
#[test]
#[ignore = "needs PYSCF_ORACLE_VENV + the vendored upstream PySCF"]
fn krhf_si_222_is_the_pseudopotential_floor() {
    let cell = silicon();
    let Some(want) = upstream(
        &cell,
        "gth-szv",
        "gth-pade",
        &Oracle {
            nk: [2, 2, 2],
            mesh: MESH_GATE,
            method: "KRHF",
            xc: "",
            xclib: "",
        },
    ) else {
        eprintln!("SKIP: {GATE} is not set");
        return;
    };
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("k-mesh");
    let df = Fftdf::with_mesh(cell, &kpts, MESH_GATE).expect("FFTDF");
    let got = Krhf::from_df(Box::new(df)).kernel(&tight()).expect("KRHF");
    assert!(got.converged);
    assert_matches(&got, &want, 1e-11, "KRHF Si 2x2x2 (no XC)");
}

/// `KUKS` on a closed-shell cell against upstream's `KUKS` — the open-shell
/// path shares no code with `nr_rks` (`nr_uks`, the full-vs-half `vk`, the
/// cross-spin `_stack_fg` assembly).
#[test]
#[ignore = "needs PYSCF_ORACLE_VENV + the vendored upstream PySCF"]
fn kuks_si_222_pbe_matches_upstream() {
    let cell = silicon();
    let Some(want) = upstream(
        &cell,
        "gth-szv",
        "gth-pade",
        &Oracle {
            nk: [2, 2, 2],
            mesh: MESH_GATE,
            method: "KUKS",
            xc: "pbe",
            xclib: "libxc",
        },
    ) else {
        eprintln!("SKIP: {GATE} is not set");
        return;
    };
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("k-mesh");
    let df = Fftdf::with_mesh(cell, &kpts, MESH_GATE).expect("FFTDF");
    let got = Kuks::from_df(Box::new(df), "pbe")
        .expect("KUKS")
        .kernel(&tight())
        .expect("KUKS");
    assert!(got.converged);
    assert_matches(&got, &want, 1e-11, "KUKS Si 2x2x2 PBE");
}

/// A HYBRID functional, which routes through the `veff.rs` J/K dispatch and
/// builds an exchange matrix the pure-functional gates never touch.
#[test]
#[ignore = "needs PYSCF_ORACLE_VENV + the vendored upstream PySCF"]
fn krks_si_222_pbe0_matches_upstream() {
    let cell = silicon();
    let Some(want) = upstream(
        &cell,
        "gth-szv",
        "gth-pade",
        &Oracle {
            nk: [2, 2, 2],
            mesh: MESH_GATE,
            method: "KRKS",
            xc: "pbe0",
            xclib: "libxc",
        },
    ) else {
        eprintln!("SKIP: {GATE} is not set");
        return;
    };
    let got = krks(cell, [2, 2, 2], MESH_GATE, "pbe0")
        .kernel(&tight())
        .expect("KRKS");
    assert!(got.converged);
    assert_matches(&got, &want, 1e-11, "KRKS Si 2x2x2 PBE0");
}

/// Not a gate — a MEASUREMENT, kept so the size of the libxc/xcfun functional
/// gap stays on the record. This port runs libxc; upstream is driven with
/// **xcfun** here, so the delta is the parameterisation difference between two
/// independent implementations of PBE and nothing else. Before 2026-08-28 this
/// measurement ran the other way round, and its magnitude is why the gates
/// could not use a default-configured upstream.
#[test]
#[ignore = "needs PYSCF_ORACLE_VENV + the vendored upstream PySCF"]
fn krks_si_222_pbe_against_xcfun() {
    let cell = silicon();
    let Some(want) = upstream(
        &cell,
        "gth-szv",
        "gth-pade",
        &Oracle {
            nk: [2, 2, 2],
            mesh: MESH_GATE,
            method: "KRKS",
            xc: "pbe",
            xclib: "xcfun",
        },
    ) else {
        eprintln!("SKIP: {GATE} is not set");
        return;
    };
    let got = krks(cell, [2, 2, 2], MESH_GATE, "pbe")
        .kernel(&tight())
        .expect("KRKS");
    let d = got.e_tot - want["e_tot"].as_f64().expect("e_tot");
    println!(
        "MEASUREMENT  KRKS Si 2x2x2 PBE vs upstream-with-{}: rust {:.15}  upstream {:.15}  delta {:.3e}",
        want["xclib"].as_str().unwrap_or("?"),
        got.e_tot,
        want["e_tot"].as_f64().expect("e_tot"),
        d
    );
    // Bounded only so a REGRESSION (a genuinely broken functional) still fails.
    assert!(
        d.abs() < 1e-5,
        "the libxc/xcfun gap grew to {:e}, which is far beyond a parameterisation difference",
        d.abs()
    );
}
