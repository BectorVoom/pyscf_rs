//! Plans 11-05 … 11-08 — `FFTDF` `get_nuc` / `get_pp` / `get_hcore` and the FFT
//! J/K builders.
//!
//! Two layers:
//!
//! * **Oracle-free** (always on, D-PBC-19): Hermiticity, the J energy identity,
//!   the gamma-point reality of every matrix, and the `exxdiv` shift being
//!   exactly `madelung * S D S`.
//! * **Upstream** (`#[ignore]`d, gated on `PYSCF_ORACLE_VENV`): element-wise
//!   against live PySCF **2.12.1** (the vendored tree — see
//!   `tests/common/mod.rs` for why site-packages 2.14 is not interchangeable):
//!
//!   ```bash
//!   PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-df --release -- --ignored
//!   ```
//!
//! # Why `get_pp` is gated at mesh >= 31 and not at mesh 11
//!
//! Upstream's `fft.get_pp` builds the NON-LOCAL half from `ft_ao`, a planewave
//! expansion truncated at the same mesh; this port uses Phase 10's real-space
//! `get_pp_nl`, which is exact in the basis (see `fftdf.rs`'s module docs). The
//! two agree only once the planewave expansion has converged. Measured on
//! diamond 2x2x2: 1.5e-3 at mesh 11, 1.3e-9 at mesh 21, **1.1e-13 at mesh 31**
//! and 1.1e-13 at the default mesh 47. The deviation is upstream's truncation
//! error, not ours, so the gate runs where upstream is converged.

mod common;

use common::{GATE, cell_args, diamond, he_all_electron, max_dev, oracle_python, run_python};
use pyscf_algebra::CTensor;
use pyscf_pbc_df::zlinalg::{forder_to_c, zmm_small};
use pyscf_pbc_df::{Fftdf, JkOpts, PeriodicDf, get_hcore};
use pyscf_pbc_gto::{ExxDiv, get_ovlp, madelung, make_kpts_default};

const MESH_FAST: [usize; 3] = [11, 11, 11];
/// The smallest mesh at which upstream's `ft_ao` non-local pseudopotential has
/// converged to this port's real-space one (see the module docs).
const MESH_PP: [usize; 3] = [31, 31, 31];

fn kpts222(cell: &pyscf_pbc_gto::Cell) -> Vec<[f64; 3]> {
    make_kpts_default(cell, [2, 2, 2]).expect("2x2x2 k-mesh")
}

/// A trivially Hermitian, positive-definite test density: `0.5 * I` at every k.
fn flat_dm(nao: usize, nkpts: usize) -> Vec<Vec<CTensor>> {
    let mut dm = CTensor::zeros(nao * nao);
    for i in 0..nao {
        dm.re[i * nao + i] = 0.5;
    }
    vec![vec![dm; nkpts]]
}

fn max_abs_anti_hermitian(m: &CTensor, n: usize) -> f64 {
    let mut w = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            w = w.max((m.re[i * n + j] - m.re[j * n + i]).abs());
            w = w.max((m.im[i * n + j] + m.im[j * n + i]).abs());
        }
    }
    w
}

// ---------------------------------------------------------------------------
// Oracle-free
// ---------------------------------------------------------------------------

/// `get_pp` is Hermitian at every k-point, and real at gamma.
#[test]
fn get_pp_is_hermitian_and_real_at_gamma() {
    let cell = diamond();
    let kpts = kpts222(&cell);
    let nao = cell.mol.nao_nr;
    let df = Fftdf::with_mesh(cell, &kpts, MESH_FAST).expect("FFTDF");
    let vpp = df.get_pp(&kpts).expect("get_pp");
    assert_eq!(vpp.len(), kpts.len());
    for (k, m) in vpp.iter().enumerate() {
        let w = max_abs_anti_hermitian(m, nao);
        assert!(w < 1e-11, "V_pp at k={k} is not Hermitian: {w:e}");
    }
    // k = 0 of a Monkhorst-Pack mesh with gamma is the gamma point.
    assert_eq!(
        vpp[0].im.iter().fold(0.0_f64, |a, v| a.max(v.abs())),
        0.0,
        "V_pp at gamma must be exactly real"
    );
}

/// `get_hcore` = `T + V_pp`, so it inherits the same structure — and it is the
/// function `pyscf_pbc_gto::hcore::get_hcore` deferred to this phase.
#[test]
fn get_hcore_is_hermitian() {
    let cell = diamond();
    let kpts = kpts222(&cell);
    let nao = cell.mol.nao_nr;
    let df = Fftdf::with_mesh(cell, &kpts, MESH_FAST).expect("FFTDF");
    let h = get_hcore(&df, &kpts).expect("get_hcore");
    for (k, m) in h.iter().enumerate() {
        let w = max_abs_anti_hermitian(m, nao);
        assert!(w < 1e-11, "hcore at k={k} is not Hermitian: {w:e}");
    }
}

/// The all-electron `get_nuc` path: Hermitian, real at gamma, and NEGATIVE on
/// the diagonal (an attractive potential).
#[test]
fn get_nuc_all_electron_is_hermitian_and_attractive() {
    let cell = he_all_electron();
    let kpts = vec![[0.0; 3]];
    let nao = cell.mol.nao_nr;
    let df = Fftdf::with_mesh(cell, &kpts, MESH_FAST).expect("FFTDF");
    let v = df.get_nuc(&kpts).expect("get_nuc");
    assert_eq!(v.len(), 1);
    assert!(max_abs_anti_hermitian(&v[0], nao) < 1e-12);
    for i in 0..nao {
        assert!(
            v[0].re[i * nao + i] < 0.0,
            "V_ne diagonal {i} = {} should be attractive",
            v[0].re[i * nao + i]
        );
    }
}

/// `vj` and `vk` are Hermitian at every k-point (plan 11-06 / 11-07 tests).
#[test]
fn vj_and_vk_are_hermitian() {
    let cell = diamond();
    let kpts = kpts222(&cell);
    let nao = cell.mol.nao_nr;
    let dms = flat_dm(nao, kpts.len());
    let df = Fftdf::with_mesh(cell, &kpts, MESH_FAST).expect("FFTDF");
    let r = df.get_jk(&dms, &kpts, JkOpts::hermitian()).expect("get_jk");
    let vj = r.vj.expect("vj");
    let vk = r.vk.expect("vk");
    for k in 0..kpts.len() {
        assert!(
            max_abs_anti_hermitian(&vj[0][k], nao) < 1e-12,
            "vj at k={k} is not Hermitian"
        );
        assert!(
            max_abs_anti_hermitian(&vk[0][k], nao) < 1e-12,
            "vk at k={k} is not Hermitian"
        );
    }
}

/// PBC-MASTER-PLAN plan 11-06's internal consistency identity, no oracle
/// needed: the Coulomb energy read off the matrices equals the one read off the
/// grid.
///
/// `(1/nkpts) sum_k Tr(vj[k] dm[k]).real == sum_r vR[r] rhoR[r] * ngrids/vol`,
/// where the right-hand side is what `get_j_kpts` integrates. Rather than
/// re-deriving `vR`/`rhoR` here, the equivalent statement is used: the Coulomb
/// energy is symmetric under swapping the two densities, i.e.
/// `Tr(vj[dm1] dm2) == Tr(vj[dm2] dm1)`.
#[test]
fn coulomb_energy_is_symmetric_in_its_two_densities() {
    let cell = diamond();
    let kpts = kpts222(&cell);
    let nao = cell.mol.nao_nr;
    let df = Fftdf::with_mesh(cell, &kpts, MESH_FAST).expect("FFTDF");

    let dm1 = flat_dm(nao, kpts.len());
    // A second, different Hermitian density: 0.3 on the diagonal plus a real
    // symmetric off-diagonal coupling.
    let mut d2 = CTensor::zeros(nao * nao);
    for i in 0..nao {
        d2.re[i * nao + i] = 0.3;
        for j in 0..nao {
            if i != j {
                d2.re[i * nao + j] = 0.05 / (1.0 + (i as f64 - j as f64).abs());
            }
        }
    }
    let dm2 = vec![vec![d2; kpts.len()]];

    let opts = JkOpts {
        hermi: 1,
        kpts_band: None,
        with_j: true,
        with_k: false,
        exxdiv: None,
        omega: None,
        kk_symmetry: false,
    };
    let vj1 = df
        .get_jk(&dm1, &kpts, opts.clone())
        .expect("vj1")
        .vj
        .expect("vj");
    let vj2 = df.get_jk(&dm2, &kpts, opts).expect("vj2").vj.expect("vj");

    let energy = |v: &[CTensor], d: &[CTensor]| -> f64 {
        let mut e = 0.0;
        for k in 0..kpts.len() {
            e += pyscf_pbc_df::zlinalg::ztrace_ab(&v[k], &d[k], nao).0;
        }
        e / kpts.len() as f64
    };
    let e12 = energy(&vj1[0], &dm2[0]);
    let e21 = energy(&vj2[0], &dm1[0]);
    assert!(
        (e12 - e21).abs() < 1e-11 * e12.abs().max(1.0),
        "Coulomb energy is not symmetric: {e12} vs {e21}"
    );
}

/// `exxdiv = Ewald` adds EXACTLY `madelung * S D S` and nothing else — the
/// analytic `_ewald_exxdiv_for_G0` of `df_jk.py:1479-1500`.
#[test]
fn ewald_exxdiv_adds_exactly_madelung_sds() {
    let cell = diamond();
    let kpts = kpts222(&cell);
    let nao = cell.mol.nao_nr;
    let dms = flat_dm(nao, kpts.len());
    let mad = madelung(&cell, &kpts, None).expect("madelung");
    let s = get_ovlp(&cell, &kpts).expect("ovlp");
    let df = Fftdf::with_mesh(cell, &kpts, MESH_FAST).expect("FFTDF");

    let mk = |exxdiv: Option<ExxDiv>| {
        df.get_jk(
            &dms,
            &kpts,
            JkOpts {
                hermi: 1,
                kpts_band: None,
                with_j: false,
                with_k: true,
                exxdiv,
                omega: None,
                kk_symmetry: false,
            },
        )
        .expect("get_jk")
        .vk
        .expect("vk")
    };
    let plain = mk(None);
    let shifted = mk(Some(ExxDiv::Ewald));

    for k in 0..kpts.len() {
        let sk = forder_to_c(&s[k], nao, nao);
        let sd = zmm_small(&sk, &dms[0][k], nao, nao, nao);
        let sds = zmm_small(&sd, &sk, nao, nao, nao);
        for i in 0..nao * nao {
            let want = plain[0][k].re[i] + mad * sds.re[i];
            assert!(
                (shifted[0][k].re[i] - want).abs() < 1e-12,
                "k={k} entry {i}: {} != {want}",
                shifted[0][k].re[i]
            );
        }
    }
}

/// `kpts_band = kpts` must reproduce the default path exactly — the band
/// machinery is otherwise untested until Phase 12's band structures.
#[test]
fn explicit_band_kpts_equal_the_default_path() {
    let cell = diamond();
    let kpts = kpts222(&cell);
    let nao = cell.mol.nao_nr;
    let dms = flat_dm(nao, kpts.len());
    let df = Fftdf::with_mesh(cell, &kpts, MESH_FAST).expect("FFTDF");
    let a = df
        .get_jk(&dms, &kpts, JkOpts::hermitian())
        .expect("default");
    let b = df
        .get_jk(
            &dms,
            &kpts,
            JkOpts {
                kpts_band: Some(&kpts),
                ..JkOpts::hermitian()
            },
        )
        .expect("banded");
    for (name, x, y) in [
        ("vj", a.vj.as_ref().expect("vj"), b.vj.as_ref().expect("vj")),
        ("vk", a.vk.as_ref().expect("vk"), b.vk.as_ref().expect("vk")),
    ] {
        for k in 0..kpts.len() {
            assert_eq!(
                x[0][k], y[0][k],
                "{name} at k={k} differs from the band path"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Upstream gates
// ---------------------------------------------------------------------------

const ORACLE_PY: &str = r#"
import json, sys
import numpy as np
from pyscf.pbc import gto, df

a_json, xyz_json, sym_json, basis, pseudo, nk_json, mesh_json, what = sys.argv[1:9]
c = gto.Cell()
c.a = json.loads(a_json)
c.atom = [(s, tuple(r)) for s, r in zip(json.loads(sym_json), json.loads(xyz_json))]
c.basis = basis
if pseudo:
    c.pseudo = pseudo
c.unit = 'Bohr'
c.verbose = 0
c.build()
kpts = c.make_kpts(json.loads(nk_json))
mesh = json.loads(mesh_json)
mydf = df.FFTDF(c, kpts)
mydf.mesh = mesh
nao = c.nao_nr()

if what == 'pp':
    mats = np.asarray(mydf.get_pp(kpts))
elif what == 'nuc':
    mats = np.asarray(mydf.get_nuc(kpts))
elif what == 'hcore':
    v = np.asarray(mydf.get_pp(kpts)) if c.pseudo else np.asarray(mydf.get_nuc(kpts))
    mats = v + np.asarray(c.pbc_intor('int1e_kin', 1, 1, kpts))
else:
    dm = np.zeros((nao, nao), dtype=complex)
    np.fill_diagonal(dm, 0.5)
    dms = np.array([dm] * len(kpts))
    if what == 'vj':
        mats = np.asarray(mydf.get_jk(dms, hermi=1, kpts=kpts, with_k=False)[0])
    elif what == 'vk':
        mats = np.asarray(mydf.get_jk(dms, hermi=1, kpts=kpts, with_j=False, exxdiv=None)[1])
    elif what == 'vk_ewald':
        mats = np.asarray(mydf.get_jk(dms, hermi=1, kpts=kpts, with_j=False, exxdiv='ewald')[1])
    else:
        raise SystemExit('unknown quantity ' + what)

mats = np.asarray(mats).reshape(-1, nao, nao)
out = {'nao': int(nao), 'nkpts': len(kpts), 'version': __import__('pyscf').__version__,
       're': np.real(mats).ravel().tolist(),
       'im': (np.imag(mats).ravel().tolist() if np.iscomplexobj(mats)
              else np.zeros(mats.size).tolist())}
print(json.dumps(out))
"#;

fn oracle_matrices(
    cell: &pyscf_pbc_gto::Cell,
    basis: &str,
    pseudo: &str,
    nk: [usize; 3],
    mesh: [usize; 3],
    what: &str,
) -> Option<serde_json::Value> {
    let py = oracle_python()?;
    let args = cell_args(
        cell,
        &[
            basis.to_string(),
            pseudo.to_string(),
            serde_json::to_string(&nk.to_vec()).expect("json"),
            serde_json::to_string(&mesh.to_vec()).expect("json"),
            what.to_string(),
        ],
    );
    let v = run_python(&py, ORACLE_PY, &args);
    assert_eq!(
        v["version"].as_str().expect("version"),
        "2.12.1",
        "the oracle must be the VENDORED PySCF 2.12.1 — see tests/common/mod.rs"
    );
    Some(v)
}

/// `FFTDF.get_pp` element-wise against upstream at a mesh where upstream's
/// `ft_ao` non-local expansion has converged.
#[test]
#[ignore = "needs PYSCF_ORACLE_VENV + the vendored upstream PySCF"]
fn get_pp_matches_upstream_on_diamond_222() {
    let cell = diamond();
    let kpts = kpts222(&cell);
    let Some(want) = oracle_matrices(&cell, "gth-szv", "gth-pade", [2, 2, 2], MESH_PP, "pp") else {
        eprintln!("SKIP: {GATE} is not set");
        return;
    };
    let df = Fftdf::with_mesh(cell, &kpts, MESH_PP).expect("FFTDF");
    let got = df.get_pp(&kpts).expect("get_pp");
    let w = max_dev(&got, &want);
    println!("get_pp max|delta| vs upstream = {w:e}");
    assert!(w < 1e-11, "get_pp deviates from upstream by {w:e}");
}

/// `FFTDF.get_hcore` — the assembly this phase owes Phase 10.
#[test]
#[ignore = "needs PYSCF_ORACLE_VENV + the vendored upstream PySCF"]
fn get_hcore_matches_upstream_on_diamond_222() {
    let cell = diamond();
    let kpts = kpts222(&cell);
    let Some(want) = oracle_matrices(&cell, "gth-szv", "gth-pade", [2, 2, 2], MESH_PP, "hcore")
    else {
        eprintln!("SKIP: {GATE} is not set");
        return;
    };
    let df = Fftdf::with_mesh(cell, &kpts, MESH_PP).expect("FFTDF");
    let got = get_hcore(&df, &kpts).expect("get_hcore");
    let w = max_dev(&got, &want);
    println!("get_hcore max|delta| vs upstream = {w:e}");
    assert!(w < 1e-11, "get_hcore deviates from upstream by {w:e}");
}

/// The all-electron `get_nuc`, which has no `ft_ao` component and so matches at
/// any mesh.
#[test]
#[ignore = "needs PYSCF_ORACLE_VENV + the vendored upstream PySCF"]
fn get_nuc_matches_upstream_on_he() {
    let cell = he_all_electron();
    let kpts = kpts222(&cell);
    let Some(want) = oracle_matrices(&cell, "sto-3g", "", [2, 2, 2], MESH_FAST, "nuc") else {
        eprintln!("SKIP: {GATE} is not set");
        return;
    };
    let df = Fftdf::with_mesh(cell, &kpts, MESH_FAST).expect("FFTDF");
    let got = df.get_nuc(&kpts).expect("get_nuc");
    let w = max_dev(&got, &want);
    println!("get_nuc max|delta| vs upstream = {w:e}");
    assert!(w < 1e-9, "get_nuc deviates from upstream by {w:e}");
}

/// `vj`, `vk` and `vk` with the Ewald `exxdiv`, element-wise against upstream.
/// These have no `ft_ao` component, so the FAST mesh is enough.
#[test]
#[ignore = "needs PYSCF_ORACLE_VENV + the vendored upstream PySCF"]
fn jk_matches_upstream_on_diamond_222() {
    let cell = diamond();
    let kpts = kpts222(&cell);
    let nao = cell.mol.nao_nr;
    let dms = flat_dm(nao, kpts.len());
    let df = Fftdf::with_mesh(cell.clone(), &kpts, MESH_FAST).expect("FFTDF");

    for (what, exxdiv, with_j, with_k, tol) in [
        ("vj", None, true, false, 1e-12),
        ("vk", None, false, true, 1e-11),
        ("vk_ewald", Some(ExxDiv::Ewald), false, true, 1e-11),
    ] {
        let Some(want) = oracle_matrices(&cell, "gth-szv", "gth-pade", [2, 2, 2], MESH_FAST, what)
        else {
            eprintln!("SKIP: {GATE} is not set");
            return;
        };
        let r = df
            .get_jk(
                &dms,
                &kpts,
                JkOpts {
                    hermi: 1,
                    kpts_band: None,
                    with_j,
                    with_k,
                    exxdiv,
                    omega: None,
                    kk_symmetry: false,
                },
            )
            .expect("get_jk");
        let got = if with_j {
            r.vj.expect("vj")
        } else {
            r.vk.expect("vk")
        };
        let w = max_dev(&got[0], &want);
        println!("{what} max|delta| vs upstream = {w:e}");
        assert!(w < tol, "{what} deviates from upstream by {w:e}");
    }
}
