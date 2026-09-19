//! `GDF.get_jk(omega)` / `MDF.get_jk(omega)` against upstream — plan 20-05.
//!
//! Both entry points refused a set `omega` (`NotYetImplemented { phase: 14 }`)
//! from the day they were written, because an ignored `omega` hands an RSH
//! functional a plausible full-range exchange matrix. Plan 20-05 replaced the
//! refusals with upstream's RSH branch, mirrored branch for branch
//! (`df.py:461-479`, `mdf.py:181-199`):
//!
//! | request | upstream route | this file's case |
//! |---|---|---|
//! | `omega > 0` (long range), 3-D | swap to `AFTDF` at `cutoff_to_mesh(estimate_ke_cutoff_for_omega(cell, omega))`, `cell.omega = omega` | `gdf_lr`, `mdf_lr` |
//! | `omega < 0` (short range) | `self.range_coulomb(omega)` → `_RSGDFBuilder` / `_RSMDFBuilder` on a cell with `cell.omega = omega` | `gdf_sr`, `mdf_sr` |
//!
//! Each case is gated against upstream's OWN answer for that request, never
//! against plain GDF/MDF: upstream's comment says outright that the AFT swap
//! "may cause small difference to the GDF integrator".
//!
//! # The floor
//!
//! **1.353e-08** — the Phase-14 Gate-3 floor (`20-.../measurements/README.md`
//! §1 row 5), on `|dvj|` and `|dvk|`, the worst element over every k-point, for
//! `exxdiv = None` AND `exxdiv = 'ewald'` (the latter is what `KRKS` passes, and
//! it exercises the attenuated Madelung constant the cell's `omega` selects).
//!
//! The density matrix is generated in Rust and handed to the oracle as literal
//! numbers (same contract as `band_kpoints.rs`).

mod common;

use pyscf_algebra::CTensor;
use pyscf_pbc_df::df_jk::KMats;
use pyscf_pbc_df::gdf::Gdf;
use pyscf_pbc_df::mdf::Mdf;
use pyscf_pbc_df::traits::{JkOpts, PeriodicDf};
use pyscf_pbc_gto::ExxDiv;

/// Phase-14 Gate-3 floor — `measurements/README.md` §1 row 5.
const FLOOR: f64 = 1.353e-08;

/// CAM-B3LYP's `omega` — long range as `+OMEGA`, short range as `-OMEGA`.
const OMEGA: f64 = 0.33;

fn kpts(cell: &pyscf_pbc_gto::Cell, km: [usize; 3]) -> Vec<[f64; 3]> {
    pyscf_pbc_gto::kpts_mesh::make_kpts(cell, km, false, true, None).expect("kpts")
}

/// Hermitian, positive, deterministic — `band_kpoints.rs`'s `model_dm`.
fn model_dm(nao: usize, nkpts: usize) -> KMats {
    (0..nkpts)
        .map(|k| {
            let mut m = CTensor::zeros(nao * nao);
            for p in 0..nao {
                for q in 0..nao {
                    let v =
                        0.3 / (1.0 + (p as f64 - q as f64).abs()) + if p == q { 1.0 } else { 0.0 };
                    m.re[p * nao + q] = v * (1.0 + 0.1 * k as f64);
                }
            }
            m
        })
        .collect()
}

const SCRIPT: &str = r#"
import json
import sys
import numpy as np
import pyscf
from pyscf.pbc import gto as pgto
from pyscf.pbc.df import df as pbcdf
from pyscf.pbc.df import mdf as pbcmdf

a = json.loads(sys.argv[1])
xyz = json.loads(sys.argv[2])
sym = json.loads(sys.argv[3])
kpts = np.array(json.loads(sys.argv[4]))
dm_flat = json.loads(sys.argv[5])
route = sys.argv[6]
omega = float(sys.argv[7])

cell = pgto.Cell()
cell.a = a
cell.atom = [(s, x) for s, x in zip(sym, xyz)]
cell.basis = 'sto-3g'
cell.unit = 'Bohr'
cell.verbose = 0
cell.build()

nao = cell.nao_nr()
dm = np.asarray(dm_flat).reshape(len(kpts), nao, nao)

# Upstream's defaults: GDF._prefer_ccdf = False, MDF._prefer_ccdf = False.
mydf = pbcdf.GDF(cell, kpts) if route == 'gdf' else pbcmdf.MDF(cell, kpts)
out = {'version': pyscf.__version__}
for tag, exxdiv in (('none', None), ('ewald', 'ewald')):
    vj, vk = mydf.get_jk(dm, hermi=1, kpts=kpts, with_j=True, with_k=True,
                         omega=omega, exxdiv=exxdiv)
    out[tag] = {
        'vj_re': vj.real.ravel().tolist(), 'vj_im': vj.imag.ravel().tolist(),
        'vk_re': vk.real.ravel().tolist(), 'vk_im': vk.imag.ravel().tolist(),
    }
# Diagnostics only: the mesh each route ran on.
if omega > 0:
    from pyscf.pbc.df import aft
    out['mesh'] = [int(x) for x in cell.cutoff_to_mesh(
        aft.estimate_ke_cutoff_for_omega(cell, omega))]
else:
    rsh = mydf._rsh_df['%.6f' % omega]
    out['mesh'] = None if rsh.mesh is None else [int(x) for x in rsh.mesh]
print(json.dumps(out))
"#;

fn worst(got: &[KMats], want: &serde_json::Value, re: &str, im: &str) -> f64 {
    let pull = |key: &str| -> Vec<f64> {
        want[key]
            .as_array()
            .unwrap_or_else(|| panic!("oracle payload has no {key} array"))
            .iter()
            .map(|v| v.as_f64().expect("f64"))
            .collect()
    };
    let (wr, wi) = (pull(re), pull(im));
    let mut w = 0.0_f64;
    let mut p = 0usize;
    for m in &got[0] {
        for i in 0..m.len() {
            w = w.max((wr[p] - m.re[i]).abs());
            w = w.max((wi[p] - m.im[i]).abs());
            p += 1;
        }
    }
    assert_eq!(p, wr.len(), "shape mismatch vs upstream");
    w
}

/// Run one route at one `omega` through `PeriodicDf::get_jk`, compare both
/// `exxdiv` settings against upstream, print, and return the worst deviations.
fn gate(route: &str, df: &dyn PeriodicDf, omega: f64) -> (f64, f64) {
    let Some(py) = common::oracle_python() else {
        eprintln!("{} unset — skipping the upstream oracle", common::GATE);
        return (0.0, 0.0);
    };
    let cell = df.cell().clone();
    let k = df.kpts().to_vec();
    let nao = cell.mol.nao_nr;
    let dms = vec![model_dm(nao, k.len())];
    let dm_json = serde_json::to_string(&dms[0].iter().map(|m| m.re.clone()).collect::<Vec<_>>())
        .expect("dm json");

    let want = common::run_python(
        &py,
        SCRIPT,
        &common::cell_args(
            &cell,
            &[
                serde_json::to_string(&k).expect("kpts json"),
                dm_json,
                route.to_string(),
                format!("{omega:.17e}"),
            ],
        ),
    );
    assert_eq!(want["version"].as_str(), Some("2.12.1"));

    let (mut wj, mut wk) = (0.0_f64, 0.0_f64);
    for (tag, exxdiv) in [("none", None), ("ewald", Some(ExxDiv::Ewald))] {
        let t0 = std::time::Instant::now();
        let res = df
            .get_jk(
                &dms,
                &k,
                JkOpts {
                    exxdiv,
                    omega: Some(omega),
                    ..JkOpts::hermitian()
                },
            )
            .unwrap_or_else(|e| panic!("{route} get_jk(omega={omega}) exxdiv={tag}: {e}"));
        let dt = t0.elapsed().as_secs_f64();
        let dj = worst(res.vj.as_ref().expect("vj"), &want[tag], "vj_re", "vj_im");
        let dk = worst(res.vk.as_ref().expect("vk"), &want[tag], "vk_re", "vk_im");
        eprintln!(
            "{route} omega={omega:+} exxdiv={tag}: |dvj|={dj:.3e} |dvk|={dk:.3e} \
             (upstream mesh {}, rust {dt:.2}s)",
            want["mesh"]
        );
        wj = wj.max(dj);
        wk = wk.max(dk);
    }
    (wj, wk)
}

fn he_222() -> (pyscf_pbc_gto::Cell, Vec<[f64; 3]>) {
    let cell = common::he_all_electron();
    let k = kpts(&cell, [2, 2, 2]);
    (cell, k)
}

/// **Oracle.** `GDF.get_jk(omega > 0)` — the AFTDF swap (`df.py:470-474`).
#[test]
#[ignore = "T2: needs PYSCF_ORACLE_VENV + the vendored upstream PySCF"]
fn gdf_lr_matches_upstream() {
    let (cell, k) = he_222();
    let df = Gdf::new(cell, &k);
    let (dj, dk) = gate("gdf", &df, OMEGA);
    assert!(dj < FLOOR, "GDF LR vj vs upstream: {dj:e} >= {FLOOR:e}");
    assert!(dk < FLOOR, "GDF LR vk vs upstream: {dk:e} >= {FLOOR:e}");
}

/// **Oracle.** `GDF.get_jk(omega < 0)` — `range_coulomb` + `_RSGDFBuilder`
/// on an attenuated cell (`df.py:476-479`, `rsdf_builder.py:83-90`).
#[test]
#[ignore = "T1: needs PYSCF_ORACLE_VENV + the vendored upstream PySCF"]
fn gdf_sr_matches_upstream() {
    let (cell, k) = he_222();
    let df = Gdf::new(cell, &k);
    let (dj, dk) = gate("gdf", &df, -OMEGA);
    assert!(dj < FLOOR, "GDF SR vj vs upstream: {dj:e} >= {FLOOR:e}");
    assert!(dk < FLOOR, "GDF SR vk vs upstream: {dk:e} >= {FLOOR:e}");
}

/// **Oracle.** `MDF.get_jk(omega > 0)` — the AFTDF swap (`mdf.py:190-194`).
#[test]
#[ignore = "T2: needs PYSCF_ORACLE_VENV + the vendored upstream PySCF"]
fn mdf_lr_matches_upstream() {
    let (cell, k) = he_222();
    let mut df = Mdf::new(cell, &k);
    df.prefer_ccdf = false;
    let (dj, dk) = gate("mdf", &df, OMEGA);
    assert!(dj < FLOOR, "MDF LR vj vs upstream: {dj:e} >= {FLOOR:e}");
    assert!(dk < FLOOR, "MDF LR vk vs upstream: {dk:e} >= {FLOOR:e}");
}

/// **Oracle.** `MDF.get_jk(omega < 0)` — `range_coulomb` + `_RSMDFBuilder`
/// on an attenuated cell, plus the attenuated plane-wave half. The parent MDF
/// is unbuilt on both sides, so the copy's mesh is derived from `omega`
/// (`rsdf_builder.py:145-147`) identically in both.
#[test]
#[ignore = "T2: needs PYSCF_ORACLE_VENV + the vendored upstream PySCF"]
fn mdf_sr_matches_upstream() {
    let (cell, k) = he_222();
    let mut df = Mdf::new(cell, &k);
    df.prefer_ccdf = false;
    let (dj, dk) = gate("mdf", &df, -OMEGA);
    assert!(dj < FLOOR, "MDF SR vj vs upstream: {dj:e} >= {FLOOR:e}");
    assert!(dk < FLOOR, "MDF SR vk vs upstream: {dk:e} >= {FLOOR:e}");
}
