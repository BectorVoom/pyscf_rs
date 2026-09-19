//! Root-cause diagnostic: compare dm-independent band inputs (hcore, overlap)
//! between Rust and upstream before blaming the eigensolver.
//!
//! Run: `PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-scf --release
//! --test band_inputs_diagnostic -- --ignored --nocapture`

mod common;

use common::{cell_args, he_all_electron, oracle_python, run_python};
use pyscf_pbc_df::{Fftdf, JkOpts, PeriodicDf};
use pyscf_pbc_gto::make_kpts_default;

const MESH: [usize; 3] = [15, 15, 15];
const NK: [usize; 3] = [2, 2, 2];

const INPUTS_PY: &str = r#"
import json, sys
import numpy as np
from pyscf.pbc import gto, scf

a_json, xyz_json, sym_json, basis, nk_json, mesh_json, kband_json = sys.argv[1:8]
c = gto.Cell()
c.a = json.loads(a_json)
c.atom = [(s, tuple(r)) for s, r in zip(json.loads(sym_json), json.loads(xyz_json))]
c.basis = basis
c.unit = 'Bohr'
c.verbose = 0
c.build()
kpts = c.make_kpts(json.loads(nk_json))
mf = scf.KRHF(c, kpts)
mf.with_df.mesh = json.loads(mesh_json)
kband = np.asarray(json.loads(kband_json), dtype=float)
hcore = mf.get_hcore(c, kband)
s1e = mf.get_ovlp(c, kband)
def pack(mats):
    mats = np.asarray(mats)
    return {'re': mats.real.ravel().tolist(), 'im': mats.imag.ravel().tolist(),
            'shape': list(mats.shape)}
print(json.dumps({
    'version': __import__('pyscf').__version__,
    'hcore': pack(hcore),
    's1e': pack(s1e),
}))
"#;

fn max_dev_mats(got: &[pyscf_algebra::CTensor], want: &serde_json::Value) -> (f64, usize, usize) {
    let re: Vec<f64> = want["re"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let im: Vec<f64> = want["im"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let mut worst = 0.0f64;
    let (mut total, mut bitwise) = (0usize, 0usize);
    let mut p = 0usize;
    for m in got {
        for i in 0..m.len() {
            total += 2;
            if m.re[i].to_bits() == re[p].to_bits() {
                bitwise += 1;
            }
            if m.im[i].to_bits() == im[p].to_bits() {
                bitwise += 1;
            }
            worst = worst.max((m.re[i] - re[p]).abs());
            worst = worst.max((m.im[i] - im[p]).abs());
            p += 1;
        }
    }
    (worst, bitwise, total)
}

#[test]
#[ignore = "diagnostic: needs PYSCF_ORACLE_VENV + vendored upstream PySCF"]
fn band_dm_independent_inputs_vs_upstream() {
    let Some(py) = oracle_python() else {
        eprintln!("SKIP: PYSCF_ORACLE_VENV is not set");
        return;
    };
    let cell = he_all_electron();
    let kband = cell
        .get_abs_kpts(&[[0.15, -0.07, 0.03], [-0.05, 0.11, 0.02]])
        .expect("scaled -> absolute k");
    let kpts = make_kpts_default(&cell, NK).expect("k-mesh");
    let df = Fftdf::with_mesh(cell.clone(), &kpts, MESH).expect("FFTDF");

    let hcore = pyscf_pbc_df::get_hcore(&df, &kband).expect("hcore");
    let s1e = pyscf_pbc_gto::get_ovlp_scf(&cell, &kband).expect("ovlp");

    let kband_json: Vec<Vec<f64>> = kband.iter().map(|k| k.to_vec()).collect();
    let want = run_python(
        &py,
        INPUTS_PY,
        &cell_args(
            &cell,
            &[
                "sto-3g".to_string(),
                serde_json::to_string(&NK.to_vec()).expect("json"),
                serde_json::to_string(&MESH.to_vec()).expect("json"),
                serde_json::to_string(&kband_json).expect("json"),
            ],
        ),
    );
    assert_eq!(want["version"].as_str(), Some("2.12.1"));

    let (dh, bh, th) = max_dev_mats(&hcore, &want["hcore"]);
    println!("hcore: {bh}/{th} bitwise identical, worst |delta| = {dh:e}");
    let (ds, bs, ts) = max_dev_mats(&s1e, &want["s1e"]);
    println!("s1e:   {bs}/{ts} bitwise identical, worst |delta| = {ds:e}");
}

const JK_PY: &str = r#"
import json, sys
import numpy as np
from pyscf.pbc import gto, scf

a_json, xyz_json, sym_json, basis, nk_json, mesh_json, kband_json = sys.argv[1:8]
c = gto.Cell()
c.a = json.loads(a_json)
c.atom = [(s, tuple(r)) for s, r in zip(json.loads(sym_json), json.loads(xyz_json))]
c.basis = basis
c.unit = 'Bohr'
c.verbose = 0
c.build()
kpts = c.make_kpts(json.loads(nk_json))
mf = scf.KRHF(c, kpts)
mf.with_df.mesh = json.loads(mesh_json)
kband = np.asarray(json.loads(kband_json), dtype=float)
# Identical model density by construction: 1x1 ones at all 8 sampling k-points.
dm = [np.ones((1, 1)) for _ in range(len(kpts))]
vj, vk = mf.get_jk(c, dm, kpts=kpts, kpts_band=kband)
def pack(mats):
    mats = np.asarray(mats)
    return {'re': mats.real.ravel().tolist(), 'im': mats.imag.ravel().tolist()}
print(json.dumps({
    'version': __import__('pyscf').__version__,
    'vj': pack(vj),
    'vk': pack(vk),
}))
"#;

#[test]
#[ignore = "diagnostic: needs PYSCF_ORACLE_VENV + vendored upstream PySCF"]
fn band_jk_same_density_vs_upstream() {
    let Some(py) = oracle_python() else {
        eprintln!("SKIP: PYSCF_ORACLE_VENV is not set");
        return;
    };
    let cell = he_all_electron();
    let kband = cell
        .get_abs_kpts(&[[0.15, -0.07, 0.03], [-0.05, 0.11, 0.02]])
        .expect("scaled -> absolute k");
    let kpts = make_kpts_default(&cell, NK).expect("k-mesh");
    let df = Fftdf::with_mesh(cell.clone(), &kpts, MESH).expect("FFTDF");

    // Same constant density on the Rust side: 1x1 ones at all sampling k-points.
    let one = pyscf_algebra::CTensor {
        re: vec![1.0],
        im: vec![0.0],
    };
    let dm: Vec<Vec<pyscf_algebra::CTensor>> =
        vec![(0..kpts.len()).map(|_| one.clone()).collect()];
    let r = df
        .get_jk(
            &dm,
            &kpts,
            JkOpts {
                hermi: 1,
                kpts_band: Some(&kband),
                with_j: true,
                with_k: true,
                exxdiv: Some(pyscf_pbc_gto::ExxDiv::Ewald),
                omega: None,
                kk_symmetry: false,
            },
        )
        .expect("jk");
    let (vj, vk) = (r.vj.expect("vj"), r.vk.expect("vk"));

    let kband_json: Vec<Vec<f64>> = kband.iter().map(|k| k.to_vec()).collect();
    let want = run_python(
        &py,
        JK_PY,
        &cell_args(
            &cell,
            &[
                "sto-3g".to_string(),
                serde_json::to_string(&NK.to_vec()).expect("json"),
                serde_json::to_string(&MESH.to_vec()).expect("json"),
                serde_json::to_string(&kband_json).expect("json"),
            ],
        ),
    );
    assert_eq!(want["version"].as_str(), Some("2.12.1"));

    let (dj, bj, tj) = max_dev_mats(&vj[0], &want["vj"]);
    println!("vj(band, same dm): {bj}/{tj} bitwise identical, worst |delta| = {dj:e}");
    let (dk, bk, tk) = max_dev_mats(&vk[0], &want["vk"]);
    println!("vk(band, same dm): {bk}/{tk} bitwise identical, worst |delta| = {dk:e}");
}
