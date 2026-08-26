//! Plan 10-06 — the GTH non-local pseudopotential `V_nl`.

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::pseudo::{fake_cell_vnl, get_pp_nl};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs, kpts_mesh::make_kpts_default};
use std::path::PathBuf;
use std::process::Command;

fn diamond() -> Cell {
    let h = 3.37032;
    let q = 1.68516;
    bohr_cell(
        [[0.0, h, h], [h, 0.0, h], [h, h, 0.0]],
        vec![("C".into(), [0.0, 0.0, 0.0]), ("C".into(), [q, q, q])],
    )
}

/// Silicon: `gth-pade` gives Si a 2x2 `h^0` block, so this is the system that
/// exercises `nproj = 2` — the `int1e_r2_origi` half-overlap and the
/// [`pyscf_pbc_gto::pseudo::PLI_FAC`] rescaling.
fn si() -> Cell {
    let h = 5.13165;
    let q = 2.565825;
    bohr_cell(
        [[0.0, h, h], [h, 0.0, h], [h, h, 0.0]],
        vec![("Si".into(), [0.0, 0.0, 0.0]), ("Si".into(), [q, q, q])],
    )
}

fn bohr_cell(a: [[f64; 3]; 3], atoms: Vec<(String, [f64; 3])>) -> Cell {
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(atoms),
            basis: BasisInput::Name("gth-szv".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix(a),
        pseudo: Some("gth-pade".into()),
        ..Default::default()
    })
    .expect("reference cell must build")
}

/// The projector basis: diamond's carbon has ONE `l = 0` channel with one
/// projector, so only rank 0 exists, one shell per atom.
#[test]
fn fake_cell_vnl_layout_for_carbon() {
    let cell = diamond();
    let fake = fake_cell_vnl(&cell).expect("fake_cell_vnl");

    assert_eq!(fake.blocks.len(), 2, "one l=0 channel per carbon");
    for (ia, b) in fake.blocks.iter().enumerate() {
        assert_eq!(b.atom, ia);
        assert_eq!(b.l, 0);
        assert_eq!(b.dim, 1);
        // nproj = 1 -> PLI_FAC[0][0] = 1 and rl^0 = 1, so h is unrescaled.
        approx::assert_abs_diff_eq!(b.h[0], 9.52284179, epsilon = 1e-12);
    }

    let c0 = fake.cells[0].as_ref().expect("rank 0 must exist");
    assert_eq!(c0.mol.nbas, 2);
    assert_eq!(c0.mol.nao_nr, 2, "two s projectors -> two AOs");
    // rcut is INHERITED, not re-estimated from the compact projector basis.
    assert_eq!(c0.rcut, cell.rcut);
    assert!(fake.cells[1].is_none(), "carbon has no second projector");
    assert!(fake.cells[2].is_none());
}

/// Silicon reaches rank 1, and its `h^0` block is rescaled by `PLI_FAC`.
#[test]
fn fake_cell_vnl_layout_for_silicon() {
    let cell = si();
    let fake = fake_cell_vnl(&cell).expect("fake_cell_vnl");

    // Two atoms x two channels (l=0 with nproj 2, l=1 with nproj 1).
    assert_eq!(fake.blocks.len(), 4);
    let b = &fake.blocks[0];
    assert_eq!((b.atom, b.l, b.dim), (0, 0, 2));
    // Raw h^0 = [[5.90692831, -1.26189397], [-1.26189397, 3.25819622]],
    // rl = 0.42273813. fac = [1, 1/sqrt(3.75)/rl^2].
    let rl = 0.42273813_f64;
    let f1 = 1.0 / 3.75_f64.sqrt() / (rl * rl);
    approx::assert_abs_diff_eq!(b.h[0], 5.90692831, epsilon = 1e-12);
    approx::assert_abs_diff_eq!(b.h[1], -1.26189397 * f1, epsilon = 1e-10);
    approx::assert_abs_diff_eq!(b.h[2], b.h[1], epsilon = 1e-15);
    approx::assert_abs_diff_eq!(b.h[3], 3.25819622 * f1 * f1, epsilon = 1e-10);

    assert!(fake.cells[0].is_some());
    assert!(fake.cells[1].is_some(), "Si needs the r^2 half-overlap");
    assert_eq!(
        fake.cells[1].as_ref().unwrap().mol.nao_nr,
        2,
        "only the l=0 channel has a second projector"
    );
    assert!(fake.cells[2].is_none());
}

/// `V_nl` is Hermitian at every k and real at gamma.
#[test]
fn vnl_is_hermitian_and_real_at_gamma() {
    for cell in [diamond(), si()] {
        let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("make_kpts");
        let v = get_pp_nl(&cell, &kpts).expect("get_pp_nl");
        let nao = cell.mol.nao_nr;
        assert_eq!(v.len(), kpts.len());

        for (k, m) in v.iter().enumerate() {
            assert_eq!(m.len(), nao * nao);
            for i in 0..nao {
                for j in 0..nao {
                    let (a, b) = (m.re[i + j * nao], m.re[j + i * nao]);
                    let (c, d) = (m.im[i + j * nao], m.im[j + i * nao]);
                    assert!(
                        (a - b).abs() < 1e-12 && (c + d).abs() < 1e-12,
                        "V_nl(k={k}) not Hermitian at ({i},{j})"
                    );
                }
            }
        }
        let gamma = v[0].im.iter().fold(0.0_f64, |a, x| a.max(x.abs()));
        assert_eq!(gamma, 0.0, "gamma V_nl must be exactly real");
        let mag = v[0].re.iter().fold(0.0_f64, |a, x| a.max(x.abs()));
        assert!(mag > 1e-3, "V_nl is suspiciously small ({mag:e})");
    }
}

/// `V_nl(-k) == conj(V_nl(k))`.
#[test]
fn vnl_obeys_time_reversal_symmetry() {
    let cell = diamond();
    let k = [0.17, -0.05, 0.23];
    let v = get_pp_nl(&cell, &[k, [-k[0], -k[1], -k[2]]]).expect("get_pp_nl");
    let mut worst = 0.0_f64;
    for i in 0..v[0].len() {
        worst = worst.max((v[0].re[i] - v[1].re[i]).abs());
        worst = worst.max((v[0].im[i] + v[1].im[i]).abs());
    }
    assert!(worst < 1e-13, "V_nl(-k) != conj(V_nl(k)): {worst:e}");
}

/// A cell with no pseudopotential has no non-local part at all.
#[test]
fn all_electron_cell_has_no_vnl() {
    let h = 3.37032;
    let cell = Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("C".into(), [0.0, 0.0, 0.0])]),
            basis: BasisInput::Name("gth-szv".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        pseudo: None,
        ..Default::default()
    })
    .expect("builds");
    let v = get_pp_nl(&cell, &[[0.0; 3]]).expect("get_pp_nl");
    assert!(v[0].re.iter().all(|x| *x == 0.0));
    assert!(v[0].im.iter().all(|x| *x == 0.0));
}

// ---------------------------------------------------------------------------
// Upstream gate
// ---------------------------------------------------------------------------

const GATE: &str = "PYSCF_ORACLE_VENV";

const ORACLE_PY: &str = r#"
import json, sys
import numpy as np
from pyscf.pbc import gto
from pyscf.pbc.gto.pseudo import pp_int

a_json, xyz_json, sym_json, nk_json = sys.argv[1:5]
c = gto.Cell()
c.a = json.loads(a_json)
c.atom = [(s, tuple(r)) for s, r in zip(json.loads(sym_json), json.loads(xyz_json))]
c.basis = 'gth-szv'
c.pseudo = 'gth-pade'
c.unit = 'Bohr'
c.verbose = 0
c.build()
kpts = c.make_kpts(json.loads(nk_json))
v = np.asarray(pp_int.get_pp_nl(c, kpts))
out = {'nao': int(c.nao_nr()), 're': [], 'im': []}
for m in v:
    m = np.asarray(m)
    out['re'].append(m.real.ravel(order='F').tolist())
    out['im'].append(m.imag.ravel(order='F').tolist() if np.iscomplexobj(m)
                     else np.zeros(m.size).tolist())
print(json.dumps(out))
"#;

#[test]
#[ignore = "needs PYSCF_ORACLE_VENV + an upstream PySCF"]
fn vnl_matches_upstream_on_diamond_222() {
    compare_with_upstream(&diamond(), [2, 2, 2], 1e-11);
}

/// Silicon exercises the `nproj = 2` path (the `r²` half-overlap + `PLI_FAC`).
#[test]
#[ignore = "needs PYSCF_ORACLE_VENV + an upstream PySCF"]
fn vnl_matches_upstream_on_silicon_222() {
    compare_with_upstream(&si(), [2, 2, 2], 1e-11);
}

fn compare_with_upstream(cell: &Cell, nk: [usize; 3], tol: f64) {
    let Some(py) = oracle_python() else {
        eprintln!("SKIP: {GATE} is not set — upstream oracle not run");
        return;
    };
    let a: Vec<Vec<f64>> = cell.a.iter().map(|r| r.to_vec()).collect();
    let xyz: Vec<Vec<f64>> = cell.mol.atom_coords().iter().map(|r| r.to_vec()).collect();
    let sym: Vec<String> = cell.mol._atom.iter().map(|(s, _)| s.clone()).collect();

    let want = run_python(
        &py,
        ORACLE_PY,
        &[
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&xyz).unwrap(),
            serde_json::to_string(&sym).unwrap(),
            serde_json::to_string(&nk.to_vec()).unwrap(),
        ],
    );
    assert_eq!(want["nao"].as_u64().unwrap() as usize, cell.mol.nao_nr);

    let kpts = make_kpts_default(cell, nk).expect("make_kpts");
    let got = get_pp_nl(cell, &kpts).expect("get_pp_nl");

    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for k in 0..kpts.len() {
        let wre: Vec<f64> = want["re"][k]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let wim: Vec<f64> = want["im"][k]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        assert_eq!(wre.len(), got[k].len(), "element count differs at k={k}");
        for p in 0..wre.len() {
            scale = scale.max(wre[p].abs()).max(wim[p].abs());
            worst = worst.max((wre[p] - got[k].re[p]).abs());
            worst = worst.max((wim[p] - got[k].im[p]).abs());
        }
    }
    println!("get_pp_nl vs upstream: max |delta| = {worst:e} (max |V_nl| = {scale:e})");
    assert!(
        worst < tol,
        "get_pp_nl differs from upstream by {worst:e} (tolerance {tol:e})"
    );
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

fn oracle_python() -> Option<PathBuf> {
    let raw = std::env::var(GATE).ok()?.trim().to_string();
    if raw.is_empty() {
        return None;
    }
    let p = if matches!(raw.as_str(), "1" | "true" | "auto" | "yes") {
        workspace_root().join(".venv/bin/python")
    } else {
        let c = PathBuf::from(&raw);
        if c.is_dir() { c.join("bin/python") } else { c }
    };
    assert!(
        p.exists(),
        "{GATE} = {raw:?} resolved to {p:?}, which does not exist"
    );
    Some(p)
}

static SCRIPT_SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn run_python(py: &PathBuf, script: &str, args: &[String]) -> serde_json::Value {
    let n = SCRIPT_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("gth_pp_nl_oracle_{}_{n}.py", std::process::id()));
    std::fs::write(&path, script).expect("write oracle script");
    let out = Command::new(py)
        .arg(&path)
        .args(args)
        .current_dir(workspace_root())
        .output()
        .expect("spawn upstream python");
    let _ = std::fs::remove_file(&path);
    assert!(
        out.status.success(),
        "upstream oracle failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .rev()
        .find(|l| l.trim_start().starts_with('{'))
        .expect("oracle produced no JSON line");
    serde_json::from_str(line).expect("oracle JSON parses")
}
