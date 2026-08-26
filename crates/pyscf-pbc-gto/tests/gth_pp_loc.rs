//! Plan 10-05 — the GTH local pseudopotential.
//!
//! Two halves, gated separately:
//!
//! * the closed-form G-space factors (`get_gth_vlocG`, `get_gth_vlocG_part1`,
//!   `get_alphas`, `get_coulG`) — compared elementwise against upstream at
//!   1e-12 on a `[5,5,5]` mesh;
//! * the real-space short-range matrix `get_pp_loc_part2` — the double lattice
//!   sum over 3-centre `origk` integrals, compared against upstream at 1e-9.

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::pseudo::{
    fake_cell_vloc, get_alphas, get_coulg, get_gth_vlocg, get_gth_vlocg_part1, get_pp_loc_part2,
    get_pp_loc_part2_gamma,
};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs, gv::get_gv};
use std::f64::consts::PI;
use std::path::PathBuf;
use std::process::Command;

const MESH: [usize; 3] = [5, 5, 5];

fn diamond() -> Cell {
    let h = 3.37032;
    let q = 1.68516;
    bohr_cell(
        [[0.0, h, h], [h, 0.0, h], [h, h, 0.0]],
        vec![("C".into(), [0.0, 0.0, 0.0]), ("C".into(), [q, q, q])],
    )
}

fn lif() -> Cell {
    let h = 3.80763;
    bohr_cell(
        [[0.0, h, h], [h, 0.0, h], [h, h, 0.0]],
        vec![("Li".into(), [0.0, 0.0, 0.0]), ("F".into(), [h, h, h])],
    )
}

fn he_fcc() -> Cell {
    let h = 2.834589;
    bohr_cell(
        [[0.0, h, h], [h, 0.0, h], [h, h, 0.0]],
        vec![("He".into(), [0.0, 0.0, 0.0])],
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

// ---------------------------------------------------------------------------
// Oracle-free gates
// ---------------------------------------------------------------------------

/// `coulG = 4π/G²` with the `G = 0` singularity zeroed.
#[test]
fn coulg_is_the_bare_coulomb_kernel() {
    let cell = diamond();
    let gv = get_gv(&cell, Some(MESH)).expect("Gv");
    let c = get_coulg(&cell, &gv).expect("coulG");

    assert_eq!(c.len(), gv.len());
    for (g, v) in gv.iter().zip(c.iter()) {
        let g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
        if g2 == 0.0 {
            assert_eq!(*v, 0.0, "coulG must be zeroed at G = 0");
        } else {
            approx::assert_relative_eq!(*v, 4.0 * PI / g2, epsilon = 1e-14, max_relative = 1e-14);
        }
    }
}

/// `alphas = −V_loc(G = 0)`, and `V_loc(0)` must equal the closed form
/// `2π Z r_loc² + (2π)^{3/2} r_loc³ (C1 + 3C2 + 15C3 + 105C4)` — the `x = 0`
/// limit of the polynomial, stated independently in PBC-MASTER-PLAN plan 10-05.
#[test]
fn alphas_match_the_closed_form_g0_limit() {
    let cell = diamond();
    let alphas = get_alphas(&cell).expect("alphas");
    assert_eq!(alphas.len(), 2);

    let pp = cell.atom_pseudo(0).expect("carbon has a PP");
    let z = cell.atom_charges()[0] as f64;
    let c = &pp.local_coeffs;
    let poly = c.first().copied().unwrap_or(0.0)
        + 3.0 * c.get(1).copied().unwrap_or(0.0)
        + 15.0 * c.get(2).copied().unwrap_or(0.0)
        + 105.0 * c.get(3).copied().unwrap_or(0.0);
    let want = 2.0 * PI * pp.rloc * pp.rloc * z + (2.0 * PI).powf(1.5) * pp.rloc.powi(3) * poly;

    approx::assert_relative_eq!(alphas[0], want, epsilon = 1e-13, max_relative = 1e-13);
    approx::assert_relative_eq!(alphas[0], alphas[1], epsilon = 1e-15);
}

/// `get_gth_vlocG` = part 1 plus the polynomial, and the two agree away from the
/// polynomial (an all-electron atom, where part 2 contributes nothing).
#[test]
fn vlocg_splits_into_part1_plus_the_polynomial() {
    let cell = diamond();
    let gv = get_gv(&cell, Some(MESH)).expect("Gv");
    let p1 = get_gth_vlocg_part1(&cell, &gv).expect("part1");
    let full = get_gth_vlocg(&cell, &gv).expect("full");
    assert_eq!(p1.len(), cell.mol.natm * gv.len());
    assert_eq!(full.len(), p1.len());

    let pp = cell.atom_pseudo(0).expect("carbon has a PP");
    let two_pi_32 = (2.0 * PI).powf(1.5);
    for (g, gg) in gv.iter().enumerate() {
        let g2 = gg[0] * gg[0] + gg[1] * gg[1] + gg[2] * gg[2];
        let x = g2 * pp.rloc * pp.rloc;
        let cfacs = pp.local_coeffs[0] + pp.local_coeffs[1] * (3.0 - x);
        let want = p1[g] - two_pi_32 * pp.rloc.powi(3) * (-0.5 * x).exp() * cfacs;
        approx::assert_relative_eq!(full[g], want, epsilon = 1e-13, max_relative = 1e-13);
    }
}

/// The `fake_cell_vloc` auxiliary expansion: carbon's `gth-pade` has `nexp = 2`,
/// so `cn = 1, 2` produce one Gaussian per atom and `cn = 3, 4` produce none.
#[test]
fn fake_cell_vloc_layout() {
    let cell = diamond();
    let pp = cell.atom_pseudo(0).expect("carbon has a PP");
    let alpha = 0.5 / (pp.rloc * pp.rloc);

    let cn0 = fake_cell_vloc(&cell, 0).expect("cn=0");
    assert_eq!(cn0.len(), 2, "the erf term exists for every atom");
    approx::assert_relative_eq!(cn0[0].alpha, alpha, epsilon = 1e-14);

    for cn in 1..=2usize {
        let aux = fake_cell_vloc(&cell, cn).expect("cn");
        assert_eq!(aux.len(), 2, "cn={cn} covers both carbons");
        for (ia, a) in aux.iter().enumerate() {
            assert_eq!(a.atom, ia);
            approx::assert_relative_eq!(a.alpha, alpha, epsilon = 1e-14);
            // pp_int.py:554 — coeff = C_cn / rloc^(2cn-2) / half_sph_norm.
            let want = pp.local_coeffs[cn - 1]
                / pp.rloc.powi(2 * cn as i32 - 2)
                / pyscf_pbc_gto::pseudo::HALF_SPH_NORM;
            approx::assert_relative_eq!(a.coeff, want, epsilon = 1e-12, max_relative = 1e-12);
        }
    }
    for cn in 3..=4usize {
        assert!(
            fake_cell_vloc(&cell, cn).expect("cn").is_empty(),
            "carbon's gth-pade has only C1 and C2"
        );
    }
    assert!(fake_cell_vloc(&cell, 5).is_err(), "cn > 4 must be refused");
}

/// `V2` is real and symmetric, and its diagonal is negative (the `C_n` terms of
/// a GTH potential are attractive at short range).
#[test]
fn part2_is_symmetric_and_attractive() {
    let cell = diamond();
    let v = get_pp_loc_part2_gamma(&cell).expect("part2");
    let nao = cell.mol.nao_nr;
    assert_eq!(v.len(), nao * nao);

    for i in 0..nao {
        assert!(
            v[i + i * nao] < 0.0,
            "V2[{i},{i}] = {} is not attractive",
            v[i + i * nao]
        );
        for j in 0..nao {
            approx::assert_abs_diff_eq!(v[i + j * nao], v[j + i * nao], epsilon = 1e-12);
        }
    }
}

/// A k-point request must be refused rather than silently returning the gamma
/// matrix.
#[test]
fn part2_refuses_k_points() {
    let cell = he_fcc();
    let err = get_pp_loc_part2(&cell, &[[0.1, 0.0, 0.0]]).expect_err("must refuse");
    assert!(
        matches!(
            err,
            pyscf_core::PyscfRsError::NotYetImplemented { phase: 13, .. }
        ),
        "unexpected error: {err}"
    );
    // …and the gamma point must still go through.
    get_pp_loc_part2(&cell, &[[0.0; 3]]).expect("gamma is supported");
}

/// An all-electron cell has no local pseudopotential at all.
#[test]
fn all_electron_cell_has_no_part2() {
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
    let v = get_pp_loc_part2_gamma(&cell).expect("part2");
    assert!(v.iter().all(|x| *x == 0.0));
}

// ---------------------------------------------------------------------------
// Upstream gates
// ---------------------------------------------------------------------------

const GATE: &str = "PYSCF_ORACLE_VENV";

const ORACLE_PY: &str = r#"
import json, sys
import numpy as np
from pyscf.pbc import gto
from pyscf.pbc.gto.pseudo import pp, pp_int
from pyscf.pbc import tools

a_json, xyz_json, sym_json, mesh_json = sys.argv[1:5]
c = gto.Cell()
c.a = json.loads(a_json)
c.atom = [(s, tuple(r)) for s, r in zip(json.loads(sym_json), json.loads(xyz_json))]
c.basis = 'gth-szv'
c.pseudo = 'gth-pade'
c.unit = 'Bohr'
c.verbose = 0
c.build()
mesh = json.loads(mesh_json)
Gv = c.get_Gv(mesh)
out = {
    'nao': int(c.nao_nr()),
    'coulG': tools.get_coulG(c, Gv=Gv).ravel().tolist(),
    'vlocG_part1': pp_int.get_gth_vlocG_part1(c, Gv).ravel().tolist(),
    'vlocG': pp.get_gth_vlocG(c, Gv).ravel().tolist(),
    'alphas': np.asarray(pp.get_alphas(c)).ravel().tolist(),
    'part2': np.asarray(pp_int.get_pp_loc_part2(c)).ravel(order='F').tolist(),
}
print(json.dumps(out))
"#;

#[test]
#[ignore = "needs PYSCF_ORACLE_VENV + an upstream PySCF"]
fn g_space_factors_match_upstream_on_diamond() {
    compare_g_space(&diamond(), 1e-12);
}

#[test]
#[ignore = "needs PYSCF_ORACLE_VENV + an upstream PySCF"]
fn g_space_factors_match_upstream_on_lif() {
    compare_g_space(&lif(), 1e-12);
}

#[test]
#[ignore = "needs PYSCF_ORACLE_VENV + an upstream PySCF"]
fn part2_matches_upstream_on_diamond() {
    compare_part2(&diamond(), 1e-9);
}

/// LiF exercises a GENERAL CONTRACTION: `Li`/`gth-szv` is one `s` shell with two
/// contractions, the only such basis among the PBC-MASTER-PLAN §9.2 elements.
///
/// It was blocked until 2026-08-26 by a cintx bug — `int3c1e_r{2,4,6}_origk`
/// panicked on `nctr > 1` — and `cn = 2, 3, 4` (both Li, `nexp = 4`, and F,
/// `nexp = 2`, reach them) are exactly the terms that need those operators.
/// cintx has since fixed it; `tests/cintx_moment_weighted_available.rs` pins the
/// corrected kernels against libcint, and this is the end-to-end gate.
#[test]
#[ignore = "needs PYSCF_ORACLE_VENV + an upstream PySCF"]
fn part2_matches_upstream_on_lif() {
    compare_part2(&lif(), 1e-9);
}

fn compare_g_space(cell: &Cell, tol: f64) {
    let Some(want) = oracle(cell) else { return };
    let gv = get_gv(cell, Some(MESH)).expect("Gv");

    check(
        "coulG",
        &floats(&want, "coulG"),
        &get_coulg(cell, &gv).expect("coulG"),
        tol,
    );
    check(
        "vlocG_part1",
        &floats(&want, "vlocG_part1"),
        &get_gth_vlocg_part1(cell, &gv).expect("part1"),
        tol,
    );
    check(
        "vlocG",
        &floats(&want, "vlocG"),
        &get_gth_vlocg(cell, &gv).expect("vlocG"),
        tol,
    );
    check(
        "alphas",
        &floats(&want, "alphas"),
        &get_alphas(cell).expect("alphas"),
        tol,
    );
}

fn compare_part2(cell: &Cell, tol: f64) {
    let Some(want) = oracle(cell) else { return };
    assert_eq!(want["nao"].as_u64().unwrap() as usize, cell.mol.nao_nr);
    let got = get_pp_loc_part2_gamma(cell).expect("part2");
    check("get_pp_loc_part2", &floats(&want, "part2"), &got, tol);
}

fn check(name: &str, want: &[f64], got: &[f64], tol: f64) {
    assert_eq!(
        want.len(),
        got.len(),
        "{name}: length {} vs {}",
        got.len(),
        want.len()
    );
    let mut worst = 0.0_f64;
    let mut at = 0usize;
    for (i, (w, g)) in want.iter().zip(got.iter()).enumerate() {
        let d = (w - g).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    println!("{name}: max |delta| = {worst:e} at element {at}");
    assert!(
        worst < tol,
        "{name} differs from upstream by {worst:e} at element {at} \
         (upstream {}, got {})",
        want[at],
        got[at]
    );
}

fn oracle(cell: &Cell) -> Option<serde_json::Value> {
    let py = oracle_python()?;
    let a: Vec<Vec<f64>> = cell.a.iter().map(|r| r.to_vec()).collect();
    let xyz: Vec<Vec<f64>> = cell.mol.atom_coords().iter().map(|r| r.to_vec()).collect();
    let sym: Vec<String> = cell.mol._atom.iter().map(|(s, _)| s.clone()).collect();
    Some(run_python(
        &py,
        ORACLE_PY,
        &[
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&xyz).unwrap(),
            serde_json::to_string(&sym).unwrap(),
            serde_json::to_string(&MESH.to_vec()).unwrap(),
        ],
    ))
}

fn floats(v: &serde_json::Value, key: &str) -> Vec<f64> {
    v[key]
        .as_array()
        .unwrap_or_else(|| panic!("oracle field {key} is not an array"))
        .iter()
        .map(|x| x.as_f64().expect("float"))
        .collect()
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
        eprintln!("SKIP: {GATE} is not set — upstream oracle not run");
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
    let path =
        std::env::temp_dir().join(format!("gth_pp_loc_oracle_{}_{n}.py", std::process::id()));
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
