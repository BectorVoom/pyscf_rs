//! Plan 09-09 — the venv-gated upstream oracle for the whole of Phase 9.
//!
//! Every test here is `#[ignore]`d and additionally short-circuits unless
//! `PYSCF_ORACLE_VENV` is set, so it is NEVER a hard CI dependency: a plain
//! `cargo test --workspace` never touches Python, and even
//! `cargo test -- --ignored` on a machine without an upstream venv prints a
//! skip line and passes.
//!
//! ```bash
//! # the repo's own venv (PySCF 2.12.1 lives in ./pyscf, ./.venv/bin/python runs it)
//! PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-gto --test oracle_phase9 -- --ignored
//!
//! # or point at any interpreter / venv directory that can `import pyscf`
//! PYSCF_ORACLE_VENV=/path/to/venv cargo test -p pyscf-pbc-gto --test oracle_phase9 -- --ignored
//! ```
//!
//! # What it compares
//!
//! For each of the five PBC-MASTER-PLAN §9.2 reference systems:
//!
//! | quantity | upstream | tolerance |
//! |---|---|---|
//! | `vol`, `rcut`, `mesh` | `cell.vol`, `cell.rcut`, `cell.mesh` | 1e-12 rel / exact ints |
//! | `b` | `cell.reciprocal_vectors()` | 1e-12 |
//! | `get_Gv` on `[5,5,5]` | `cell.get_Gv([5,5,5])` | 1e-12 element-wise |
//! | `get_SI` on `[5,5,5]` | `cell.get_SI(mesh=[5,5,5])` | 1e-12 element-wise |
//! | `get_lattice_Ls` | `cell.get_lattice_Ls()` | count EXACT + values 1e-12 |
//! | `make_kpts` | `cell.make_kpts([2,2,2])`, `([3,2,1])` | 1e-12 |
//! | `get_kconserv` | `kpts_helper.get_kconserv` | EXACT ints |
//! | `ewald()` | `cell.ewald()` | 1e-9 Ha |
//!
//! # Geometry is specified in BOHR
//!
//! `pyscf_core::Unit::Ang` is CODATA-2014 and upstream is CODATA-2010 — the
//! 4.951e-9 relative gap of plan 09-03. Comparing at 1e-12 therefore requires
//! both sides to start from bit-identical Bohr geometry, which is what
//! [`bohr_cell`] and the emitted Python both do (the same resolution plans 09-07
//! and 09-08 already use). The Angstrom conversion path is NOT left uncovered:
//! [`angstrom_lattices_match_upstream_within_the_codata_gap`] compares the §9.2
//! Angstrom builds against upstream's Angstrom builds and asserts the deviation
//! is exactly the unit gap and nothing more.
//!
//! `pseudo` is unset on both sides — this port has no GTH parser before plan
//! 10-01 (D-PBC-11), so `atom_charges()` is the all-electron `Z`.

mod common;

use common::ewald_reference::{EWALD_REFERENCES, EwaldReference};
use common::systems;
use pyscf_core::{PyscfRsError, Unit};
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::{
    ALattice, Cell, CellBuildArgs, ewald, get_kconserv, get_lattice_ls_default, get_si, make_kpts,
    make_kpts_default,
};
use serde_json::Value;
use std::path::PathBuf;
use std::process::Command;

/// The env var that arms this file. Absent -> every test skips.
const GATE: &str = "PYSCF_ORACLE_VENV";

/// The mesh every grid comparison uses. Odd and small, so `fftfreq` covers both
/// signs and the JSON stays readable.
const ORACLE_MESH: [usize; 3] = [5, 5, 5];

/// The Python emitted into a temp file and run by the gate interpreter.
///
/// It takes the system name, the Bohr lattice and the Bohr coordinates on the
/// command line, rebuilds the cell EXACTLY as [`bohr_cell`] does, and prints one
/// JSON object on stdout. No file in the repo is imported, so the harness works
/// against any upstream PySCF the user points it at.
const ORACLE_PY: &str = r#"
import json, sys
import numpy as np
import pyscf
from pyscf.pbc import gto
from pyscf.pbc.lib import kpts_helper as kh

name, dim, a_json, xyz_json, sym_json, mesh_json = sys.argv[1:7]
a = json.loads(a_json)
xyz = json.loads(xyz_json)
syms = json.loads(sym_json)
mesh = json.loads(mesh_json)

c = gto.Cell()
c.a = a
c.atom = [(s, tuple(r)) for s, r in zip(syms, xyz)]
c.basis = 'gth-szv'
c.unit = 'Bohr'
c.dimension = int(dim)
c.verbose = 0
c.build()

out = {
    'pyscf_version': pyscf.__version__,
    'pyscf_file': pyscf.__file__,
    'name': name,
    'natm': int(c.natm),
    'vol': float(c.vol),
    'rcut': float(c.rcut),
    'mesh': [int(x) for x in c.mesh],
    'b': c.reciprocal_vectors().ravel().tolist(),
    'gv': c.get_Gv(mesh).ravel().tolist(),
    'lattice_ls': c.get_lattice_Ls().ravel().tolist(),
    'kpts222': c.make_kpts([2, 2, 2]).ravel().tolist(),
    'kpts321': c.make_kpts([3, 2, 1]).ravel().tolist(),
    'kconserv222': [int(x) for x in kh.get_kconserv(c, c.make_kpts([2, 2, 2])).ravel()],
}
si = c.get_SI(mesh=mesh)
out['si_re'] = si.real.ravel().tolist()
out['si_im'] = si.imag.ravel().tolist()
try:
    out['ewald'] = float(c.ewald())
except Exception as e:
    out['ewald'] = None
    out['ewald_error'] = f'{type(e).__name__}: {e}'
print(json.dumps(out))
"#;

/// The Angstrom-input variant of [`ORACLE_PY`] — used only by the CODATA-gap
/// test, which needs upstream's own Angstrom -> Bohr conversion.
const ORACLE_ANG_PY: &str = r#"
import json, sys
import numpy as np
from pyscf.pbc import gto

dim, a_json, xyz_json, sym_json = sys.argv[1:5]
c = gto.Cell()
c.a = json.loads(a_json)
c.atom = [(s, tuple(r)) for s, r in zip(json.loads(sym_json), json.loads(xyz_json))]
c.basis = 'gth-szv'
c.unit = 'Angstrom'
c.dimension = int(dim)
c.verbose = 0
c.build()
print(json.dumps({'a_bohr': c.lattice_vectors().ravel().tolist(),
                  'vol': float(c.vol)}))
"#;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

fn workspace_root() -> PathBuf {
    // CARGO_MANIFEST_DIR = <root>/crates/pyscf-pbc-gto
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

/// Resolve `PYSCF_ORACLE_VENV` to an interpreter, or `None` when the gate is
/// closed.
///
/// * unset or empty -> `None` (skip);
/// * `1` / `true` / `auto` -> `<workspace>/.venv/bin/python`;
/// * a directory -> `<dir>/bin/python`;
/// * anything else -> used verbatim as the interpreter path.
fn oracle_python() -> Option<PathBuf> {
    let raw = std::env::var(GATE).ok()?;
    let raw = raw.trim().to_string();
    if raw.is_empty() {
        return None;
    }
    let p = if matches!(raw.as_str(), "1" | "true" | "auto" | "yes") {
        workspace_root().join(".venv/bin/python")
    } else {
        let candidate = PathBuf::from(&raw);
        if candidate.is_dir() {
            candidate.join("bin/python")
        } else {
            candidate
        }
    };
    assert!(
        p.exists(),
        "{GATE} = {raw:?} resolved to {p:?}, which does not exist"
    );
    Some(p)
}

/// Print the skip line and return `true` when the gate is closed.
macro_rules! gate {
    ($py:ident) => {
        let Some($py) = oracle_python() else {
            eprintln!("SKIP: {GATE} is not set — upstream oracle not run");
            return;
        };
    };
}

/// Serial number for the emitted scripts. `cargo test` runs these tests in
/// PARALLEL threads of ONE process, so a fixed filename would let one test
/// overwrite the script another is about to run — which is exactly the race
/// that showed up as two spurious failures before this counter existed.
static SCRIPT_SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn run_python(py: &PathBuf, script: &str, args: &[String]) -> Value {
    let dir = std::env::temp_dir().join(format!("pyscf_rs_oracle_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let seq = SCRIPT_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = dir.join(format!("oracle_{seq}.py"));
    std::fs::write(&path, script).expect("write oracle script");

    let mut cmd = Command::new(py);
    cmd.arg(&path).args(args);
    // The vendored upstream tree at <root>/pyscf is the source the PORT blocks
    // were read from; prefer it unless the caller has already chosen one.
    if std::env::var_os("PYTHONPATH").is_none() {
        cmd.env("PYTHONPATH", workspace_root());
    }
    let out = cmd.output().expect("spawn the oracle interpreter");
    assert!(
        out.status.success(),
        "oracle python failed ({}):\nstdout: {}\nstderr: {}",
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    // PySCF prints plugin banners to stdout; take the last non-empty line.
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let line = stdout
        .lines()
        .rev()
        .find(|l| l.trim_start().starts_with('{'))
        .unwrap_or_else(|| panic!("no JSON on oracle stdout:\n{stdout}"));
    serde_json::from_str(line).expect("parse oracle JSON")
}

/// Run [`ORACLE_PY`] for one reference system.
fn oracle(py: &PathBuf, r: &EwaldReference) -> Value {
    let args = vec![
        r.name.to_string(),
        r.dimension.to_string(),
        serde_json::to_string(&r.a_bohr).expect("a"),
        serde_json::to_string(r.coords_bohr).expect("xyz"),
        serde_json::to_string(r.symbols).expect("syms"),
        serde_json::to_string(&ORACLE_MESH).expect("mesh"),
    ];
    run_python(py, ORACLE_PY, &args)
}

/// The Rust `Cell` matching what [`ORACLE_PY`] builds — Bohr in, no pseudo.
fn bohr_cell(r: &EwaldReference) -> Cell {
    let atoms: Vec<(String, [f64; 3])> = r
        .symbols
        .iter()
        .zip(r.coords_bohr.iter())
        .map(|(s, xyz)| ((*s).to_string(), *xyz))
        .collect();
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(atoms),
            basis: BasisInput::Name("gth-szv".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix(r.a_bohr),
        dimension: r.dimension,
        ..Default::default()
    })
    .expect("reference cell must build")
}

fn floats(v: &Value, key: &str) -> Vec<f64> {
    v[key]
        .as_array()
        .unwrap_or_else(|| panic!("oracle field {key} is not an array"))
        .iter()
        .map(|x| x.as_f64().expect("float"))
        .collect()
}

fn ints(v: &Value, key: &str) -> Vec<i64> {
    v[key]
        .as_array()
        .unwrap_or_else(|| panic!("oracle field {key} is not an array"))
        .iter()
        .map(|x| x.as_i64().expect("int"))
        .collect()
}

fn assert_close(name: &str, got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(
        got.len(),
        want.len(),
        "{name}: length {} vs upstream {}",
        got.len(),
        want.len()
    );
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "{name}[{i}]: {g:.17} vs upstream {w:.17}, dev {:.3e} > {tol:.1e}",
            (g - w).abs()
        );
    }
}

fn flat3(v: &[[f64; 3]]) -> Vec<f64> {
    v.iter().flat_map(|r| r.iter().copied()).collect()
}

// ---------------------------------------------------------------------------
// The oracle tests
// ---------------------------------------------------------------------------

/// Success criterion 4 — `Cell::build` scalars: `vol`, `rcut`, `mesh`, `b`.
#[test]
#[ignore = "venv-gated upstream oracle; set PYSCF_ORACLE_VENV and pass --ignored"]
fn cell_scalars_match_upstream() {
    gate!(py);
    for r in EWALD_REFERENCES.iter() {
        let up = oracle(&py, r);
        let cell = bohr_cell(r);
        eprintln!(
            "[{}] upstream PySCF {} ({})",
            r.name, up["pyscf_version"], up["pyscf_file"]
        );

        assert_eq!(cell.mol.natm as i64, up["natm"].as_i64().expect("natm"));

        let vol = up["vol"].as_f64().expect("vol");
        assert!(
            (cell.vol() - vol).abs() <= vol.abs() * 1e-12,
            "{}: vol {} vs {vol}",
            r.name,
            cell.vol()
        );

        let rcut = up["rcut"].as_f64().expect("rcut");
        assert!(
            (cell.rcut - rcut).abs() <= rcut.abs() * 1e-12,
            "{}: rcut {} vs {rcut}",
            r.name,
            cell.rcut
        );

        let mesh: Vec<i64> = ints(&up, "mesh");
        assert_eq!(
            cell.mesh.map(|m| m as i64).to_vec(),
            mesh,
            "{}: mesh",
            r.name
        );

        let b = cell.reciprocal_vectors_2pi().expect("b");
        assert_close(
            &format!("{}: b", r.name),
            &b.iter()
                .flat_map(|row| row.iter().copied())
                .collect::<Vec<_>>(),
            &floats(&up, "b"),
            1e-12,
        );

        // b . a^T == 2*pi*I (success criterion 4, second half).
        let a = cell.lattice_vectors();
        for (i, bi) in b.iter().enumerate() {
            for (j, aj) in a.iter().enumerate() {
                let dot: f64 = (0..3).map(|k| bi[k] * aj[k]).sum();
                let want = if i == j {
                    2.0 * std::f64::consts::PI
                } else {
                    0.0
                };
                assert!(
                    (dot - want).abs() < 1e-12,
                    "{}: (b.a^T)[{i}][{j}] = {dot}, want {want}",
                    r.name
                );
            }
        }
    }
}

/// Success criterion 5 — `get_Gv` element-wise, and `|SI| == 1`.
#[test]
#[ignore = "venv-gated upstream oracle; set PYSCF_ORACLE_VENV and pass --ignored"]
fn gv_and_si_match_upstream() {
    gate!(py);
    for r in EWALD_REFERENCES.iter() {
        let up = oracle(&py, r);
        let cell = bohr_cell(r);

        let gv = pyscf_pbc_gto::get_gv(&cell, Some(ORACLE_MESH)).expect("get_Gv");
        assert_close(
            &format!("{}: Gv", r.name),
            &flat3(&gv),
            &floats(&up, "gv"),
            1e-12,
        );

        let si = get_si(&cell, None, Some(ORACLE_MESH), None).expect("get_SI");
        assert_close(
            &format!("{}: SI.re", r.name),
            &si.re,
            &floats(&up, "si_re"),
            1e-12,
        );
        assert_close(
            &format!("{}: SI.im", r.name),
            &si.im,
            &floats(&up, "si_im"),
            1e-12,
        );

        for (i, (re, im)) in si.re.iter().zip(&si.im).enumerate() {
            let mag = (re * re + im * im).sqrt();
            assert!(
                (mag - 1.0).abs() < 1e-12,
                "{}: |SI[{i}]| = {mag}, must be 1",
                r.name
            );
        }
    }
}

/// Success criterion 6, first half — `get_lattice_Ls` count and values.
#[test]
#[ignore = "venv-gated upstream oracle; set PYSCF_ORACLE_VENV and pass --ignored"]
fn lattice_ls_match_upstream() {
    gate!(py);
    for r in EWALD_REFERENCES.iter() {
        let up = oracle(&py, r);
        let cell = bohr_cell(r);
        let ls = get_lattice_ls_default(&cell).expect("get_lattice_Ls");
        let want = floats(&up, "lattice_ls");
        assert_eq!(
            ls.len() * 3,
            want.len(),
            "{}: nimgs {} vs upstream {}",
            r.name,
            ls.len(),
            want.len() / 3
        );
        assert_close(&format!("{}: Ls", r.name), &flat3(&ls), &want, 1e-12);
    }
}

/// Success criterion 6, second half — `make_kpts` and the `get_kconserv` table.
#[test]
#[ignore = "venv-gated upstream oracle; set PYSCF_ORACLE_VENV and pass --ignored"]
fn kpts_and_kconserv_match_upstream() {
    gate!(py);
    for r in EWALD_REFERENCES.iter() {
        let up = oracle(&py, r);
        let cell = bohr_cell(r);

        let k222 = make_kpts_default(&cell, [2, 2, 2]).expect("make_kpts [2,2,2]");
        assert_close(
            &format!("{}: kpts [2,2,2]", r.name),
            &flat3(&k222),
            &floats(&up, "kpts222"),
            1e-12,
        );

        let k321 = make_kpts_default(&cell, [3, 2, 1]).expect("make_kpts [3,2,1]");
        assert_close(
            &format!("{}: kpts [3,2,1]", r.name),
            &flat3(&k321),
            &floats(&up, "kpts321"),
            1e-12,
        );

        let kc = get_kconserv(&cell, &k222);
        let want = ints(&up, "kconserv222");
        assert_eq!(
            kc.data.iter().map(|x| *x as i64).collect::<Vec<_>>(),
            want,
            "{}: get_kconserv([2,2,2]) must match EXACTLY",
            r.name
        );
    }
}

/// Success criterion 7 — `cell.ewald()` to 1e-9 Ha, on every branch this port
/// ships. Graphene (`dimension = 2`) must instead defer with the typed Phase 12
/// error while upstream returns a number.
#[test]
#[ignore = "venv-gated upstream oracle; set PYSCF_ORACLE_VENV and pass --ignored"]
fn ewald_matches_upstream() {
    gate!(py);
    for r in EWALD_REFERENCES.iter() {
        let up = oracle(&py, r);
        let cell = bohr_cell(r);
        let want = up["ewald"].as_f64();
        match (ewald(&cell, None, None), want) {
            (Ok(got), Some(w)) => assert!(
                (got - w).abs() < 1e-9,
                "{}: ewald() = {got:.15} vs upstream {w:.15}, dev {:.3e}",
                r.name,
                (got - w).abs()
            ),
            (Err(PyscfRsError::NotYetImplemented { phase: 12, .. }), Some(w)) => {
                assert_eq!(r.dimension, 2, "{}: only the 2D branch is deferred", r.name);
                eprintln!(
                    "[{}] DEFERRED to plan 12-08; upstream value {w} recorded",
                    r.name
                );
            }
            (got, w) => panic!("{}: unexpected pair {got:?} / {w:?}", r.name),
        }
    }
}

/// `make_kpts` non-default flag combinations, so the oracle covers the whole
/// surface plan 09-07 shipped and not just the default.
#[test]
#[ignore = "venv-gated upstream oracle; set PYSCF_ORACLE_VENV and pass --ignored"]
fn make_kpts_variants_match_upstream() {
    gate!(py);
    const VARIANT_PY: &str = r#"
import json, sys
from pyscf.pbc import gto
dim, a_json, xyz_json, sym_json = sys.argv[1:5]
c = gto.Cell()
c.a = json.loads(a_json)
c.atom = [(s, tuple(r)) for s, r in zip(json.loads(sym_json), json.loads(xyz_json))]
c.basis = 'gth-szv'; c.unit = 'Bohr'; c.dimension = int(dim); c.verbose = 0
c.build()
print(json.dumps({
    'no_gamma':   c.make_kpts([2,2,2], with_gamma_point=False).ravel().tolist(),
    'wrap':       c.make_kpts([2,2,2], wrap_around=True).ravel().tolist(),
    'no_gamma_wrap': c.make_kpts([2,2,2], with_gamma_point=False, wrap_around=True).ravel().tolist(),
    'centered':   c.make_kpts([2,2,2], scaled_center=[0.1,0.2,0.3]).ravel().tolist(),
}))
"#;
    let r = EWALD_REFERENCES
        .iter()
        .find(|r| r.name == "diamond")
        .expect("diamond");
    let args = vec![
        r.dimension.to_string(),
        serde_json::to_string(&r.a_bohr).expect("a"),
        serde_json::to_string(r.coords_bohr).expect("xyz"),
        serde_json::to_string(r.symbols).expect("syms"),
    ];
    let up = run_python(&py, VARIANT_PY, &args);
    let cell = bohr_cell(r);

    let cases: [(&str, Vec<[f64; 3]>); 4] = [
        (
            "no_gamma",
            make_kpts(&cell, [2, 2, 2], false, false, None).expect("no gamma"),
        ),
        (
            "wrap",
            make_kpts(&cell, [2, 2, 2], true, true, None).expect("wrap"),
        ),
        (
            "no_gamma_wrap",
            make_kpts(&cell, [2, 2, 2], true, false, None).expect("no gamma + wrap"),
        ),
        (
            "centered",
            make_kpts(&cell, [2, 2, 2], false, true, Some([0.1, 0.2, 0.3])).expect("centered"),
        ),
    ];
    for (key, got) in cases {
        assert_close(key, &flat3(&got), &floats(&up, key), 1e-12);
    }
}

/// The Angstrom conversion path. This port's `Unit::Ang` is CODATA-2014 and
/// upstream's is CODATA-2010, so the lattices differ by 4.951e-9 RELATIVE and
/// nothing more. Asserting a two-sided bound turns a silent constant drift into
/// a test failure if either side ever changes its constant.
#[test]
#[ignore = "venv-gated upstream oracle; set PYSCF_ORACLE_VENV and pass --ignored"]
fn angstrom_lattices_match_upstream_within_the_codata_gap() {
    gate!(py);
    /// `1.8897261339213 / (1 / 0.52917721092) - 1`
    const CODATA_GAP: f64 = 4.951e-9;

    // The §9.2 systems' Angstrom inputs, mirroring
    // `pyscf_pbc_gto::test_systems`.
    let a0 = 2.46_f64;
    let ang: [AngstromSystem; 5] = [
        (
            "diamond",
            fcc(3.5668),
            vec![("C", [0.0; 3]), ("C", [3.5668 / 4.0; 3])],
            3,
        ),
        (
            "si",
            fcc(5.4306),
            vec![("Si", [0.0; 3]), ("Si", [5.4306 / 4.0; 3])],
            3,
        ),
        (
            "lif",
            fcc(4.03),
            vec![("Li", [0.0; 3]), ("F", [4.03 / 2.0; 3])],
            3,
        ),
        ("he_fcc", fcc(3.0), vec![("He", [0.0; 3])], 3),
        (
            "graphene",
            [
                [a0, 0.0, 0.0],
                [-a0 / 2.0, a0 * 3.0_f64.sqrt() / 2.0, 0.0],
                [0.0, 0.0, 20.0],
            ],
            vec![("C", [0.0; 3]), ("C", [0.0, a0 / 3.0_f64.sqrt(), 0.0])],
            2,
        ),
    ];

    for (name, a, atoms, dim) in ang {
        let args = vec![
            dim.to_string(),
            serde_json::to_string(&a).expect("a"),
            serde_json::to_string(&atoms.iter().map(|(_, r)| *r).collect::<Vec<_>>()).expect("xyz"),
            serde_json::to_string(&atoms.iter().map(|(s, _)| *s).collect::<Vec<_>>())
                .expect("syms"),
        ];
        let up = run_python(&py, ORACLE_ANG_PY, &args);
        let want = floats(&up, "a_bohr");

        let cell = systems::all()
            .into_iter()
            .find(|(n, _)| *n == name)
            .map(|(_, c)| c)
            .expect("reference system");
        let got: Vec<f64> = cell
            .lattice_vectors()
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();

        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            if w.abs() < 1e-14 {
                assert!(g.abs() < 1e-14, "{name}: a[{i}] {g} should be zero");
                continue;
            }
            let rel = (g - w).abs() / w.abs();
            assert!(
                (rel - CODATA_GAP).abs() < 1e-11,
                "{name}: a[{i}] relative deviation {rel:.4e} is not the \
                 {CODATA_GAP:.4e} CODATA gap — one side changed its Bohr constant"
            );
        }
    }
}

/// One Angstrom-input reference system: `(name, lattice, [(symbol, coords)], dimension)`.
type AngstromSystem = (
    &'static str,
    [[f64; 3]; 3],
    Vec<(&'static str, [f64; 3])>,
    u8,
);

/// fcc PRIMITIVE lattice for conventional cube edge `a0`, in the input unit.
fn fcc(a0: f64) -> [[f64; 3]; 3] {
    let h = a0 / 2.0;
    [[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]
}
