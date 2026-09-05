//! Shared fixtures for the `pyscf-pbc-df` integration tests.
//!
//! # Geometry is specified in BOHR
//!
//! `pyscf_core::Unit::Ang` is CODATA-2014 and upstream is CODATA-2010, so an
//! Angstrom cell differs in the 8th digit of every lattice vector before an
//! integral is evaluated. Same note as `pyscf-pbc-gto`'s tests.
//!
//! # The upstream oracle is the VENDORED tree, not site-packages
//!
//! Two PySCF installs are reachable from this workspace: the vendored source
//! tree at `<root>/pyscf` (2.12.1 — the port target, whose line numbers every
//! `PORT` comment cites) and `.venv/lib/.../site-packages/pyscf` (2.14.0).
//! They are NOT interchangeable here: 2.14 rewrote `fft_jk.get_k_kpts` to fold
//! the `exxdiv='ewald'` correction into `get_coulG` instead of applying
//! `_ewald_exxdiv_for_G0` analytically, which moves the exchange matrix by
//! ~1e-5. [`run_python`] therefore pins `PYTHONPATH` to the workspace root so
//! `import pyscf` resolves to the vendored tree; a plain script run would
//! silently pick up 2.14 (its own directory lands on `sys.path[0]`, not the
//! CWD).

#![allow(dead_code)]

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs};
use std::path::PathBuf;
use std::process::Command;

/// Diamond, fcc `a0 = 6.74064` Bohr, `gth-szv` / `gth-pade`. 8 AOs.
pub fn diamond() -> Cell {
    let h = 3.37032;
    let q = 1.68516;
    bohr_cell(
        [[0.0, h, h], [h, 0.0, h], [h, h, 0.0]],
        vec![("C".into(), [0.0, 0.0, 0.0]), ("C".into(), [q, q, q])],
        Some("gth-pade"),
    )
}

/// Si, fcc `a0 = 10.2622` Bohr, `gth-szv` / `gth-pade`.
pub fn silicon() -> Cell {
    let h = 5.1311;
    let q = 2.55555;
    bohr_cell(
        [[0.0, h, h], [h, 0.0, h], [h, h, 0.0]],
        vec![("Si".into(), [0.0, 0.0, 0.0]), ("Si".into(), [q, q, q])],
        Some("gth-pade"),
    )
}

/// He on an fcc lattice, ALL-ELECTRON (`sto-3g`, no pseudopotential) — the
/// `get_nuc` path, which a `gth-pade` cell never reaches.
pub fn he_all_electron() -> Cell {
    let h = 2.834589;
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("He".into(), [0.0, 0.0, 0.0])]),
            basis: BasisInput::Name("sto-3g".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        ..Default::default()
    })
    .expect("He cell must build")
}

pub fn bohr_cell(a: [[f64; 3]; 3], atoms: Vec<(String, [f64; 3])>, pseudo: Option<&str>) -> Cell {
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(atoms),
            basis: BasisInput::Name("gth-szv".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix(a),
        pseudo: pseudo.map(str::to_string),
        ..Default::default()
    })
    .expect("reference cell must build")
}

/// Diamond built with an explicit `precision`.
///
/// **Build time, not a post-hoc mutation.** `cell.rcut` is a cached field
/// computed during `Cell::build`, so assigning `cell.precision = p` afterwards
/// tightens only the estimators that read `precision` at CALL time (`ft_ao`'s
/// `estimate_rcut`) and leaves `cell.rcut` — and therefore `pbc_intor`, Ewald
/// and `eval_gto` — on the original target.
pub fn diamond_prec(precision: f64) -> Cell {
    let h = 3.37032;
    let q = 1.68516;
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("C".into(), [0.0, 0.0, 0.0]), ("C".into(), [q, q, q])]),
            basis: BasisInput::Name("gth-szv".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        pseudo: Some("gth-pade".into()),
        precision,
        ..Default::default()
    })
    .expect("reference cell must build")
}

// ---------------------------------------------------------------------------
// The venv-gated upstream oracle
// ---------------------------------------------------------------------------

pub const GATE: &str = "PYSCF_ORACLE_VENV";

pub fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

/// Resolve `PYSCF_ORACLE_VENV` to an interpreter, or `None` when the gate is
/// unset (in which case the caller must SKIP, not fail).
pub fn oracle_python() -> Option<PathBuf> {
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

/// Run `script` under the oracle interpreter and parse its LAST JSON line.
pub fn run_python(py: &PathBuf, script: &str, args: &[String]) -> serde_json::Value {
    let n = SCRIPT_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("pbc_df_oracle_{}_{n}.py", std::process::id()));
    std::fs::write(&path, script).expect("write oracle script");
    let root = workspace_root();
    let out = Command::new(py)
        .arg(&path)
        .args(args)
        .env("PYTHONPATH", &root)
        .current_dir(&root)
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
        .unwrap_or_else(|| panic!("oracle produced no JSON line:\n{stdout}"))
        .to_string();
    serde_json::from_str(&line).expect("oracle JSON parses")
}

/// JSON encoding of a cell's geometry, for the oracle scripts.
pub fn cell_args(cell: &Cell, extra: &[String]) -> Vec<String> {
    let a: Vec<Vec<f64>> = cell.a.iter().map(|r| r.to_vec()).collect();
    let xyz: Vec<Vec<f64>> = cell.mol.atom_coords().iter().map(|r| r.to_vec()).collect();
    let sym: Vec<String> = cell.mol._atom.iter().map(|(s, _)| s.clone()).collect();
    let mut v = vec![
        serde_json::to_string(&a).expect("json"),
        serde_json::to_string(&xyz).expect("json"),
        serde_json::to_string(&sym).expect("json"),
    ];
    v.extend_from_slice(extra);
    v
}

/// Largest element-wise deviation between a stack of row-major Rust matrices
/// and the upstream `{"re": [...], "im": [...]}` payload.
pub fn max_dev(got: &[pyscf_algebra::CTensor], want: &serde_json::Value) -> f64 {
    let pull = |key: &str| -> Vec<f64> {
        want[key]
            .as_array()
            .unwrap_or_else(|| panic!("oracle payload has no {key} array"))
            .iter()
            .map(|v| v.as_f64().expect("f64"))
            .collect()
    };
    let re = pull("re");
    let im = pull("im");
    let n: usize = got.iter().map(pyscf_algebra::CTensor::len).sum();
    assert_eq!(re.len(), n, "shape mismatch vs upstream");
    let mut w = 0.0_f64;
    let mut p = 0usize;
    for m in got {
        for i in 0..m.len() {
            w = w.max((re[p] - m.re[i]).abs());
            w = w.max((im[p] - m.im[i]).abs());
            p += 1;
        }
    }
    w
}

/// He with a **two**-function basis. [`he_all_electron`] is `sto-3g`, i.e. one
/// AO, which makes every MO transform on it a 1x1 identity — see
/// `pbc_ao2mo_mofirst.rs::mo_first_matches_ao_first_with_complex_coefficients`.
pub fn he_631g() -> Cell {
    let h = 2.834589;
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("He".into(), [0.0, 0.0, 0.0])]),
            basis: BasisInput::Name("6-31g".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        mesh: Some([9, 9, 9]),
        ..Default::default()
    })
    .expect("He 6-31g cell must build")
}
