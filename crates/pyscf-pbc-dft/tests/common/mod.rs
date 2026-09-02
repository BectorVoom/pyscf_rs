//! Shared fixtures for the `pyscf-pbc-dft` integration tests.
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
    spin_cell(a, atoms, "gth-szv", pseudo, 0, 0)
}

/// The general fixture builder: an explicit basis, pseudopotential, `spin`
/// (`2S = Nalpha - Nbeta`, PER CELL) and `charge`.
///
/// # The k-mesh parity trap — read before adding an open-shell fixture
///
/// `Kuks::nelec()` (`kuks.rs:102-117`, port of `kuhf.py:442-458`) forms
/// `nalpha = (Ne_supercell + spin) / 2` where `Ne_supercell =
/// cell.tot_electrons(nkpts)` but `spin` is PER CELL. An odd-electron cell with
/// an EVEN k-count therefore fails `nalpha + nbeta == Ne` and is rejected — by
/// upstream and by this port identically (`kuhf.py:450-453` raises the same
/// `RuntimeError`). Gate such a cell at Gamma or at an ODD k-count such as
/// `[1,1,3]`. This is an inherited upstream constraint, not a port bug; do not
/// "fix" it.
pub fn spin_cell(
    a: [[f64; 3]; 3],
    atoms: Vec<(String, [f64; 3])>,
    basis: &str,
    pseudo: Option<&str>,
    spin: i32,
    charge: i32,
) -> Cell {
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(atoms),
            basis: BasisInput::Name(basis.into()),
            unit: Unit::Bohr,
            spin,
            charge,
            ..Default::default()
        },
        a: ALattice::Matrix(a),
        pseudo: pseudo.map(str::to_string),
        ..Default::default()
    })
    .expect("reference cell must build")
}

/// A cubic lattice of edge `a` Bohr.
fn cube(a: f64) -> [[f64; 3]; 3] {
    [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]
}

// ---------------------------------------------------------------------------
// The OPEN-SHELL fixtures (KUKS-OPTIMISATION-PLAN U-00, RULE U)
//
// RULE U: no KUKS work item may be validated on a closed-shell cell. On a
// closed-shell cell `dm_a == dm_b` bit-identically and PERMANENTLY — it is an
// exact fixed point of this port's SCF map (§2.2.1) — so the unrestricted path
// degenerates to the restricted one and a passing test proves nothing about it.
//
// Both fixtures are ALL-ELECTRON on purpose. The `gth-pade` cells floor at
// ~4e-12 Ha for structural reasons inherited from `get_pp`, and the
// all-electron control is what proves a tighter number is reachable at all
// (`KRKS He-fcc` sits at 9.81e-14). Do not gate the open-shell work on a
// pseudopotential cell.
// ---------------------------------------------------------------------------

/// A lithium atom in a 6-Bohr cubic box, all-electron `sto-3g`, **`spin = 1`**.
///
/// 3 electrons, 5 AOs (1s, 2s, 2p). The genuinely spin-POLARISED case: it
/// exercises the `cell.spin != 0` path, where `_break_dm_spin_symm`
/// short-circuits and the per-channel renormalisation of `kuhf.py:476-486` is
/// the only thing that polarises the initial guess at all.
///
/// Odd electron count ⇒ Gamma or an ODD k-count only (see [`spin_cell`]).
pub fn li_atom_spin1() -> Cell {
    spin_cell(
        cube(6.0),
        vec![("Li".into(), [0.0, 0.0, 0.0])],
        "sto-3g",
        None,
        1,
        0,
    )
}

/// Stretched H2 (bond 3.0 Bohr) in an 8-Bohr cubic box, all-electron `6-31g`,
/// **`spin = 0`**.
///
/// 2 electrons, 4 AOs — two per atom, so `_break_dm_spin_symm`'s `breaksym == 1`
/// branch (keep only the intra-atomic 2x2 blocks of the beta guess) is a
/// genuine, visible break ON THE GUESS. It does NOT converge to a spin-broken
/// solution: upstream lands at `<S^2> = 0` here, and at every separation from
/// 2.0 to 6.0 Bohr, molecular UHF included — see the MEASURED CAVEAT in
/// `tests/gate_openshell.rs`. The guess-level assertions are in
/// `pyscf-pbc-scf/tests/init_guess_spin.rs`.
pub fn h2_stretched_spin0() -> Cell {
    spin_cell(
        cube(8.0),
        vec![
            ("H".into(), [0.0, 0.0, -1.5]),
            ("H".into(), [0.0, 0.0, 1.5]),
        ],
        "6-31g",
        None,
        0,
        0,
    )
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
    let path = std::env::temp_dir().join(format!("pbc_dft_oracle_{}_{n}.py", std::process::id()));
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
///
/// U-00: `spin` and `charge` are emitted as the 4th and 5th positional
/// arguments. They used to be absent entirely, and `ORACLE_PY` never set
/// `c.spin` — which is why no test in this crate could express an open-shell
/// cell, and why `grep -rn spin tests/` found exactly one hit, a doc-comment
/// word. Every oracle script that consumes `cell_args` must unpack all five.
pub fn cell_args(cell: &Cell, extra: &[String]) -> Vec<String> {
    let a: Vec<Vec<f64>> = cell.a.iter().map(|r| r.to_vec()).collect();
    let xyz: Vec<Vec<f64>> = cell.mol.atom_coords().iter().map(|r| r.to_vec()).collect();
    let sym: Vec<String> = cell.mol._atom.iter().map(|(s, _)| s.clone()).collect();
    let mut v = vec![
        serde_json::to_string(&a).expect("json"),
        serde_json::to_string(&xyz).expect("json"),
        serde_json::to_string(&sym).expect("json"),
        cell.mol.spin.to_string(),
        cell.mol.charge.to_string(),
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
