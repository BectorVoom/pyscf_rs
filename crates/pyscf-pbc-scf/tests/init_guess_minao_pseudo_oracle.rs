//! `init_guess_by_minao` on cells with a GTH pseudopotential, against live
//! upstream PySCF 2.12.1 (`pyscf/scf/hf.py:348-474`).
//!
//! With core electrons removed (`mol.atom_nelec_core(ia) > 0`) upstream drops
//! the core shells and, when the input basis has AO character, occupies the
//! INPUT basis functions with the valence configuration instead of projecting
//! ANO functions. The port used to project the all-electron ANO occupation
//! and let the electron-count renormalisation absorb the difference — a
//! different starting density (KTaO3 SZV: init E −134.969 vs −135.378 Ha).
//!
//! ```bash
//! PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-scf --release \
//!     --test init_guess_minao_pseudo_oracle -- --ignored --nocapture
//! ```

mod common;

use common::{GATE, cell_args, diamond, oracle_python, run_python};
use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs};
use pyscf_scf::InitGuessMode;
use pyscf_scf::init_guess::default_get_init_guess;

const ORACLE_PY: &str = r#"
import json, sys
import numpy as np
from pyscf.pbc import gto
from pyscf.scf import hf as mol_hf

a_json, xyz_json, sym_json, basis, pseudo = sys.argv[1:6]
c = gto.Cell()
c.a = json.loads(a_json)
c.atom = [(s, tuple(r)) for s, r in zip(json.loads(sym_json), json.loads(xyz_json))]
c.basis = basis
c.pseudo = pseudo
c.unit = 'Bohr'
c.verbose = 0
c.build()
dm = np.asarray(mol_hf.init_guess_by_minao(c))
print(json.dumps({'version': __import__('pyscf').__version__, 'nao': int(c.nao_nr()),
                  'core': [int(c.atom_nelec_core(i)) for i in range(c.natm)],
                  'dm': dm.ravel().tolist()}))
"#;

/// Cubic KTaO3, Bohr.
fn ktao3(basis: &str) -> Cell {
    let a = 7.54_f64;
    let h = 0.5 * a;
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![
                ("K".into(), [0.0, 0.0, 0.0]),
                ("Ta".into(), [h, h, h]),
                ("O".into(), [h, h, 0.0]),
                ("O".into(), [h, 0.0, h]),
                ("O".into(), [0.0, h, h]),
            ]),
            basis: BasisInput::Name(basis.into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]),
        pseudo: Some("gth-pbe".into()),
        ..Default::default()
    })
    .expect("KTaO3 cell must build")
}

/// Largest element-wise deviation of the port's guess from upstream's.
fn guess_dev(cell: &Cell, basis: &str, pseudo: &str) -> Option<f64> {
    let py = oracle_python()?;
    let want = run_python(&py, ORACLE_PY, &cell_args(cell, &[basis.to_string(), pseudo.to_string()]));
    assert_eq!(want["version"].as_str(), Some("2.12.1"), "oracle must be vendored 2.12.1");
    let nao = want["nao"].as_u64().expect("nao") as usize;
    assert_eq!(nao, cell.mol.nao_nr, "AO count differs");
    let core: Vec<i64> = want["core"].as_array().expect("core").iter().map(|v| v.as_i64().expect("int")).collect();
    assert!(core.iter().all(|n| *n > 0), "the fixture must carry core electrons on every atom: {core:?}");
    let dm = default_get_init_guess(&cell.mol, &InitGuessMode::Minao).expect("minao");
    let w = want["dm"].as_array().expect("dm");
    assert_eq!(w.len(), dm.data.len(), "density size differs");
    Some(
        dm.data
            .iter()
            .zip(w)
            .fold(0.0_f64, |m, (g, u)| m.max((g - u.as_f64().expect("f64")).abs())),
    )
}

#[test]
#[ignore = "oracle: set PYSCF_ORACLE_VENV (upstream PySCF 2.12.1)"]
fn minao_guess_matches_upstream_on_ktao3_szv_and_dzvp() {
    for basis in ["gth-szv-molopt-sr", "gth-dzvp-molopt-sr"] {
        let cell = ktao3(basis);
        let Some(w) = guess_dev(&cell, basis, "gth-pbe") else {
            eprintln!("SKIP: {GATE} is not set");
            return;
        };
        println!("KTaO3 {basis}: minao guess max|delta| vs upstream = {w:e}");
        assert!(w < 1e-10, "{basis}: minao guess deviates by {w:e}");
    }
}

#[test]
#[ignore = "oracle: set PYSCF_ORACLE_VENV (upstream PySCF 2.12.1)"]
fn minao_guess_matches_upstream_on_diamond() {
    let cell = diamond();
    let Some(w) = guess_dev(&cell, "gth-szv", "gth-pade") else {
        eprintln!("SKIP: {GATE} is not set");
        return;
    };
    println!("diamond gth-szv: minao guess max|delta| vs upstream = {w:e}");
    assert!(w < 1e-10, "diamond: minao guess deviates by {w:e}");
}
