//! `_env` bit-identical to upstream PySCF 2.12.1 (F6).
//!
//! `make_env` normalises contraction coefficients with numpy ARRAY arithmetic
//! (`gto_norm` + `_nomalize_contracted_ao`): `scipy.special.gamma(l+1.5)`
//! (cephes, 1 ulp off the correctly-rounded value) and numpy's SVML `a**n1`
//! (`__svml_pow8` on this AVX-512 host). This test pins the whole `_env` for
//! the basis sets `get_nuc` and friends load, gated on `PYSCF_ORACLE_VENV`:
//!
//! ```bash
//! PYSCF_ORACLE_VENV=1 cargo test --release -p pyscf-gto --test make_env_bitexact -- --ignored
//! ```

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};
use std::path::PathBuf;
use std::process::Command;

const GATE: &str = "PYSCF_ORACLE_VENV";

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
    assert!(p.exists(), "{GATE} = {raw:?} -> {p:?}");
    Some(p)
}

fn run_python(py: &PathBuf, script: &str, args: &[String]) -> serde_json::Value {
    let dir = std::env::temp_dir();
    let base = format!("make_env_oracle_{}", std::process::id());
    let path = dir.join(format!("{base}.py"));
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
        "oracle failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .rev()
        .find(|l| l.trim_start().starts_with('{'))
        .unwrap_or_else(|| panic!("oracle produced no JSON:\n{stdout}"))
        .to_string();
    serde_json::from_str(&line).expect("oracle JSON parses")
}

const ORACLE_PY: &str = r#"
import os
os.environ['OMP_NUM_THREADS'] = '1'
import json, sys
import numpy as np
from pyscf import gto

def bits(x):
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64)).ravel()
    return [int(v) for v in x.view(np.uint64)]

atoms, basis, cart = sys.argv[1:4]
c = gto.Mole()
c.atom = [(s, tuple(r)) for s, r in json.loads(atoms)]
c.basis = basis
c.cart = json.loads(cart)
c.unit = 'Bohr'
c.verbose = 0
c.build()
print(json.dumps({'version': __import__('pyscf').__version__,
                  'nao': int(c.nao_nr()), 'env': bits(c._env)}))
"#;

fn pull(v: &serde_json::Value, key: &str) -> Vec<u64> {
    v[key]
        .as_array()
        .unwrap_or_else(|| panic!("oracle payload has no {key}"))
        .iter()
        .map(|x| x.as_u64().expect("u64 bit pattern"))
        .collect()
}

fn run_case(atoms: Vec<(String, [f64; 3])>, basis: &str, cart: bool) {
    let Some(py) = oracle_python() else {
        eprintln!("SKIP: {GATE} is not set");
        return;
    };
    let mol = M(MoleBuildArgs {
        atom: AtomInput::Tuples(atoms.clone()),
        basis: BasisInput::Name(basis.into()),
        cart,
        unit: Unit::Bohr,
        ..Default::default()
    })
    .expect("molecule builds");
    let args = vec![
        serde_json::to_string(&atoms).expect("json"),
        basis.to_string(),
        serde_json::to_string(&cart).expect("json"),
    ];
    let want = run_python(&py, ORACLE_PY, &args);
    assert_eq!(
        want["version"].as_str(),
        Some("2.12.1"),
        "vendored PySCF only"
    );
    assert_eq!(
        want["nao"].as_u64(),
        Some(mol.nao_nr as u64),
        "nao mismatch for {atoms:?}/{basis} (cart={cart})"
    );
    let want_env = pull(&want, "env");
    assert_eq!(mol._env.len(), want_env.len(), "env length mismatch");
    let mut n = 0usize;
    for (i, (g, &w)) in mol._env.iter().zip(&want_env).enumerate() {
        if g.to_bits() != w {
            if n < 12 {
                eprintln!(
                    "  _env[{i}]: got {:016x} ({}) want {:016x} ({})",
                    g.to_bits(),
                    g,
                    w,
                    f64::from_bits(w)
                );
            }
            n += 1;
        }
    }
    assert_eq!(
        n, 0,
        "{atoms:?}/{basis} (cart={cart}): _env differs in {n}/{} entries",
        want_env.len()
    );
}

#[test]
#[ignore = "F6: needs PYSCF_ORACLE_VENV + the vendored upstream PySCF"]
fn env_bit_identical_across_basis_sets() {
    // Segmented basis sets only: Pople-style 6-31g/6-311g carry SP shells whose
    // split ordering differs from upstream (`format_basis` splits SP into an
    // interleaved s/p pair, upstream `sort_basis=True` regroups all s then all
    // p). That is a pre-existing shell-order gap outside `make_env`'s
    // normalisation; every segmented basis below matches bit-for-bit.
    for sym in ["He", "Ne", "C", "O", "Fe"] {
        for basis in ["sto-3g", "cc-pvdz", "cc-pvtz", "def2-svp"] {
            run_case(vec![(sym.to_string(), [0.0, 0.0, 0.0])], basis, false);
        }
    }
}

#[test]
#[ignore = "F6: needs PYSCF_ORACLE_VENV + the vendored upstream PySCF"]
fn env_bit_identical_multi_atom_and_cart() {
    // Same-symbol multi-atom (no symbol-order ambiguity: upstream iterates a
    // `set` of unique atoms, so a mixed-symbol `_env` order is not even
    // reproducible in Python), and a Cartesian single-atom.
    run_case(
        vec![("He".to_string(), [0.0, 0.0, 0.0]), ("He".to_string(), [1.0, 0.5, 0.25])],
        "sto-3g",
        false,
    );
    run_case(
        vec![("C".to_string(), [0.0, 0.0, 0.0]), ("C".to_string(), [0.0, 0.0, 2.1])],
        "cc-pvdz",
        false,
    );
    run_case(
        vec![("Ne".to_string(), [0.1, 0.2, 0.3])],
        "cc-pvtz",
        true,
    );
}
