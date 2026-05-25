//! GTO-07 oracle helper — invoked by `tests/oracle/test_eval_gto.py`.
//!
//! Builds a `Mole` from env-var inputs, parses the JSON-encoded coords
//! list, calls `pyscf_gto::eval_gto(&mol, name, &coords)`, and dumps
//! `{values, shape}` to JSON. The Python harness reshapes via F-order
//! and diffs against upstream `mol.eval_gto(name, coords)`.
//!
//! Phase 2 ships `GTOval`, `GTOval_sph`, `GTOval_cart` for s-shells; the
//! cc-pVDZ p-shell xfail in the Python suite tracks the Phase 4 DFT
//! deferral.
//!
//! Env vars:
//!   - `PYSCF_RS_ORACLE_ATOM`      : atom string
//!   - `PYSCF_RS_ORACLE_BASIS`     : basis name
//!   - `PYSCF_RS_ORACLE_EVAL_NAME` : eval_gto variant (e.g. `"GTOval_sph"`)
//!   - `PYSCF_RS_ORACLE_COORDS`    : JSON-encoded `Vec<[f64;3]>`
//!   - `PYSCF_RS_ORACLE_OUT`       : output JSON path

#![cfg(feature = "release-oracle-tests")]

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs, eval_gto};
use std::path::PathBuf;

#[test]
#[ignore = "oracle helper — invoked by python via `cargo test --features release-oracle-tests -- --ignored`"]
fn dump_eval_gto_for_oracle() {
    let atom = std::env::var("PYSCF_RS_ORACLE_ATOM").expect("PYSCF_RS_ORACLE_ATOM env var");
    let basis = std::env::var("PYSCF_RS_ORACLE_BASIS").expect("PYSCF_RS_ORACLE_BASIS env var");
    let eval_name =
        std::env::var("PYSCF_RS_ORACLE_EVAL_NAME").expect("PYSCF_RS_ORACLE_EVAL_NAME env var");
    let coords_json =
        std::env::var("PYSCF_RS_ORACLE_COORDS").expect("PYSCF_RS_ORACLE_COORDS env var");
    let out = std::env::var("PYSCF_RS_ORACLE_OUT").expect("PYSCF_RS_ORACLE_OUT env var");

    let coords: Vec<[f64; 3]> = serde_json::from_str(&coords_json)
        .expect("PYSCF_RS_ORACLE_COORDS must be JSON Vec<[f64;3]>");

    let mol = M(MoleBuildArgs {
        atom: AtomInput::String(atom),
        basis: BasisInput::Name(basis),
        unit: Unit::Bohr,
        ..Default::default()
    })
    .expect("M(...) must succeed for oracle fixture");

    let result = eval_gto(&mol, &eval_name, &coords).expect("eval_gto(...) must succeed");

    let payload = serde_json::json!({
        "values": result.values,
        "shape":  result.shape,
    });
    std::fs::write(PathBuf::from(out), payload.to_string()).expect("write json");
}
