//! GTO-09 oracle helper — invoked by `tests/oracle/test_json_interop.py`.
//!
//! Builds a `Mole` from env-var inputs and writes the
//! `pyscf_gto::dumps(&mol)` JSON string to disk. The Python oracle
//! parses it and confirms the user-input shape is preserved well enough
//! to feed back into upstream `pyscf.M(...)` and produce a structurally
//! equivalent `_atm` array.
//!
//! Per CONTEXT D-09 the round-trip contract is **semantic** (snapshot
//! preserves user inputs; deterministic rebuild reproduces derived
//! arrays) — NOT byte-identical to upstream PySCF JSON, which is the
//! Phase 8 ORACLE-08 chkfile job.
//!
//! Env vars:
//!   - `PYSCF_RS_ORACLE_ATOM`  : atom string
//!   - `PYSCF_RS_ORACLE_BASIS` : basis name
//!   - `PYSCF_RS_ORACLE_OUT`   : output path for the dumps() JSON

#![cfg(feature = "release-oracle-tests")]

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs, dumps};
use std::path::PathBuf;

#[test]
#[ignore = "oracle helper — invoked by python via `cargo test --features release-oracle-tests -- --ignored`"]
fn dump_mole_dumps_for_oracle() {
    let atom = std::env::var("PYSCF_RS_ORACLE_ATOM").expect("PYSCF_RS_ORACLE_ATOM env var");
    let basis = std::env::var("PYSCF_RS_ORACLE_BASIS").expect("PYSCF_RS_ORACLE_BASIS env var");
    let out = std::env::var("PYSCF_RS_ORACLE_OUT").expect("PYSCF_RS_ORACLE_OUT env var");

    let mol = M(MoleBuildArgs {
        atom: AtomInput::String(atom),
        basis: BasisInput::Name(basis),
        unit: Unit::Bohr,
        ..Default::default()
    })
    .expect("M(...) must succeed for oracle fixture");

    let json = dumps(&mol).expect("dumps(...) must succeed");
    std::fs::write(PathBuf::from(out), json).expect("write json");
}
