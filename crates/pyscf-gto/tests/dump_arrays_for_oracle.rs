//! GTO-04 oracle helper — invoked by `tests/oracle/test_byte_identity.py`.
//!
//! Reads the molecule selection from environment variables, builds the
//! `Mole` via the same `pyscf_gto::M(...)` front-door real users go
//! through, and dumps `_atm` / `_bas` / `_env` / `ao_loc_nr` / `nao_nr`
//! to a JSON file. The Python oracle then loads this JSON and diffs
//! against upstream `pyscf.M(...)` arrays byte-for-byte.
//!
//! Gated behind the `release-oracle-tests` Cargo feature **and** the
//! `#[ignore]` attribute, so it never runs in `cargo test --workspace`
//! by accident. The Python harness invokes it explicitly with
//! `--features release-oracle-tests -- --ignored`.
//!
//! Env vars:
//!   - `PYSCF_RS_ORACLE_ATOM`  : atom string passed to `AtomInput::String`
//!   - `PYSCF_RS_ORACLE_BASIS` : basis name passed to `BasisInput::Name`
//!   - `PYSCF_RS_ORACLE_OUT`   : path to write the JSON to

#![cfg(feature = "release-oracle-tests")]

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs, M};
use std::path::PathBuf;

#[test]
#[ignore = "oracle helper — invoked by python via `cargo test --features release-oracle-tests -- --ignored`"]
fn dump_arrays_for_oracle() {
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

    let payload = serde_json::json!({
        "_atm":      mol._atm,
        "_bas":      mol._bas,
        "_env":      mol._env,
        "ao_loc_nr": mol.ao_loc_nr,
        "nao_nr":    mol.nao_nr,
        "nbas":      mol.nbas,
        "natm":      mol.natm,
    });
    std::fs::write(PathBuf::from(out), payload.to_string()).expect("write json");
}
