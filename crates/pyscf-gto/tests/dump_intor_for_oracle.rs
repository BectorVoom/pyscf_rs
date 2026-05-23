//! GTO-06 oracle helper — invoked by `tests/oracle/test_intor_oracle.py`.
//!
//! Builds a `Mole` from env-var inputs, calls `pyscf_gto::intor(&mol, name)`,
//! and dumps `{values, shape}` to JSON. The Python harness reshapes via
//! `numpy.reshape(..., order="F")` and diffs against upstream
//! `mol.intor(name)`.
//!
//! Pitfall 1 / 8 coverage: this helper preserves the F-order layout
//! returned by the dispatcher (component-leading axis intact for
//! `int1e_ipovlp_sph` etc.). The Python side reshapes in F-order so the
//! component axis ends up where upstream expects it.
//!
//! Env vars:
//!   - `PYSCF_RS_ORACLE_ATOM`  : atom string
//!   - `PYSCF_RS_ORACLE_BASIS` : basis name
//!   - `PYSCF_RS_ORACLE_INTOR` : intor symbol (e.g. `"int1e_ovlp_sph"`)
//!   - `PYSCF_RS_ORACLE_OUT`   : output JSON path

#![cfg(feature = "release-oracle-tests")]

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs, intor};
use std::path::PathBuf;

#[test]
#[ignore = "oracle helper — invoked by python via `cargo test --features release-oracle-tests -- --ignored`"]
fn dump_intor_for_oracle() {
    let atom = std::env::var("PYSCF_RS_ORACLE_ATOM").expect("PYSCF_RS_ORACLE_ATOM env var");
    let basis = std::env::var("PYSCF_RS_ORACLE_BASIS").expect("PYSCF_RS_ORACLE_BASIS env var");
    let intor_name = std::env::var("PYSCF_RS_ORACLE_INTOR").expect("PYSCF_RS_ORACLE_INTOR env var");
    let out = std::env::var("PYSCF_RS_ORACLE_OUT").expect("PYSCF_RS_ORACLE_OUT env var");

    let mol = M(MoleBuildArgs {
        atom: AtomInput::String(atom),
        basis: BasisInput::Name(basis),
        unit: Unit::Bohr,
        ..Default::default()
    })
    .expect("M(...) must succeed for oracle fixture");

    let result = intor(&mol, &intor_name).expect("intor(...) must succeed");

    let layout_tag = match result.layout {
        pyscf_gto::layout_table::IntorLayout::ScalarFOrder => "scalar".to_string(),
        pyscf_gto::layout_table::IntorLayout::ComponentLeadingFOrder { components } => {
            format!("component_leading:{components}")
        }
    };
    let payload = serde_json::json!({
        "values": result.values,
        "shape":  result.shape,
        "layout": layout_tag,
    });
    std::fs::write(PathBuf::from(out), payload.to_string()).expect("write json");
}
