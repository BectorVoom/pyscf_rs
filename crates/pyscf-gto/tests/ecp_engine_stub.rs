//! Plan 02-07 Task 1 acceptance tests: the `EcpEngineNotAvailable` stub
//! routes through `intor` and the `EcpEngine` trait.

use pyscf_core::{EcpEngine, PyscfRsError, Unit};
use pyscf_gto::{AtomInput, BasisInput, EcpEngineNotAvailable, M, MoleBuildArgs, intor};

fn h_mol() -> pyscf_core::Mole {
    M(MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0".into()),
        basis: BasisInput::Name("sto-3g".into()),
        unit: Unit::Bohr,
        max_memory: 4000.0,
        axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ..Default::default()
    })
    .expect("H/STO-3G build")
}

#[test]
fn int1e_ecp_routes_through_engine_stub() {
    let mol = h_mol();
    let r = intor(&mol, "int1e_ecp");
    assert!(
        matches!(r, Err(PyscfRsError::EcpEngineNotAvailable)),
        "expected EcpEngineNotAvailable, got {:?}",
        r
    );
}

#[test]
fn int1e_ecp_iprinv_routes_to_engine() {
    let mol = h_mol();
    // Suffix variant (e.g. `int1e_ecp_iprinv`) should match the prefix
    // branch in the dispatcher and end up in the engine stub.
    let r = intor(&mol, "int1e_ecp_iprinv");
    assert!(
        matches!(r, Err(PyscfRsError::EcpEngineNotAvailable)),
        "got {:?}",
        r
    );
}

#[test]
#[allow(non_snake_case)]
fn ECPscalar_prefix_routes_to_engine() {
    let mol = h_mol();
    let r = intor(&mol, "ECPscalar");
    assert!(
        matches!(r, Err(PyscfRsError::EcpEngineNotAvailable)),
        "got {:?}",
        r
    );
}

#[test]
fn engine_ipnuc_returns_phase_7_not_yet_implemented() {
    let mol = h_mol();
    let stub = EcpEngineNotAvailable;
    let r = stub.ecp_int1e_ipnuc(&mol, "int1e_ecp_ipnuc");
    assert!(
        matches!(r, Err(PyscfRsError::NotYetImplemented { phase: 7, .. })),
        "got {:?}",
        r
    );
}

#[test]
fn engine_int1e_returns_engine_not_available() {
    // Direct trait-method call (not through the dispatcher) also errs
    // with the canonical EcpEngineNotAvailable.
    let mol = h_mol();
    let stub = EcpEngineNotAvailable;
    let r = stub.ecp_int1e(&mol, "int1e_ecp");
    assert!(
        matches!(r, Err(PyscfRsError::EcpEngineNotAvailable)),
        "got {:?}",
        r
    );
}
