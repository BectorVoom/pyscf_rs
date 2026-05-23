//! Plan 02-07 Task 1 acceptance tests — UPDATED by gap-closure plan 02-10.
//!
//! 02-07 shipped the `EcpEngineNotAvailable` stub as the default engine.
//! 02-10 swapped `pyscf_gto::ecp_engine()` to the cintx-backed
//! `CintxEcpEngine`. The stub stays in the codebase (documentation +
//! testable error path); these tests now exercise it DIRECTLY via
//! `EcpEngineNotAvailable` rather than through `pyscf_gto::ecp_engine()`.
//!
//! The dispatcher-routing tests below still hold: an ECP-less molecule
//! (H/STO-3G) routed through `intor("int1e_ecp")` / `intor("ECPscalar")`
//! reaches the cintx engine, which returns the canonical
//! `PyscfRsError::EcpEngineNotAvailable` for a molecule with no ECP entries
//! (`mol._ecp.is_empty()`). Derivative ECP names (`int1e_ecp_iprinv`,
//! `int1e_ecp_ipnuc`) instead return `NotYetImplemented{phase:7}` up front —
//! independent of ECP presence — per 02-10 code-review WR-01.

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
fn int1e_ecp_on_ecpless_mol_returns_engine_not_available() {
    // H/STO-3G has no ECP. Routing int1e_ecp through the (now cintx-backed)
    // dispatcher must still surface EcpEngineNotAvailable, because the engine
    // guards mol._ecp.is_empty().
    let mol = h_mol();
    let r = intor(&mol, "int1e_ecp");
    assert!(
        matches!(r, Err(PyscfRsError::EcpEngineNotAvailable)),
        "expected EcpEngineNotAvailable for ECP-less mol, got {:?}",
        r
    );
}

#[test]
fn int1e_ecp_iprinv_returns_phase_7_not_yet_implemented() {
    // WR-01 (02-10 code review): `int1e_ecp_iprinv` is an ECP *gradient*
    // (derivative) name. The cintx engine rejects derivative ECP names up
    // front with NotYetImplemented{phase:7}, INDEPENDENT of whether the
    // molecule carries an ECP — so it can never silently resolve to the
    // scalar operator. (Previously this passed only because the ECP-less
    // guard happened to fire first.) Phase 7 GRAD-07 wires the real gradient.
    let mol = h_mol();
    let r = intor(&mol, "int1e_ecp_iprinv");
    assert!(
        matches!(r, Err(PyscfRsError::NotYetImplemented { phase: 7, .. })),
        "expected NotYetImplemented{{phase:7}} for derivative ECP name, got {:?}",
        r
    );
}

#[test]
#[allow(non_snake_case)]
fn ECPscalar_prefix_on_ecpless_mol_returns_engine_not_available() {
    let mol = h_mol();
    let r = intor(&mol, "ECPscalar");
    assert!(
        matches!(r, Err(PyscfRsError::EcpEngineNotAvailable)),
        "got {:?}",
        r
    );
}

#[test]
fn stub_ipnuc_returns_phase_7_not_yet_implemented() {
    // Direct stub instantiation (NOT via ecp_engine(), which now returns the
    // cintx engine). The stub's default ecp_int1e_ipnuc still gates Phase 7.
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
fn stub_int1e_returns_engine_not_available() {
    // Direct stub trait-method call — the stub always errs with the canonical
    // EcpEngineNotAvailable. Demoted from the default engine by 02-10 but
    // retained as a documented, testable error path.
    let mol = h_mol();
    let stub = EcpEngineNotAvailable;
    let r = stub.ecp_int1e(&mol, "int1e_ecp");
    assert!(
        matches!(r, Err(PyscfRsError::EcpEngineNotAvailable)),
        "got {:?}",
        r
    );
}
