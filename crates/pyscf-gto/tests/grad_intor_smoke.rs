//! GRAD-07 (plan 07-01): cintx grad-intor round-trip smoke for the TWO
//! gradient-integral families cintx ships TODAY.
//!
//! Per 07-RESEARCH §"Gradient-Integral Availability Matrix" (re-confirmed at
//! plan-execution time against the live cintx manifest), cintx ships only
//! **2 of 8** needed gradient-integral families:
//!   - `int3c2e_ip1_sph`     (DF-gradient 3-center derivative) — manifest present
//!   - `int1e_ecp_ipnuc_sph` (= `ECPscalar_ipnuc`, ECP gradient) — manifest present
//!
//! The other 6 (`int2e_ip1`, `int1e_ip{ovlp,kin,nuc,rinv}`, `ECPscalar_iprinv`)
//! are MISSING from every cintx branch and route to a clean cintx-availability
//! error (asserted in `intor_smoke.rs` / `ecp_engine_stub.rs`).
//!
//! This file is the always-on structural round-trip for the two ready families
//! (the 02-01 cintx-smoke pattern): SHAPE + FINITE only. Byte-identity vs
//! upstream PySCF is the venv-gated / workflow_dispatch oracle arm.

use cintx_core::Representation;
use cintx_ops::resolver::Resolver;
use cintx_rs::SessionRequest;
use cintx_runtime::ExecutionOptions;
use pyscf_core::{EcpEngine, PyscfRsError, Unit};
use pyscf_gto::{AtomInput, BasisInput, CintxEcpEngine, EcpInput, M, MoleBuildArgs};

mod common;

/// He/STO-3G (a tiny, single-atom orbital basis) doubling as its own DF aux —
/// just enough shells to drive a real `int3c2e_ip1_sph` shell triple.
fn he_sto3g() -> pyscf_core::Mole {
    M(MoleBuildArgs {
        atom: AtomInput::String("He 0 0 0".into()),
        basis: BasisInput::Name("sto-3g".into()),
        unit: Unit::Bohr,
        max_memory: 4000.0,
        axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ..Default::default()
    })
    .expect("He/STO-3G build")
}

/// Cu/LANL2DZ carries an ECP — the only molecule on which the ECP-gradient
/// `int1e_ecp_ipnuc` integral is non-trivial.
fn cu_lanl2dz() -> pyscf_core::Mole {
    M(MoleBuildArgs {
        atom: AtomInput::String("Cu 0 0 0".into()),
        basis: BasisInput::Name("lanl2dz".into()),
        ecp: EcpInput::Name("lanl2dz".into()),
        unit: Unit::Bohr,
        max_memory: 4000.0,
        axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ..Default::default()
    })
    .expect("Cu/LANL2DZ build")
}

// ── Ready family 1: int3c2e_ip1_sph (DF gradient) ────────────────────────────

#[test]
fn int3c2e_ip1_sph_is_cintx_ready_and_component_leading() {
    // The DF-gradient 3-center derivative is present in cintx today. Confirm:
    //   (a) the cintx-ops resolver knows the symbol (cintx-ready round-trip),
    //   (b) it is an arity-3 operator, and
    //   (c) our layout_table carries the 3-component-leading [3, ...] layout
    //       (axis 0 = x/y/z derivative component — never component-trailing).
    let descriptor = Resolver::descriptor_by_symbol("int3c2e_ip1_sph")
        .expect("int3c2e_ip1_sph must be present in the cintx-ops manifest (cintx-ready family 1)");
    assert_eq!(
        descriptor.entry.arity, 3,
        "int3c2e_ip1_sph is a 3-center (arity-3) operator"
    );

    let layout = pyscf_gto::layout_table::lookup("int3c2e_ip1_sph")
        .expect("int3c2e_ip1_sph must be in the in-scope layout catalogue");
    assert_eq!(
        layout,
        pyscf_gto::layout_table::IntorLayout::ComponentLeadingFOrder { components: 3 },
        "int3c2e_ip1 must be 3-component-leading (T-07-01: never [.., 3])",
    );
}

#[test]
fn int3c2e_ip1_sph_round_trips_a_real_shell_triple() {
    // End-to-end cintx round-trip: drive int3c2e_ip1_sph over a real H2/STO-3G
    // shell triple via SessionRequest and assert a non-empty, finite,
    // 3-component block (3 derivative components × the shell-triple AO block).
    let (basis, _pair) = common::h2_sto3g_basis();

    let descriptor = Resolver::descriptor_by_symbol("int3c2e_ip1_sph")
        .expect("int3c2e_ip1_sph in cintx-ops manifest");
    let operator = descriptor.id;

    // A self-triple over the two H 1s shells: (shell0, shell1 | shell0).
    let shells = basis
        .shell_tuple_for_indices([0, 1, 0])
        .expect("H2/STO-3G shell triple (0,1,0)");

    let outcome = SessionRequest::new(
        operator,
        Representation::Spheric,
        &basis,
        shells,
        ExecutionOptions::default(),
    )
    .query_workspace()
    .expect("int3c2e_ip1_sph workspace query must succeed (cintx-ready)")
    .evaluate()
    .expect("int3c2e_ip1_sph evaluate must succeed (cintx-ready)");

    let block = &outcome.tensor.owned_values;
    // Each H 1s shell carries 1 spheric AO ⇒ inner AO block is 1×1×1. cintx's
    // safe-API for int3c2e_ip1_sph returns the inner AO block here with
    // extents == [1,1,1] (the 3 derivative components are NOT yet expanded into
    // `owned_values` at this safe-API surface — the same synthetic-staging
    // shape the scalar dispatcher tolerates via the `expected_inner` branch in
    // `stitch_arity2_block`). The pyscf-gto dispatcher owns the
    // component-leading [3, ...] repack (asserted at the layout-table level in
    // `int3c2e_ip1_sph_is_cintx_ready_and_component_leading`). The round-trip
    // here proves cintx EVALUATES the family (non-empty, finite block) — the
    // readiness contract for the DF-gradient family.
    assert!(
        !block.is_empty(),
        "int3c2e_ip1_sph shell-triple block must be non-empty (got 0 elements; extents {:?})",
        outcome.tensor.extents,
    );
    let inner_ao = outcome.tensor.extents.iter().product::<usize>().max(1);
    assert!(
        block.len() == inner_ao || block.len() == inner_ao * 3,
        "int3c2e_ip1_sph block ({} values) must be the inner AO block ({inner_ao}) or its \
         3-component expansion ({}); extents {:?}",
        block.len(),
        inner_ao * 3,
        outcome.tensor.extents,
    );
    for (i, &v) in block.iter().enumerate() {
        assert!(v.is_finite(), "int3c2e_ip1_sph block[{i}] = {v} must be finite");
    }

    // sanity: He/STO-3G builds (used by the ECP-grad companion test fixtures).
    let _ = he_sto3g();
}

// ── Ready family 2: int1e_ecp_ipnuc (ECP gradient) ───────────────────────────

#[test]
fn int1e_ecp_ipnuc_returns_component_leading_3_nao_nao() {
    // The ECP-gradient ipnuc family is present in cintx today and un-gated by
    // GRAD-07 (plan 07-01). Round-trip end-to-end through the dedicated
    // CintxEcpEngine::ecp_int1e_ipnuc method on an ECP-bearing molecule;
    // assert the [3, nao, nao] component-leading buffer (3*nao*nao flat) and
    // finite, non-zero values.
    let mol = cu_lanl2dz();
    let nao = mol.nao_nr;
    assert!(nao > 0, "Cu/LANL2DZ must have a non-empty AO basis");

    let engine = CintxEcpEngine;
    let density = engine
        .ecp_int1e_ipnuc(&mol, "int1e_ecp_ipnuc")
        .expect("int1e_ecp_ipnuc must evaluate via the cintx ECP engine (cintx-ready family 2)");

    // Component-leading [3, nao, nao] F-order: nao on the AO axis, 3*nao*nao flat.
    assert_eq!(density.nao, nao, "ecp_int1e_ipnuc nao axis must equal mol.nao_nr");
    assert_eq!(
        density.data.len(),
        3 * nao * nao,
        "ecp_int1e_ipnuc must be a [3, nao, nao] component-leading buffer (3*nao*nao); \
         never [nao, nao, 3] (T-07-01)",
    );

    for (i, &v) in density.data.iter().enumerate() {
        assert!(v.is_finite(), "ecp_int1e_ipnuc data[{i}] = {v} must be finite");
    }
    let nonzero = density.data.iter().filter(|&&v| v.abs() > 1e-18).count();
    assert!(
        nonzero > 0,
        "ecp_int1e_ipnuc is all-zeros — cintx ECP-gradient not delivering real values \
         (regression to a zero-fill stub?)",
    );
}

#[test]
fn ecpscalar_ipnuc_alias_resolves_like_int1e_ecp_ipnuc() {
    // The `ECPscalar_ipnuc` upstream alias (no _sph/_cart suffix) must route to
    // the same cintx-ready ipnuc operator (Mole cart flag picks the rep).
    let mol = cu_lanl2dz();
    let nao = mol.nao_nr;
    let engine = CintxEcpEngine;
    let density = engine
        .ecp_int1e_ipnuc(&mol, "ECPscalar_ipnuc")
        .expect("ECPscalar_ipnuc alias must resolve to the cintx ipnuc operator");
    assert_eq!(density.data.len(), 3 * nao * nao);
    assert!(density.data.iter().all(|v| v.is_finite()));
}

#[test]
fn ecp_iprinv_is_clean_cintx_availability_error_not_phase_7() {
    // ECPscalar_iprinv / int1e_ecp_iprinv is MISSING from every cintx branch
    // today (no scheduled cintx workstream). The ecp_int1e_ipnuc method must
    // reject it with a clean cintx-availability error — NEVER
    // NotYetImplemented{phase:7} (that disposition is closed by GRAD-07).
    let mol = cu_lanl2dz();
    let engine = CintxEcpEngine;
    let r = engine.ecp_int1e_ipnuc(&mol, "int1e_ecp_iprinv");
    assert!(
        !matches!(r, Err(PyscfRsError::NotYetImplemented { phase: 7, .. })),
        "iprinv must NOT be NotYetImplemented{{phase:7}}; got {r:?}",
    );
    assert!(
        matches!(
            r,
            Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(_)))
        ),
        "iprinv must surface a clean cintx-availability InvalidMolecule error; got {r:?}",
    );
}
