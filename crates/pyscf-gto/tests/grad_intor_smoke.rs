//! GRAD-07 (plan 07-01): cintx grad-intor round-trip smoke for the TWO
//! gradient-integral families cintx ships TODAY.
//!
//! cintx grad-intor availability matrix (re-confirmed at plan-execution time
//! against the live cintx manifest) — the ready families are growing:
//!   - `int3c2e_ip1_sph`     (DF-gradient 3-center derivative) — present
//!   - `int1e_ecp_ipnuc_sph` (= `ECPscalar_ipnuc`, ECP gradient) — present
//!   - `int1e_ecp_iprinv_sph` (ECP per-atom iprinv) — present (cintx 21-07, F-05)
//!   - `int1e_iprinv_sph`    (non-ECP nuclear-attraction per-atom derivative) —
//!     present and EVALUATES once a `rinv` origin is supplied
//!     (`intor_with_rinv_origin` / `with_rinv_at_nucleus`, F-08 / quick 260601-sln).
//!
//! `int1e_iprinv` is NOT cintx-missing: it is a real `AllCint1e` operator that
//! evaluates the moment a `rinv_orig` is threaded into `ExecutionOptions`. Plain
//! origin-less `intor(mol, "int1e_iprinv")` still fails the cintx validator
//! (`InvalidEnvParam{PTR_RINV_ORIG}`) by design — the origin is mandatory.
//!
//! Still genuinely MISSING from cintx (route to a clean availability error):
//! `int2e_ip1`, `int1e_ip{ovlp,kin,nuc}`.
//!
//! This file is the always-on structural round-trip for the ready families
//! (the 02-01 cintx-smoke pattern): SHAPE + FINITE, plus the F-08
//! translational-invariance physics oracle (Σ_atoms (-Z)·iprinv|@atom ==
//! int1e_ipnuc) — NO live PySCF, NO libxc. Upstream byte-identity is the
//! venv-gated / workflow_dispatch arm.

use cintx_core::Representation;
use cintx_ops::resolver::Resolver;
use cintx_rs::SessionRequest;
use cintx_runtime::ExecutionOptions;
use pyscf_algebra::oracle_sum;
use pyscf_core::{EcpEngine, Unit};
use pyscf_gto::layout_table::IntorLayout;
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

/// H2/STO-3G as a fully-built Mole (2 nuclei, non-ECP, 2 AOs) — the fixture for
/// the F-08 `int1e_iprinv` origin entry points + the translational-invariance
/// oracle (multiple nuclei, so the Σ-over-atoms identity is non-trivial).
fn h2_sto3g_mol() -> pyscf_core::Mole {
    M(MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0; H 0 0 1.4".into()),
        basis: BasisInput::Name("sto-3g".into()),
        unit: Unit::Bohr,
        max_memory: 4000.0,
        axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ..Default::default()
    })
    .expect("H2/STO-3G build")
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
        assert!(
            v.is_finite(),
            "int3c2e_ip1_sph block[{i}] = {v} must be finite"
        );
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
    assert_eq!(
        density.nao, nao,
        "ecp_int1e_ipnuc nao axis must equal mol.nao_nr"
    );
    assert_eq!(
        density.data.len(),
        3 * nao * nao,
        "ecp_int1e_ipnuc must be a [3, nao, nao] component-leading buffer (3*nao*nao); \
         never [nao, nao, 3] (T-07-01)",
    );

    for (i, &v) in density.data.iter().enumerate() {
        assert!(
            v.is_finite(),
            "ecp_int1e_ipnuc data[{i}] = {v} must be finite"
        );
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

// ── F-05: ECP per-atom iprinv (un-gated by cintx 21-07) ──────────────────────
// F-05 / cintx 21-07: iprinv un-gated; the prior assertion (iprinv → cintx
// -availability error) is superseded. iprinv now evaluates through the dedicated
// CintxEcpEngine::ecp_int1e_iprinv method with a per-atom rinv origin.

#[test]
fn ecp_iprinv_evaluates_real_per_atom_buffer() {
    // F-05: the per-atom iprinv ECP-gradient family is cintx-READY (workstream
    // 21-07). Round-trip end-to-end through CintxEcpEngine::ecp_int1e_iprinv on
    // an ECP-bearing molecule, with the rinv origin set to the Cu nucleus; assert
    // the [3, nao, nao] component-leading buffer (3*nao*nao flat), finite, and
    // carrying real (non-zero-fill) values.
    let mol = cu_lanl2dz();
    let nao = mol.nao_nr;
    assert!(nao > 0, "Cu/LANL2DZ must have a non-empty AO basis");

    let engine = CintxEcpEngine;
    let origin = mol.atom_coord(0); // the Cu nucleus (Bohr).
    let density = engine
        .ecp_int1e_iprinv(&mol, "ECPscalar_iprinv", origin)
        .expect("int1e_ecp_iprinv must evaluate via the cintx ECP engine (F-05 / cintx 21-07)");

    assert_eq!(
        density.nao, nao,
        "ecp_int1e_iprinv nao axis must equal mol.nao_nr"
    );
    assert_eq!(
        density.data.len(),
        3 * nao * nao,
        "ecp_int1e_iprinv must be a [3, nao, nao] component-leading buffer (3*nao*nao); \
         never [nao, nao, 3] (T-07-01)",
    );
    for (i, &v) in density.data.iter().enumerate() {
        assert!(
            v.is_finite(),
            "ecp_int1e_iprinv data[{i}] = {v} must be finite"
        );
    }
    let nonzero = density.data.iter().filter(|&&v| v.abs() > 1e-18).count();
    assert!(
        nonzero > 0,
        "ecp_int1e_iprinv is all-zeros at the Cu nucleus — cintx ECP-iprinv not delivering \
         real values (regression to a zero-fill stub?)",
    );
}

#[test]
fn ecp_iprinv_at_cu_equals_ipnuc_single_nucleus() {
    // SELF-CONSISTENCY / structural smoke (NOT an external oracle): for the
    // single-ECP-atom Cu/LANL2DZ fixture (exactly ONE ECP nucleus, at atom index
    // 0) the per-nucleus rinv selection degenerates — `iprinv@Cu == ipnuc`,
    // because with a single ECP nucleus the per-atom rinv selection coincides
    // with the all-slot accumulation. This compares the cintx iprinv kernel
    // against the cintx ipnuc kernel through the SAME engine + the SAME stitch;
    // if cintx returned the same WRONG value for both, this check would still
    // pass. The math (single-ECP-atom degeneracy) is what makes the compare
    // meaningful, but the EXTERNAL byte-identity vs upstream PySCF nr_ecp_deriv
    // is owned by cintx's own cintx-oracle/tests/ecp_iprinv_parity.rs, NOT here.
    let mol = cu_lanl2dz();
    let nao = mol.nao_nr;
    let engine = CintxEcpEngine;

    let ipnuc = engine
        .ecp_int1e_ipnuc(&mol, "ECPscalar_ipnuc")
        .expect("ECPscalar_ipnuc cintx-ready");
    let iprinv = engine
        .ecp_int1e_iprinv(&mol, "ECPscalar_iprinv", mol.atom_coord(0))
        .expect("ECPscalar_iprinv cintx-ready (F-05)");

    assert_eq!(ipnuc.data.len(), 3 * nao * nao);
    assert_eq!(iprinv.data.len(), ipnuc.data.len());
    for (k, (&a, &b)) in iprinv.data.iter().zip(ipnuc.data.iter()).enumerate() {
        assert!(
            (a - b).abs() <= 1e-12,
            "single-ECP-atom self-consistency broke at element {k}: iprinv={a} vs ipnuc={b} \
             (|Δ|={} > 1e-12)",
            (a - b).abs(),
        );
    }
}

#[test]
fn ecp_iprinv_origin_matching_no_atom_is_all_zeros() {
    // F-05 / T-rhc-02: an iprinv rinv origin matching NO ECP atom must yield an
    // all-zero [3, nao, nao] buffer (the cintx no-match → zero-fill path), never
    // a panic and never a wrong-atom selection.
    let mol = cu_lanl2dz();
    let nao = mol.nao_nr;
    let engine = CintxEcpEngine;
    let density = engine
        .ecp_int1e_iprinv(&mol, "ECPscalar_iprinv", [100.0, 100.0, 100.0])
        .expect("iprinv with a no-match origin must be Ok (zero-fill), not an error");
    assert_eq!(density.data.len(), 3 * nao * nao);
    assert!(
        density.data.iter().all(|&v| v == 0.0),
        "iprinv at an origin matching no ECP atom must be all-zeros",
    );
}

// ── F-08: non-ECP int1e_iprinv via a caller-supplied rinv origin ─────────────
// quick 260601-sln: plumb `rinv_origin` through `intor` for the iprinv arm.
// `int1e_iprinv` is a real cintx AllCint1e operator — it EVALUATES once an
// origin is supplied; only the origin-less default (rinv_orig: None) fails the
// validator (PTR_RINV_ORIG), by design.

#[test]
fn int1e_iprinv_with_origin_evaluates_component_leading() {
    // intor_with_rinv_origin returns a finite, component-leading [3, nao, nao]
    // buffer on a real molecule (was: validator/availability error with the
    // origin-less default None).
    let mol = h2_sto3g_mol();
    let nao = mol.nao_nr;
    assert!(nao > 0, "H2/STO-3G must have a non-empty AO basis");

    let origin = mol.atom_coord(0); // pin rinv at H1 (Bohr).
    let out = pyscf_gto::intor_with_rinv_origin(&mol, "int1e_iprinv", origin)
        .expect("int1e_iprinv must EVALUATE once a rinv origin is supplied (F-08)");

    assert_eq!(
        out.layout,
        IntorLayout::ComponentLeadingFOrder { components: 3 },
        "int1e_iprinv must be 3-component-leading (never [.., 3])",
    );
    assert_eq!(
        out.shape,
        vec![3, nao, nao],
        "int1e_iprinv shape must be [3, nao, nao]",
    );
    assert_eq!(out.values.len(), 3 * nao * nao);
    for (i, &v) in out.values.iter().enumerate() {
        assert!(
            v.is_finite(),
            "int1e_iprinv values[{i}] = {v} must be finite"
        );
    }
}

#[test]
fn int1e_iprinv_at_nucleus_matches_explicit_origin() {
    // The convenience entry point resolves the origin from mol.atom_coord(atm_id)
    // and must agree elementwise with the explicit-origin call.
    let mol = h2_sto3g_mol();
    for atm_id in 0..mol.natm {
        let via_nucleus = pyscf_gto::intor_with_rinv_at_nucleus(&mol, "int1e_iprinv", atm_id)
            .expect("intor_with_rinv_at_nucleus must evaluate");
        let via_origin =
            pyscf_gto::intor_with_rinv_origin(&mol, "int1e_iprinv", mol.atom_coord(atm_id))
                .expect("intor_with_rinv_origin must evaluate");
        assert_eq!(via_nucleus.values.len(), via_origin.values.len());
        for (k, (&a, &b)) in via_nucleus
            .values
            .iter()
            .zip(via_origin.values.iter())
            .enumerate()
        {
            assert_eq!(
                a, b,
                "with_rinv_at_nucleus(atm={atm_id}) must equal with_rinv_origin(atom_coord) at \
                 element {k}: {a} vs {b}",
            );
        }
    }
}

#[test]
fn int1e_iprinv_sum_over_nuclei_equals_ipnuc() {
    // PHYSICS ORACLE (translational-invariance identity, the rigorous in-tree
    // anchor — NO live PySCF, NO libxc):
    //   int1e_ipnuc == Σ_atoms (-Z_atom) · int1e_iprinv|rinv@atom_coord(atom)
    // because the nuclear-attraction operator is Σ_A (-Z_A)/|r - R_A| and
    // int1e_iprinv|@R_A is its per-nucleus contribution. This is exactly the
    // relation hcore_deriv exploits, checked against the already-available
    // int1e_ipnuc (plain intor).
    let mol = h2_sto3g_mol();
    let nao = mol.nao_nr;
    let charges = mol.atom_charges();

    // Reference: int1e_ipnuc (full nuclear-attraction derivative), available today.
    let ipnuc = pyscf_gto::intor(&mol, "int1e_ipnuc").expect("int1e_ipnuc must evaluate");
    assert_eq!(ipnuc.values.len(), 3 * nao * nao);

    // Accumulate Σ_atoms (-Z) · int1e_iprinv|@atom via the project's pairwise sum.
    let mut acc = vec![0.0_f64; 3 * nao * nao];
    for atm_id in 0..mol.natm {
        let z = charges[atm_id] as f64;
        let iprinv = pyscf_gto::intor_with_rinv_at_nucleus(&mol, "int1e_iprinv", atm_id)
            .expect("per-atom int1e_iprinv must evaluate");
        assert_eq!(iprinv.values.len(), 3 * nao * nao);
        for (slot, &v) in acc.iter_mut().zip(iprinv.values.iter()) {
            *slot = oracle_sum(&[*slot, -z * v]);
        }
    }

    let mut max_abs = 0.0_f64;
    for (k, (&got, &want)) in acc.iter().zip(ipnuc.values.iter()).enumerate() {
        let d = (got - want).abs();
        if d > max_abs {
            max_abs = d;
        }
        assert!(
            d <= 1e-10,
            "translational-invariance oracle broke at element {k}: \
             Σ(-Z)·iprinv={got} vs int1e_ipnuc={want} (|Δ|={d} > 1e-10)",
        );
    }
    assert!(
        max_abs <= 1e-10,
        "max |Σ(-Z)·iprinv - int1e_ipnuc| = {max_abs} exceeds 1e-10",
    );
}

#[test]
fn intor_with_rinv_origin_rejects_non_iprinv_name() {
    // The entry point is scoped to the iprinv family — the rinv origin is only
    // meaningful/validated for iprinv. A non-iprinv name (e.g. int1e_ovlp) must
    // error cleanly (never silently ignore the origin / evaluate the wrong op).
    let mol = h2_sto3g_mol();
    let err = pyscf_gto::intor_with_rinv_origin(&mol, "int1e_ovlp", [0.0, 0.0, 0.0])
        .expect_err("a non-iprinv name must error cleanly");
    let msg = format!("{err}");
    assert!(
        msg.contains("iprinv"),
        "error must name the iprinv-family scope, got: {msg}",
    );
}
