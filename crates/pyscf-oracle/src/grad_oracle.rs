//! Phase 7 gradient + geomopt oracle fixtures (plan 07-10 — closes the Nyquist
//! validation contract for GRAD-01..07 + GEOMOPT-07).
//!
//! REGISTER-BUT-DEFER-DISPATCH, mirroring the MP2 (`mp2-oracle-upstream-manual`,
//! 05-01/05-08) and CCSD (`ccsd_oracle.rs` / `ccsd-oracle-upstream-manual`,
//! 06-11) precedent and the cintx-gating split locked in 07-01-SUMMARY.
//!
//! TWO TIERS:
//!
//!   * SMALL, ALWAYS-ON (the `cargo test -p pyscf-oracle` default build, NO
//!     `--features python`, NO libxc): the dispatch-layer behaviour — the eight
//!     Phase-7 method names (`nuc_grad_rhf`/`uhf`/`rks`/`uks`/`mp2`/`ccsd`/`ecp`
//!     + `geomopt_h2o`) are REGISTERED in `KNOWN_METHODS` (runner.rs), so an
//!     `oracle_check!` call on a grad/geomopt method is RECOGNISED (it is NOT
//!     rejected as an unknown method); a genuinely unknown grad-shaped name
//!     STILL surfaces `"unknown method"`. These assertions need no libpython and
//!     make NO live PySCF call.
//!
//!     The real ALWAYS-ON NUMERIC proofs are the in-tree FD self-verification
//!     (`verify_fd`, D-01, ≤ 1e-6 Ha/Bohr — NO upstream PySCF) + the structural
//!     arms (`single_cphf_impl`, `atmlst`, optimizer convergence + bmatrix/rfo/
//!     conv_defaults) living in `crates/pyscf-grad/tests` + `crates/pyscf-geomopt/
//!     tests`, run by the always-on `grad-structural` CI arm. This module is the
//!     ORACLE-side registration + the gated byte-identity arms, not a duplicate
//!     of those numeric smokes.
//!
//!   * HEAVY, `#[cfg(feature = "python")]`-GATED + `#[ignore]`'d (the
//!     `workflow_dispatch` / human-verify `grad-oracle-upstream-manual` CI arm
//!     ONLY): per-method analytical-gradient byte-identity (≤ 1e-7 Ha/Bohr vs
//!     `pyscf/grad/*`) + the geomopt trajectory / stationary-point parity vs
//!     `geometric_solver`. These require an installed upstream PySCF (+
//!     geomeTRIC) — NEVER always-on.
//!
//! cintx gate (07-01-SUMMARY, D-02): SIX of the eight gradient-integral families
//! (`int2e_ip1`, `int1e_ip{ovlp,kin,nuc,rinv}`, `ECPscalar_iprinv` + the
//! `with_rinv_at_nucleus` origin shift) are MISSING from cintx today, with NO
//! scheduled cintx workstream — so the per-method numeric byte-identity arms
//! stay `#[ignore]`'d behind BOTH the `python` feature AND that (future) cintx
//! prerequisite. Only `int3c2e_ip1` (DF-grad) and `int1e_ecp_ipnuc`
//! (= `ECPscalar_ipnuc`, the ECP `get_hcore` term) are cintx-ready, but their
//! upstream byte-identity still needs an installed PySCF, so they too live on
//! the `workflow_dispatch` arm. The DEFAULT `pyscf-oracle` build pulls NO libxc
//! and makes NO live PySCF call (ORACLE_NO_LIBXC discipline, 06-11).

// ── Always-on: dispatch-layer behaviour for the grad/geomopt arms (no Python) ─
//
// The eight Phase-7 method names registered in `KNOWN_METHODS` (runner.rs) are
// recognised by the dispatch guard. Without `--features python` each known
// method short-circuits to `PythonFeatureNotEnabled` (NOT `UnknownMethod`) —
// this is the always-on proof that the grad/geomopt names are registered without
// pulling libpython or making any live PySCF call.

#[cfg(test)]
mod dispatch_layer {
    use crate::run_oracle_check;
    use crate::runner::OracleError;

    /// The eight Phase-7 grad/geomopt arm names registered in 07-10.
    const GRAD_GEOMOPT_METHODS: &[&str] = &[
        "nuc_grad_rhf",
        "nuc_grad_uhf",
        "nuc_grad_rks",
        "nuc_grad_uks",
        "nuc_grad_mp2",
        "nuc_grad_ccsd",
        "nuc_grad_ecp",
        "geomopt_h2o",
    ];

    /// Every registered grad/geomopt method is RECOGNISED by the dispatch guard:
    /// it must NEVER surface `UnknownMethod`. In a NO-`python` build it
    /// short-circuits to `PythonFeatureNotEnabled` (the always-on proof that the
    /// names are registered); in a `--features python` build it would reach the
    /// live dispatch. Either way it is distinctly NOT `UnknownMethod`.
    #[test]
    fn registered_grad_geomopt_methods_are_never_unknown() {
        for &m in GRAD_GEOMOPT_METHODS {
            let r = run_oracle_check(m, "h2o_ccpvdz", 1e-6);
            assert!(
                !matches!(r, Err(OracleError::UnknownMethod(_))),
                "grad/geomopt method '{}' must be registered (never UnknownMethod), got {:?}",
                m,
                r
            );
        }
    }

    /// Without `--features python`, a KNOWN grad method short-circuits to
    /// `PythonFeatureNotEnabled` — the always-on proof that the name is
    /// registered (NOT `UnknownMethod`) and that the default build makes no live
    /// PySCF call.
    #[cfg(not(feature = "python"))]
    #[test]
    fn known_grad_method_without_python_is_feature_gated_not_unknown() {
        let r = run_oracle_check("nuc_grad_rhf", "h2o_ccpvdz", 1e-6);
        assert!(
            matches!(r, Err(OracleError::PythonFeatureNotEnabled)),
            "expected PythonFeatureNotEnabled for nuc_grad_rhf without python, got {:?}",
            r
        );
    }

    /// A genuinely unknown grad-shaped method name still surfaces
    /// `UnknownMethod` — the dispatch table rejects it BEFORE any Python work, in
    /// every build mode. (Guards against a typo'd method name silently passing.)
    #[test]
    fn unknown_grad_method_is_rejected() {
        let r = run_oracle_check("nuc_grad_does_not_exist_xyz", "h2o_ccpvdz", 1.0);
        match r {
            Err(OracleError::UnknownMethod(name)) => {
                assert_eq!(name, "nuc_grad_does_not_exist_xyz");
            }
            other => panic!(
                "expected UnknownMethod for an unregistered grad name, got {:?}",
                other
            ),
        }
    }

    /// All eight Phase-7 names are present in the catalogue (the lockstep with
    /// the `known_methods_list_has_all_arms` len==32 assertion in runner.rs).
    #[test]
    fn all_eight_phase7_names_registered() {
        assert_eq!(GRAD_GEOMOPT_METHODS.len(), 8);
        for &m in GRAD_GEOMOPT_METHODS {
            // A round-trip through the dispatch guard proves registration: an
            // unregistered name would return UnknownMethod here.
            let r = run_oracle_check(m, "h2o_ccpvdz", 1e-6);
            assert!(
                !matches!(r, Err(OracleError::UnknownMethod(_))),
                "'{}' is not registered in KNOWN_METHODS",
                m
            );
        }
    }
}

// ── Python-feature-gated: the live grad/geomopt byte-identity oracle arms ─────
//
// `workflow_dispatch` / human-verify ONLY. Each arm is `#[ignore]`'d so even the
// `--features python` CI build does not auto-run it on push; the
// `grad-oracle-upstream-manual` arm runs them explicitly with
// `-- --include-ignored` against an installed upstream PySCF (+ geomeTRIC). The
// live dispatch wiring (the per-method `check_*` helpers in runner.rs
// `python_impl`) is the documented phase-gate follow-up that lands WITH the cintx
// grad-intor workstream — until both arrive these arms drive the registered names
// against upstream and are the tracked human-verify step.

#[cfg(all(test, feature = "python"))]
mod live_arms {
    use crate::oracle_check;

    /// GRAD-01: RHF analytical nuclear gradient byte-identity (≤ 1e-7 Ha/Bohr)
    /// vs upstream `scf.RHF(mol).run().nuc_grad_method().kernel()`. Gated on the
    /// cintx grad-intor workstream (`int2e_ip1` + `int1e_ip{ovlp,kin,nuc,rinv}`
    /// MISSING from cintx today, 07-01).
    #[test]
    #[ignore = "workflow_dispatch human-verify: needs upstream PySCF + the cintx grad-intor workstream (int2e_ip1 + int1e_ip* MISSING)"]
    fn arm_nuc_grad_rhf() {
        oracle_check!("nuc_grad_rhf", crate::H2O_CC_PVDZ, 1e-7);
    }

    /// GRAD-02: UHF open-shell analytical gradient byte-identity vs upstream
    /// `scf.UHF(mol).run().nuc_grad_method().kernel()`.
    #[test]
    #[ignore = "workflow_dispatch human-verify: needs upstream PySCF + the cintx grad-intor workstream"]
    fn arm_nuc_grad_uhf() {
        oracle_check!("nuc_grad_uhf", crate::H2O_TRIPLET_CCPVDZ, 1e-7);
    }

    /// GRAD-03: RKS analytical gradient (with the `grid_response` Becke-weight
    /// derivative term) byte-identity vs upstream
    /// `dft.RKS(mol).run().nuc_grad_method().kernel()`.
    #[test]
    #[ignore = "workflow_dispatch human-verify: needs upstream PySCF + the cintx grad-intor workstream"]
    fn arm_nuc_grad_rks() {
        oracle_check!("nuc_grad_rks", crate::H2O_CC_PVDZ, 1e-7);
    }

    /// GRAD-04: UKS analytical gradient byte-identity vs upstream
    /// `dft.UKS(mol).run().nuc_grad_method().kernel()`.
    #[test]
    #[ignore = "workflow_dispatch human-verify: needs upstream PySCF + the cintx grad-intor workstream"]
    fn arm_nuc_grad_uks() {
        oracle_check!("nuc_grad_uks", crate::H2O_TRIPLET_CCPVDZ, 1e-7);
    }

    /// GRAD-05: MP2 analytical gradient (Z-vector via the single CPHF solver,
    /// `max_cycle=30`) byte-identity vs upstream
    /// `mp.MP2(mf).run().nuc_grad_method().kernel()`.
    #[test]
    #[ignore = "workflow_dispatch human-verify: needs upstream PySCF + the cintx grad-intor workstream"]
    fn arm_nuc_grad_mp2() {
        oracle_check!("nuc_grad_mp2", crate::H2O_CC_PVDZ, 1e-7);
    }

    /// GRAD-06: CCSD analytical gradient (Λ + Z-vector, consuming the Phase-6
    /// `solve_lambda` + `make_rdm1`/`make_rdm2`, D-04) byte-identity vs upstream
    /// `cc.CCSD(mf).run().nuc_grad_method().kernel()`.
    #[test]
    #[ignore = "workflow_dispatch human-verify: needs upstream PySCF + the cintx grad-intor workstream"]
    fn arm_nuc_grad_ccsd() {
        oracle_check!("nuc_grad_ccsd", crate::H2O_CC_PVDZ, 1e-7);
    }

    /// GRAD-07: ECP analytical gradient byte-identity vs upstream. The
    /// `ECPscalar_ipnuc` `get_hcore` term is cintx-ready (07-01); the
    /// `ECPscalar_iprinv` `hcore_deriv` per-atom term is MISSING from cintx —
    /// the full byte-identity therefore still rides the cintx workstream.
    #[test]
    #[ignore = "workflow_dispatch human-verify: needs upstream PySCF; ECPscalar_iprinv still MISSING from cintx (07-01)"]
    fn arm_nuc_grad_ecp() {
        oracle_check!("nuc_grad_ecp", crate::H2O_CC_PVDZ, 1e-7);
    }

    /// GEOMOPT-07: H2O geometry-optimizer trajectory / stationary-point parity
    /// vs upstream `geometric_solver`. The bar is "same stationary point within
    /// chemical accuracy", not bit-for-bit (geomeTRIC not importable in the
    /// sandbox; the optimizer drives the analytical gradient, so this also rides
    /// the cintx grad-intor gate).
    #[test]
    #[ignore = "workflow_dispatch human-verify: needs upstream PySCF + geomeTRIC + the cintx grad-intor workstream"]
    fn arm_geomopt_h2o() {
        oracle_check!("geomopt_h2o", crate::H2O_CC_PVDZ, 1e-7);
    }
}
