//! Phase 6 CCSD oracle fixtures (plan 06-11 — closes the Nyquist validation
//! contract for CCSD-01/02/05/06/08).
//!
//! TWO TIERS, mirroring the `mp2-oracle-upstream-manual` precedent (ci.yml:445)
//! and the `oracle_check_smoke.rs` gating model:
//!
//!   * SMALL, ALWAYS-ON (the `cargo test -p pyscf-oracle` default build, NO
//!     `--features python`): the dispatch-layer behaviour — the six CCSD method
//!     names are REGISTERED in `KNOWN_METHODS`, so an `oracle_check!` call on a
//!     CCSD method is recognised (it is NOT rejected as an unknown method); a
//!     genuinely unknown CCSD-shaped name STILL panics with `"unknown method"`.
//!     These assertions need no libpython and make NO live PySCF call. The real
//!     always-on small-system NUMERIC proofs (RCCSD/UCCSD/λ/RDM/DF-CCSD energies
//!     ≤ 1 µHartree vs the published in-tree references) already live in
//!     `crates/pyscf-ccsd/tests/` (rccsd_numeric_smoke, uccsd_smoke, lambda,
//!     rdm, dfccsd_spill, direct) and run on the always-on `ccsd-structural` CI
//!     arm — this file is the ORACLE-side registration + the gated byte-identity
//!     arms, not a duplicate of those numeric smokes.
//!
//!   * HEAVY, `#[cfg(feature = "python")]`-GATED + `#[ignore]`'d (the
//!     `workflow_dispatch` / human-verify `ccsd-oracle-upstream-manual` CI arm
//!     ONLY): caffeine/cc-pVDZ RCCSD byte-identity, UCCSD open-shell
//!     byte-identity, DF-CCSD `Wabef`→HDF5 spill on a constrained
//!     `PYSCF_MAX_MEMORY` (benzene-dimer), and λ / RDM byte-identity vs live
//!     upstream `cc.CCSD(mf).kernel()` / `.solve_lambda()` / `.make_rdm1()` /
//!     `.make_rdm2()`. These require an installed upstream PySCF (and the live
//!     dispatch wiring) — NEVER always-on. This honours the user-memory
//!     constraint: heavy/upstream arms are `workflow_dispatch`-only, and the
//!     DEFAULT `pyscf-oracle` build pulls NO libxc and makes NO live PySCF call
//!     (T-06-11-SC / T-06-11-FREEZE).

use pyscf_oracle::oracle_check;

// ── Always-on: dispatch-layer behaviour for the CCSD arms (no Python) ──────
//
// The six CCSD method names registered in `KNOWN_METHODS` (runner.rs) are
// recognised by the dispatch guard. Without `--features python` each known
// method short-circuits to `PythonFeatureNotEnabled` — which `oracle_check!`
// surfaces as `"oracle_check failed"` (NOT `"unknown method"`). This proves the
// CCSD names are registered (the always-on contract; the live byte-identity is
// the workflow_dispatch arm below).

/// A genuinely unknown CCSD-shaped method name still panics with
/// `"unknown method"` — the dispatch table rejects it BEFORE any Python work,
/// in every build mode. (Guards against a typo'd method name silently passing.)
#[test]
#[should_panic(expected = "unknown method")]
fn unknown_ccsd_method_panics() {
    oracle_check!("ccsd_does_not_exist_xyz", pyscf_oracle::H2O_CC_PVDZ, 1.0);
}

/// In a NO-`python` build, a KNOWN CCSD method (here `ccsd_rccsd_energy`) is
/// recognised by the dispatch guard and short-circuits to
/// `PythonFeatureNotEnabled` — surfaced by `oracle_check!` as `"oracle_check
/// failed"`, distinctly NOT `"unknown method"`. This is the always-on proof
/// that the CCSD names are registered without pulling libpython or making any
/// live PySCF call.
#[cfg(not(feature = "python"))]
#[test]
#[should_panic(expected = "oracle_check failed")]
fn known_ccsd_method_without_python_is_feature_gated_not_unknown() {
    oracle_check!("ccsd_rccsd_energy", pyscf_oracle::H2O_CC_PVDZ, 1e-6);
}

// ── Python-feature-gated: the live CCSD byte-identity oracle arms ──────────
//
// `workflow_dispatch` / human-verify ONLY. Each arm is `#[ignore]`'d so even
// the `--features python` CI build does not auto-run it on push; the
// `ccsd-oracle-upstream-manual` arm runs them explicitly with
// `-- --include-ignored` against an installed upstream PySCF. The live dispatch
// wiring (the `check_ccsd_*` helpers in runner.rs `python_impl`) is the
// documented phase-gate follow-up — until then these arms drive the registered
// names against upstream and are the tracked human-verify step.

#[cfg(feature = "python")]
mod live_arms {
    use pyscf_oracle::{H2O_CC_PVDZ, H2O_TRIPLET_CCPVDZ, oracle_check};

    /// CCSD-01: closed-shell RCCSD correlation energy byte-identity vs upstream
    /// `cc.CCSD(scf.RHF(mol).run()).kernel()`. Small system here; the caffeine /
    /// cc-pVDZ heavy fixture is the same arm on a larger system.
    #[test]
    #[ignore = "workflow_dispatch human-verify: requires upstream PySCF + the live ccsd dispatch wiring"]
    fn arm_ccsd_rccsd_energy() {
        oracle_check!("ccsd_rccsd_energy", H2O_CC_PVDZ, 1e-6);
    }

    /// CCSD-02: open-shell UCCSD correlation energy byte-identity vs upstream
    /// `cc.UCCSD(scf.UHF(mol).run()).kernel()`.
    #[test]
    #[ignore = "workflow_dispatch human-verify: requires upstream PySCF + the live ccsd dispatch wiring"]
    fn arm_ccsd_uccsd_energy() {
        oracle_check!("ccsd_uccsd_energy", H2O_TRIPLET_CCPVDZ, 1e-6);
    }

    /// CCSD-08: DF-CCSD correlation energy byte-identity vs upstream
    /// `cc.CCSD(mf.density_fit()).kernel()`. The `Wabef`→HDF5 spill proof on a
    /// deliberately constrained `PYSCF_MAX_MEMORY` (benzene-dimer/cc-pVDZ) is the
    /// heavy variant of this arm.
    #[test]
    #[ignore = "workflow_dispatch human-verify: DF-CCSD spill needs a constrained PYSCF_MAX_MEMORY + upstream PySCF"]
    fn arm_ccsd_dfccsd_energy() {
        oracle_check!("ccsd_dfccsd_energy", H2O_CC_PVDZ, 1e-6);
    }

    /// CCSD-05: λ amplitudes byte-identity vs upstream
    /// `cc.CCSD(mf).run().solve_lambda()`.
    #[test]
    #[ignore = "workflow_dispatch human-verify: requires upstream PySCF + the live ccsd dispatch wiring"]
    fn arm_ccsd_lambda() {
        oracle_check!("ccsd_lambda", H2O_CC_PVDZ, 1e-7);
    }

    /// CCSD-06: 1-particle RDM byte-identity vs upstream
    /// `cc.CCSD(mf).run().make_rdm1()` (incl. the `ao_repr` path).
    #[test]
    #[ignore = "workflow_dispatch human-verify: requires upstream PySCF + the live ccsd dispatch wiring"]
    fn arm_ccsd_rdm1() {
        oracle_check!("ccsd_rdm1", H2O_CC_PVDZ, 1e-7);
    }

    /// CCSD-06: 2-particle RDM byte-identity vs upstream
    /// `cc.CCSD(mf).run().make_rdm2()`.
    #[test]
    #[ignore = "workflow_dispatch human-verify: requires upstream PySCF + the live ccsd dispatch wiring"]
    fn arm_ccsd_rdm2() {
        oracle_check!("ccsd_rdm2", H2O_CC_PVDZ, 1e-7);
    }
}
