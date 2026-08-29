//! Auxiliary-basis default resolution table — port of pyscf/df/addons.py
//! `DEFAULT_AUXBASIS` (plan 03-05 Task 1 RED).
//!
//! Bit-exact value assertion vs upstream lives in plan 03-10 (oracle harness
//! wave 2); plan 03-05 only asserts table-lookup behaviour.

use pyscf_df::{default_jkfit, default_ri};

#[test]
fn cc_pvdz_resolves_to_jkfit() {
    assert_eq!(default_jkfit("cc-pvdz"), "cc-pvdz-jkfit");
    assert_eq!(default_ri("cc-pvdz"), "cc-pvdz-ri");
}

#[test]
fn def2_svp_resolves() {
    assert_eq!(default_jkfit("def2-svp"), "def2-svp-jkfit");
    assert_eq!(default_ri("def2-svp"), "def2-svp-ri");
}

#[test]
fn unknown_basis_falls_back_to_weigend() {
    assert_eq!(default_jkfit("invented-basis-2030"), "weigend");
    assert_eq!(default_ri("invented-basis-2030"), "weigend");
}

#[test]
fn pople_falls_back_to_weigend() {
    assert_eq!(default_jkfit("6-31g"), "weigend");
    assert_eq!(default_jkfit("6-31g*"), "weigend");
    assert_eq!(default_jkfit("6-31g**"), "weigend");
}

/// Corrected by plan 14-01. Upstream's `DEFAULT_AUXBASIS['sto3g']` is
/// `('def2-svp-jkfit', 'def2-svp-ri')`; this table said `weigend`, which is the
/// UNIVERSAL fallback rather than this row. The upstream-faithful chain (Psi4
/// table, then BSE metadata, then even-tempered) is
/// `pyscf_df::predefined_auxbasis` / `pyscf_df::make_auxbasis`, exercised in
/// `tests/make_auxbasis.rs`.
#[test]
fn sto_3g_resolves_to_def2_svp_jkfit() {
    assert_eq!(default_jkfit("sto-3g"), "def2-svp-jkfit");
    assert_eq!(default_ri("sto-3g"), "def2-svp-ri");
    assert_eq!(
        pyscf_df::predefined_auxbasis("sto-3g", true, false),
        Some("def2-svp-jkfit")
    );
}
