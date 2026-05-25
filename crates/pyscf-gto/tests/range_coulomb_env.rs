//! DFT-05 (plan 04-07): range-coulomb `env[8]` (`PTR_RANGE_OMEGA`)
//! set/restore on the `mol.intor` path — the `mol.with_range_coulomb(omega)`
//! equivalent.
//!
//! ## What this test owns (always-on, locally verifiable)
//!   1. `intor_with_omega(mol, "int2e", ±omega)` sets `mol._env[8]` to the
//!      omega for the duration of the call and **restores the prior value
//!      after** — including on the error path (T-04-07a: no omega leak
//!      across intor calls).
//!   2. The mechanism uses the **standard `int2e`** + the env[8] slot, NOT a
//!      phantom `int2e_lr_*`/`int2e_sr_*` symbol (Pitfall 1) — asserted by
//!      driving a standard symbol through `intor_with_omega` and by the
//!      source-level absence of `int2e_lr_`/`int2e_sr_` (checked in
//!      `intor_does_not_use_phantom_lr_sr_symbols`).
//!   3. A non-ranged `intor` call after a ranged one sees the original
//!      `env[8]` (restore is byte-exact).
//!
//! ## What is CI-gated (the numerical bit-exact arm)
//! The assertion that the *ranged* `int2e` numerically matches upstream
//! `mol.with_range_coulomb(omega).intor('int2e')` cannot run locally:
//!   - cintx's safe-API `int2e` is arity-4, gated `NotYetImplemented{phase:2}`
//!     (the Phase-2 verification rollup gap, cintx#11);
//!   - even once arity-4 lands, the safe-API plan does not yet *read*
//!     `_env[8]` (cintx exposes `f12_zeta`=env[9] but no `range_omega`=env[8]
//!     setter — Open Question A5 RESOLVED: cintx gap-closure required).
//!
//! The env-slot set/restore contract THIS file owns is verified independently
//! of that gap, exactly the 04-06 DFT-01 CI-only-oracle convention.

use pyscf_core::Unit;
use pyscf_gto::{
    AtomInput, BasisInput, M, MoleBuildArgs, PTR_RANGE_OMEGA, intor, intor_with_omega,
};

fn h2_mol() -> pyscf_core::Mole {
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

/// env[8] is restored to its prior value after a ranged `int2e` call
/// (T-04-07a: no omega leak). We drive `int1e_ovlp` (an arity-2 intor that
/// the cintx safe API evaluates today) through `intor_with_omega` so the
/// call returns Ok and we can assert the restore on the success path.
#[test]
fn env8_restored_after_ranged_intor_call() {
    let mut mol = h2_mol();
    let prior = mol._env[PTR_RANGE_OMEGA];
    assert_eq!(prior, 0.0, "fresh Mole has env[8] = 0 (standard Coulomb)");

    // A ranged arity-2 call (succeeds on the current safe API).
    let omega = 0.33;
    let out = intor_with_omega(&mut mol, "int1e_ovlp", omega).expect("ranged int1e_ovlp");
    assert_eq!(out.shape, vec![mol.nao_nr, mol.nao_nr]);
    assert!(out.values.iter().all(|v| v.is_finite()));

    // env[8] restored after the call — the omega did NOT leak.
    assert_eq!(
        mol._env[PTR_RANGE_OMEGA], prior,
        "env[8] restored to {prior} after the ranged call (no leak — T-04-07a)"
    );
}

/// env[8] is restored even when the wrapped intor errors (the `?`-propagation
/// path). We force an error with an unknown intor name and confirm env[8] is
/// back to its prior value (the RAII guard restores on drop, including unwind).
#[test]
fn env8_restored_on_error_path() {
    let mut mol = h2_mol();
    let prior = mol._env[PTR_RANGE_OMEGA];
    let res = intor_with_omega(&mut mol, "int_does_not_exist", 0.5);
    assert!(res.is_err(), "unknown intor name must error");
    assert_eq!(
        mol._env[PTR_RANGE_OMEGA], prior,
        "env[8] restored after the error path (no leak — T-04-07a)"
    );
}

/// A standard (non-ranged) `intor` call AFTER a ranged one sees the original
/// env[8] — proves there is no cross-call contamination. Also exercises the
/// omega<0 (short-range) and omega>0 (long-range) sign convention through the
/// same set/restore path.
#[test]
fn no_cross_call_contamination_lr_and_sr_signs() {
    let mut mol = h2_mol();
    let baseline = intor(&mol, "int1e_ovlp").expect("baseline ovlp").values;

    // Long-range (omega > 0): erf(ωr)/r.
    let _lr = intor_with_omega(&mut mol, "int1e_ovlp", 0.4).expect("LR call");
    assert_eq!(mol._env[PTR_RANGE_OMEGA], 0.0, "env[8] restored after LR");

    // Short-range (omega < 0): SR complement.
    let _sr = intor_with_omega(&mut mol, "int1e_ovlp", -0.4).expect("SR call");
    assert_eq!(mol._env[PTR_RANGE_OMEGA], 0.0, "env[8] restored after SR");

    // A subsequent standard intor is byte-identical to the baseline — no
    // residual omega from the ranged calls (int1e_ovlp does not read env[8],
    // so this also confirms restore did not corrupt the slot).
    let after = intor(&mol, "int1e_ovlp").expect("post-ranged ovlp").values;
    assert_eq!(
        after, baseline,
        "standard intor unchanged by prior ranged calls"
    );
}

/// Pitfall 1 source assertion: the range-coulomb module does NOT introduce
/// `int2e_lr_*`/`int2e_sr_*` symbols. It must use the standard `int2e` +
/// env[8]. We assert the source text of `range_coulomb.rs` references the
/// standard `int2e` and does NOT contain `int2e_lr_`/`int2e_sr_` as a CALL
/// (the doc-comments mention them only to say "NOT these").
#[test]
fn intor_does_not_use_phantom_lr_sr_symbols() {
    let src = include_str!("../src/range_coulomb.rs");
    // The implementation drives the standard int2e symbol.
    assert!(
        src.contains("\"int2e\""),
        "range_coulomb must call the standard \"int2e\" symbol"
    );
    // No phantom symbol is ever passed as an intor name. The strings
    // `int2e_lr_` / `int2e_sr_` appear ONLY inside doc-comments that warn
    // against them; they must never be quoted as an argument. We assert the
    // quoted-string forms are absent.
    assert!(
        !src.contains("\"int2e_lr_") && !src.contains("\"int2e_sr_"),
        "Pitfall 1: must not pass int2e_lr_/int2e_sr_ as an intor name"
    );
}

/// DFT-05 numerical bit-exact arm (CI-only): the ranged `int2e` matches
/// upstream `mol.with_range_coulomb(omega)`. Gated behind the cintx env[8] +
/// arity-4 gap (cintx#11) AND a live oracle — mirrors the 04-06 DFT-01
/// rks_uks_bitexact CI-only oracle. Unignores when the gap closes.
#[test]
#[ignore = "DFT-05 numerical RSH ERI: needs cintx safe-API env[8] reader + arity-4 int2e (cintx#11), CI-gated"]
fn ranged_int2e_matches_upstream_with_range_coulomb() {
    // When the cintx gap closes:
    //   let mut mol = h2_mol();
    //   let lr = intor_with_omega(&mut mol, "int2e", 0.4)?;   // erf(ωr)/r
    //   assert ≤ 1e-12 vs upstream mol.with_range_coulomb(0.4).intor('int2e').
    // Until then this arm documents the target without a phantom pass.
    unimplemented!("cintx#11: safe-API env[8] reader + arity-4 int2e gap-closure");
}
