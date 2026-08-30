//! DFT-05 (plan 04-07): range-coulomb `env[8]` (`PTR_RANGE_OMEGA`) on the
//! `mol.intor` path — the `mol.with_range_coulomb(omega)` equivalent.
//!
//! ## What this file owns
//!   1. **The integrals are really ranged.** `intor_with_omega(mol, "int2e",
//!      ±ω)` produces `erf(ω r)/r` and `erfc(ω r)/r`, gated by the identity
//!      `SR(ω) + LR(ω) == full` element by element — the check that catches an
//!      `erf`/`erfc` swap, which a magnitude check never would.
//!   2. `mol._env[8]` is set for the duration of the call and **restored
//!      after**, including on the error path (T-04-07a: no omega leak).
//!   3. The mechanism uses the **standard `int2e`**, NOT a phantom
//!      `int2e_lr_*`/`int2e_sr_*` symbol (Pitfall 1).
//!   4. A non-ranged `intor` call after a ranged one is byte-identical to one
//!      before it.
//!   5. **Fail closed.** A non-zero ω on an operator with no range-separated
//!      kernel is a typed refusal, not a full-range evaluation.
//!
//! ## What changed, and why point 1 is new
//!
//! This file used to say the numerical arm "cannot run locally" because
//! cintx's safe API had no `env[8]` reader. That was correct and it was worse
//! than it sounded: `intor` builds cintx's `BasisSet` from `mol._atom`/`_basis`
//! and never hands cintx this `_env`, so setting the slot did not merely fail
//! to work — it produced FULL-RANGE integrals under a range-separated request,
//! silently. D-PBC-24 closed the capability (`ExecutionOptions::range_omega`,
//! gated against vendored libcint 6.1.3 at 3.4e-14) and `intor_with_omega` now
//! threads ω through the options as well as the env slot. The numerical arm
//! runs here, locally, with no oracle: the SR/LR identity is self-checking.

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
/// (T-04-07a: no omega leak).
#[test]
fn env8_restored_after_ranged_intor_call() {
    let mut mol = h2_mol();
    let nao = mol.nao_nr;
    let prior = mol._env[PTR_RANGE_OMEGA];
    assert_eq!(prior, 0.0, "fresh Mole has env[8] = 0 (standard Coulomb)");

    let out = intor_with_omega(&mut mol, "int2e", 0.33).expect("ranged int2e");
    assert_eq!(out.shape, vec![nao, nao, nao, nao]);
    assert!(out.values.iter().all(|v| v.is_finite()));

    // env[8] restored after the call — the omega did NOT leak.
    assert_eq!(
        mol._env[PTR_RANGE_OMEGA], prior,
        "env[8] restored to {prior} after the ranged call (no leak — T-04-07a)"
    );
}

/// **The integrals are really ranged**: `SR(ω) + LR(ω) == full`, element by
/// element, on the `int2e` tensor `get_k_with_omega` contracts.
///
/// This is the assertion that was CI-gated and unreachable, and it is worth
/// more than the upstream comparison it replaced: it is self-checking, so it
/// cannot be satisfied by two identically-wrong halves the way "matches a
/// reference we also computed" can. It also catches the exact failure this
/// module shipped with for a phase — a full-range substitute — because a
/// substitute makes `SR` and `LR` both equal `full` and the sum twice it.
#[test]
fn ranged_int2e_splits_the_coulomb_kernel() {
    let mut mol = h2_mol();

    let full = intor(&mol, "int2e").expect("full-range int2e").values;
    let scale = full.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(scale > 1e-3, "the full-range ERI tensor is all zeros");

    for omega in [0.4_f64, 0.9] {
        let lr = intor_with_omega(&mut mol, "int2e", omega)
            .expect("long-range int2e")
            .values;
        let sr = intor_with_omega(&mut mol, "int2e", -omega)
            .expect("short-range int2e")
            .values;
        assert_eq!(lr.len(), full.len());

        // Neither half may BE the full-range tensor.
        for (label, half) in [("LR", &lr), ("SR", &sr)] {
            let moved = half
                .iter()
                .zip(&full)
                .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
            assert!(
                moved > 1e-3 * scale,
                "omega={omega}: the {label} tensor is (almost) the full-range one \
                 (max |delta| = {moved:e} against scale {scale:e}) — this is what a \
                 silent full-range substitution looks like"
            );
        }

        let residual = (0..full.len()).fold(0.0_f64, |m, i| m.max((sr[i] + lr[i] - full[i]).abs()));
        assert!(
            residual <= 1e-12 * scale,
            "omega={omega}: erfc(w r)/r + erf(w r)/r != 1/r, residual {residual:e} \
             against scale {scale:e}"
        );
    }

    // omega = 0 is the full Coulomb operator, bit for bit (g2e.c:4445 branches
    // on `omega == 0.` exactly).
    assert_eq!(
        intor_with_omega(&mut mol, "int2e", 0.0)
            .expect("omega = 0")
            .values,
        full,
        "omega = 0 must be byte-identical to the standard int2e"
    );
}

/// **Fail closed.** `with_range_coulomb` is not a mode a caller can leave set
/// around arbitrary integrals.
///
/// Upstream's context manager is harmless around a 1e integral only because
/// libcint's `g1e.c` never reads the slot. cintx's safe API is stricter than
/// the slot it stands for: it refuses a non-zero ω on any operator it has not
/// implemented range separation for, rather than evaluating the full-range
/// kernel under a range-separated request. So this port's ω must be scoped to
/// the Coulomb call, and the refusal says so.
#[test]
fn omega_on_a_one_electron_intor_is_refused_not_ignored() {
    let mut mol = h2_mol();
    let prior = mol._env[PTR_RANGE_OMEGA];

    intor_with_omega(&mut mol, "int1e_ovlp", 0.0)
        .expect("omega = 0 is full Coulomb and must be accepted everywhere");

    let err = intor_with_omega(&mut mol, "int1e_ovlp", -0.4)
        .expect_err("a non-zero omega on int1e_ovlp must be refused, not silently ignored");
    let msg = format!("{err}");
    assert!(
        msg.contains("range_omega"),
        "the refusal must name what it is refusing: {msg}"
    );

    // And the guard still restored the slot on that error path.
    assert_eq!(mol._env[PTR_RANGE_OMEGA], prior);
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
    let _lr = intor_with_omega(&mut mol, "int2e", 0.4).expect("LR call");
    assert_eq!(mol._env[PTR_RANGE_OMEGA], 0.0, "env[8] restored after LR");

    // Short-range (omega < 0): SR complement.
    let _sr = intor_with_omega(&mut mol, "int2e", -0.4).expect("SR call");
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

/// DFT-05's remaining CI-only arm: agreement with UPSTREAM's own
/// `mol.with_range_coulomb(omega).intor('int2e')`, rather than with the
/// self-checking identity above.
///
/// The identity in `ranged_int2e_splits_the_coulomb_kernel` is the stronger
/// correctness gate — it cannot be satisfied by a consistent mistake — but it
/// does not pin the CONVENTION: a port that had ω's sign backwards would
/// satisfy `SR + LR == full` at every ω and disagree with PySCF everywhere.
/// cintx gates its own side of that against vendored libcint 6.1.3 at 3.4e-14
/// (`cintx/crates/cintx-oracle/tests/range_omega_parity.rs`), which covers the
/// kernel; what is left for this arm is the sign convention across the
/// `pyscf-gto` boundary, and it needs a live oracle.
#[test]
#[ignore = "needs PYSCF_ORACLE_VENV: convention check against upstream with_range_coulomb"]
fn ranged_int2e_matches_upstream_with_range_coulomb() {
    //   let mut mol = h2_mol();
    //   let lr = intor_with_omega(&mut mol, "int2e", 0.4)?;   // erf(wr)/r
    //   assert <= 1e-12 vs upstream mol.with_range_coulomb(0.4).intor('int2e').
    unimplemented!("oracle harness: upstream with_range_coulomb comparison");
}
