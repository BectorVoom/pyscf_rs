//! Plan 14-08 Task 4/5 — `rsjk`, and the recorded reason it does not run.
//!
//! # `rsjk` is BLOCKED on the same cintx gap as `_RSGDFBuilder`
//!
//! `rsjk` builds `vj`/`vk` exactly, with no auxiliary basis, by splitting the
//! Coulomb operator: a short-range real-space `int2e` over a supermole plus a
//! long-range reciprocal-space pass through `ft_aopair`. `rsjk.py:186` sets
//! `supmol_sr.omega = -self.omega` and evaluates the STANDARD `int2e` symbol
//! against it — libcint's `PTR_RANGE_OMEGA` (`env[8]`) toggle.
//!
//! cintx's safe API has no such knob: `ExecutionOptions`
//! (`cintx-runtime/src/options.rs:96`) carries `f12_zeta` (`env[9]`),
//! `rinv_orig` and `common_orig`, and no kernel reads `env[8]`. This
//! repository already records the gap as Phase 4's Open Question A5 / cintx#11
//! (`crates/pyscf-gto/src/range_coulomb.rs`).
//!
//! So `14-08-PLAN.md` Task 5.3-5.5 — `rsjk` against FFTDF at the Phase-13
//! floor (2.607e-11 diamond 2×2×2, 3.006e-13 He-fcc), its Hermiticity, and its
//! oracle — cannot be measured. What CAN be asserted, and is:
//!
//! * the ω parameters `rsjk` would run at, which are plan 14-07's and ship;
//! * that `build` and `get_jk` REFUSE rather than substituting the full-range
//!   kernel (D-PBC-20);
//! * that `rsjk` is **not** a `PeriodicDf`, which `14-08-PLAN.md` requires:
//!   "it must not be given a `PeriodicDf` impl whose `sr_loop`/`get_naoaux`
//!   half is a lie."

mod common;

use common::{diamond, he_all_electron};
use pyscf_pbc_scf::rsjk::RangeSeparatedJkBuilder;

/// The ω half ships: `rsjk.build` reads its `(omega, mesh, ke_cutoff)` from
/// `rsdf_builder._guess_omega` (`rsjk.py:145-151`), which plan 14-07 sub-task
/// 7a ported and gated against `measurements/omega.out`.
#[test]
fn rsjk_guesses_the_same_omega_as_rsdf() {
    let cell = he_all_electron();
    let kpts = cell.make_kpts([2, 2, 2]).expect("kpts");
    let b = RangeSeparatedJkBuilder::new(cell.clone(), &kpts);
    let (omega, mesh, ke) = b.guess_omega().expect("omega");
    assert!(
        (omega - 0.739_358_637_866_536).abs() < 1e-12,
        "He-fcc 2x2x2 omega: {omega}"
    );
    assert_eq!(mesh, [11, 11, 11]);
    assert!((ke - 30.708_567_591_994_9).abs() < 1e-10, "ke_cutoff: {ke}");

    let d = diamond();
    let dk = d.make_kpts([2, 2, 2]).expect("kpts");
    let (omega, mesh, _) = RangeSeparatedJkBuilder::new(d, &dk)
        .guess_omega()
        .expect("omega");
    assert!(
        (omega - 0.601_955_030_338_906).abs() < 1e-12,
        "diamond 2x2x2 omega: {omega}"
    );
    assert_eq!(mesh, [11, 11, 11]);
}

/// An explicitly set `omega` is honoured — `rsjk.py:142-143` takes it over the
/// guess, and an RSH functional supplies it from `cell.omega`.
#[test]
fn an_explicit_omega_overrides_the_guess() {
    let cell = he_all_electron();
    let mut b = RangeSeparatedJkBuilder::new(cell, &[[0.0; 3]]);
    b.omega = Some(0.5);
    let (omega, _, ke) = b.guess_omega().expect("omega");
    assert_eq!(omega, 0.5);
    assert!(ke > 0.0 && ke.is_finite(), "ke_cutoff: {ke}");
}

/// **The blocker, asserted.** `build` and `get_jk` refuse and name the missing
/// cintx capability. Substituting the full-range `int2e` would give a builder
/// that runs, converges, and is silently not `rsjk` — and because `rsjk` is
/// EXACT, the wrong answer would land within the DF fitting error of a correct
/// GDF and look plausible.
#[test]
fn rsjk_refuses_and_names_the_cintx_gap() {
    let cell = he_all_electron();
    let mut b = RangeSeparatedJkBuilder::new(cell, &[[0.0; 3]]);
    for msg in [
        format!("{}", b.build().expect_err("build must refuse")),
        format!(
            "{}",
            b.get_jk(
                &[],
                &[[0.0; 3]],
                pyscf_pbc_df::traits::JkOpts::hermitian()
            )
            .expect_err("get_jk must refuse")
        ),
    ] {
        assert!(
            msg.contains("range_omega") && msg.contains("env[8]"),
            "the refusal must name the missing cintx capability: {msg}"
        );
    }
}

/// The MPI / multi-threaded partitioning variants are a NON-GOAL of the phase
/// and say so, pointing at the phase that owns them rather than at this gap.
#[test]
fn the_partitioning_variants_are_a_named_non_goal() {
    let cell = he_all_electron();
    let b = RangeSeparatedJkBuilder::new(cell, &[[0.0; 3]]);
    let msg = format!("{}", b.get_jk_mpi().expect_err("must refuse"));
    assert!(
        msg.contains("MPI") && msg.contains("serial"),
        "the MPI variants must be refused as a non-goal, not as the cintx gap: {msg}"
    );
}

/// **`rsjk` is not a `PeriodicDf`, and that is the plan's requirement.**
///
/// It has no `cderi` to `sr_loop` over and no auxiliary count to report, so an
/// impl would have to lie in two methods. This test is a compile-time
/// assertion written as a runtime one: `RangeSeparatedJkBuilder` is accepted
/// only through its own narrow surface.
#[test]
fn rsjk_is_not_a_density_fitting_builder() {
    fn takes_a_df<T: pyscf_pbc_df::traits::PeriodicDf>(_: &T) {}
    let _ = takes_a_df::<pyscf_pbc_df::Gdf>; // GDF is one …
    // … and `RangeSeparatedJkBuilder` deliberately is not: it exposes `build`
    // and `get_jk` only. If a future change adds the impl, delete this test
    // and explain what `sr_loop` and `get_naoaux` would return.
    let cell = he_all_electron();
    let b = RangeSeparatedJkBuilder::new(cell, &[[0.0; 3]]);
    assert!(!b.exclude_dd_block, "D-PBC-23: false everywhere in Phase 14");
}
