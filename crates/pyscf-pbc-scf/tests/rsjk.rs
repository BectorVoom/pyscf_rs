//! Plan 14-08 Task 4/5 — `rsjk`, and the recorded reason it does not run.
//!
//! # HISTORICAL header — the cintx gap below is CLOSED
//!
//! Plan 20-06 (2026-09-13) re-assessed the blocker: cintx now honours
//! `range_omega` and the supermole types exist; what is missing is `rsjk.py`'s
//! own screened SR body (see `src/rsjk.rs` module docs and the carryover
//! D-PBC-24). The text below is kept as the Phase-14 record.
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

/// The ω half ships, and it is `rsjk.py`'s OWN `_guess_omega` (`rsjk.py:1263`),
/// NOT `rsdf_builder._guess_omega`: `build` (`rsjk.py:145-150`) resolves the
/// module-level names defined in `rsjk.py` itself.
///
/// Plan 20-06 found this test previously pinned RSDF's ω (0.7393586378665364 /
/// mesh 11 on He-fcc) under the belief that `rsjk` imports it. Targets below
/// are upstream 2.12.1, `RangeSeparatedJKBuilder(cell, kpts).build()` and
/// `rsjk._guess_omega`, measured 2026-09-13 with `PYTHONPATH` pinned to the
/// vendored `pyscf/` tree.
#[test]
fn rsjk_guesses_rsjk_py_omega_not_rsdf() {
    let cell = he_all_electron();
    let kpts = cell.make_kpts([2, 2, 2]).expect("kpts");
    let (omega, mesh, ke) = RangeSeparatedJkBuilder::new(cell.clone(), &kpts)
        .guess_omega()
        .expect("omega");
    assert!(
        (omega - 1.312_754_030_266_949).abs() < 1e-12,
        "He-fcc 2x2x2 omega: {omega}"
    );
    assert_eq!(mesh, [15, 15, 15]);
    assert!(
        (ke - 60.188_792_480_310_04).abs() < 1e-10,
        "ke_cutoff: {ke}"
    );

    let (omega, mesh, ke) = RangeSeparatedJkBuilder::new(cell.clone(), &[[0.0; 3]])
        .guess_omega()
        .expect("omega");
    assert!(
        (omega - 1.533_462_187_962_161).abs() < 1e-12,
        "He-fcc gamma omega: {omega}"
    );
    assert_eq!(mesh, [17, 17, 17]);
    assert!((ke - 78.613_933_035_507).abs() < 1e-10, "ke_cutoff: {ke}");

    let mut preset = RangeSeparatedJkBuilder::new(cell, &kpts);
    preset.mesh = Some([9, 9, 9]);
    let (omega, mesh, ke) = preset.guess_omega().expect("omega");
    assert!(
        (omega - 0.717_644_070_562_487_7).abs() < 1e-12,
        "He-fcc mesh-9 omega: {omega}"
    );
    assert_eq!(mesh, [9, 9, 9]);
    assert!(
        (ke - 19.653_483_258_876_75).abs() < 1e-10,
        "ke_cutoff: {ke}"
    );

    let d = diamond();
    let dk = d.make_kpts([2, 2, 2]).expect("kpts");
    let (omega, mesh, ke) = RangeSeparatedJkBuilder::new(d.clone(), &dk)
        .guess_omega()
        .expect("omega");
    assert!(
        (omega - 0.764_541_213_395_792_8).abs() < 1e-12,
        "diamond 2x2x2 omega: {omega}"
    );
    assert_eq!(mesh, [11, 11, 11]);
    assert!(
        (ke - 21.721_883_440_437_864).abs() < 1e-10,
        "ke_cutoff: {ke}"
    );

    let mut preset = RangeSeparatedJkBuilder::new(d, &dk);
    preset.mesh = Some([9, 9, 9]);
    let (omega, _, ke) = preset.guess_omega().expect("omega");
    assert!(
        (omega - 0.604_341_256_511_272_1).abs() < 1e-12,
        "diamond mesh-9 omega: {omega}"
    );
    assert!(
        (ke - 13.902_005_401_880_233).abs() < 1e-10,
        "ke_cutoff: {ke}"
    );
}

/// An explicitly set `omega` is honoured — `rsjk.py:142-149` takes it over the
/// guess AND discards a preset mesh: upstream `build` with `omega = 0.5`,
/// `mesh = [5,5,5]` on He-fcc runs at mesh `[7,7,7]`, `ke_cutoff`
/// 9.545851180141373 (`rsjk.estimate_ke_cutoff_for_omega`).
#[test]
fn an_explicit_omega_overrides_the_guess() {
    let cell = he_all_electron();
    for (w, ke_ref, mesh_ref) in [
        (0.5, 9.545_851_180_141_373, [7, 7, 7]),
        (0.3, 3.531_257_370_504_433, [5, 5, 5]),
    ] {
        let mut b = RangeSeparatedJkBuilder::new(cell.clone(), &[[0.0; 3]]);
        b.omega = Some(w);
        b.mesh = Some([5, 5, 5]);
        let (omega, mesh, ke) = b.guess_omega().expect("omega");
        assert_eq!(omega, w);
        assert_eq!(mesh, mesh_ref, "omega {w}: preset mesh must be discarded");
        assert!((ke - ke_ref).abs() < 1e-10, "omega {w} ke_cutoff: {ke}");
    }
}

/// **What is unported, asserted.** `build` and `get_jk` refuse and say what is
/// actually missing.
///
/// The short-range `int2e` this builder needs EXISTS now — D-PBC-24 put
/// `range_omega` (libcint `env[8]`) on cintx's safe API — so the refusal no
/// longer names another repository. What is missing is `rsjk`'s own body: the
/// supermole, the screened real-space sweep, the `ft_aopair` long-range half
/// and the `vj`/`vk` assembly.
///
/// The assertion survives the change of cause for the reason it was written:
/// substituting the full-range `int2e` would give a builder that runs,
/// converges, and is silently not `rsjk` — and because `rsjk` is EXACT, the
/// wrong answer would land within the DF fitting error of a correct GDF and
/// look plausible. Delete it in the commit that ships `rsjk`, not before.
#[test]
fn rsjk_refuses_and_names_what_is_unported() {
    let cell = he_all_electron();
    let mut b = RangeSeparatedJkBuilder::new(cell, &[[0.0; 3]]);
    for msg in [
        format!("{}", b.build().expect_err("build must refuse")),
        format!(
            "{}",
            b.get_jk(&[], &[[0.0; 3]], pyscf_pbc_df::traits::JkOpts::hermitian())
                .expect_err("get_jk must refuse")
        ),
    ] {
        assert!(
            msg.contains("range_omega") && msg.contains("env[8]"),
            "the refusal must say the integral capability EXISTS, so the next reader \
             does not re-derive a blocker that is gone: {msg}"
        );
        assert!(
            !msg.contains("cintx's safe API has no"),
            "the old cintx blocker text must not come back: {msg}"
        );
        // Plan 20-06: the refusal names the constructs that are actually
        // absent, not the closed cintx / supermole blockers.
        assert!(
            msg.contains("PBCVHF_direct_drv")
                && msg.contains("PBCVHFnr_int2e_q_cond")
                && msg.contains("rsjk.estimate_rcut"),
            "the refusal must name rsjk's missing SR body: {msg}"
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
    assert!(
        !b.exclude_dd_block,
        "D-PBC-23: false everywhere in Phase 14"
    );
}
