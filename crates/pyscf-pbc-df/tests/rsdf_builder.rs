//! Plan 14-07 acceptance — sub-task **7a**, the ω machinery.
//!
//! # 7b / 7c / 7d are BLOCKED, and this file records the block
//!
//! `_RSGDFBuilder`'s `get_2c2e` needs a short-range `int2c2e` and its
//! `outcore_auxe2` a short-range `int3c2e`. The cintx safe API cannot request
//! either — `ExecutionOptions` has no `range_omega` (libcint `env[8]`) knob,
//! no kernel reads that slot, and the periodic 3-centre driver reaches cintx
//! through `build_image_expanded_with_aux`, which builds its `BasisSet` from
//! the parsed per-element basis rather than from an `_env` array, so even
//! `pyscf-gto`'s `range_coulomb.rs` workaround is out of reach. The gap is
//! already on this repository's record as Phase 4's Open Question A5 /
//! cintx#11. See `crates/pyscf-pbc-df/src/rsdf_builder/mod.rs`.
//!
//! Plan 14-07 Task 7b says exactly what to do about that: report it, do not
//! work around it with a numerically different kernel. So
//! [`RsGdfBuilder::build`] refuses (asserted below), and 7a — which needs no
//! integral at all — ships in full.
//!
//! # Every number below is from `measurements/omega.out`
//!
//! Recorded BEFORE the port was written (plan 14-07 Task 0, the Phase-9
//! precedent). A wrong `ω` does not fail loudly; it produces a plausible 1e-6,
//! which is why these are asserted at 1e-12 against upstream rather than
//! re-derived.

mod common;

use pyscf_pbc_df::rsdf_builder as rs;
use pyscf_pbc_df::{Aftdf, RsGdfBuilder};
use pyscf_pbc_gto::Cell;

fn kpts_of(cell: &Cell, mesh: [usize; 3]) -> Vec<[f64; 3]> {
    cell.make_kpts(mesh).expect("make_kpts")
}

fn close(got: f64, want: f64, tol: f64, what: &str) {
    let d = (got - want).abs();
    assert!(
        d < tol,
        "{what}: got {got:.15}, upstream {want:.15}, |d| = {d:e} (tol {tol:e})"
    );
}

const TOL: f64 = 1e-12;

// ---------------------------------------------------------------------------
// Task 7e.1 — the estimators, against measurements/omega.out
// ---------------------------------------------------------------------------

/// `_guess_omega` on all four measured configurations. The `(omega, mesh,
/// ke_cutoff)` triple is what every other quantity in the scheme hangs off.
#[test]
fn guess_omega_matches_upstream() {
    let he = common::he_all_electron();
    let dia = common::diamond();
    /// `(label, cell, kmesh, omega, mesh, ke_cutoff)` — one row of
    /// `measurements/omega.out`.
    type Case<'a> = (&'a str, &'a Cell, [usize; 3], f64, [usize; 3], f64);
    let cases: [Case<'_>; 4] = [
        (
            "He-fcc 2x2x2",
            &he,
            [2, 2, 2],
            0.739_358_637_866_536,
            [11, 11, 11],
            30.708_567_591_994_9,
        ),
        (
            "He-fcc gamma",
            &he,
            [1, 1, 1],
            1.035_102_093_013_15,
            [15, 15, 15],
            60.188_792_480_31,
        ),
        (
            "diamond 2x2x2",
            &dia,
            [2, 2, 2],
            0.601_955_030_338_906,
            [11, 11, 11],
            21.721_883_440_437_9,
        ),
        (
            "diamond gamma",
            &dia,
            [1, 1, 1],
            0.720_159_089_892_724,
            [13, 13, 13],
            31.279_512_154_230_5,
        ),
    ];
    for (name, cell, km, w, m, ke) in cases {
        let kpts = kpts_of(cell, km);
        let (omega, mesh, ke_cutoff) = rs::guess_omega(cell, &kpts, None).expect("guess_omega");
        assert_eq!(mesh, m, "{name}: mesh");
        close(omega, w, TOL, &format!("{name}: omega"));
        close(ke_cutoff, ke, 1e-10, &format!("{name}: ke_cutoff"));
    }
}

/// The four scalar ω estimators, plus `OMEGA_MIN`.
#[test]
fn omega_estimators_match_upstream() {
    assert_eq!(rs::OMEGA_MIN, 0.08, "OMEGA_MIN — rsdf_builder.py:52");
    assert_eq!(
        rs::RCUT_THRESHOLD,
        1.0,
        "RCUT_THRESHOLD — rsdf_builder.py:56"
    );

    let he = common::he_all_electron();
    close(
        rs::estimate_omega_min(&he, None),
        0.324_467_042_356_544,
        TOL,
        "He estimate_omega_min",
    );
    close(
        rs::estimate_ke_cutoff_for_omega(&he, rs::OMEGA_MIN, None),
        0.335_874_219_073_685,
        TOL,
        "He estimate_ke_cutoff_for_omega(OMEGA_MIN)",
    );
    close(
        rs::estimate_ke_cutoff_for_omega(&he, 0.739_358_637_866_536, None),
        24.247_277_384_978_9,
        1e-10,
        "He estimate_ke_cutoff_for_omega(omega)",
    );
    close(
        rs::estimate_omega_for_ke_cutoff(&he, 20.0, None),
        0.596_678_472_583_996,
        TOL,
        "He estimate_omega_for_ke_cutoff(20)",
    );

    let dia = common::diamond();
    close(
        rs::estimate_omega_min(&dia, None),
        0.180_615_163_949_558,
        TOL,
        "diamond estimate_omega_min",
    );
    close(
        rs::estimate_ke_cutoff_for_omega(&dia, rs::OMEGA_MIN, None),
        0.330_921_403_072_804,
        TOL,
        "diamond estimate_ke_cutoff_for_omega(OMEGA_MIN)",
    );
    close(
        rs::estimate_ke_cutoff_for_omega(&dia, 0.601_955_030_338_906, None),
        16.873_349_934_088_5,
        1e-10,
        "diamond estimate_ke_cutoff_for_omega(omega)",
    );
    close(
        rs::estimate_omega_for_ke_cutoff(&dia, 20.0, None),
        0.578_002_439_380_595,
        TOL,
        "diamond estimate_omega_for_ke_cutoff(20)",
    );
}

/// `estimate_omega_for_ke_cutoff` is the exact inverse of `_guess_omega`'s
/// last line — upstream's own round trip, and the tightest available check
/// that both are ported right.
#[test]
fn omega_and_ke_cutoff_round_trip() {
    for (cell, km) in [
        (common::he_all_electron(), [2usize, 2, 2]),
        (common::diamond(), [1, 1, 1]),
    ] {
        let kpts = kpts_of(&cell, km);
        let (omega, _, ke) = rs::guess_omega(&cell, &kpts, None).expect("guess_omega");
        close(
            rs::estimate_omega_for_ke_cutoff(&cell, ke, None),
            omega,
            1e-15,
            "omega(ke(omega))",
        );
    }
}

/// `_round_off_to_odd_mesh` and `_estimate_meshz`. Both look trivial and are
/// not: an even axis has plane waves without a `-G` counterpart, which breaks
/// the `k` / `-k` conjugation symmetry `_make_j3c` relies on.
#[test]
fn mesh_helpers_match_upstream() {
    assert_eq!(
        rs::round_off_to_odd_mesh([8, 9, 10]),
        [9, 9, 11],
        "_round_off_to_odd_mesh"
    );
    assert_eq!(rs::round_off_to_odd_mesh([1, 2, 3]), [1, 3, 3]);
    assert_eq!(rs::round_off_to_odd_mesh([0, 0, 0]), [1, 1, 1]);

    assert_eq!(
        rs::estimate_meshz(&common::he_all_electron(), None).expect("meshz"),
        43,
        "He-fcc _estimate_meshz"
    );
    assert_eq!(
        rs::estimate_meshz(&common::diamond(), None).expect("meshz"),
        47,
        "diamond _estimate_meshz"
    );
}

/// The three radius estimators, per shell.
///
/// `estimate_rcut` and `estimate_ft_rcut` take a `_RangeSeparatedCell`
/// upstream; this port has none (D-PBC-21/23) and calls them with the plain
/// cell. `measurements/omega.out` records both, and the MAXIMA agree exactly —
/// the split only refines the smaller radii, so the plain-cell call is the
/// conservative one.
#[test]
fn radius_estimators_match_upstream() {
    let he = common::he_all_electron();
    // `measurements/omega.py` uses `df.make_modrho_basis`, and that matters:
    // the modrho rewrite rescales the `_env` contraction coefficients, so the
    // auxiliary cell's own `rcut` — which `estimate_rs_2c2e_rcut` reads — is
    // not the one a plain `make_auxcell` would give.
    let aux_he = pyscf_pbc_df::make_modrho_basis(&he, None, None)
        .expect("He modrho auxcell")
        .cell;
    close(
        rs::estimate_rs_2c2e_rcut(&aux_he, 0.739_358_637_866_536, None),
        9.266_297_611_723_78,
        1e-11,
        "He estimate_rs_2c2e_rcut(omega)",
    );
    close(
        rs::estimate_rs_2c2e_rcut(&aux_he, 0.0, None),
        6.865_405_159_992_42,
        1e-11,
        "He estimate_rs_2c2e_rcut(0)",
    );
    let r = rs::estimate_rcut(&he, &aux_he, 0.739_358_637_866_536, None);
    assert_eq!(r.len(), he.mol.nbas, "one radius per orbital shell");
    close(r[0], 11.130_814_509_359_459, 1e-11, "He estimate_rcut");
    let f = rs::estimate_ft_rcut(&he, None);
    close(f[0], 12.274_614_504_009_085, 1e-11, "He estimate_ft_rcut");

    let dia = common::diamond();
    let aux_dia = pyscf_pbc_df::make_modrho_basis(&dia, None, None)
        .expect("diamond modrho auxcell")
        .cell;
    close(
        rs::estimate_rs_2c2e_rcut(&aux_dia, 0.601_955_030_338_906, None),
        16.415_367_989_46,
        1e-10,
        "diamond estimate_rs_2c2e_rcut(omega)",
    );
    let r = rs::estimate_rcut(&dia, &aux_dia, 0.601_955_030_338_906, None);
    assert_eq!(r.len(), 4, "diamond has 4 orbital shells");
    let want = [
        17.731_949_787_234_566,
        18.568_121_481_440_436,
        17.731_949_787_234_566,
        18.568_121_481_440_436,
    ];
    for (i, w) in want.iter().enumerate() {
        close(r[i], *w, 1e-11, &format!("diamond estimate_rcut[{i}]"));
    }
    let f = rs::estimate_ft_rcut(&dia, None);
    let want = [
        20.148_587_596_092_714,
        20.481_090_224_560_27,
        20.148_587_596_092_714,
        20.481_090_224_560_27,
    ];
    for (i, w) in want.iter().enumerate() {
        close(f[i], *w, 1e-11, &format!("diamond estimate_ft_rcut[{i}]"));
    }
}

/// `_gaussian_int(auxcell)` is the auxiliary monopole, and the modrho
/// normalisation makes it exactly 1 on every fitted function — the same
/// convention 14-01 Task 3 fixed. `rsdf.get_aux_chg` (plan 14-08) is this
/// quantity, so asserting it here is what lets 14-08 assert equality rather
/// than recompute.
#[test]
fn gaussian_int_is_the_monopole() {
    for (name, cell) in [
        ("He-fcc", common::he_all_electron()),
        ("diamond", common::diamond()),
    ] {
        let aux = pyscf_pbc_df::make_modrho_basis(&cell, None, None).expect("modrho");
        let g = rs::gaussian_int(&aux.cell).expect("gaussian_int");
        assert_eq!(g.len(), aux.naux(), "{name}: one integral per auxiliary AO");
        // `make_modrho_basis` normalises every s function to unit monopole and
        // leaves l > 0 at exactly zero (they integrate to zero by symmetry).
        let mut ones = 0usize;
        for v in &g {
            assert!(
                v.abs() < 1e-14 || (v - 1.0).abs() < 1e-13,
                "{name}: a monopole is neither 0 nor 1: {v}"
            );
            if (v - 1.0).abs() < 1e-13 {
                ones += 1;
            }
        }
        assert!(ones > 0, "{name}: no charged auxiliary function at all");
    }
}

// ---------------------------------------------------------------------------
// Task 7e.2 — the identity that catches an erf/erfc swap
// ---------------------------------------------------------------------------

/// **`weighted_coulG_SR + weighted_coulG_LR == weighted_coulG`, at every `G`.**
///
/// The single sharpest test in 7a: it holds only if the short-range half is
/// `erfc(w r)/r` and the long-range half is its complement. Swap them and this
/// still sums correctly — which is why the test also pins the SHAPE of each
/// half separately: the SR kernel must be finite at `G = 0` (it has no
/// `4 pi / G^2` pole) while the LR kernel must vanish there.
///
/// Upstream's residual is exactly 0 at both `k = 0` and `k = kpts[1]`
/// (`measurements/omega.out`), because `weighted_coulG_LR` is defined AS the
/// difference. This port defines it the same way, so 0 is the target, not a
/// tolerance.
#[test]
fn sr_and_lr_coulg_sum_to_the_full_kernel() {
    for (name, cell, km) in [
        ("He-fcc", common::he_all_electron(), [2usize, 2, 2]),
        ("diamond", common::diamond(), [2, 2, 2]),
    ] {
        let kpts = kpts_of(&cell, km);
        let (omega, mesh, _) = rs::guess_omega(&cell, &kpts, None).expect("guess_omega");
        let df = Aftdf::with_mesh(cell.clone(), &kpts, mesh).expect("aftdf");

        for (label, kpt) in [("k=0", [0.0; 3]), ("k=kpts[1]", kpts[1])] {
            let full = df.weighted_coulg(kpt, None, mesh, None).expect("full");
            let sr = rs::weighted_coulg_sr(&df, kpt, mesh, omega).expect("sr");
            let lr = rs::weighted_coulg_lr(&df, kpt, None, mesh, omega).expect("lr");
            let worst = (0..full.len())
                .map(|g| (lr[g] + sr[g] - full[g]).abs())
                .fold(0.0f64, f64::max);
            assert!(
                worst == 0.0,
                "{name} {label}: LR + SR != full by {worst:e}; upstream's residual \
                 is exactly 0 because LR is DEFINED as the difference"
            );

            // The shape check that a sum cannot make: SR is regular at G+k=0,
            // LR is not — it carries the whole 4 pi / |G+k|^2 pole and must
            // therefore be zero there, where the pole is removed.
            let sr_max = sr.iter().fold(0.0f64, |a, v| a.max(v.abs()));
            let lr_max = lr.iter().fold(0.0f64, |a, v| a.max(v.abs()));
            assert!(
                sr_max.is_finite() && lr_max.is_finite(),
                "{name} {label}: a non-finite kernel"
            );
            if label == "k=0" {
                assert_eq!(lr[0], 0.0, "{name}: LR must vanish at G = 0");
                assert_eq!(sr[0], 0.0, "{name}: the G = 0 term is removed entirely");
                // Away from the pole the SR half is the SMALLER one at large G
                // and the LARGER one at small G — the whole point of the split.
                assert!(
                    sr[1] > lr[1],
                    "{name}: at the smallest non-zero G the SHORT-range half must \
                     dominate ({} vs {}); if it does not, erf and erfc are swapped",
                    sr[1],
                    lr[1]
                );
            }
        }
    }
}

/// The recorded first four values of each half, so a future refactor that
/// keeps the SUM right but moves the SPLIT is still caught.
#[test]
fn sr_and_lr_coulg_match_upstream_values() {
    let cell = common::he_all_electron();
    let kpts = kpts_of(&cell, [2, 2, 2]);
    let (omega, mesh, _) = rs::guess_omega(&cell, &kpts, None).expect("guess_omega");
    let df = Aftdf::with_mesh(cell, &kpts, mesh).expect("aftdf");
    let lr = rs::weighted_coulg_lr(&df, [0.0; 3], None, mesh, omega).expect("lr");
    let sr = rs::weighted_coulg_sr(&df, [0.0; 3], mesh, omega).expect("sr");

    let want_lr = [
        0.0,
        1.387_917_293_470_906e-2,
        2.210_996_447_283_078e-5,
        2.152_185_039_558_385e-9,
    ];
    let want_sr = [
        0.0,
        0.060_984_093_235_164,
        0.018_693_706_577_995,
        0.008_318_138_533_356,
    ];
    for i in 0..4 {
        close(
            lr[i],
            want_lr[i],
            1e-14,
            &format!("He weighted_coulG_LR[{i}]"),
        );
        close(
            sr[i],
            want_sr[i],
            1e-14,
            &format!("He weighted_coulG_SR[{i}]"),
        );
    }
}

// ---------------------------------------------------------------------------
// 7b / 7c / 7d — still unported, asserted
// ---------------------------------------------------------------------------

/// **What is missing is a REFUSAL, not a silent substitution** (D-PBC-20).
///
/// This test used to say "cintx cannot request a short-range
/// `int3c2e`/`int2c2e`". D-PBC-24 made it able to, and
/// `tests/incore.rs::aux_e2_splits_the_coulomb_kernel_at_omega` gates that
/// end to end — so the assertion moved with the reason rather than being
/// deleted. What `_RSGDFBuilder` lacks now is its own body: sub-tasks 7b/7c.
///
/// Building anyway with the full-range kernel would give a builder that runs,
/// converges, and is silently a different method — the one outcome
/// `14-07-PLAN.md` Task 7b forbids, and the reason this assertion survives the
/// change of cause. Delete it in the commit that ships `_RSGDFBuilder`, not
/// before.
#[test]
fn rs_gdf_builder_refuses_and_names_what_is_unported() {
    let cell = common::he_all_electron();
    let mut b = RsGdfBuilder::new(cell.clone(), &[[0.0; 3]]);
    // The ω half of the builder works — that is 7a, and it ships.
    let (omega, mesh, _) = b.guess().expect("guess must work");
    assert!(omega > 0.0 && mesh[0] > 1);

    let e = b
        .build()
        .expect_err("the SR 3-centre route must be refused");
    let msg = format!("{e}");
    assert!(
        msg.contains("_RSGDFBuilder") && msg.contains("7b/7c"),
        "the refusal must name the unported sub-tasks, not a stale blocker: {msg}"
    );
    assert!(
        msg.contains("range_omega") && msg.contains("env[8]"),
        "the refusal must say the integral capability EXISTS, so the next reader \
         does not re-derive a blocker that is gone: {msg}"
    );
    assert!(
        !msg.contains("cintx's safe API has no"),
        "the old cintx blocker text must not come back: {msg}"
    );
}

/// `GDF::prefer_ccdf` therefore stays `true`: plan 14-07's Task 7d flip cannot
/// happen while the RS route is unbuildable, and a committed reference energy
/// must not move on a route that does not exist.
#[test]
fn gdf_default_route_has_not_flipped() {
    let cell = common::he_all_electron();
    let g = pyscf_pbc_df::Gdf::new(cell, &[[0.0; 3]]);
    assert!(
        g.prefer_ccdf,
        "Task 7d flips this to false ONLY once _RSGDFBuilder builds; until then \
         the default must stay on the route that works"
    );
}
