//! Plan 14-08 acceptance — `get_aux_chg`, the shared `density_fit` shim, and
//! the recorded refusal of everything range-separated.
//!
//! # What this file can and cannot gate
//!
//! `14-08-PLAN.md` Task 5 asks for four things. Two of them ship:
//!
//! * **5.1** `get_aux_chg == 14-01's monopole` at 1e-14 — here.
//! * **Task 3** the `density_fit` shims, one implementation for all four
//!   builders — here.
//!
//! Two cannot:
//!
//! * **5.2, GATE 3** — `|E_KRHF(GDF) − E_KRHF(RSDF)|` against upstream's floor
//!   (1.353e-08 diamond 2×2×2, 4.566e-09 gamma, 1.113e-10 He-fcc). It needs
//!   RSDF, which needs `_RSGDFBuilder`, which needs a short-range `int3c2e`.
//! * **5.3, 5.4, 5.5** — `rsjk` against FFTDF. `rsjk`'s short-range half is a
//!   short-range `int2e` (`rsjk.py:186` sets `supmol_sr.omega = -self.omega`).
//!
//! Both are the SAME missing capability: cintx's safe API has no `range_omega`
//! (libcint `env[8]`) knob. `ExecutionOptions` carries `f12_zeta` (`env[9]`),
//! `rinv_orig` and `common_orig`; no kernel reads `env[8]`; and this repository
//! already records the gap as Phase 4's Open Question A5 / cintx#11 in
//! `crates/pyscf-gto/src/range_coulomb.rs`. See
//! `crates/pyscf-pbc-df/src/rsdf_builder/mod.rs` for the full evidence.
//!
//! The tests below therefore assert the REFUSAL, so that "RSDF is missing" is a
//! fact the suite states rather than an absence a reader has to notice.

mod common;

use pyscf_pbc_df::{DfKind, DfOpts, Rsdf, density_fit, get_aux_chg};

// ---------------------------------------------------------------------------
// Task 5.1 — get_aux_chg
// ---------------------------------------------------------------------------

/// **Task 5.1** — `get_aux_chg` IS 14-01's monopole, to 1e-14. No oracle.
///
/// `make_modrho_basis` normalises every fitted s function to unit charge
/// (plan 14-01 Task 3 gated that at 1e-14 against upstream), so this is not a
/// tautology: it asserts that `rsdf.get_aux_chg`'s `ft_ao(auxcell, G=0).real`
/// and `incore.make_modrho_basis`'s `gaussian_int` normalisation are the same
/// convention. Upstream computes the charge one way and normalises the other,
/// and a mismatch would make range separation treat the wrong functions as
/// charged.
#[test]
fn get_aux_chg_is_the_modrho_monopole() {
    for (name, cell) in [
        ("He-fcc", common::he_all_electron()),
        ("diamond", common::diamond()),
    ] {
        let aux = pyscf_pbc_df::make_modrho_basis(&cell, None, None).expect("modrho");
        let chg = get_aux_chg(&aux.cell).expect("get_aux_chg");
        assert_eq!(chg.len(), aux.naux(), "{name}: one charge per auxiliary AO");

        // Every s function carries exactly unit charge; every l > 0 function
        // integrates to zero by symmetry.
        let charged = chg.iter().filter(|v| (**v - 1.0).abs() < 1e-14).count();
        let neutral = chg.iter().filter(|v| v.abs() < 1e-14).count();
        assert_eq!(
            charged + neutral,
            chg.len(),
            "{name}: a monopole is neither 0 nor 1 — the modrho normalisation and \
             get_aux_chg disagree on the convention"
        );
        assert!(charged > 0, "{name}: no charged auxiliary function");

        // And it agrees with rsdf_builder's `_gaussian_int`, which is the same
        // integral under upstream's other name.
        let gi = pyscf_pbc_df::rsdf_builder::gaussian_int(&aux.cell).expect("gaussian_int");
        let w = chg
            .iter()
            .zip(&gi)
            .fold(0.0f64, |a, (x, y)| a.max((x - y).abs()));
        assert!(w == 0.0, "{name}: get_aux_chg != _gaussian_int by {w:e}");
    }
}

// ---------------------------------------------------------------------------
// Task 3 — the density_fit shims
// ---------------------------------------------------------------------------

/// **Task 3** — one `density_fit` produces every builder, each naming itself.
///
/// Upstream has four copies of this function; the plan asks for one. The
/// builders come back as `Box<dyn PeriodicDf>`, which is what D-PBC-22 made the
/// drivers accept.
#[test]
fn density_fit_produces_every_shipped_builder() {
    let cell = common::he_all_electron();
    let kpts = [[0.0; 3]];
    for kind in [DfKind::Fftdf, DfKind::Aftdf, DfKind::Gdf, DfKind::Mdf] {
        let df = density_fit(cell.clone(), &kpts, kind, DfOpts::default())
            .unwrap_or_else(|e| panic!("density_fit({kind:?}) failed: {e}"));
        assert_eq!(df.name(), kind.name(), "the builder must name itself");
        assert_eq!(df.kpts().len(), 1);
    }
}

/// `mesh` reaches the builders that have one, and `auxbasis` the builders that
/// have one. A shim that silently drops its arguments is worse than no shim.
#[test]
fn density_fit_honours_its_options() {
    let cell = common::he_all_electron();
    let kpts = [[0.0; 3]];
    let opts = DfOpts {
        auxbasis: None,
        mesh: Some([13, 13, 13]),
    };
    for kind in [DfKind::Fftdf, DfKind::Aftdf, DfKind::Mdf] {
        let df = density_fit(cell.clone(), &kpts, kind, opts.clone()).expect("density_fit");
        assert_eq!(
            df.mesh(),
            [13, 13, 13],
            "{kind:?} must honour the requested mesh"
        );
    }
    // GDF's own mesh is the compensating-charge one and is NOT settable — its
    // job is to resolve the model charge, not the density (14-04's defect 2).
    let df = density_fit(cell, &kpts, DfKind::Gdf, opts).expect("density_fit");
    assert_ne!(
        df.mesh(),
        [13, 13, 13],
        "GDF's mesh is the model-charge mesh and must not take the caller's"
    );
}

// ---------------------------------------------------------------------------
// The blocker, asserted
// ---------------------------------------------------------------------------

/// `density_fit(RSDF)` REFUSES and names the cintx gap — D-PBC-20. A shim that
/// quietly returned a GDF instead would make Gate 3 pass by comparing GDF with
/// itself.
#[test]
fn density_fit_refuses_rsdf_and_names_the_gap() {
    let cell = common::he_all_electron();
    let e = density_fit(cell, &[[0.0; 3]], DfKind::Rsdf, DfOpts::default())
        .expect_err("RSDF must be refused");
    let msg = format!("{e}");
    assert!(
        msg.contains("range_omega") && msg.contains("env[8]"),
        "the refusal must name the missing cintx capability: {msg}"
    );
}

/// `RSDF`'s ω half works — that is plan 14-07's 7a — and its build does not.
///
/// Recording both in one test is the point: it says precisely how far the port
/// got, so `14-VERIFICATION.md`'s "Gate 3 unreachable" is a measured statement
/// and not a shrug.
#[test]
fn rsdf_guesses_omega_but_cannot_build() {
    let cell = common::he_all_electron();
    let kpts = cell.make_kpts([2, 2, 2]).expect("kpts");
    let mut d = Rsdf::new(cell, &kpts);

    let (omega, mesh, ke) = d.guess_omega().expect("the omega half ships");
    assert!((omega - 0.739_358_637_866_536).abs() < 1e-12, "omega: {omega}");
    assert_eq!(mesh, [11, 11, 11]);
    assert!((ke - 30.708_567_591_994_9).abs() < 1e-10, "ke_cutoff: {ke}");

    let e = d.build().expect_err("the SR 3-centre route must be refused");
    let msg = format!("{e}");
    assert!(
        msg.contains("range_omega"),
        "the refusal must name the gap: {msg}"
    );
}
