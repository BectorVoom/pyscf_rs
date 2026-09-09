//! K-02 — the k-symmetric KRKS SCF running on the k-point-resolved
//! multigrid (`KsymAdaptedKrks::ni = KsNumInt::multigrid2()`).
//!
//! # What this suite can and cannot claim
//!
//! Multigrid is a **different quadrature** from the reference `KNumInt`, not
//! a tighter-tolerance target — 17-CONTEXT §2.2 Gate E, restated by
//! `tests/multigrid2.rs`'s module doc and by `17-VERIFICATION.md`. So the
//! comparison that means something here is **multigrid-vs-multigrid**:
//!
//! * `ibz_energy_matches_full_bz_on_multigrid` — the IBZ SCF against the
//!   FULL-BZ SCF, both on multigrid, same cell, same pinned mesh. This is
//!   the symmetry gate proper and it is oracle-free: it is what "the IBZ
//!   determines the zone" means. A symmetry defect moves it; a quadrature
//!   difference cannot, because both sides use the same quadrature.
//! * `multigrid_and_grid_agree_at_the_multigrid_floor` — the same IBZ SCF
//!   through both quadratures, reported at the floor 17-12's Gate E measured
//!   rather than at the symmetry gate's tolerance. Recorded as a
//!   MEASUREMENT with its own loose bound, so a regression is still caught
//!   without the number pretending to be an agreement it is not.
//!
//! # Why the multigrid arm needs no symmetry of its own
//!
//! `KsymAdaptedKrks` already unfolds the density to the full BZ once per
//! cycle (S-01) and asks for the potential at `kpts_band = kpts_ibz`. The
//! k-resolved multigrid consumes exactly that: its collocation is over
//! lattice IMAGES and does not grow with `nkpts`, and `kpts_band` is one
//! more phase table over the same real grid integrals. So there is no
//! multigrid analogue of S-03's symmetrised quadrature, and none is
//! invented — see `multigrid::kpts`'s module doc.

use pyscf_pbc_dft::krks::Krks;
use pyscf_pbc_dft::krks_ksymm::KsymAdaptedKrks;
use pyscf_pbc_dft::numint::KsNumInt;
use pyscf_pbc_gto::make_kpts_default;
use pyscf_pbc_gto::test_systems::si_precision;
use pyscf_pbc_scf::{KInitGuess, KScfConfig};
use pyscf_pbc_symm::kpts::make_kpts;

/// The same fixture tightness `krks_ksymm.rs` uses, and for the same reason:
/// the residual rides on `transform_dm`, whose accuracy 17-05 measured as a
/// joint function of `cell.precision` and `conv_tol_grad`.
const FIXTURE_PRECISION: f64 = 1e-10;
const FIXTURE_CONV_TOL_GRAD: f64 = 1e-10;

/// D-17-07-01 — `little_cogroup_ops` indexes `k2opk`'s doubled column space
/// while its consumers index `ops`, an upstream mismatch that surfaces at Γ.
const TIME_REVERSAL: bool = false;

/// How much WORSE than the reference grid's own IBZ-vs-full-BZ gap the
/// multigrid's gap is allowed to be, as a ratio.
///
/// **Why this is a ratio against a measured control and not an absolute
/// tolerance.** The first version of this gate asserted `|dE| < 1e-9`
/// outright and measured **6.875e-7** on this fixture. The multigrid was not
/// at fault: the multigrid IBZ run agrees with the *grid* IBZ run to
/// **1.307e-9** (`multigrid_and_grid_agree_at_the_multigrid_floor`), so the
/// outlier is the full-BZ side, and it is the outlier for BOTH quadratures.
/// The cause is the PINNED COARSE MESH this file uses to keep the two sides
/// comparable — the same effect already recorded for `KRHF` (a pinned mesh
/// 15 moves the energy by 1.35e-5 where the default mesh gives 4.8e-11), and
/// `krks_ksymm.rs`'s own `krks_ibz_energy_matches_full_bz` avoids it by not
/// pinning the mesh at all.
///
/// A mesh artefact shared by both quadratures is not a symmetry defect in
/// the one under test, and hiding it behind a loosened absolute tolerance
/// would throw away the gate's ability to catch a real one. So the control
/// is measured in the same test and the assertion is relative to it.
const SYMMETRY_GAP_RATIO: f64 = 4.0;

/// A floor under the ratio, so a fixture whose control gap happens to be
/// near zero cannot make the assertion impossible to satisfy.
const SYMMETRY_GAP_FLOOR: f64 = 1e-9;

/// A coarse mesh, PINNED on both sides. `make_kpts` can itself change
/// `cell.mesh` through `check_mesh_symmetry`, and an unpinned comparison
/// would measure that instead of the symmetry.
const MESH: [usize; 3] = [15, 15, 15];

fn cfg() -> KScfConfig {
    KScfConfig {
        conv_tol: 1e-11,
        conv_tol_grad: Some(FIXTURE_CONV_TOL_GRAD),
        max_cycle: 50,
        init_guess: KInitGuess::Minao,
        ..KScfConfig::default()
    }
}

/// **The symmetry gate.** The multigrid's IBZ-vs-full-BZ gap against the
/// reference grid's own gap on the identical fixture — see
/// `SYMMETRY_GAP_RATIO` for why the control is measured rather than assumed
/// to be zero.
#[test]
fn ibz_energy_matches_full_bz_on_multigrid() {
    let mut cell = si_precision(FIXTURE_PRECISION);
    cell.mesh = MESH;
    let kpts_abs = make_kpts_default(&cell, [2, 2, 2]).expect("make_kpts_default");
    let kpts = make_kpts(&cell, &kpts_abs, true, TIME_REVERSAL).expect("make_kpts");
    assert!(
        kpts.nkpts_ibz() < kpts.nkpts(),
        "the fixture must actually fold: {} IBZ of {} BZ",
        kpts.nkpts_ibz(),
        kpts.nkpts()
    );
    cell.mesh = MESH;

    // Full BZ, on multigrid.
    let mut full = Krks::new(cell.clone(), &kpts.kpts, "lda,vwn").expect("Krks");
    full.ni = KsNumInt::multigrid2();
    let r_full = full.kernel(&cfg()).expect("full-BZ KRKS on multigrid");
    assert!(r_full.converged, "full-BZ multigrid KRKS did not converge");

    // IBZ, on multigrid. `use_ao_symmetry = false` keeps D-17-09-02 (the
    // non-canonical-orbital finding) out of this measurement: it is a
    // property of the symmetry-adapted eigensolve, not of the quadrature,
    // and `krks_ksymm.rs` gates it already.
    let mut ibz =
        KsymAdaptedKrks::new(cell.clone(), kpts.clone(), "lda,vwn").expect("KsymAdaptedKrks");
    ibz.use_ao_symmetry = false;
    ibz.ni = KsNumInt::multigrid2();
    let r_ibz = ibz.kernel(&cfg()).expect("IBZ KRKS on multigrid");
    assert!(r_ibz.converged, "IBZ multigrid KRKS did not converge");

    // THE CONTROL: the same IBZ-vs-full-BZ comparison through the reference
    // grid quadrature, on the identical fixture and the identical pinned
    // mesh. Whatever the mesh does to the symmetry, it does to both.
    let ctrl_full = Krks::new(cell.clone(), &kpts.kpts, "lda,vwn").expect("Krks");
    let r_ctrl_full = ctrl_full.kernel(&cfg()).expect("full-BZ KRKS on the grid");
    assert!(
        r_ctrl_full.converged,
        "control full-BZ KRKS did not converge"
    );
    let mut ctrl_ibz =
        KsymAdaptedKrks::new(cell.clone(), kpts.clone(), "lda,vwn").expect("KsymAdaptedKrks");
    ctrl_ibz.use_ao_symmetry = false;
    let r_ctrl_ibz = ctrl_ibz.kernel(&cfg()).expect("IBZ KRKS on the grid");
    assert!(r_ctrl_ibz.converged, "control IBZ KRKS did not converge");

    let de = (r_full.e_tot - r_ibz.e_tot).abs();
    let de_ctrl = (r_ctrl_full.e_tot - r_ctrl_ibz.e_tot).abs();
    println!(
        "multigrid symmetry ({} IBZ of {} BZ, mesh {MESH:?}):\n  \
         multigrid: e_full = {:.12}, e_ibz = {:.12}, |dE| = {de:e}\n  \
         grid CTRL: e_full = {:.12}, e_ibz = {:.12}, |dE| = {de_ctrl:e}",
        kpts.nkpts_ibz(),
        kpts.nkpts(),
        r_full.e_tot,
        r_ibz.e_tot,
        r_ctrl_full.e_tot,
        r_ctrl_ibz.e_tot,
    );
    let bound = (de_ctrl * SYMMETRY_GAP_RATIO).max(SYMMETRY_GAP_FLOOR);
    assert!(
        de < bound,
        "the multigrid's IBZ-vs-full-BZ gap is {de:e} against the reference \
         grid's {de_ctrl:e} on the IDENTICAL fixture — more than {SYMMETRY_GAP_RATIO}x \
         worse, so this is the multigrid's own symmetry defect and not the \
         pinned mesh both sides share"
    );
}

/// The multigrid IBZ SCF against the reference-grid IBZ SCF.
///
/// **Reported at multigrid's own floor, not at the symmetry gate's.** These
/// are two different quadratures; 17-12's Gate E measured that difference on
/// the gamma point and this is its k-symmetric analogue. The bound is loose
/// on purpose — it exists to catch a regression of ORDERS, not to assert an
/// agreement multigrid does not have.
#[test]
fn multigrid_and_grid_agree_at_the_multigrid_floor() {
    let mut cell = si_precision(FIXTURE_PRECISION);
    cell.mesh = MESH;
    let kpts_abs = make_kpts_default(&cell, [2, 2, 2]).expect("make_kpts_default");
    let kpts = make_kpts(&cell, &kpts_abs, true, TIME_REVERSAL).expect("make_kpts");
    cell.mesh = MESH;

    let mut grid =
        KsymAdaptedKrks::new(cell.clone(), kpts.clone(), "lda,vwn").expect("KsymAdaptedKrks");
    grid.use_ao_symmetry = false;
    let r_grid = grid.kernel(&cfg()).expect("IBZ KRKS on the grid");
    assert!(r_grid.converged, "grid IBZ KRKS did not converge");

    let mut mg =
        KsymAdaptedKrks::new(cell.clone(), kpts.clone(), "lda,vwn").expect("KsymAdaptedKrks");
    mg.use_ao_symmetry = false;
    mg.ni = KsNumInt::multigrid2();
    let r_mg = mg.kernel(&cfg()).expect("IBZ KRKS on multigrid");
    assert!(r_mg.converged, "multigrid IBZ KRKS did not converge");

    let de = (r_grid.e_tot - r_mg.e_tot).abs();
    println!(
        "GATE E (k-symmetric): e_grid = {:.12}, e_multigrid = {:.12}, |dE| = {de:e} Ha \
         at mesh {MESH:?} — the two quadratures' own difference, measured",
        r_grid.e_tot, r_mg.e_tot
    );
    assert!(
        de < 1e-3,
        "multigrid and the reference grid differ by {de:e} Ha on the same IBZ SCF. \
         Gate E's floor is a small multiple of the collocation error, not this — \
         a residual this large is a defect, not a quadrature difference"
    );
}
