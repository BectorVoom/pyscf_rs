//! **M-00 — `nr_uks` for both multigrid drivers.**
//!
//! Plan item M-00 of
//! `.planning/pbc/KUKS-KSYMM-MULTIGRID-OPTIMISATION-PLAN.md` §2.3.1. 17-11
//! shipped `MultiGridNumInt::nr_rks` and 17-12 shipped
//! `MultiGridNumInt2::nr_rks`; neither shipped `nr_uks`, while upstream has
//! `multigrid.py:1166`. So "KUKS on multigrid" was a phrase and not a code
//! path — and, more consequentially for this plan, **no multigrid
//! optimisation could be validated on an open-shell density at all**, which
//! RULE U requires.
//!
//! # What each test can see
//!
//! | test | what only it can see |
//! |---|---|
//! | `*_nr_uks_matches_the_reference_numint` | that the open-shell quadrature agrees with `KNumInt::nr_uks` on the SAME grid — the Gate E comparison, for two channels |
//! | `*_nr_uks_reduces_to_nr_rks_on_a_closed_shell_density` | that the spin machinery is wired the right way round: `dm_a == dm_b == dm/2` must reproduce `nr_rks(dm)` |
//! | `*_open_shell_fixture_is_genuinely_polarised` | RULE U — that the fixture is not silently closed-shell, which would make every other row here vacuous |
//!
//! The middle row is the one that catches a transposed channel or a dropped
//! spin sum, and it needs no reference implementation at all: it is an
//! identity of the unrestricted functional itself. `exc` and `ecoul` must
//! match to machine precision; `veff` per channel must equal the restricted
//! `veff`.
//!
//! # Tolerances
//!
//! Inherited from the existing Gate E rows rather than invented: v1 sits at
//! the reference quadrature's own floor (1e-6 on `nelec`/`exc`,
//! `multigrid.rs::gate_e_nr_rks_lda_vs_reference`), v2 an order looser
//! (1e-3, `multigrid2.rs::gate_e_nr_rks_lda_vs_reference_v2`) because its
//! per-image screening threshold is `precision * EXTRA_PREC` rather than
//! `precision / vol`. Neither number is relaxed here.

mod common;

use pyscf_pbc_dft::multigrid::{MultiGridNumInt, MultiGridNumInt2};
use pyscf_pbc_dft::numint::KNumInt;
use pyscf_pbc_gto::Cell;

const MESH: [usize; 3] = [25, 25, 25];
const GAMMA: [[f64; 3]; 1] = [[0.0, 0.0, 0.0]];

fn small_silicon() -> Cell {
    let mut c = common::silicon();
    c.mesh = MESH;
    c
}

fn small_diamond() -> Cell {
    let mut c = common::diamond();
    c.mesh = MESH;
    c
}

fn lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed;
    move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64 / (1u64 << 53) as f64) * 0.2
    }
}

fn random_symmetric_dm(nao: usize, seed: u64) -> Vec<f64> {
    let mut next = lcg(seed);
    let mut dm = vec![0.0f64; nao * nao];
    for v in dm.iter_mut() {
        *v = next();
    }
    for i in 0..nao {
        for j in 0..nao {
            let v = 0.5 * (dm[i * nao + j] + dm[j * nao + i]);
            dm[i * nao + j] = v;
            dm[j * nao + i] = v;
        }
        dm[i * nao + i] += 1.0;
    }
    dm
}

/// A genuinely spin-polarised pair: two INDEPENDENT symmetric densities, not
/// one scaled twice. RULE U — a pair related by a scalar would still exercise
/// two channels but could not distinguish `rho_a + rho_b` from `2 rho_a`.
fn polarised_pair(nao: usize) -> (Vec<f64>, Vec<f64>) {
    let a = random_symmetric_dm(nao, 0xA1FA_0000);
    let b = random_symmetric_dm(nao, 0xBE7A_0000);
    (a, b)
}

fn to_kdms(dm: &[f64], nao: usize) -> pyscf_pbc_scf::types::KDms {
    vec![vec![pyscf_algebra::CTensor::from_planes(
        dm.to_vec(),
        vec![0.0; nao * nao],
    )]]
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

// ---------------------------------------------------------------------------
// Gate E for the open-shell path
// ---------------------------------------------------------------------------

#[test]
fn v1_nr_uks_matches_the_reference_numint() {
    let ni = MultiGridNumInt::new();
    for (name, cell) in [("diamond", small_diamond()), ("si", small_silicon())] {
        let nao = cell.mol.nao_nr;
        let (dma, dmb) = polarised_pair(nao);
        let out = ni
            .nr_uks(&cell, "lda,vwn", &[&dma, &dmb])
            .expect("multigrid v1 nr_uks");

        let refni = KNumInt::new(&GAMMA);
        let grids = pyscf_pbc_dft::gen_grid::PeriodicGrids::uniform(&cell, Some(cell.mesh))
            .expect("uniform grid");
        let sets = [to_kdms(&dma, nao), to_kdms(&dmb, nao)];
        let refout = refni
            .nr_uks(&cell, &grids, "lda,vwn", &sets, 1, None)
            .expect("reference nr_uks");

        let dna = (out.nelec.0 - refout.nelec[0].0).abs();
        let dnb = (out.nelec.1 - refout.nelec[0].1).abs();
        let dexc = (out.exc - refout.excsum[0]).abs();
        println!(
            "{name}: v1 nr_uks(lda,vwn) |d nelec| = ({dna:.3e}, {dnb:.3e})  |d exc| = {dexc:.3e}"
        );
        assert!(dna < 1e-6, "{name}: v1 alpha nelec diff {dna:.3e}");
        assert!(dnb < 1e-6, "{name}: v1 beta nelec diff {dnb:.3e}");
        assert!(dexc < 1e-6, "{name}: v1 exc diff {dexc:.3e}");
    }
}

#[test]
fn v2_nr_uks_matches_the_reference_numint() {
    let ni = MultiGridNumInt2::new();
    // ONE cell: a v2 density evaluation is seconds, and `nr_uks` runs four
    // sweeps. `si` is the harder of the two (its v2-vs-reference floor is
    // ~5x diamond's, 17-12), so it is the one kept.
    let cell = small_silicon();
    let nao = cell.mol.nao_nr;
    let (dma, dmb) = polarised_pair(nao);
    let out = ni
        .nr_uks(&cell, "lda,vwn", &[&dma, &dmb])
        .expect("multigrid v2 nr_uks");

    let refni = KNumInt::new(&GAMMA);
    let grids =
        pyscf_pbc_dft::gen_grid::PeriodicGrids::uniform(&cell, Some(cell.mesh)).expect("grid");
    let sets = [to_kdms(&dma, nao), to_kdms(&dmb, nao)];
    let refout = refni
        .nr_uks(&cell, &grids, "lda,vwn", &sets, 1, None)
        .expect("reference nr_uks");

    let dna = (out.nelec.0 - refout.nelec[0].0).abs();
    let dnb = (out.nelec.1 - refout.nelec[0].1).abs();
    let dexc = (out.exc - refout.excsum[0]).abs();
    println!("si: v2 nr_uks(lda,vwn) |d nelec| = ({dna:.3e}, {dnb:.3e})  |d exc| = {dexc:.3e}");
    assert!(dna < 1e-3, "v2 alpha nelec diff {dna:.3e}");
    assert!(dnb < 1e-3, "v2 beta nelec diff {dnb:.3e}");
    assert!(dexc < 1e-3, "v2 exc diff {dexc:.3e}");
}

// ---------------------------------------------------------------------------
// The oracle-free identity: UKS on a closed shell IS RKS
// ---------------------------------------------------------------------------

/// `nr_uks(dm/2, dm/2)` must reproduce `nr_rks(dm)` — an identity of the
/// unrestricted functional, needing no reference implementation and no
/// converged state.
///
/// This is what catches the errors a Gate E comparison cannot: a transposed
/// channel, a spin sum applied to the wrong axis, `vG` added to only one
/// channel, or `excsum` counting one spin twice. All of those leave the
/// per-channel numbers plausible and only this identity refuses them.
///
/// The tolerance is machine-precision-ish rather than a quadrature floor: both
/// sides run the SAME collocation on the SAME grid, so the only difference is
/// the association of a few sums.
#[test]
fn v1_nr_uks_reduces_to_nr_rks_on_a_closed_shell_density() {
    let ni = MultiGridNumInt::new();
    for (name, cell) in [("diamond", small_diamond()), ("si", small_silicon())] {
        let nao = cell.mol.nao_nr;
        let dm = random_symmetric_dm(nao, 0xC0FF_EE00);
        let half: Vec<f64> = dm.iter().map(|x| x * 0.5).collect();

        for xc in ["lda,vwn", "pbe"] {
            let r = ni.nr_rks(&cell, xc, &dm).expect("nr_rks");
            let u = ni
                .nr_uks(&cell, xc, &[&half, &half])
                .expect("nr_uks on a closed shell");

            let dn = (u.nelec.0 + u.nelec.1 - r.nelec).abs();
            let dcoul = (u.ecoul - r.ecoul).abs();
            let dexc = (u.exc - r.exc).abs();
            let dva = max_abs_diff(&u.veff[0], &r.veff);
            let dvb = max_abs_diff(&u.veff[1], &r.veff);
            println!(
                "{name} {xc}: v1 UKS(dm/2,dm/2) vs RKS(dm)  |d nelec| = {dn:.3e}  \
                 |d ecoul| = {dcoul:.3e}  |d exc| = {dexc:.3e}  |d veff| = ({dva:.3e}, {dvb:.3e})"
            );
            let scale = r.exc.abs().max(r.ecoul.abs()).max(1.0);
            assert!(
                u.nelec.0 == u.nelec.1,
                "{name} {xc}: identical channels gave different electron counts \
                 ({} vs {}) — the two channels are not being treated identically",
                u.nelec.0,
                u.nelec.1
            );
            assert!(
                dn < 1e-10 * scale,
                "{name} {xc}: nelec identity broke, {dn:.3e}"
            );
            assert!(
                dcoul < 1e-10 * scale,
                "{name} {xc}: ecoul identity broke, {dcoul:.3e}"
            );
            assert!(
                dexc < 1e-10 * scale,
                "{name} {xc}: exc identity broke, {dexc:.3e}"
            );
            assert!(
                dva < 1e-10 * scale,
                "{name} {xc}: alpha veff identity broke, {dva:.3e}"
            );
            assert!(
                dvb < 1e-10 * scale,
                "{name} {xc}: beta veff identity broke, {dvb:.3e}"
            );
        }
    }
}

#[test]
fn v2_nr_uks_reduces_to_nr_rks_on_a_closed_shell_density() {
    let ni = MultiGridNumInt2::new();
    let cell = small_silicon();
    let nao = cell.mol.nao_nr;
    let dm = random_symmetric_dm(nao, 0xC0FF_EE00);
    let half: Vec<f64> = dm.iter().map(|x| x * 0.5).collect();

    let r = ni.nr_rks(&cell, "lda,vwn", &dm).expect("nr_rks");
    let u = ni
        .nr_uks(&cell, "lda,vwn", &[&half, &half])
        .expect("nr_uks on a closed shell");

    let dn = (u.nelec.0 + u.nelec.1 - r.nelec).abs();
    let dcoul = (u.ecoul - r.ecoul).abs();
    let dexc = (u.exc - r.exc).abs();
    let dva = max_abs_diff(&u.veff[0], &r.veff);
    let dvb = max_abs_diff(&u.veff[1], &r.veff);
    println!(
        "si lda,vwn: v2 UKS(dm/2,dm/2) vs RKS(dm)  |d nelec| = {dn:.3e}  \
         |d ecoul| = {dcoul:.3e}  |d exc| = {dexc:.3e}  |d veff| = ({dva:.3e}, {dvb:.3e})"
    );
    let scale = r.exc.abs().max(r.ecoul.abs()).max(1.0);
    assert!(dn < 1e-10 * scale, "v2 nelec identity broke, {dn:.3e}");
    assert!(
        dcoul < 1e-10 * scale,
        "v2 ecoul identity broke, {dcoul:.3e}"
    );
    assert!(dexc < 1e-10 * scale, "v2 exc identity broke, {dexc:.3e}");
    assert!(
        dva < 1e-10 * scale,
        "v2 alpha veff identity broke, {dva:.3e}"
    );
    assert!(
        dvb < 1e-10 * scale,
        "v2 beta veff identity broke, {dvb:.3e}"
    );
}

/// RULE U, asserted rather than assumed: if the fixture's two channels were
/// (near-)equal, every open-shell row in this file would be a closed-shell row
/// wearing an open-shell name.
#[test]
fn the_open_shell_fixture_is_genuinely_polarised() {
    let cell = small_silicon();
    let nao = cell.mol.nao_nr;
    let (dma, dmb) = polarised_pair(nao);
    let d = max_abs_diff(&dma, &dmb);
    println!("fixture: max |dm_a - dm_b| = {d:.3e}");
    assert!(
        d > 1e-3,
        "the two channels differ by only {d:.3e} — RULE U: this fixture cannot \
         see anything in the unrestricted path"
    );
}
