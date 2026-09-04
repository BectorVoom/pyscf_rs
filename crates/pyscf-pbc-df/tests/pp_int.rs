//! `pyscf_pbc_df::pp_int::get_pp_loc_part2_kpts` — upstream's
//! `aft._IntPPBuilder`, the k-resolved short-range local pseudopotential.
//!
//! Plan 14-03 found that Phase 10's `pseudo::vloc_part2::get_pp_loc_part2` is
//! GAMMA-ONLY, which silently limited AFTDF, GDF, MDF and RSDF to gamma. This
//! closes it.

mod common;

use pyscf_pbc_df::pp_int::get_pp_loc_part2_kpts;

fn kpts(cell: &pyscf_pbc_gto::Cell, km: [usize; 3]) -> Vec<[f64; 3]> {
    pyscf_pbc_gto::kpts_mesh::make_kpts(cell, km, false, true, None).expect("kpts")
}

/// **The gate on the port.** At gamma the new k-resolved route must reproduce
/// Phase 10's already-oracle-verified `get_pp_loc_part2_gamma` — same lattice
/// sum, same operators, one extra (trivial) Bloch phase. Anything else means
/// the generalisation changed the algebra.
#[test]
fn gamma_reproduces_the_phase_10_route() {
    let cell = common::diamond();
    let nao = cell.mol.nao_nr;
    let want = pyscf_pbc_gto::pseudo::vloc_part2::get_pp_loc_part2_gamma(&cell)
        .expect("phase-10 gamma route");
    let got = get_pp_loc_part2_kpts(&cell, &[[0.0; 3]]).expect("k-resolved route");
    assert_eq!(got.len(), 1);

    let mut worst = 0.0_f64;
    for mu in 0..nao {
        for nu in 0..nao {
            // Phase 10's output is F-ORDER; this route is row-major. The matrix
            // is symmetric at gamma, so the comparison is order-free — but the
            // transpose is asserted explicitly so a future non-symmetric
            // operator cannot slip through.
            let a = got[0].re[mu * nao + nu];
            let b = want[mu + nu * nao];
            worst = worst.max((a - b).abs());
        }
    }
    assert!(
        worst < 1e-12,
        "gamma: k-resolved route differs from Phase 10's by {worst:e}"
    );
    let max_im = got[0].im.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    assert!(
        max_im < 1e-14,
        "gamma block should be real, |Im| = {max_im:e}"
    );
}

/// Away from gamma the matrix must be HERMITIAN — the property the gamma test
/// cannot check, and the one a wrong Bloch phase breaks.
#[test]
fn k_points_are_hermitian() {
    let cell = common::diamond();
    let nao = cell.mol.nao_nr;
    let k = kpts(&cell, [2, 2, 2]);
    let mats = get_pp_loc_part2_kpts(&cell, &k).expect("k-resolved route");
    assert_eq!(mats.len(), k.len());
    for (ik, m) in mats.iter().enumerate() {
        let mut worst = 0.0_f64;
        for p in 0..nao {
            for q in 0..nao {
                let (a, b) = (p * nao + q, q * nao + p);
                worst = worst.max((m.re[a] - m.re[b]).abs());
                worst = worst.max((m.im[a] + m.im[b]).abs());
            }
        }
        assert!(worst < 1e-11, "k={ik}: asymmetry {worst:e}");
    }
    // NOTE every k-point of a 2x2x2 Monkhorst-Pack mesh is HALF a reciprocal
    // lattice vector and therefore self-conjugate, so `T(k) = conj(T(k))` and
    // these blocks are REAL. That is physics, not a missing Bloch phase — the
    // test for the phase is `bloch_phase_reaches_a_generic_k_point` below,
    // which uses a k-point that is not its own negative.
}

/// A GENERIC k-point — not a Monkhorst-Pack grid point, so not self-conjugate —
/// must give a genuinely COMPLEX block. Without this, a dropped Bloch phase
/// would sail through both the Hermiticity and the time-reversal tests.
#[test]
fn bloch_phase_reaches_a_generic_k_point() {
    let cell = common::diamond();
    let k = [0.11_f64, -0.07, 0.19];
    let mats = get_pp_loc_part2_kpts(&cell, &[k]).expect("k-resolved route");
    let max_im = mats[0].im.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    assert!(
        max_im > 1e-8,
        "a generic k-point came out real (|Im| = {max_im:e}) — the Bloch phase \
         never reached the accumulation"
    );
}

/// `T(-k) = conj(T(k))` — time-reversal symmetry of a real operator.
#[test]
fn obeys_time_reversal_symmetry() {
    let cell = common::diamond();
    let nao = cell.mol.nao_nr;
    let k = [0.11_f64, -0.07, 0.19];
    let mats = get_pp_loc_part2_kpts(&cell, &[k, [-k[0], -k[1], -k[2]]]).expect("k-resolved route");
    let mut worst = 0.0_f64;
    for p in 0..nao * nao {
        worst = worst.max((mats[0].re[p] - mats[1].re[p]).abs());
        worst = worst.max((mats[0].im[p] + mats[1].im[p]).abs());
    }
    assert!(worst < 1e-12, "T(-k) != conj(T(k)): {worst:e}");
}

/// An all-electron cell has no local pseudopotential, so part 2 is identically
/// zero — and must not error.
#[test]
fn all_electron_cell_has_no_part_two() {
    let cell = common::he_all_electron();
    let k = kpts(&cell, [2, 2, 2]);
    let mats = get_pp_loc_part2_kpts(&cell, &k).expect("all-electron");
    assert_eq!(mats.len(), k.len());
    for m in &mats {
        assert!(m.re.iter().all(|v| *v == 0.0));
        assert!(m.im.iter().all(|v| *v == 0.0));
    }
}
