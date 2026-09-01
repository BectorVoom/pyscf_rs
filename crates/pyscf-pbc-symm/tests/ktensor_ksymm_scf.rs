//! `KsymmArray` fed from a REAL `khf_ksymm` SCF — the acceptance test
//! `17-06-SUMMARY.md` explicitly handed to 17-07.
//!
//! # What 17-06 asked for, and why it could not do it itself
//!
//! > **17-07 — the acceptance test the plan actually named.** Once
//! > `khf_ksymm` has a Fock store, add a test that builds it as a
//! > `KsymmArray` over the IBZ [...] **directly from the SCF's own container
//! > rather than from a test-local projection**, reads back every BZ
//! > k-point, and compares to the dense full-BZ [quantity].
//! >
//! > — `17-06-SUMMARY.md`, "What 17-07 (and 17-09) must add"
//!
//! When 17-06 ran, `khf_ksymm` did not exist, so its own
//! `acceptance_real_converged_krhf_mo_blocks_round_trip_through_ksymmarray`
//! projected MO blocks out of a **full-BZ** KRHF. That stand-in is kept — it
//! pins the algebra independently of `khf_ksymm`'s bookkeeping. This file adds
//! the missing half: the store is now filled from
//! [`KsymAdaptedKrhf`]'s **own IBZ-length output**.
//!
//! # The gauge problem, and how the comparison avoids it
//!
//! The obvious test — compare against MO blocks from a separate full-BZ KRHF —
//! is **not sound**. Orbitals are defined only up to a unitary rotation within
//! each degenerate subspace, so the ksymm SCF's `C` at an IBZ k-point and a
//! full-BZ SCF's `C` at the same k-point need not agree, and `C^H h C` then
//! legitimately differs. 17-CONTEXT §3.1 states the general rule (never
//! compare `mo_coeff` elementwise) and 17-06 met the same wall from the other
//! side, where Schur's lemma made three of its four blocks blind to `trans`.
//!
//! So both sides here start from **one** ksymm SCF's orbitals, and the test
//! compares two independent ways of unfolding them:
//!
//! * `KsymmArray`'s `get_2d` (17-06's `transform_2d`, driven by
//!   `MORotationMatrix`), against
//! * projecting with the orbitals `KPoints::transform_mo_coeff` unfolds
//!   (17-05 Task 3, independently gated at 1e-12).
//!
//! Two routes to the same number inside one process — the idiom 17-10 used for
//! the MO-factorised `get_k_kpts` and 17-07 Task 6 uses for the fast `get_jk`.
//! It is gauge-safe by construction, and it is a real cross-check: the two
//! share no code below `KPoints`.

use num_complex::Complex64;

use pyscf_algebra::CTensor;
use pyscf_pbc_df::{Fftdf, JkOpts, PeriodicDf};
use pyscf_pbc_gto::test_systems::si_precision;
use pyscf_pbc_gto::make_kpts_default;
use pyscf_pbc_scf::khooks::KOverrideHooks;
use pyscf_pbc_scf::krhf::to_row_major;
use pyscf_pbc_scf::{KInitGuess, KScfConfig, KsymAdaptedKrhf};
use pyscf_pbc_symm::basis::{self, SymmAdaptedBasisInput};
use pyscf_pbc_symm::kpts::{MORotationMatrix, make_kpts};
use pyscf_pbc_symm::ktensor::{
    Conj, KsymmArray, KsymmMeta, OrbSpace, SubarrayOrder, parse_label, parse_trans,
};

/// 17-01's Gate-B floor, as 17-06 used it.
const GATE_B_TOL: f64 = 1e-9;

/// The joint precision/convergence fixture — `17-04-MEASUREMENT.md`.
const FIXTURE_PRECISION: f64 = 1e-10;
const FIXTURE_CONV_TOL_GRAD: f64 = 1e-10;

/// `time_reversal_symmetry = false` — D-17-07-01 in `17-07-SUMMARY.md`:
/// `little_cogroup_ops` indexes `k2opk`'s doubled column space while
/// `symm_adapted_basis` indexes `ops`, an upstream mismatch that surfaces at Γ.
const TIME_REVERSAL: bool = false;

/// Column-major `nao x nmo` `CTensor` -> row-major `Vec<Complex64>`.
fn col_major_to_rowmajor(c: &CTensor, nao: usize, nmo: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); nao * nmo];
    for i in 0..nao {
        for p in 0..nmo {
            out[i * nmo + p] = Complex64::new(c.re[p * nao + i], c.im[p * nao + i]);
        }
    }
    out
}

/// `C_a^H M C_b` for row-major `C` (`nao x nmo`) column ranges and row-major
/// `M` (`nao x nao`).
fn project(
    c: &[Complex64],
    m: &[Complex64],
    nao: usize,
    nmo: usize,
    a: std::ops::Range<usize>,
    b: std::ops::Range<usize>,
) -> Vec<Complex64> {
    let (na, nb) = (a.len(), b.len());
    let mut out = vec![Complex64::new(0.0, 0.0); na * nb];
    for (ia, p) in a.clone().enumerate() {
        for (ib, q) in b.clone().enumerate() {
            let mut acc = Complex64::new(0.0, 0.0);
            for i in 0..nao {
                let cip = c[i * nmo + p].conj();
                for j in 0..nao {
                    acc += cip * m[i * nao + j] * c[j * nmo + q];
                }
            }
            out[ia * nb + ib] = acc;
        }
    }
    out
}

#[test]
fn ksymm_scf_fock_store_unfolds_through_ksymmarray() {
    // ---- one k-symmetric SCF, and everything below comes from it ----------
    let mut cell = si_precision(FIXTURE_PRECISION);
    let kpts_abs = make_kpts_default(&cell, [2, 2, 2]).expect("make_kpts_default");
    let kpts = make_kpts(&cell, &kpts_abs, true, TIME_REVERSAL).expect("make_kpts");
    assert!(
        kpts.nkpts_ibz() < kpts.nkpts(),
        "the fixture must fold: {} IBZ of {} BZ",
        kpts.nkpts_ibz(),
        kpts.nkpts()
    );

    let input = SymmAdaptedBasisInput {
        kpts_scaled_ibz: kpts.kpts_scaled_ibz.clone(),
        little_cogroup_ops: kpts.little_cogroup_ops.clone(),
        ops: kpts.symmetry.ops.clone(),
        dmats: kpts.symmetry.dmats.clone(),
    };
    basis::build_symmetry(&mut cell, &input).expect("build_symmetry");

    let df = Fftdf::new(cell.clone(), &kpts.kpts).expect("Fftdf");
    let mf = KsymAdaptedKrhf::from_df(Box::new(df), kpts.clone());
    let r = mf
        .kernel(&KScfConfig {
            conv_tol: 1e-11,
            conv_tol_grad: Some(FIXTURE_CONV_TOL_GRAD),
            max_cycle: 50,
            init_guess: KInitGuess::Minao,
            ..KScfConfig::default()
        })
        .expect("ksymm KRHF");
    assert!(r.converged, "ksymm KRHF did not converge");

    let nao = cell.mol.nao_nr;
    let nibz = kpts.nkpts_ibz();
    let nkpts = kpts.nkpts();
    let nmo = r.mo_occ[0].len();
    let nocc = r.mo_occ[0].iter().filter(|&&o| o > 0.0).count();
    let nvir = nmo - nocc;
    assert_eq!(
        r.mo_coeff.len(),
        nibz,
        "the ksymm SCF must produce IBZ-length orbitals — that is the whole \
         point of the adapter"
    );
    println!(
        "  ksymm SCF: nkpts = {nkpts}, nkpts_ibz = {nibz}, nocc = {nocc}, nvir = {nvir}"
    );

    // The one-electron operator the blocks are built from. `hcore` rather than
    // the Fock: in the MO basis the Fock is DIAGONAL by construction, so
    // `C^H F C` would compare two diagonal matrices and could not see a wrong
    // rotation. `hcore` has genuine off-diagonal structure — the same reason
    // 17-06 gated `C_o^H h C_o` / `C_o^H h C_v` / `C_v^H h C_v` alongside the
    // Fock block.
    let hcore_ibz = to_row_major(
        pyscf_pbc_df::get_hcore(mf.with_df.as_ref(), mf.kpts()).expect("get_hcore ibz"),
        nao,
    );
    let hcore_bz = to_row_major(
        pyscf_pbc_df::get_hcore(mf.with_df.as_ref(), &kpts.kpts).expect("get_hcore bz"),
        nao,
    );
    let to_rm = |t: &CTensor| -> Vec<Complex64> {
        t.re.iter()
            .zip(t.im.iter())
            .map(|(&a, &b)| Complex64::new(a, b))
            .collect()
    };

    // ---- route A: unfold the ORBITALS (17-05), then project ---------------
    let mo_ibz_rm: Vec<Vec<Complex64>> = r
        .mo_coeff
        .iter()
        .map(|c| col_major_to_rowmajor(c, nao, nmo))
        .collect();
    let mo_bz_rm = kpts
        .transform_mo_coeff(&cell, &mo_ibz_rm, nao, nmo)
        .expect("transform_mo_coeff");
    assert_eq!(mo_bz_rm.len(), nkpts, "orbitals must unfold to the full BZ");

    // ---- route B: store the IBZ blocks in a KsymmArray, read back the BZ --
    let mut rmat = MORotationMatrix::new(nocc, nmo);
    let ovlp_bz: Vec<Vec<Complex64>> =
        pyscf_pbc_gto::get_ovlp(&cell, &kpts.kpts)
            .expect("get_ovlp")
            .iter()
            .map(|m| {
                // `get_ovlp` is F-order; `build` wants row-major nao x nao.
                let mut out = vec![Complex64::new(0.0, 0.0); nao * nao];
                for i in 0..nao {
                    for j in 0..nao {
                        out[i * nao + j] = Complex64::new(m.re[i + j * nao], m.im[i + j * nao]);
                    }
                }
                out
            })
            .collect();
    rmat.build(&kpts, &cell, &mo_bz_rm, &ovlp_bz, nao)
        .expect("MORotationMatrix::build");

    let cases: [(&str, usize, usize, std::ops::Range<usize>, std::ops::Range<usize>); 3] = [
        ("oo", nocc, nocc, 0..nocc, 0..nocc),
        ("ov", nocc, nvir, 0..nocc, nocc..nmo),
        ("vv", nvir, nvir, nocc..nmo, nocc..nmo),
    ];

    let mut overall = 0.0_f64;
    for (label_s, di, dj, ra, rb) in cases.iter() {
        let label: Vec<OrbSpace> = parse_label(label_s, 2).expect("label");
        let trans: Vec<Conj> = parse_trans("cn", 2).expect("trans");

        // Route A's reference: project with the UNFOLDED orbitals.
        let ref_bz: Vec<Vec<Complex64>> = (0..nkpts)
            .map(|k| {
                project(
                    &mo_bz_rm[k],
                    &to_rm(&hcore_bz[k]),
                    nao,
                    nmo,
                    ra.clone(),
                    rb.clone(),
                )
            })
            .collect();

        // Route B: build the store from the SCF's OWN IBZ blocks. Only the
        // irreducible representatives are ever written.
        let meta = KsymmMeta {
            kpts: &kpts,
            kqrts: None,
            rmat: Some(&rmat),
            label: Some(&label),
            trans: Some(&trans),
            incore: true,
        };
        let mut arr =
            KsymmArray::zeros(&[*di, *dj], SubarrayOrder::C, meta).expect("KsymmArray::zeros");
        for (i, &bz) in kpts.ibz2bz.iter().enumerate() {
            let block = project(
                &mo_ibz_rm[i],
                &to_rm(&hcore_ibz[i]),
                nao,
                nmo,
                ra.clone(),
                rb.clone(),
            );
            arr.set_2d_at(bz, &block).expect("set_2d_at");
        }

        // Read back EVERY full-BZ k-point.
        let mut worst = 0.0_f64;
        let mut worst_at = (0usize, 0usize);
        for k in 0..nkpts {
            let got = arr.get_2d(k).expect("get_2d");
            assert_eq!(got.len(), di * dj);
            for p in 0..di * dj {
                let d = (got[p] - ref_bz[k][p]).norm();
                if d > worst {
                    worst = d;
                    worst_at = (k, p);
                }
            }
        }
        println!(
            "  label '{label_s}' trans 'cn': max |KsymmArray(get_2d) - \
             project(transform_mo_coeff)| = {worst:e}  (k = {}, elem {})",
            worst_at.0, worst_at.1
        );
        overall = overall.max(worst);
        assert!(
            worst < GATE_B_TOL,
            "label '{label_s}': the KsymmArray unfold of the ksymm SCF's own \
             IBZ store disagrees with the independent transform_mo_coeff route \
             by {worst:e} (> {GATE_B_TOL:e}) at k = {}, elem {}. Both start \
             from ONE SCF's orbitals, so this is not a gauge difference.",
            worst_at.0,
            worst_at.1
        );
    }
    println!("  worst over all three blocks: {overall:e}");
}
