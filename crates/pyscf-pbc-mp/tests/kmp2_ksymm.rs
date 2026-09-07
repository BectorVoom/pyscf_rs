//! `KsymAdaptedKMP2` — plan 17-09, the MP2 half.
//!
//! # The gate is TIGHTER than the SCF's, and that is deliberate
//!
//! 17-CONTEXT §2.3: upstream gates `KMP2` ksymm-vs-full-BZ at **10 decimals**
//! (`mp/test/test_ksym.py:56`) and `rdm1` at 10 (`:66`), while gating the *SCF*
//! comparison at 7. The reason is structural — MP2 reads ONE converged SCF's
//! orbitals through two index paths, so there is no second convergence to add
//! noise. Relaxing this to the SCF's 5e-8 by analogy would let a real
//! index-map defect through.
//!
//! This file does better than that where it can, because it does not have to
//! compare two SCFs at all for the tests that matter most: the route
//! comparison, the thread comparison and `kernel_with_t2` all run against
//! **one** reference. Only `ksymm_e_corr_matches_full_bz` compares two
//! independently converged SCFs, and its residual is quoted with the SCF
//! residual beside it so the two are never confused.
//!
//! # What is compared, and what is deliberately not
//!
//! `mo_coeff` is never compared elementwise (17-CONTEXT §3.1) — the ksymm SCF
//! and the full-BZ SCF are free to differ by a unitary rotation inside every
//! degenerate subspace, and `si` has degeneracies everywhere. `e_corr` and
//! `rdm1` eigen-invariants are gauge-safe; the coefficients are not.

#![allow(clippy::needless_range_loop)]

use std::time::Instant;

use pyscf_pbc_df::{Fftdf, Gdf, PeriodicDf};
use pyscf_pbc_gto::test_systems::si;
use pyscf_pbc_gto::{Cell, make_kpts_default};
use pyscf_pbc_mp::{EriRoute, Kmp2, KsymAdaptedKmp2, PartialT2, RdmKind, unfold_kscf_result};
use pyscf_pbc_scf::{KInitGuess, KScfConfig, KScfResult, Krhf, KsymAdaptedKrhf};
use pyscf_pbc_symm::basis::{self, SymmAdaptedBasisInput};
use pyscf_pbc_symm::kpts::{KPoints, make_kpts};

/// D-17-07-01 (`17-07-SUMMARY.md`): `little_cogroup_ops` indexes `k2opk`'s
/// doubled column space while `symm_adapted_basis` indexes `ops`, so
/// `use_ao_symmetry = true` needs the space-group fold alone. Same setting as
/// `pyscf-pbc-scf/tests/khf_ksymm.rs`.
const TIME_REVERSAL: bool = false;

/// 17-01 Task 4 measured upstream's own ksymm-vs-full-BZ `e_corr` residual at
/// **1.067e-9** on si `[2,2,2]` with `density_fit`
/// (`measurements/gate_mp2.out`) — but that number mixes two effects, because
/// upstream's k-symmetric kernel uses `ao2mo` while its full-BZ kernel uses
/// the `Lov` route for a GDF reference (see `kmp2_ksymm`'s module doc).
///
/// With the route matched on both sides this port MEASURES **1.138e-10**
/// (FFTDF, si `[2,2,2]`) against a `|d E_scf|` of `8.793e-14` — an order
/// better than upstream on the same cell, and no longer SCF-bound: what is
/// left is the integral-transform floor of evaluating one s2 class instead of
/// its whole orbit.
///
/// **The gate is set at the measured value with one order of headroom, not at
/// upstream's `test_ksym.py` 10 decimals (5e-11).** That number is upstream's
/// gate for its HE cell, where the residual is `3.096e-16`; si is a
/// pseudopotential solid at a coarse mesh and 5e-11 is below its floor. 17-09
/// Task 3's ruling — "do not relax this to the SCF's 5e-8 by analogy" — is
/// respected: 5e-10 is **two orders tighter** than the SCF gate and it is
/// backed by a measurement rather than an analogy.
const E_CORR_TOL: f64 = 5e-10;

/// Two routes to the same number inside one process, against ONE reference —
/// no convergence noise at all. MEASURED at **3.278e-12** (si `[2,2,2]`, GDF)
/// between the s2-class kernel and the dense `nkpts^3` one; that residual is
/// pure re-association of the same terms (both take the `Lov` route on the
/// same `Lov` table), so it is `oracle_sum` grouping and nothing else.
const SAME_REFERENCE_TOL: f64 = 5e-11;

fn build(mesh: [usize; 3]) -> (Cell, KPoints) {
    let cell = si();
    let kpts_abs = make_kpts_default(&cell, mesh).expect("make_kpts_default");
    let kpts = make_kpts(&cell, &kpts_abs, true, TIME_REVERSAL).expect("make_kpts");
    let mut cell = cell;
    let input = SymmAdaptedBasisInput {
        kpts_scaled_ibz: kpts.kpts_scaled_ibz.clone(),
        little_cogroup_ops: kpts.little_cogroup_ops.clone(),
        ops: kpts.symmetry.ops.clone(),
        dmats: kpts.symmetry.dmats.clone(),
    };
    basis::build_symmetry(&mut cell, &input).expect("build_symmetry");
    (cell, kpts)
}

fn cfg() -> KScfConfig {
    KScfConfig {
        conv_tol: 1e-11,
        conv_tol_grad: Some(1e-9),
        max_cycle: 60,
        init_guess: KInitGuess::Minao,
        ..KScfConfig::default()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Route {
    Fft,
    Gdf,
}

fn df_for(route: Route, cell: &Cell, kpts: &[[f64; 3]]) -> Box<dyn PeriodicDf> {
    match route {
        Route::Fft => Box::new(Fftdf::new(cell.clone(), kpts).expect("Fftdf")),
        Route::Gdf => Box::new(Gdf::new(cell.clone(), kpts)),
    }
}

/// One k-symmetric SCF over the IBZ, plus the full-BZ MO set it unfolds to.
///
/// **The mean field's OWN density-fitting object is returned with it.** A GDF
/// `_cderi` build on si `[2,2,2]` costs ~90 s here, and building a second one
/// for the KMP2 would double every test's wall clock for no benefit — the two
/// would be bit-identical by construction.
fn ksymm_reference(route: Route, cell: &Cell, kpts: &KPoints) -> (KsymAdaptedKrhf, KScfResult) {
    ksymm_reference_with(route, cell, kpts, true)
}

/// As [`ksymm_reference`], with `use_ao_symmetry` explicit.
///
/// It has to be explicit, because **`true` is unsound on a k-mesh whose
/// symmetry is lower than the lattice's** — see
/// `use_ao_symmetry_is_unsound_when_the_kmesh_breaks_the_lattice_symmetry`,
/// which measures it.
fn ksymm_reference_with(
    route: Route,
    cell: &Cell,
    kpts: &KPoints,
    use_ao_symmetry: bool,
) -> (KsymAdaptedKrhf, KScfResult) {
    let mut mf = KsymAdaptedKrhf::from_df(df_for(route, cell, &kpts.kpts), kpts.clone());
    mf.use_ao_symmetry = use_ao_symmetry;
    // `exxdiv = None`, as 17-01's own `gate_mp2.py` fixture does — the Ewald
    // correction is a constant shift on both sides but it is one more thing
    // that could differ between two SCF runs.
    mf.exxdiv = None;
    let ibz = mf.kernel(&cfg()).expect("ksymm SCF");
    assert!(ibz.converged, "ksymm SCF did not converge");
    assert_eq!(ibz.nkpts, kpts.nkpts_ibz());
    let bz = unfold_kscf_result(&ibz, kpts, cell).expect("unfold");
    (mf, bz)
}

fn full_bz_reference(route: Route, cell: &Cell, kpts: &[[f64; 3]]) -> (Krhf, KScfResult) {
    let mut mf = Krhf::from_df(df_for(route, cell, kpts));
    mf.exxdiv = None;
    let r = mf.kernel(&cfg()).expect("full-BZ SCF");
    assert!(r.converged, "full-BZ SCF did not converge");
    (mf, r)
}

// =====================================================================

/// The unfold is the contract: an IBZ-length result must be rejected, and the
/// unfolded one must have full-BZ length with the IBZ representatives'
/// occupations and energies carried across unchanged.
#[test]
fn unfold_produces_a_full_bz_reference() {
    let (cell, kpts) = build([2, 2, 2]);
    let mut mf = KsymAdaptedKrhf::from_df(df_for(Route::Fft, &cell, &kpts.kpts), kpts.clone());
    mf.exxdiv = None;
    let ibz = mf.kernel(&cfg()).expect("ksymm SCF");
    let df = mf.with_df.as_ref();

    // The IBZ result is NOT what KsymAdaptedKmp2 takes.
    assert!(
        KsymAdaptedKmp2::new(&ibz, df, &kpts).is_err(),
        "an IBZ-length reference must be refused, not silently reinterpreted"
    );

    let bz = unfold_kscf_result(&ibz, &kpts, &cell).expect("unfold");
    assert_eq!(bz.nkpts, kpts.nkpts());
    assert_eq!(bz.mo_coeff.len(), kpts.nkpts());
    assert_eq!(bz.e_tot, ibz.e_tot, "the unfold must not touch the energy");
    for k in 0..kpts.nkpts() {
        let i = kpts.bz2ibz[k];
        assert_eq!(
            bz.mo_energy[k], ibz.mo_energy[i],
            "mo_energy is a pure index map (kpts.py:644-661)"
        );
        assert_eq!(bz.mo_occ[k], ibz.mo_occ[i]);
    }
    KsymAdaptedKmp2::new(&bz, df, &kpts).expect("full-BZ reference accepted");
}

/// **The gate.** `e_corr` from the k-symmetric kernel against `e_corr` from
/// the full-BZ kernel, per DF route, with the SCF residual printed beside it.
///
/// Two independently converged SCFs, so the residual has an SCF-noise floor;
/// that is why it is reported next to `|d E_scf|` rather than alone.
#[test]
fn ksymm_e_corr_matches_full_bz() {
    for route in [Route::Fft, Route::Gdf] {
        let (cell, kpts) = build([2, 2, 2]);
        let (bz_mf, bz_ref) = full_bz_reference(route, &cell, &kpts.kpts);
        let (sym_mf, sym_ref) = ksymm_reference(route, &cell, &kpts);

        let full = Kmp2::new(&bz_ref, bz_mf.with_df.as_ref())
            .expect("KMP2")
            .kernel()
            .expect("full-BZ KMP2");

        let mp =
            KsymAdaptedKmp2::new(&sym_ref, sym_mf.with_df.as_ref(), &kpts).expect("ksymm KMP2");
        let sym = mp.kernel(&cell).expect("ksymm KMP2 kernel");

        let de = (sym.e_corr - full.e_corr).abs();
        let dscf = (sym_ref.e_tot - bz_ref.e_tot).abs();
        println!("--- si [2,2,2] {route:?}");
        println!("e_corr full BZ = {:.15}", full.e_corr);
        println!("e_corr ksymm   = {:.15}", sym.e_corr);
        println!("|d e_corr| = {de:e}   |d E_scf| = {dscf:e}");
        assert!(
            de < E_CORR_TOL,
            "{route:?}: ksymm e_corr differs from the full-BZ one by {de:e} \
             (> {E_CORR_TOL:e}); the SCF references differ by {dscf:e}"
        );
        // The spin components must fold the same way as the total.
        assert!((sym.e_corr_ss + sym.e_corr_os - sym.e_corr).abs() < 1e-15);
        assert!((sym.e_corr_os - full.e_corr_os).abs() < E_CORR_TOL);
    }
}

/// The two ERI routes, against ONE reference. `EriRoute::Ao2mo` is upstream's
/// literal choice (`kmp2_ksymm.py:46`); `EriRoute::MatchFullBz` takes the
/// `Lov` route a GDF reference makes available. They compute the same
/// integrals two different ways, so this is the strongest oracle-free check in
/// the file — and it is the one that isolates the `1.067e-9` upstream reports
/// on this fixture into "route" and "symmetry" halves.
#[test]
fn the_two_eri_routes_agree_on_one_reference() {
    let (cell, kpts) = build([2, 2, 2]);
    let (sym_mf, sym_ref) = ksymm_reference(Route::Gdf, &cell, &kpts);
    let df = sym_mf.with_df.as_ref();
    assert!(df.has_cderi(), "the GDF fixture must actually carry _cderi");

    let mut mp = KsymAdaptedKmp2::new(&sym_ref, df, &kpts).expect("ksymm KMP2");
    mp.route = EriRoute::MatchFullBz;
    let lov = mp.kernel(&cell).expect("Lov route");
    mp.route = EriRoute::Ao2mo;
    let ao2mo = mp.kernel(&cell).expect("ao2mo route");

    let d = (lov.e_corr - ao2mo.e_corr).abs();
    println!("e_corr Lov   = {:.15}", lov.e_corr);
    println!("e_corr ao2mo = {:.15}", ao2mo.e_corr);
    println!("|d| = {d:e}");
    assert!(
        d < 1e-9,
        "the Lov and ao2mo routes disagree by {d:e} on one reference"
    );
}

/// `kernel_with_t2` (`kmp2_ksymm.py:120-126`) runs the PLAIN full-BZ kernel by
/// upstream's own design. On one reference it must give the same energy as the
/// symmetry-reduced kernel — that is the whole claim the symmetry makes.
#[test]
fn kernel_with_t2_matches_the_symmetry_reduced_kernel() {
    let (cell, kpts) = build([2, 2, 2]);
    let (sym_mf, sym_ref) = ksymm_reference(Route::Gdf, &cell, &kpts);
    let mp = KsymAdaptedKmp2::new(&sym_ref, sym_mf.with_df.as_ref(), &kpts).expect("ksymm KMP2");

    let reduced = mp.kernel(&cell).expect("ksymm kernel");
    let dense = mp.kernel_with_t2().expect("kernel_with_t2");
    let d = (reduced.e_corr - dense.e_corr).abs();
    println!("e_corr reduced = {:.15}", reduced.e_corr);
    println!("e_corr dense   = {:.15}", dense.e_corr);
    println!("|d| = {d:e}");
    assert!(
        d < SAME_REFERENCE_TOL,
        "the symmetry-reduced kernel and the dense one disagree by {d:e} on ONE reference; \
         with no second convergence in play this is algebra, not tolerance"
    );
    assert!(dense.t2.is_some(), "kernel_with_t2 must return amplitudes");
}

/// `Tr(gamma)` and Hermiticity — the two oracle-free `rdm1` gates
/// 15-CONTEXT set for the non-symmetric KMP2, plus the identity that actually
/// holds at k-resolution.
///
/// **`Tr(gamma_k)` is NOT `nelec` at each k-point**, and upstream does not
/// satisfy that either (`tests/kmp2.rs` measured upstream missing it by 2.8e-2
/// on the very first k-point of its own anchor): the MP2 correction moves
/// charge BETWEEN k-points. What holds is the WEIGHTED average over the zone —
/// and under symmetry the weights are `weights_ibz`, not `1/nkpts_ibz`.
#[test]
fn rdm1_trace_and_hermiticity() {
    let (cell, kpts) = build([2, 2, 2]);
    let (sym_mf, sym_ref) = ksymm_reference(Route::Gdf, &cell, &kpts);
    let mp = KsymAdaptedKmp2::new(&sym_ref, sym_mf.with_df.as_ref(), &kpts).expect("ksymm KMP2");

    let t2 = mp.make_t2_for_rdm1(&cell).expect("make_t2_for_rdm1");
    println!(
        "make_t2_for_rdm1 built {} of {} blocks",
        t2.filled(),
        kpts.nkpts().pow(3)
    );
    assert!(
        t2.filled() < kpts.nkpts().pow(3),
        "make_t2_for_rdm1 exists to build FEWER blocks than the dense t2"
    );

    let dm = mp.make_rdm1(&t2, RdmKind::Padded).expect("rdm1");
    assert_eq!(dm.len(), kpts.nkpts_ibz());

    let mut weighted = 0.0;
    for (i, d) in dm.iter().enumerate() {
        let nmo = (d.re.len() as f64).sqrt() as usize;
        let trace: f64 = (0..nmo).map(|p| d.re[p * nmo + p]).sum();
        println!(
            "Tr(gamma_ibz{i}) = {trace:.12}  weight = {}",
            kpts.weights_ibz[i]
        );
        weighted += kpts.weights_ibz[i] * trace;
        for p in 0..nmo {
            for q in 0..nmo {
                let pq = p * nmo + q;
                let qp = q * nmo + p;
                assert!(
                    (d.re[pq] - d.re[qp]).abs() < 2e-12 && (d.im[pq] + d.im[qp]).abs() < 2e-12,
                    "rdm1 at IBZ point {i} is not Hermitian at ({p},{q})"
                );
            }
        }
    }
    let nelec = cell.tot_electrons(1) as f64;
    println!("weighted Tr = {weighted:.12}, nelec = {nelec}");
    assert!(
        (weighted - nelec).abs() < 1e-8,
        "the weights_ibz-averaged rdm1 trace must be the electron count: \
         {weighted} vs {nelec}"
    );
}

/// The k-symmetric `rdm1` at each IBZ point against the full-BZ `rdm1` at the
/// corresponding BZ point — `measurements/gate_mp2.py`'s own comparison, which
/// upstream measures at 5.028e-9 on si `[2,2,2]` (route-mixed) and 1.332e-15
/// on He.
///
/// Both sides here come from ONE reference: the same unfolded full-BZ MO set
/// drives the symmetric `make_t2_for_rdm1`/`_gamma1_intermediates` and the
/// dense `Kmp2::make_rdm1`. So this compares the two INDEX PATHS, which is
/// what the plan asks for, with no convergence noise in it at all — and it is
/// the test that catches the `t2[kj,ki,kb].transpose(1,0,3,2)` reconstruction
/// being wrong, which no trace or Hermiticity check can see.
#[test]
fn ksymm_rdm1_matches_the_dense_rdm1_on_one_reference() {
    let (cell, kpts) = build([2, 2, 2]);
    let (sym_mf, sym_ref) = ksymm_reference(Route::Gdf, &cell, &kpts);
    let mp = KsymAdaptedKmp2::new(&sym_ref, sym_mf.with_df.as_ref(), &kpts).expect("ksymm KMP2");

    let dense = mp.kernel_with_t2().expect("kernel_with_t2");
    let t2_dense = dense.t2.as_ref().expect("t2");
    let ref_dm = mp
        .mp
        .make_rdm1(t2_dense, RdmKind::Padded)
        .expect("dense rdm1");

    let t2_sym = mp.make_t2_for_rdm1(&cell).expect("make_t2_for_rdm1");
    let sym_dm = mp.make_rdm1(&t2_sym, RdmKind::Padded).expect("ksymm rdm1");

    let mut worst = 0.0f64;
    for (i, &k) in kpts.ibz2bz.iter().enumerate() {
        for p in 0..sym_dm[i].re.len() {
            worst = worst
                .max((sym_dm[i].re[p] - ref_dm[k].re[p]).abs())
                .max((sym_dm[i].im[p] - ref_dm[k].im[p]).abs());
        }
    }
    println!("rdm1 max residual (ibz vs corresponding full-BZ) = {worst:e}");
    assert!(
        worst < 1e-11,
        "the k-symmetric rdm1 differs from the dense one by {worst:e} on ONE reference; \
         the prime suspect is the t2[kj,ki,kb].transpose(1,0,3,2) reconstruction \
         (kmp2_ksymm.py:214-216)"
    );

    // The dense amplitudes, promoted, must reproduce the same IBZ blocks —
    // this isolates `make_t2_for_rdm1`'s block SELECTION from
    // `_gamma1_intermediates`' index algebra.
    let promoted = PartialT2::from_full(t2_dense);
    let via_dense = mp.make_rdm1(&promoted, RdmKind::Padded).expect("rdm1");
    for i in 0..sym_dm.len() {
        for p in 0..sym_dm[i].re.len() {
            assert!(
                (sym_dm[i].re[p] - via_dense[i].re[p]).abs() < 1e-11
                    && (sym_dm[i].im[p] - via_dense[i].im[p]).abs() < 1e-11,
                "make_t2_for_rdm1's block selection changed the density at IBZ point {i}"
            );
        }
    }
}

/// §9.3: `e_corr` bit-identical at `RAYON_NUM_THREADS` 1 and 8, inside ONE
/// process with explicit thread pools — strictly stronger than an env-var
/// sweep across processes, because both runs share every cached input.
#[test]
fn e_corr_is_bit_identical_across_thread_counts() {
    let (cell, kpts) = build([2, 2, 2]);
    let (sym_mf, sym_ref) = ksymm_reference(Route::Gdf, &cell, &kpts);
    let mp = KsymAdaptedKmp2::new(&sym_ref, sym_mf.with_df.as_ref(), &kpts).expect("ksymm KMP2");

    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool")
            .install(|| mp.kernel(&cell).expect("kernel"))
    };
    let one = run(1);
    let eight = run(8);
    println!("e_corr @1 = {:.17}", one.e_corr);
    println!("e_corr @8 = {:.17}", eight.e_corr);
    assert_eq!(
        one.e_corr.to_bits(),
        eight.e_corr.to_bits(),
        "e_corr must be BIT-identical at 1 and 8 threads"
    );
    assert_eq!(one.e_corr_ss.to_bits(), eight.e_corr_ss.to_bits());
    assert_eq!(one.e_corr_os.to_bits(), eight.e_corr_os.to_bits());
}

/// Cost, REPORTED not gated (17-09 Task 3's last bullet). The class count is
/// the exact, deterministic part of the saving and IS asserted; wall time is
/// printed.
#[test]
fn cost_is_reported() {
    let (cell, kpts) = build([2, 2, 2]);
    let n3 = kpts.nkpts().pow(3);
    let s1 = kpts.make_k4_ibz(&cell, "s1").expect("s1").k4.len();
    let s2 = kpts.make_k4_ibz(&cell, "s2").expect("s2").k4.len();
    println!("si [2,2,2]: {n3} k-triples, {s1} s1 classes, {s2} s2 classes");
    assert!(s2 < s1 && s1 < n3);

    let (sym_mf, sym_ref) = ksymm_reference(Route::Gdf, &cell, &kpts);
    let mp = KsymAdaptedKmp2::new(&sym_ref, sym_mf.with_df.as_ref(), &kpts).expect("ksymm KMP2");

    let t0 = Instant::now();
    let reduced = mp.kernel(&cell).expect("ksymm kernel");
    let t_sym = t0.elapsed();
    let t0 = Instant::now();
    let dense = mp.kernel_with_t2().expect("dense kernel");
    let t_dense = t0.elapsed();
    println!(
        "wall: ksymm {:?}, dense {:?}, ratio {:.2}x",
        t_sym,
        t_dense,
        t_dense.as_secs_f64() / t_sym.as_secs_f64().max(f64::MIN_POSITIVE)
    );
    println!("e_corr {:.12} / {:.12}", reduced.e_corr, dense.e_corr);
}

/// `KsymAdaptedKMP2.make_rdm2` is `raise NotImplementedError` upstream
/// (`kmp2_ksymm.py:253-254`). The port ships the refusal, and this test is
/// what stops a later plan from filling it in with a number no oracle checks.
#[test]
fn make_rdm2_refuses() {
    let (cell, kpts) = build([1, 1, 2]);
    let (sym_mf, sym_ref) = ksymm_reference(Route::Gdf, &cell, &kpts);
    let mp = KsymAdaptedKmp2::new(&sym_ref, sym_mf.with_df.as_ref(), &kpts).expect("ksymm KMP2");
    let err = mp.make_rdm2().expect_err("make_rdm2 must refuse");
    let msg = err.to_string();
    assert!(msg.contains("kmp2_ksymm.py:253-254"), "{msg}");
}

/// A second cell shape, cheap: `[1,1,2]` folds nothing in k but still exercises
/// the dummy-index fold (8 triples -> 6 classes, `measurements/gate_k4_s2.out`).
///
/// **`use_ao_symmetry = false` here, and that is a FINDING, not a
/// convenience** — D-17-09-02, measured by the test below.
#[test]
fn si_112_ksymm_matches_full_bz() {
    let (cell, kpts) = build([1, 1, 2]);
    let (bz_mf, bz_ref) = full_bz_reference(Route::Gdf, &cell, &kpts.kpts);
    // `use_ao_symmetry = false` — see the doc comment above and D-17-09-02.
    let (sym_mf, sym_ref) = ksymm_reference_with(Route::Gdf, &cell, &kpts, false);
    let full = Kmp2::new(&bz_ref, bz_mf.with_df.as_ref())
        .expect("KMP2")
        .kernel()
        .expect("full-BZ KMP2");
    let mp = KsymAdaptedKmp2::new(&sym_ref, sym_mf.with_df.as_ref(), &kpts).expect("ksymm KMP2");
    let sym = mp.kernel(&cell).expect("ksymm KMP2 kernel");
    let de = (sym.e_corr - full.e_corr).abs();
    let dscf = (sym_ref.e_tot - bz_ref.e_tot).abs();
    println!(
        "si [1,1,2] nkpts {} nkpts_ibz {}",
        kpts.nkpts(),
        kpts.nkpts_ibz()
    );
    println!(
        "si [1,1,2] E_scf full {:.15} ksymm {:.15} |d| = {dscf:e}",
        bz_ref.e_tot, sym_ref.e_tot
    );
    for k in 0..kpts.nkpts() {
        let d: f64 = sym_ref.mo_energy[k]
            .iter()
            .zip(bz_ref.mo_energy[k].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        println!("  k={k}: max |d mo_energy| = {d:e}");
    }
    println!(
        "si [1,1,2] e_corr full {:.15} ksymm {:.15} |d| = {de:e}",
        full.e_corr, sym.e_corr
    );
    // **`|d E_scf|` is checked FIRST and on purpose.** A `[1,1,2]` mesh keeps
    // no point-group operation that moves a k-point, so the s2 fold here is the
    // dummy-index interchange ALONE (8 triples -> 6 classes) and that
    // interchange is an EXACT symmetry of `(ia|jb)`. If `e_corr` moves, the
    // mean field moved -- D-17-08-03's shape -- and the number to look at is
    // the one above.
    assert!(
        dscf < 1e-9,
        "the two SCFs disagree by {dscf:e} before any correlation is computed; \
         the k-symmetric and full-BZ mean fields are not the same solution, so \
         the e_corr comparison would measure them, not the symmetry"
    );
    assert!(de < E_CORR_TOL, "|d e_corr| = {de:e}");
}

/// **D-17-09-02 — on a k-mesh with lower symmetry than the lattice,
/// `use_ao_symmetry = true` produces orbitals that are NOT Fock eigenvectors,
/// and `E_scf` cannot see it.**
///
/// `make_kpts_ibz` snapshots `k2opk` BEFORE wiping the columns of operations
/// that move a k-point out of the mesh (`kpts.py:60-64`), and
/// `little_cogroup_ops` is filled from that UNWIPED table (`:109-113`). So on
/// `si [1,1,2]` — where the point group maps `b3/2` onto `b1/2` and `b2/2`,
/// neither of which is in the mesh — **36 of 48 operations are outside the
/// k-mesh subgroup and every one of them appears in `little_cogroup_ops`**
/// (`crates/pyscf-pbc-symm/tests/kpts_k4_s2.rs`, no SCF needed).
///
/// `symm_adapted_basis` builds `symm_orb` from that group and
/// `eig_symm_adapted` then solves `F c = S c e` **one irrep block at a time**.
/// Schur's lemma makes that exact only if `F` has no matrix elements between
/// the blocks — and here it does: `v_J` comes from `rho(r) = Σ_k rho_k(r)` over
/// a k-mesh that is not point-group invariant, so neither `rho` nor `v_J` is
/// invariant under the operations `symm_orb` was built from.
///
/// # The consequence, and why nothing before this test could see it
///
/// A per-block solve of a Fock matrix that is not block-diagonal still spans
/// the right OCCUPIED SUBSPACE — so the density, and therefore `E_scf`, are
/// unaffected. **Measured: `|d E_scf|` is `5.329e-15`, i.e. the two routes
/// agree.** 17-07's `ao_symmetry_eig_matches_the_plain_route` compares exactly
/// that quantity, which is why this went unnoticed.
///
/// What the per-block solve does NOT give is CANONICAL orbitals: the returned
/// vectors are eigenvectors of the projected Fock, not of `F`. Every post-SCF
/// method whose denominators assume `F` is diagonal in the MO basis — MP2,
/// CCSD — is then wrong. **Measured: `KMP2`'s `e_corr` differs by `3.629e-04`,
/// against `0e0` with `use_ao_symmetry = false`.**
///
/// So the assertions below are, deliberately, in this order: `E_scf` AGREES,
/// the orbital energies DIFFER, and `e_corr` differs. An assertion that only
/// looked at `E_scf` would report all-clear.
///
/// `si [2,2,2]`, whose mesh IS closed under the cubic group (0 of 48
/// operations outside the subgroup), is unaffected —
/// `ksymm_e_corr_matches_full_bz` lands at 1.138e-10 there WITH
/// `use_ao_symmetry = true`.
#[test]
fn use_ao_symmetry_is_unsound_when_the_kmesh_breaks_the_lattice_symmetry() {
    let (cell, kpts) = build([1, 1, 2]);
    let offenders = kpts.little_cogroup_ops_outside_kmesh_subgroup();
    assert!(
        !offenders.is_empty(),
        "the fixture must be a symmetry-BREAKING mesh for this test to mean anything"
    );
    println!(
        "si [1,1,2]: {} little-co-group operations outside the k-mesh subgroup",
        offenders.len()
    );

    let (bz_mf, bz_ref) = full_bz_reference(Route::Gdf, &cell, &kpts.kpts);
    let (off_mf, off_ref) = ksymm_reference_with(Route::Gdf, &cell, &kpts, false);
    let (on_mf, on_ref) = ksymm_reference_with(Route::Gdf, &cell, &kpts, true);

    let d_scf_off = (off_ref.e_tot - bz_ref.e_tot).abs();
    let d_scf_on = (on_ref.e_tot - bz_ref.e_tot).abs();
    println!(
        "E_scf full BZ                        = {:.15}",
        bz_ref.e_tot
    );
    println!(
        "E_scf ksymm, use_ao_symmetry = false = {:.15}  |d| = {d_scf_off:e}",
        off_ref.e_tot
    );
    println!(
        "E_scf ksymm, use_ao_symmetry = true  = {:.15}  |d| = {d_scf_on:e}",
        on_ref.e_tot
    );

    // The orbital energies, which is where the two routes actually part.
    let mut d_mo_off = 0.0f64;
    let mut d_mo_on = 0.0f64;
    for k in 0..kpts.nkpts() {
        for p in 0..bz_ref.mo_energy[k].len() {
            d_mo_off = d_mo_off.max((off_ref.mo_energy[k][p] - bz_ref.mo_energy[k][p]).abs());
            d_mo_on = d_mo_on.max((on_ref.mo_energy[k][p] - bz_ref.mo_energy[k][p]).abs());
        }
    }
    println!("max |d mo_energy| vs full BZ: false = {d_mo_off:e}, true = {d_mo_on:e}");

    let full = Kmp2::new(&bz_ref, bz_mf.with_df.as_ref())
        .expect("KMP2")
        .kernel()
        .expect("full-BZ KMP2");
    let e_off = KsymAdaptedKmp2::new(&off_ref, off_mf.with_df.as_ref(), &kpts)
        .expect("ksymm KMP2")
        .kernel(&cell)
        .expect("kernel")
        .e_corr;
    let e_on = KsymAdaptedKmp2::new(&on_ref, on_mf.with_df.as_ref(), &kpts)
        .expect("ksymm KMP2")
        .kernel(&cell)
        .expect("kernel")
        .e_corr;
    let d_corr_off = (e_off - full.e_corr).abs();
    let d_corr_on = (e_on - full.e_corr).abs();
    println!(
        "e_corr full BZ                        = {:.15}",
        full.e_corr
    );
    println!("e_corr ksymm, use_ao_symmetry = false = {e_off:.15}  |d| = {d_corr_off:e}");
    println!("e_corr ksymm, use_ao_symmetry = true  = {e_on:.15}  |d| = {d_corr_on:e}");

    // 1. With the constraint OFF, everything reproduces the full-BZ run.
    assert!(d_scf_off < 1e-12, "|d E_scf| (off) = {d_scf_off:e}");
    assert!(d_mo_off < 1e-12, "max |d mo_energy| (off) = {d_mo_off:e}");
    assert!(d_corr_off < 1e-12, "|d e_corr| (off) = {d_corr_off:e}");

    // 2. With it ON, `E_scf` STILL agrees — the occupied subspace is right, so
    //    the density and the total energy are blind to the defect. This is the
    //    assertion that says why the phase's existing SCF-level gates could not
    //    have caught it.
    assert!(
        d_scf_on < 1e-9,
        "|d E_scf| (on) = {d_scf_on:e} — if this ever grows, the defect has \
         changed character: the constrained solve would then be finding a \
         different variational minimum, not merely a non-canonical basis of the \
         same one"
    );

    // 3. And the orbitals are NOT canonical, which `e_corr` does see.
    assert!(
        d_mo_on > 1e-6,
        "max |d mo_energy| (on) = {d_mo_on:e} — the per-irrep solve is supposed \
         to return non-Fock-eigenvectors on this mesh; if it no longer does, \
         `little_cogroup_ops` was fixed and this test should be deleted"
    );
    assert!(
        d_corr_on > 1e-6,
        "|d e_corr| (on) = {d_corr_on:e} — same reasoning as the assertion above"
    );
}
