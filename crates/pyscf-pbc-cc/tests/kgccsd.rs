//! Plan 16-07 Task 4 test 2 and plan 16-08 test 6 — the ORACLE-FREE gates that
//! only became possible once both drivers existed.
//!
//! `--release`: each converges an SCF.

mod common;

use std::sync::Arc;

use pyscf_pbc_cc::kccsd::Kgccsd;
use pyscf_pbc_cc::kccsd_rhf::Krccsd;
use pyscf_pbc_df::Fftdf;
use pyscf_pbc_lib::KptsHelper;
use pyscf_pbc_scf::{KScfConfig, Kghf, Krhf};
use pyscf_runtime::ZWorkspacePool;

/// **16-07 test 2 — `KGCCSD.e_corr == KRCCSD.e_corr` on a closed shell.**
///
/// The plan's main correctness gate for 16-07 and it needs no PySCF: a
/// spin-orbital CCSD on a closed-shell reference must reproduce the restricted
/// one. It is what would catch a spin-factor error in either specialisation,
/// which is the whole reason the two modules exist separately.
///
/// **G3 = `1e-8`, and that is measured, not chosen.** 16-01 ran upstream's own
/// `KGCCSD` and `KRCCSD` on this fixture and they differ by `4.95e-9`
/// (`measurements/README.md §5`); the plan's `1e-10` would fail a correct
/// implementation. Here BOTH sides are this port's, on their own mean fields.
#[test]
#[ignore = "converges two SCFs; run with --release"]
fn kgccsd_equals_krccsd_on_a_closed_shell() {
    let cell = common::diamond([15, 15, 15]);
    let kpts = cell.make_kpts([1, 1, 2]).expect("kpts");

    // --- restricted
    let mut rmf = Krhf::from_df(Box::new(Fftdf::new(cell.clone(), &kpts).expect("fftdf")));
    rmf.exxdiv = None;
    let mut cfg = KScfConfig::for_cell(&cell);
    cfg.conv_tol = 1e-10;
    let rscf = rmf.kernel(&cfg).expect("KRHF converges");
    assert!(rscf.converged);
    let rdf = Fftdf::new(cell.clone(), &kpts).expect("fftdf");
    let mut rcc = Krccsd::new(&rscf, &rdf).expect("KRCCSD builds");
    let reris = rcc.ao2mo().expect("_ERIS");
    let rres = rcc.kernel_with(&reris).expect("KRCCSD kernel");
    assert!(rres.converged, "KRCCSD must converge");

    // --- spin-orbital
    let mut gmf = Kghf::from_df(Box::new(Fftdf::new(cell.clone(), &kpts).expect("fftdf")));
    gmf.exxdiv = None;
    let gscf = gmf.kernel(&cfg).expect("KGHF converges");
    assert!(gscf.converged);
    println!(
        "mean fields: KRHF {} vs KGHF {}  |Δ| {:e}",
        rscf.e_tot,
        gscf.e_tot,
        (rscf.e_tot - gscf.e_tot).abs()
    );
    assert!(
        (rscf.e_tot - gscf.e_tot).abs() < 1e-8,
        "the two mean fields disagree, so the CC comparison would measure them"
    );

    let gcc = Kgccsd::new(&gscf, &mut gmf).expect("KGCCSD builds");
    let gdf = Fftdf::new(cell.clone(), &kpts).expect("fftdf");
    let khelper = KptsHelper::without_symm_map(&cell.a, &kpts);
    let geris = gcc.ao2mo(&gdf, &khelper).expect("spin-orbital _ERIS");
    let gres = gcc.kernel(&geris, &khelper.kconserv).expect("KGCCSD kernel");
    assert!(gres.converged, "KGCCSD must converge");

    let d = (gres.e_corr - rres.e_corr).abs();
    println!(
        "KGCCSD e_corr {} ({} cycles) vs KRCCSD {} ({} cycles)  |Δ| {d:e}",
        gres.e_corr, gres.cycles, rres.e_corr, rres.cycles
    );
    assert!(
        d < 1e-8,
        "KGCCSD and KRCCSD disagree by {d:e} on a closed shell, above G3 1e-8"
    );
}

/// **16-08 test 6 — the peak `t3`-class memory is bounded by ONE virtual
/// block, not by `nkpts³ · nvir³ · nocc³`.**
///
/// `16-REVIEW.md §4.2` rules that "(T) is a streaming problem and the blocking
/// IS the algorithm"; this makes the claim testable. The bound is a LITERAL,
/// derived in `16-08-SUMMARY.md`: the `w`/`v` cache is
/// `2 · nkpts³ · na·nb·nc · nocc³ · 16` bytes, so blocking the virtuals at
/// `blksize` reduces the peak by `(blksize/nvir)³` — and the energy does not
/// move.
#[test]
#[ignore = "converges an SCF; run with --release"]
fn ccsd_t_peak_memory_is_bounded_by_one_block() {
    let cell = common::diamond([15, 15, 15]);
    let kpts = cell.make_kpts([1, 1, 2]).expect("kpts");
    let mut mf = Krhf::from_df(Box::new(Fftdf::new(cell.clone(), &kpts).expect("fftdf")));
    mf.exxdiv = None;
    let mut cfg = KScfConfig::for_cell(&cell);
    cfg.conv_tol = 1e-10;
    let scf = mf.kernel(&cfg).expect("KRHF converges");
    let df = Fftdf::new(cell.clone(), &kpts).expect("fftdf");
    let mut cc = Krccsd::new(&scf, &df).expect("KRCCSD builds");
    let eris = cc.ao2mo().expect("_ERIS");
    let res = cc.kernel_with(&eris).expect("KRCCSD kernel");
    assert!(res.converged);

    let (nk, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let full = 2 * nk.pow(3) * nvir.pow(3) * nocc.pow(3) * 16;
    let mut last = 0.0_f64;
    for (blksize, expect_blocks) in [(None, 1usize), (Some(2), 2), (Some(1), 1)] {
        let n = blksize.unwrap_or(nvir);
        let (e, peak) = pyscf_pbc_cc::kccsd_t_rhf::kernel_with_stats(
            &eris,
            &cc.padded,
            &res.t1,
            &res.t2,
            &cc.khelper.kconserv,
            &cell.a,
            &kpts,
            blksize,
        )
        .expect("(T)");
        let want = 2 * nk.pow(3) * n.pow(3) * nocc.pow(3) * 16;
        println!(
            "(T) blksize {n}: e_t {e}  peak t3-cache {peak} bytes (derived {want}, \
             unblocked would be {full})"
        );
        assert_eq!(peak, want, "the peak cache is not one block's worth");
        assert!(
            peak <= full,
            "blocking must not increase the peak"
        );
        let _ = expect_blocks;
        if last != 0.0 {
            assert!(
                (e - last).abs() / e.abs() < 1e-12,
                "the (T) energy moved with the block size"
            );
        }
        last = e;
    }
    // The literal the plan asks for: at `blksize = 1` the peak is
    // `nvir³ = 64` times smaller than the unblocked one.
    let (_, peak1) = pyscf_pbc_cc::kccsd_t_rhf::kernel_with_stats(
        &eris,
        &cc.padded,
        &res.t1,
        &res.t2,
        &cc.khelper.kconserv,
        &cell.a,
        &kpts,
        Some(1),
    )
    .expect("(T)");
    assert_eq!(
        full / peak1,
        nvir.pow(3),
        "blocking at 1 must shrink the peak by exactly nvir^3"
    );
    let _ = Arc::new(ZWorkspacePool::new(0));
}

/// **§9.3 across THREAD COUNTS** — `t1`, `t2` and `e_corr` are bit-identical
/// when the same computation runs in a 1-thread and an 8-thread rayon pool.
///
/// 16-05 test 7 gated the in-process repeat; this gates the property the
/// determinism ruling is actually about. It holds by construction — every
/// `ZArr::einsum` output element is one `oracle_zsum` over a fixed-length
/// buffer whose recursion tree depends only on that length — and this proves
/// the construction was not broken somewhere.
#[test]
#[ignore = "converges an SCF; run with --release"]
fn amplitudes_are_bit_identical_across_thread_counts() {
    let cell = common::diamond([15, 15, 15]);
    let kpts = cell.make_kpts([1, 1, 2]).expect("kpts");
    let mut mf = Krhf::from_df(Box::new(Fftdf::new(cell.clone(), &kpts).expect("fftdf")));
    mf.exxdiv = None;
    let mut cfg = KScfConfig::for_cell(&cell);
    cfg.conv_tol = 1e-10;
    let scf = mf.kernel(&cfg).expect("KRHF converges");
    let df = Fftdf::new(cell.clone(), &kpts).expect("fftdf");
    let mut cc = Krccsd::new(&scf, &df).expect("KRCCSD builds");
    let eris = cc.ao2mo().expect("_ERIS");

    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("rayon pool")
            .install(|| cc.kernel_with(&eris).expect("KRCCSD kernel"))
    };
    let one = run(1);
    let eight = run(8);
    println!(
        "RAYON_NUM_THREADS 1: e_corr {} ({} cycles);  8: e_corr {} ({} cycles)",
        one.e_corr, one.cycles, eight.e_corr, eight.cycles
    );
    assert_eq!(one.e_corr, eight.e_corr, "e_corr is thread-count dependent");
    assert_eq!(one.t1, eight.t1, "t1 is thread-count dependent");
    assert_eq!(one.t2, eight.t2, "t2 is thread-count dependent");
    assert_eq!(one.cycles, eight.cycles);
}
