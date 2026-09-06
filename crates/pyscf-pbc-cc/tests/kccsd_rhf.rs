//! Plan 16-05 Task 4 — the ORACLE-FREE `KRCCSD` gates.
//!
//! These are the tests that survive an oracle drift, and `16-14` Task 2 collects
//! them as "the phase's real proof". They consume no Python.
//!
//! `--release`: each converges an SCF.

mod common;

use std::sync::Arc;

use pyscf_pbc_cc::kccsd_rhf::{Krccsd, init_amps};
use pyscf_pbc_cc::keris::{Blk, ErisMethod, KEris, KErisOpts};
use pyscf_pbc_cc::{Tier, ZArr};
use pyscf_pbc_df::Fftdf;
use pyscf_pbc_scf::{KScfConfig, Krhf};
use pyscf_runtime::ZWorkspacePool;

struct Fixture {
    scf: pyscf_pbc_scf::KScfResult,
    df: Fftdf,
}

fn diamond_scf(nk: [usize; 3], mesh: [usize; 3]) -> Fixture {
    let cell = common::diamond(mesh);
    let kpts = cell.make_kpts(nk).expect("kpts");
    let df = Fftdf::new(cell.clone(), &kpts).expect("fftdf");
    let mut mf = Krhf::from_df(Box::new(Fftdf::new(cell.clone(), &kpts).expect("fftdf")));
    mf.exxdiv = None;
    let mut cfg = KScfConfig::for_cell(&cell);
    cfg.conv_tol = 1e-10;
    let scf = mf.kernel(&cfg).expect("KRHF converges");
    assert!(scf.converged);
    Fixture { scf, df }
}

fn all_blocks(eris: &KEris) -> Vec<(Blk, ZArr)> {
    let nk = eris.nkpts;
    Blk::ALL
        .iter()
        .map(|&b| {
            let dims = b.dims(eris.nocc, eris.nvir);
            let mut shape = vec![nk, nk, nk];
            shape.extend_from_slice(&dims);
            let mut got = ZArr::zeros(&shape);
            for k0 in 0..nk {
                for k1 in 0..nk {
                    for k2 in 0..nk {
                        got.set_leading(&[k0, k1, k2], &eris.blk(b, k0, k1, k2).expect("block"))
                            .expect("shape");
                    }
                }
            }
            (b, got)
        })
        .collect()
}

fn maxdiff(a: &ZArr, b: &ZArr) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..a.len() {
        m = m
            .max((a.data().re[i] - b.data().re[i]).abs())
            .max((a.data().im[i] - b.data().im[i]).abs());
    }
    m
}

/// **Test 4 — the tier crossing.** The same fixture built incore and spilled
/// gives BIT-IDENTICAL blocks, and the test asserts WHICH TIER each side used.
///
/// `16-REVIEW.md §2.3`: every `§9.2` fixture is `gth-szv`, where `vvvv` at
/// 2×2×2 is 2 MiB — a phase gated only on those would ship the HDF5 spill path
/// never once executed, which is 17-12's exit-137 shape. A test that silently
/// stayed incore must FAIL, not pass, so the tier is asserted rather than
/// hoped for.
///
/// Upstream gates its own incore-vs-outcore analogue at 12 decimals
/// (`test_krccsd.py:250-256`); this one is same-process, same code, same
/// inputs, so it is a **bit-identity** (gate G8).
#[test]
#[ignore = "converges an SCF; run with --release"]
fn eris_incore_and_spilled_are_bit_identical() {
    let f = diamond_scf([1, 1, 2], [15, 15, 15]);
    let mut cc = Krccsd::new(&f.scf, &f.df).expect("KRCCSD builds");

    cc.eris_opts = KErisOpts {
        method: ErisMethod::Incore,
        ..Default::default()
    };
    let incore = cc.ao2mo().expect("incore _ERIS");
    for b in Blk::ALL {
        assert_eq!(
            incore.tier(b).expect("tier"),
            Tier::InMemory,
            "{} must be in memory at the default budget",
            b.name()
        );
    }

    // Force the spill tier: a budget below one block's exact byte count.
    cc.eris_opts = KErisOpts {
        method: ErisMethod::Outcore,
        max_memory: 1e-6, // bytes, i.e. essentially zero — everything spills
        ..Default::default()
    };
    let spilled = cc.ao2mo().expect("spilled _ERIS");
    for b in Blk::ALL {
        assert_eq!(
            spilled.tier(b).expect("tier"),
            Tier::Spilled,
            "{} must have SPILLED — a fixture that silently stayed incore is \
             the defect 16-REVIEW §2.3 names",
            b.name()
        );
    }

    for ((ba, a), (bb, b)) in all_blocks(&incore).iter().zip(all_blocks(&spilled).iter()) {
        assert_eq!(ba, bb);
        assert_eq!(
            a, b,
            "{} differs between the incore and spilled tiers",
            ba.name()
        );
    }
    println!(
        "tier crossing OK: 7 blocks, {} bytes each way",
        incore.total_bytes()
    );
}

/// **Test 5 — the `symm_map` loop against the all-triples loop.**
///
/// `16-05-PLAN.md` asks for BIT-IDENTITY here. **16-01 measured that upstream's
/// own two paths differ by up to `1.32e-7`** on this fixture
/// (`measurements/README.md §7`), because a symmetry-related k-quadruple's FFT
/// transform and its transposed sibling are not the same floating-point
/// computation. The gate is therefore `1e-6` (G10), and the plan's requirement
/// is corrected by the measurement rather than by negotiation.
///
/// `vvvv` is the control: it is built by `ao2mo_7d` in BOTH paths, so it must
/// be bit-identical, and upstream's own measurement agrees (`2.08e-17`).
#[test]
#[ignore = "converges an SCF; run with --release"]
fn symm_map_loop_matches_the_all_triples_loop() {
    let f = diamond_scf([1, 1, 2], [15, 15, 15]);
    let mut cc = Krccsd::new(&f.scf, &f.df).expect("KRCCSD builds");

    cc.eris_opts = KErisOpts::default();
    let with_symm = cc.ao2mo().expect("symmetry loop");
    cc.eris_opts = KErisOpts {
        use_symm_map: false,
        ..Default::default()
    };
    let all_triples = cc.ao2mo().expect("all-triples loop");

    let mut worst = 0.0_f64;
    for ((b, a), (_, c)) in all_blocks(&with_symm)
        .iter()
        .zip(all_blocks(&all_triples).iter())
    {
        let d = maxdiff(a, c);
        println!("max|{}_symm - {}_all| = {d:e}", b.name(), b.name());
        if *b == Blk::Vvvv {
            assert_eq!(
                a, c,
                "vvvv is built by ao2mo_7d in BOTH paths and must be bit-identical"
            );
        }
        worst = worst.max(d);
    }
    assert!(
        worst < 1e-6,
        "the symmetry loop and the all-triples loop differ by {worst:e}, above \
         the measured 1e-6 gate (upstream's own two paths differ by 1.32e-7)"
    );
}

/// **Test 7 — determinism (§9.3).** Bit-identical `t1`, `t2` AND `e_corr`.
///
/// Gating `e_corr` alone would pass a non-deterministic `t2` that happens to
/// converge to the same energy, and 16-09/10/11 read `t2` directly
/// (`16-CONTEXT §3.7`). Run the whole file under `RAYON_NUM_THREADS=1` and
/// again under `RAYON_NUM_THREADS=8` for the cross-thread half; this test pins
/// the in-process half, which is what a regression in the accumulation order
/// would break first.
#[test]
#[ignore = "converges an SCF; run with --release"]
fn amplitudes_and_energy_are_bit_reproducible() {
    let f = diamond_scf([1, 1, 2], [15, 15, 15]);
    let mut cc = Krccsd::new(&f.scf, &f.df).expect("KRCCSD builds");
    let eris = cc.ao2mo().expect("_ERIS");
    let first = cc.kernel_with(&eris).expect("run 1");
    let second = cc.kernel_with(&eris).expect("run 2");
    assert_eq!(first.e_corr, second.e_corr, "e_corr is not reproducible");
    assert_eq!(first.t1, second.t1, "t1 is not reproducible");
    assert_eq!(first.t2, second.t2, "t2 is not reproducible");
    assert_eq!(first.cycles, second.cycles);
    println!(
        "determinism OK: e_corr {} bit-identical over two runs ({} cycles)",
        first.e_corr, first.cycles
    );
}

/// **Test 3 — the MP2 cross-phase check.** `init_amps`' `emp2` equals Phase
/// 15's `KMP2` `e_corr` on the same fixture.
///
/// **With `keep_exxdiv = true`**, and that is not a workaround. Upstream's own
/// log line calls it "MP2 energy (**with fock eigenvalue shift**)"
/// (`kccsd_rhf.py:594`): with the default `keep_exxdiv = false` the CC orbital
/// energies carry the Madelung correction that `_adjust_occ` re-adds
/// (`16-CONTEXT §3.5`), while `KMP2` uses the mean field's own eigenvalues. The
/// two quantities are then DIFFERENT BY CONSTRUCTION, and a test that compared
/// them anyway would be asserting the Madelung shift is zero. Suppressing the
/// re-add on the CC side is what makes them the same quantity.
#[test]
#[ignore = "converges an SCF; run with --release"]
fn init_amps_emp2_equals_kmp2() {
    let f = diamond_scf([1, 1, 2], [15, 15, 15]);
    let mut cc = Krccsd::new(&f.scf, &f.df).expect("KRCCSD builds");
    cc.eris_opts = KErisOpts {
        keep_exxdiv: true,
        exxdiv: None,
        ..Default::default()
    };
    let eris = cc.ao2mo().expect("_ERIS");
    let (emp2, _, _) = init_amps(&eris, &cc.padded, &cc.khelper.kconserv).expect("init_amps");

    let df2 = Fftdf::new(
        pyscf_pbc_df::PeriodicDf::cell(&f.df).clone(),
        pyscf_pbc_df::PeriodicDf::kpts(&f.df),
    )
    .expect("fftdf");
    let mp2 = pyscf_pbc_mp::Kmp2::new(&f.scf, &df2).expect("KMP2 builds");
    let res = mp2.kernel().expect("KMP2 kernel");

    let d = (emp2 - res.e_corr).abs();
    println!("init_amps emp2 {emp2} vs KMP2 e_corr {}  |Δ| {d:e}", res.e_corr);
    // MEASURED at `2.17e-10`. The two are the same quantity but not the same
    // code path — the CC side takes its orbital energies from a REBUILT Fock
    // matrix's diagonal (`kccsd_rhf.py:750-754`) while `KMP2` uses the SCF's
    // own `mo_energy` — so they agree to the SCF's own `conv_tol = 1e-10`, not
    // below it. Gating at 1e-10 would be gating below the convergence
    // threshold both sides were run at.
    assert!(
        d < 1e-9,
        "the CC initial amplitudes and Phase 15's KMP2 disagree by {d:e}"
    );
}

/// The complex arena's peak in-memory bytes stay at the exact per-tensor count
/// — D-PBC-29 clause 1's accounting, measured on a real `_ERIS`.
#[test]
#[ignore = "converges an SCF; run with --release"]
fn eris_charges_exactly_what_it_allocates() {
    let f = diamond_scf([1, 1, 2], [15, 15, 15]);
    let mut cc = Krccsd::new(&f.scf, &f.df).expect("KRCCSD builds");
    let eris = cc.ao2mo().expect("_ERIS");
    let (nk, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let derived: usize = Blk::ALL
        .iter()
        .map(|b| {
            let d = b.dims(nocc, nvir);
            nk.pow(3) * d.iter().product::<usize>() * 16
        })
        .sum();
    assert_eq!(
        eris.total_bytes(),
        derived,
        "the arena's byte count and the derived one disagree"
    );
    assert_eq!(
        eris.live_inmem_bytes(),
        derived,
        "the arena charges a different number than it allocated"
    );
    println!("7 blocks: {derived} bytes, exactly");
    let _ = Arc::new(ZWorkspacePool::new(0));
}

/// **G9 / 16-05 test 1 — SUPERCELL EQUIVALENCE, oracle-free.**
///
/// `KRCCSD` on a `[1,1,2]` k-mesh must equal `KRCCSD` on the `1×1×2` supercell
/// at Γ, divided by 2. Upstream asserts this at 4 decimals
/// (`test_krccsd.py:478`) and its own two routes differ by `2.97e-8`
/// (`measurements/README.md §2`), which is where **G9 = `1e-7`** comes from.
///
/// **It alone catches a wrong `kconserv` argument order, a transposed `t2`
/// index order and a misplaced `1/nkpts`** — and it needs no PySCF, which is
/// why the plan calls it test 1.
///
/// `exxdiv = None` on both sides: with the Ewald correction the two are NOT
/// equal, because the Madelung constant of a cell and of its supercell differ.
#[test]
#[ignore = "converges two SCFs, one of them a 4-atom supercell; run with --release"]
fn krccsd_matches_the_supercell_at_gamma() {
    // --- the k-mesh side
    let cell = common::diamond([15, 15, 15]);
    let kpts = cell.make_kpts([1, 1, 2]).expect("kpts");
    let nk = kpts.len();
    let mut mf = Krhf::from_df(Box::new(Fftdf::new(cell.clone(), &kpts).expect("fftdf")));
    mf.exxdiv = None;
    let mut cfg = KScfConfig::for_cell(&cell);
    cfg.conv_tol = 1e-10;
    let scf = mf.kernel(&cfg).expect("KRHF converges");
    assert!(scf.converged);
    let df = Fftdf::new(cell.clone(), &kpts).expect("fftdf");
    let mut cc = Krccsd::new(&scf, &df).expect("KRCCSD builds");
    let eris = cc.ao2mo().expect("_ERIS");
    let kres = cc.kernel_with(&eris).expect("KRCCSD kernel");
    assert!(kres.converged, "the k-mesh KRCCSD must converge");

    // --- the supercell side, at Γ
    let sup = pyscf_pbc_gto::super_cell(&cell, [1, 1, 2], false).expect("supercell");
    let gamma = sup.make_kpts([1, 1, 1]).expect("gamma");
    let mut smf = Krhf::from_df(Box::new(Fftdf::new(sup.clone(), &gamma).expect("fftdf")));
    smf.exxdiv = None;
    let mut scfg = KScfConfig::for_cell(&sup);
    scfg.conv_tol = 1e-10;
    let sscf = smf.kernel(&scfg).expect("supercell KRHF converges");
    assert!(sscf.converged);
    println!(
        "mean fields: k-mesh {} vs supercell/2 {}  |Δ| {:e}",
        scf.e_tot,
        sscf.e_tot / nk as f64,
        (scf.e_tot - sscf.e_tot / nk as f64).abs()
    );
    let sdf = Fftdf::new(sup.clone(), &gamma).expect("fftdf");
    let mut scc = Krccsd::new(&sscf, &sdf).expect("supercell KRCCSD builds");
    let seris = scc.ao2mo().expect("supercell _ERIS");
    let sres = scc.kernel_with(&seris).expect("supercell KRCCSD kernel");
    assert!(sres.converged, "the supercell KRCCSD must converge");

    let per_cell = sres.e_corr / nk as f64;
    let d = (kres.e_corr - per_cell).abs();
    println!(
        "e_corr: k-mesh {} ({} cycles) vs supercell/nk {} ({} cycles)  |Δ| {d:e}  (G9 = 1e-7)",
        kres.e_corr, kres.cycles, per_cell, sres.cycles
    );
    assert!(
        d < 1e-7,
        "supercell equivalence broken by {d:e}: this is where a wrong kconserv \
         argument order, a transposed t2 or a misplaced 1/nkpts shows up"
    );
}
