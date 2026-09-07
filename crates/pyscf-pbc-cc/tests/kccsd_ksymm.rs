//! `KsymAdaptedRCCSD` — plan 17-09, the CC half.
//!
//! # The fixture is upstream's own, and it is deliberately tiny
//!
//! `cc/test/test_kccsd_ksymm.py:25-41`: a two-GTO He in a 2 Bohr cube at
//! `[2,2,2]`, `KRHF(exxdiv=None)` on GDF. `nocc = nvir = 1`, so every
//! amplitude block is a single number and an index error cannot hide in the
//! magnitude of a large tensor. There are **120 IBZ k-quartets against 512
//! k-triples** (`measurements/gate_kccsd_ksymm.out`), so the whole quartet
//! machinery is exercised.
//!
//! # What is gated against what
//!
//! The primary gate is **this port's k-symmetric `e_corr` against this port's
//! own full-BZ `KRCCSD`, driven from the SAME unfolded mean field.** That is
//! the comparison upstream's `test_vs_krccsd` makes, and it is the right one:
//! it has no second SCF in it, so the residual is the symmetry algebra and the
//! integral-rotation floor and nothing else.
//!
//! Upstream's own k-symmetric number is reported beside it but **not** used as
//! the gate, because of D-17-09-01 (`kccsd_rhf_ksymm`'s module doc): upstream's
//! second T1 quartet term guards on a stale loop variable and therefore drops a
//! contribution worth `1.117e-10` on this fixture. Measured, upstream's ksymm
//! sits `6.917e-11` from its own full-BZ answer and the corrected version sits
//! `4.250e-11` from it — closer. Gating on upstream's number would mean
//! reproducing its defect.

#![allow(clippy::needless_range_loop)]

use num_complex::Complex64;
use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_cc::kccsd_rhf::Krccsd;
use pyscf_pbc_cc::keris::KErisOpts;
use pyscf_pbc_cc::{KrccsdOpts, KsymRccsdInputs, KsymRccsdOpts, run_ksym_rccsd};
use pyscf_pbc_df::Gdf;
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs, make_kpts_default};
use pyscf_pbc_mp::unfold_kscf_result;
use pyscf_pbc_scf::{KInitGuess, KScfConfig, KScfResult, KsymAdaptedKrhf};
use pyscf_pbc_symm::kpts::{KPoints, make_kpts};

/// D-17-07-01: `little_cogroup_ops` indexes `k2opk`'s doubled column space
/// while `MORotationMatrix` indexes `ops`, so the k-symmetric fixtures run
/// with time reversal OFF. Upstream's oracle was re-run both ways
/// (`measurements/gate_kccsd_ksymm.out`) and gives the same 4 IBZ k-points,
/// the same 120 quartets and the same energies to 1e-13, so nothing is lost.
const TIME_REVERSAL: bool = false;

/// Upstream PySCF 2.12.1, `measurements/gate_kccsd_ksymm.out`,
/// `time_reversal_symmetry = False`.
const UP_E_SCF: f64 = -2.0931518935294924;
/// The FFT mesh upstream's build derives for that cell. A different mesh here
/// is the single most likely reason `E_scf` would not match.
const UP_MESH: [usize; 3] = [35, 35, 35];
const UP_EMP2_FULL: f64 = -0.006182522881429103;
const UP_ECORR_FULL: f64 = -0.007379123090006057;
/// Upstream's k-symmetric `e_corr`, WITH the D-17-09-01 defect.
const UP_ECORR_KSYMM_STALE: f64 = -0.007379123020831889;
/// Upstream's k-symmetric `e_corr` with the corrected guard
/// (`measurements/gate_kccsd_stale_kc.out`) — what this port targets.
const UP_ECORR_KSYMM_FIXED: f64 = -0.007379123132508862;

/// `He` in a 2 ANGSTROM cube with upstream's two uncontracted s functions.
///
/// **The unit is the trap.** `test_kccsd_ksymm.py:29-31` writes
/// `He.a = np.eye(3)*L` and `He.atom = [['He', (L/2, L/2, L/2)]]` with
/// `L = 2.` and never sets `He.unit`, so PySCF's default -- ANGSTROM --
/// applies. Reading it as Bohr gives a cell 1.89x smaller, a `[19,19,19]`
/// mesh instead of `[35,35,35]`, and `E_scf = -1.402208` instead of
/// `-2.093152`. That was measured, not guessed: the `UP_E_SCF` assertion
/// below is what caught it.
///
/// PySCF's `{'He': [[0, (4.0, 1.0)], [0, (1.0, 1.0)]]}` is two SEPARATE `s`
/// shells, each one primitive with contraction coefficient 1; the NWChem form
/// below is the same basis, and `assert_upstream_scf` checks that by comparing
/// the converged `E_scf` to upstream's to 1e-9.
fn he_cell() -> Cell {
    let l = 2.0;
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("He".into(), [l / 2.0, l / 2.0, l / 2.0])]),
            basis: BasisInput::NwchemText(
                "BASIS \"ao basis\" PRINT\nHe    S\n      4.0    1.0\nHe    S\n      1.0    1.0\nEND\n"
                    .into(),
            ),
            // Angstrom -- upstream's default, and what its fixture relies on.
            unit: Unit::Ang,
            ..Default::default()
        },
        a: ALattice::Matrix([[l, 0.0, 0.0], [0.0, l, 0.0], [0.0, 0.0, l]]),
        ..Default::default()
    })
    .expect("He cell")
}

fn kpoints(cell: &Cell, mesh: [usize; 3]) -> KPoints {
    let kpts_abs = make_kpts_default(cell, mesh).expect("make_kpts_default");
    make_kpts(cell, &kpts_abs, true, TIME_REVERSAL).expect("make_kpts")
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

/// One k-symmetric SCF over the IBZ, unfolded to the full BZ — the ONLY mean
/// field either CC route sees.
fn unfolded_reference(cell: &Cell, kpts: &KPoints) -> KScfResult {
    let df = Gdf::new(cell.clone(), &kpts.kpts);
    let mut mf = KsymAdaptedKrhf::from_df(Box::new(df), kpts.clone());
    mf.exxdiv = None;
    // `use_ao_symmetry = false`: the symmetry-adapted `eig` is 17-04/17-07's
    // subject and `pyscf-pbc-scf/tests/khf_ksymm.rs` already gates the two
    // routes against each other at 1e-10. What this file needs from the SCF is
    // only that the IBZ orbitals be converged; which eig produced them is
    // irrelevant to the CC algebra.
    //
    // It is also the safe default after D-17-09-02: `use_ao_symmetry = true`
    // is unsound on a k-mesh whose symmetry is lower than the lattice's,
    // because `little_cogroup_ops` is filled from `k2opk` as snapshotted
    // BEFORE `make_kpts_ibz`'s column wipe. This fixture's simple-cubic
    // `[2,2,2]` mesh is symmetry-closed and would be fine either way, but
    // nothing here needs to depend on that.
    mf.use_ao_symmetry = false;
    let ibz = mf.kernel(&cfg()).expect("k-symmetric SCF");
    assert!(ibz.converged, "the k-symmetric SCF did not converge");
    assert_eq!(ibz.nkpts, kpts.nkpts_ibz());
    unfold_kscf_result(&ibz, kpts, cell).expect("unfold")
}

#[test]
fn he_222_matches_the_full_bz_krccsd() {
    let cell = he_cell();
    let kpts = kpoints(&cell, [2, 2, 2]);
    assert_eq!(kpts.nkpts(), 8);
    assert_eq!(
        kpts.nkpts_ibz(),
        4,
        "upstream measures 4 IBZ k-points on this fixture"
    );

    println!("mesh = {:?} (upstream {UP_MESH:?})", cell.mesh);
    assert_eq!(
        cell.mesh, UP_MESH,
        "the He fixture's FFT mesh is not upstream's"
    );
    let scf = unfolded_reference(&cell, &kpts);
    println!("E_scf = {:.15} (upstream {UP_E_SCF:.15})", scf.e_tot);
    // MEASURED: this port's `KRHF` and upstream's differ by **1.282e-06 Ha**
    // on this fixture at the default `cell.precision`. That is the same class
    // of mean-field residual 16-01 found on diamond (`1.348e-05` at a pinned
    // coarse mesh), and it is why every comparison below that could be
    // contaminated by it is either driven from ONE mean field or reported
    // rather than gated tightly. The check here is a FIXTURE check — it
    // catches a wrong unit, a wrong mesh or a wrong basis, all of which move
    // `E_scf` by 0.1 Ha or more — not a precision gate.
    let d_scf = (scf.e_tot - UP_E_SCF).abs();
    println!("|d E_scf| vs upstream = {d_scf:e}");
    assert!(
        d_scf < 1e-4,
        "the He fixture does not reproduce upstream's mean field: {} vs {UP_E_SCF} \
         (|d| = {d_scf:e}); a wrong unit gives -1.402208 and a wrong mesh moves it \
         by more than 1e-3",
        scf.e_tot
    );

    // ---- the k-symmetric route ------------------------------------------
    let df = Gdf::new(cell.clone(), &kpts.kpts);
    let inputs = KsymRccsdInputs::build(&scf, &df, &kpts, KErisOpts::default()).expect("inputs");
    assert_eq!(
        inputs.kqrts.kqrts_ibz.len(),
        120,
        "upstream measures 120 IBZ k-quartets of 512 k-triples"
    );
    let opts = KsymRccsdOpts::default();
    let (sym, n_ao2mo) = run_ksym_rccsd(&inputs, &df, &kpts, &opts).expect("ksymm KRCCSD");
    println!(
        "ao2mo transforms: {n_ao2mo} (a full-BZ build needs one per symm_map orbit of {} triples)",
        kpts.nkpts().pow(3)
    );
    assert!(sym.converged, "the k-symmetric CC did not converge");

    // ---- the full-BZ route, SAME mean field ------------------------------
    let df_full = Gdf::new(cell.clone(), &kpts.kpts);
    let mut full = Krccsd::new(&scf, &df_full).expect("KRCCSD");
    full.opts = KrccsdOpts::default();
    let ref_r = full.kernel().expect("full-BZ KRCCSD");
    assert!(ref_r.converged, "the full-BZ CC did not converge");

    println!(
        "emp2   ksymm = {:.15}  full BZ = {:.15}",
        sym.emp2, ref_r.emp2
    );
    println!(
        "e_corr ksymm = {:.15}  full BZ = {:.15}",
        sym.e_corr, ref_r.e_corr
    );
    println!("upstream full BZ  = {UP_ECORR_FULL:.15}");
    println!("upstream ksymm (stale kc) = {UP_ECORR_KSYMM_STALE:.15}");
    println!("upstream ksymm (corrected) = {UP_ECORR_KSYMM_FIXED:.15}");

    let d_self = (sym.e_corr - ref_r.e_corr).abs();
    println!("|d e_corr| ksymm vs this port's full BZ = {d_self:e}");
    println!(
        "|d e_corr| ksymm vs upstream corrected  = {:e}",
        (sym.e_corr - UP_ECORR_KSYMM_FIXED).abs()
    );
    println!(
        "|d e_corr| this port's full BZ vs upstream = {:e}",
        (ref_r.e_corr - UP_ECORR_FULL).abs()
    );
    println!(
        "|d emp2| this port's full BZ vs upstream = {:e}",
        (ref_r.emp2 - UP_EMP2_FULL).abs()
    );

    // Upstream's own two routes on this fixture differ by 4.250e-11
    // (`measurements/gate_kccsd_stale_kc.out`, corrected guard), so THAT is the
    // floor this comparison can reach — not zero.
    assert!(
        d_self < 1e-8,
        "the k-symmetric e_corr differs from this port's own full-BZ KRCCSD by \
         {d_self:e}; upstream's own two routes differ by 4.250e-11 on this fixture"
    );
    // The ABSOLUTE comparison against upstream is gated LOOSELY and on
    // purpose: this port's mean field is 1.282e-06 Ha from upstream's on this
    // fixture (printed above), and a correlation energy compared across two
    // different mean fields measures the mean fields. 1e-6 is the honest bar
    // here; the tight statement is the one-mean-field comparison above.
    assert!(
        (ref_r.e_corr - UP_ECORR_FULL).abs() < 1e-6,
        "the full-BZ KRCCSD does not reproduce upstream on this fixture: {:e}",
        (ref_r.e_corr - UP_ECORR_FULL).abs()
    );
    assert!(
        (sym.e_corr - UP_ECORR_KSYMM_FIXED).abs() < 1e-6,
        "the k-symmetric KRCCSD does not reproduce upstream's CORRECTED number: {:e}",
        (sym.e_corr - UP_ECORR_KSYMM_FIXED).abs()
    );
    // `emp2` is `init_amps` alone — no amplitude iteration, so it is the
    // tightest single check of the quartet weights.
    assert!(
        (sym.emp2 - ref_r.emp2).abs() < 1e-9,
        "init_amps: the IBZ-weighted MP2 differs from the full-BZ one by {:e}; \
         a wrong `weights_ibz * nkpts^3` shows up here first",
        (sym.emp2 - ref_r.emp2).abs()
    );
}

/// The amplitudes themselves, unfolded — upstream's `test_vs_krccsd`
/// (`test_kccsd_ksymm.py:52-63`), which gates `t1`/`t2` at 6 decimals.
///
/// This is the test that catches a wrong `(label, trans)` pair: `e_corr` is a
/// full contraction and several wrong rotations leave it invariant, while an
/// element-wise amplitude comparison does not.
#[test]
fn he_222_amplitudes_unfold_to_the_full_bz_ones() {
    let cell = he_cell();
    let kpts = kpoints(&cell, [2, 2, 2]);
    let scf = unfolded_reference(&cell, &kpts);

    let df = Gdf::new(cell.clone(), &kpts.kpts);
    let inputs = KsymRccsdInputs::build(&scf, &df, &kpts, KErisOpts::default()).expect("inputs");
    let (sym, _) =
        run_ksym_rccsd(&inputs, &df, &kpts, &KsymRccsdOpts::default()).expect("ksymm KRCCSD");
    let ctx = inputs.ctx(&kpts);
    let (no, nv) = (inputs.nocc, inputs.nvir);
    let t1 = ctx
        .dense2(&sym.t1, [no, nv], &ctx.labels.ov, &ctx.labels.nc)
        .expect("unfold t1");
    let t2 = ctx
        .dense4(
            &sym.t2,
            [no, no, nv, nv],
            &ctx.labels.oovv,
            &ctx.labels.nncc,
        )
        .expect("unfold t2");

    let df_full = Gdf::new(cell.clone(), &kpts.kpts);
    let mut full = Krccsd::new(&scf, &df_full).expect("KRCCSD");
    let ref_r = full.kernel().expect("full-BZ KRCCSD");

    let mut d1 = 0.0f64;
    for i in 0..t1.len() {
        d1 = d1
            .max((t1.data().re[i] - ref_r.t1.data().re[i]).abs())
            .max((t1.data().im[i] - ref_r.t1.data().im[i]).abs());
    }
    let mut d2 = 0.0f64;
    for i in 0..t2.len() {
        d2 = d2
            .max((t2.data().re[i] - ref_r.t2.data().re[i]).abs())
            .max((t2.data().im[i] - ref_r.t2.data().im[i]).abs());
    }
    println!("|d t1|max = {d1:e}   |d t2|max = {d2:e}");
    // Upstream measures 1.749e-08 / 1.392e-10 for the same comparison
    // (`measurements/gate_kccsd_ksymm.out`), so 1e-6 — upstream's own gate —
    // is the honest bar.
    assert!(d1 < 1e-6, "|d t1|max = {d1:e}");
    assert!(d2 < 1e-6, "|d t2|max = {d2:e}");

    // The oracle-free half: the unfolded `t2` must satisfy the RCCSD amplitude
    // symmetry `t2[ki,kj,ka][i,j,a,b] == t2[kj,ki,kb][j,i,b,a]` over the WHOLE
    // zone, not only at the IBZ quartets that were written.
    //
    // This is the property a wrong `(label, trans)` pair breaks and `e_corr`
    // does not see: the energy is a full contraction, and several wrong
    // rotations leave it invariant while destroying this identity. The
    // k-symmetric `t2` is stored ONLY at 120 of 512 triples, so at the other
    // 392 both sides of the comparison come out of `transform_4d`.
    let kconserv = pyscf_pbc_lib::kpts_helper::get_kconserv(&cell.a, &kpts.kpts);
    let nk = kpts.nkpts();
    let mut worst = 0.0f64;
    for ki in 0..nk {
        for kj in 0..nk {
            for ka in 0..nk {
                let kb = kconserv.get(ki, ka, kj) as usize;
                let a = t2.slice_leading(&[ki, kj, ka]).expect("t2");
                let b = t2.slice_leading(&[kj, ki, kb]).expect("t2");
                for i in 0..no {
                    for j in 0..no {
                        for x in 0..nv {
                            for y in 0..nv {
                                let p = ((i * no + j) * nv + x) * nv + y;
                                let q = ((j * no + i) * nv + y) * nv + x;
                                worst = worst
                                    .max((a.data().re[p] - b.data().re[q]).abs())
                                    .max((a.data().im[p] - b.data().im[q]).abs());
                            }
                        }
                    }
                }
            }
        }
    }
    println!("max |t2[ki,kj,ka] - t2[kj,ki,kb]^T| over the whole BZ = {worst:e}");
    assert!(
        worst < 1e-10,
        "the unfolded t2 breaks the RCCSD amplitude symmetry by {worst:e}; the \
         prime suspect is a wrong (label, trans) pair on the `oovv` container"
    );
    let _ = Complex64::new(0.0, 0.0);
}

/// `ktensor_direct = true` is refused by name rather than silently ignored.
#[test]
fn ktensor_direct_is_refused() {
    let cell = he_cell();
    let kpts = kpoints(&cell, [1, 1, 2]);
    let scf = unfolded_reference(&cell, &kpts);
    let df = Gdf::new(cell.clone(), &kpts.kpts);
    let inputs = KsymRccsdInputs::build(&scf, &df, &kpts, KErisOpts::default()).expect("inputs");
    let opts = KsymRccsdOpts {
        ktensor_direct: true,
        ..KsymRccsdOpts::default()
    };
    let err = run_ksym_rccsd(&inputs, &df, &kpts, &opts).expect_err("must refuse");
    let msg = err.to_string();
    assert!(msg.contains("ktensor_direct"), "{msg}");
    assert!(msg.contains("kccsd_rhf_ksymm.py:389-395"), "{msg}");
}

/// §9.3: `e_corr` and the IBZ amplitude stores bit-identical at 1 and 8
/// threads, inside ONE process.
#[test]
fn e_corr_and_amplitudes_are_bit_identical_across_thread_counts() {
    let cell = he_cell();
    let kpts = kpoints(&cell, [1, 1, 2]);
    let scf = unfolded_reference(&cell, &kpts);
    let df = Gdf::new(cell.clone(), &kpts.kpts);
    let inputs = KsymRccsdInputs::build(&scf, &df, &kpts, KErisOpts::default()).expect("inputs");

    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool")
            .install(|| {
                run_ksym_rccsd(&inputs, &df, &kpts, &KsymRccsdOpts::default())
                    .expect("ksymm KRCCSD")
                    .0
            })
    };
    let one = run(1);
    let eight = run(8);
    println!("e_corr @1 = {:.17}", one.e_corr);
    println!("e_corr @8 = {:.17}", eight.e_corr);
    assert_eq!(one.e_corr.to_bits(), eight.e_corr.to_bits());
    assert_eq!(one.emp2.to_bits(), eight.emp2.to_bits());
    for i in 0..one.t1.len() {
        assert_eq!(
            one.t1.data().re[i].to_bits(),
            eight.t1.data().re[i].to_bits()
        );
        assert_eq!(
            one.t1.data().im[i].to_bits(),
            eight.t1.data().im[i].to_bits()
        );
    }
    for i in 0..one.t2.len() {
        assert_eq!(
            one.t2.data().re[i].to_bits(),
            eight.t2.data().re[i].to_bits()
        );
        assert_eq!(
            one.t2.data().im[i].to_bits(),
            eight.t2.data().im[i].to_bits()
        );
    }
}
