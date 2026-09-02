//! **GATE B for the k-symmetric DFT drivers** — plan item P-01 of
//! `.planning/pbc/KUKS-KSYMM-MULTIGRID-OPTIMISATION-PLAN.md`.
//!
//! D-PBC-17 requires every reduction reaching an energy, a density matrix or
//! a convergence test to be bit-identical under any worker count. That has
//! been measured for `KNumInt` (`numint_threads.rs`) and for the six
//! `gate.rs` energies, and **never for `KsymAdaptedKrks` / `KsymAdaptedKuks`**
//! — which is where plan §2.2.2 found the naive folds: `weighted_trace` and
//! `weighted_trace_uks` accumulate `nao²` products and then `nkpts_ibz`
//! weighted partials with plain running sums, and they feed `ecoul`, the
//! hybrid `exc` correction AND `energy_elec`'s `e1` for every ksymm KS driver.
//!
//! Two of the three parallel surfaces underneath are genuinely
//! thread-scheduled: `KPoints::sandwich_unfold` is a `par_iter` over the full
//! BZ (`kpts.rs`), and `eval_rho_one` / `vxc_mat_one` are `par_chunks_mut`
//! over disjoint outputs. So a divergence here would be real, not theoretical.
//!
//! # Why the SCF is run for a FIXED, SMALL number of cycles
//!
//! Bit-identity does not need convergence, and a *non*-converged iterate is
//! the sharper probe: convergence is an attractor that can mask a last-bit
//! difference by pulling two trajectories back together, whereas cycle 3 of
//! two runs either agrees bit-for-bit or does not. `max_cycle` is therefore
//! pinned and `converged` is deliberately NOT asserted.
//!
//! The worker count is varied INSIDE one process with explicit
//! `rayon::ThreadPool`s — `numint_threads.rs`'s reasoning, and strictly
//! stronger than a cross-process `RAYON_NUM_THREADS` sweep because it also
//! catches a result that depends on which worker stole a chunk. (P-01's text
//! proposed separate processes; this is the stronger form and the existing
//! convention, so it is what shipped.)

mod common;

use pyscf_algebra::CTensor;
use pyscf_pbc_dft::krks_ksymm::{KsymAdaptedKrks, KsymAdaptedKuks};
use pyscf_pbc_gto::{make_kpts_default, Cell};
use pyscf_pbc_scf::{KInitGuess, KScfConfig, KScfResult};
use pyscf_pbc_symm::kpts::{make_kpts, KPoints};

/// `3` is deliberately not a divisor of `nkpts` (8), `nkpts_ibz` (3) or `nao`.
const THREADS: [usize; 4] = [1, 2, 3, 8];

/// Small enough that four SCF starts fit a unit-test budget; large enough
/// that `RHO_CHUNK` (512) splits inside `eval_rho_one`. 11³ = 1331 points.
const MESH: [usize; 3] = [11, 11, 11];

/// Enough cycles that DIIS has engaged and `get_veff` has run several times,
/// few enough to stay cheap. See the module doc on why convergence is not
/// wanted here.
const CYCLES: u32 = 3;

/// D-17-07-01 (`17-07-SUMMARY.md`): `little_cogroup_ops` indexes `k2opk`'s
/// doubled column space while its consumers index `ops`, an upstream mismatch
/// that surfaces at Γ with time reversal on. The ksymm tests in this
/// repository all fold on the space group alone for that reason.
const TIME_REVERSAL: bool = false;

fn pool(n: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .expect("thread pool")
}

fn cfg() -> KScfConfig {
    KScfConfig {
        conv_tol: 1e-11,
        conv_tol_grad: Some(1e-10),
        max_cycle: CYCLES,
        init_guess: KInitGuess::Minao,
        ..KScfConfig::default()
    }
}

fn symmetric_kpts(cell: &mut Cell, nk: [usize; 3]) -> KPoints {
    let kpts_abs = make_kpts_default(cell, nk).expect("make_kpts_default");
    let kpts = make_kpts(cell, &kpts_abs, true, TIME_REVERSAL).expect("make_kpts");
    assert!(
        kpts.nkpts_ibz() < kpts.nkpts(),
        "the fixture must actually fold: {} IBZ of {} BZ",
        kpts.nkpts_ibz(),
        kpts.nkpts()
    );
    {
        use pyscf_pbc_symm::basis::{self, SymmAdaptedBasisInput};
        let input = SymmAdaptedBasisInput {
            kpts_scaled_ibz: kpts.kpts_scaled_ibz.clone(),
            little_cogroup_ops: kpts.little_cogroup_ops.clone(),
            ops: kpts.symmetry.ops.clone(),
            dmats: kpts.symmetry.dmats.clone(),
        };
        basis::build_symmetry(cell, &input).expect("build_symmetry");
    }
    kpts
}

fn same_bits(what: &str, t: usize, a: f64, b: f64) {
    assert_eq!(
        a.to_bits(),
        b.to_bits(),
        "{what} moved between 1 and {t} threads: {a:.17e} vs {b:.17e} (delta {:e})",
        b - a
    );
}

fn same_ctensor(what: &str, t: usize, a: &CTensor, b: &CTensor) {
    assert_eq!(a.re.len(), b.re.len(), "{what}: length moved at {t} threads");
    for i in 0..a.re.len() {
        assert_eq!(
            a.re[i].to_bits(),
            b.re[i].to_bits(),
            "{what}.re[{i}] moved between 1 and {t} threads"
        );
        assert_eq!(
            a.im[i].to_bits(),
            b.im[i].to_bits(),
            "{what}.im[{i}] moved between 1 and {t} threads"
        );
    }
}

/// Compare everything an SCF result carries that a later item could move:
/// the total and its three components, every density-matrix element, and
/// every orbital energy.
fn assert_same_result(tag: &str, t: usize, a: &KScfResult, b: &KScfResult) {
    same_bits(&format!("{tag} e_tot"), t, a.e_tot, b.e_tot);
    same_bits(&format!("{tag} e_elec"), t, a.e_elec, b.e_elec);
    same_bits(&format!("{tag} e_coul"), t, a.e_coul, b.e_coul);
    same_bits(&format!("{tag} e_nuc"), t, a.e_nuc, b.e_nuc);
    assert_eq!(a.dm.len(), b.dm.len(), "{tag}: channel count moved");
    for (s, (sa, sb)) in a.dm.iter().zip(&b.dm).enumerate() {
        assert_eq!(sa.len(), sb.len(), "{tag}: k-count moved in channel {s}");
        for (k, (ma, mb)) in sa.iter().zip(sb).enumerate() {
            same_ctensor(&format!("{tag} dm[{s}][{k}]"), t, ma, mb);
        }
    }
    for (i, (ea, eb)) in a.mo_energy.iter().zip(&b.mo_energy).enumerate() {
        assert_eq!(ea.len(), eb.len(), "{tag}: nmo moved at block {i}");
        for (j, (x, y)) in ea.iter().zip(eb).enumerate() {
            same_bits(&format!("{tag} mo_energy[{i}][{j}]"), t, *x, *y);
        }
    }
}

// ---------------------------------------------------------------------------
// KRKS over an IBZ k-set
// ---------------------------------------------------------------------------

#[test]
fn ksymm_krks_is_bit_identical_across_thread_counts() {
    // `silicon()`'s DEFAULT precision is right here: this is a bit-identity
    // comparison of one computation against itself, not an accuracy
    // comparison against a second route, so the joint precision floor
    // `17-04-MEASUREMENT.md` documents does not apply and the tight fixture
    // would only cost time.
    let mut cell = common::silicon();
    cell.mesh = MESH;
    let kpts = symmetric_kpts(&mut cell, [2, 2, 2]);

    // Both `eig` branches: the symmetry-adapted one (17-04) and the plain
    // generalised eigenproblem. They are different code, and only the first
    // touches `symm_orb`.
    for use_ao_symmetry in [true, false] {
        let run = |t: usize| {
            pool(t).install(|| {
                let mut mf = KsymAdaptedKrks::new(cell.clone(), kpts.clone(), "lda,vwn")
                    .expect("KsymAdaptedKrks");
                mf.use_ao_symmetry = use_ao_symmetry;
                mf.kernel(&cfg()).expect("ksymm KRKS")
            })
        };
        let reference = run(THREADS[0]);
        for &t in &THREADS[1..] {
            assert_same_result(
                &format!("ksymm KRKS (use_ao_symmetry={use_ao_symmetry})"),
                t,
                &reference,
                &run(t),
            );
        }
        println!(
            "ksymm KRKS use_ao_symmetry={use_ao_symmetry}: e_tot = {:.15} bit-identical at {THREADS:?} threads",
            reference.e_tot
        );
    }
}

// ---------------------------------------------------------------------------
// KUKS over an IBZ k-set — RULE U: a genuinely open-shell fixture
// ---------------------------------------------------------------------------

/// The open-shell He cell `krks_ksymm.rs::kuks_ibz_runs_and_stays_symmetric`
/// uses: `spin = 2`, so `dm_a != dm_b` and the unrestricted path does not
/// degenerate into the restricted one (RULE U).
fn open_shell_he() -> Cell {
    use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs, Unit};
    use pyscf_pbc_gto::types::{ALattice, CellBuildArgs};
    let h = 2.834589;
    let mut cell = Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("He".into(), [0.0, 0.0, 0.0])]),
            basis: BasisInput::Name("6-31g".into()),
            unit: Unit::Bohr,
            spin: 2,
            ..Default::default()
        },
        a: ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        ..Default::default()
    })
    .expect("open-shell He cell must build");
    cell.mesh = MESH;
    cell
}

#[test]
fn ksymm_kuks_is_bit_identical_across_thread_counts() {
    let mut cell = open_shell_he();
    let kpts = symmetric_kpts(&mut cell, [2, 2, 2]);

    let run = |t: usize| {
        pool(t).install(|| {
            let mf = KsymAdaptedKuks::new(cell.clone(), kpts.clone(), "lda,vwn")
                .expect("KsymAdaptedKuks");
            mf.kernel(&cfg()).expect("ksymm KUKS")
        })
    };
    let reference = run(THREADS[0]);
    assert_eq!(reference.dm.len(), 2, "KUKS must carry two density channels");
    // RULE U, asserted rather than assumed: if the two channels had collapsed
    // this test would be a KRKS test wearing a KUKS name.
    let spin_diff = reference.dm[0]
        .iter()
        .zip(&reference.dm[1])
        .map(|(a, b)| {
            a.re.iter()
                .zip(&b.re)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0_f64, f64::max)
        })
        .fold(0.0_f64, f64::max);
    assert!(
        spin_diff > 1e-6,
        "RULE U: the fixture collapsed to a restricted solution \
         (max |dm_a - dm_b| = {spin_diff:e})"
    );

    for &t in &THREADS[1..] {
        assert_same_result("ksymm KUKS", t, &reference, &run(t));
    }
    println!(
        "ksymm KUKS: e_tot = {:.15}, max |dm_a - dm_b| = {spin_diff:e}, \
         bit-identical at {THREADS:?} threads",
        reference.e_tot
    );
}
