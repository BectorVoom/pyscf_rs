//! **GATE B for the multigrid drivers** — plan item P-01 of
//! `.planning/pbc/KUKS-KSYMM-MULTIGRID-OPTIMISATION-PLAN.md`.
//!
//! D-PBC-17 requires every reduction that reaches an energy to be
//! bit-identical under any worker count. `multigrid2.rs` already gates
//! `MultiGridNumInt2::eval_rho_g` that way, but **neither driver's `nr_rks`
//! had ever been measured** — and `nr_rks` is where the energies actually
//! are: `ecoul`, `nelec` and `exc`. Plan §2.3.4 records that all six of those
//! reductions were naive `Iterator::sum::<f64>()` over `ngrids` when this
//! file was written (the largest naive folds on any energy path in the tree,
//! 15 625 terms at `25³`), which is exactly the shape P-03 replaces with
//! `oracle_sum`.
//!
//! **This gate is written BEFORE P-03 lands, deliberately.** A determinism
//! gate authored after the fix can only ever confirm the fix; authored
//! before, it says whether the defect was observable. Both answers are
//! recorded in the plan's execution summary.
//!
//! # Why a naive `.sum()` can still pass this
//!
//! `Iterator::sum` over a `Vec`/range is a strict sequential fold in index
//! order — it is not itself thread-dependent. It is thread-dependent only
//! when the *terms* are produced in a thread-dependent order or value. So a
//! PASS here does not make the naive folds acceptable (their error bound is
//! still `O(n·eps)` against `oracle_sum`'s `O(log n·eps)`, which is P-03's
//! actual argument); it establishes that the collocation and the FFT feeding
//! them are order-stable, which is the property this gate owns.
//!
//! The worker count is varied INSIDE one process with explicit
//! `rayon::ThreadPool`s, following `numint_threads.rs`'s reasoning: it is
//! strictly stronger than a `RAYON_NUM_THREADS` sweep across processes,
//! because it also catches a result that depends on which worker stole a
//! chunk. (The plan's P-01 text proposed separate processes; this is the
//! stronger form and the repository's existing convention, so it is what
//! shipped.)

mod common;

use pyscf_pbc_dft::multigrid::{MultiGridNumInt, MultiGridNumInt2};
use pyscf_pbc_gto::Cell;

/// `3` is deliberately not a divisor of the block count or of `nao`, so a
/// ragged final partition is exercised.
const THREADS: [usize; 4] = [1, 2, 3, 8];

/// The mesh `multigrid2.rs` uses. Small enough for a unit-test budget, large
/// enough that both drivers populate several levels of the ladder.
const MESH: [usize; 3] = [25, 25, 25];

fn small_diamond() -> Cell {
    let mut c = common::diamond();
    c.mesh = MESH;
    c
}

fn small_silicon() -> Cell {
    let mut c = common::silicon();
    c.mesh = MESH;
    c
}

/// The same deterministic Hermitian density `multigrid2.rs` uses — a fixed
/// LCG, so the input is bit-identical across runs and any divergence is the
/// driver's.
fn random_symmetric_dm(nao: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64 / (1u64 << 53) as f64) * 0.2
    };
    let mut dm = vec![0.0f64; nao * nao];
    for i in 0..nao {
        for j in 0..nao {
            dm[i * nao + j] = next();
        }
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

fn pool(n: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .expect("thread pool")
}

/// Assert two `f64` are the SAME BITS, naming the quantity and the worker
/// count. Not an approximate comparison — D-PBC-17 is bit-identity or it is
/// nothing.
fn same_bits(what: &str, t: usize, a: f64, b: f64) {
    assert_eq!(
        a.to_bits(),
        b.to_bits(),
        "{what} moved between 1 and {t} threads: {a:.17e} vs {b:.17e} \
         (delta {:e})",
        b - a
    );
}

fn same_bits_slice(what: &str, t: usize, a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len(), "{what}: length moved at {t} threads");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{what}[{i}] moved between 1 and {t} threads: {x:.17e} vs {y:.17e}"
        );
    }
}

// ---------------------------------------------------------------------------
// v1 — `MultiGridNumInt`
// ---------------------------------------------------------------------------

#[test]
fn v1_nr_rks_is_bit_identical_across_thread_counts() {
    let ni = MultiGridNumInt::new();
    for cell in [small_diamond(), small_silicon()] {
        let nao = cell.mol.nao_nr;
        let dm = random_symmetric_dm(nao, 0x0BAD_F00D);
        // LDA exercises `nvar = 1`; PBE adds the G-space gradient path, which
        // is three extra FFT round trips per call and its own reduction.
        for xc in ["LDA,VWN", "PBE"] {
            let run = |t: usize| {
                pool(t)
                    .install(|| ni.nr_rks(&cell, xc, &dm))
                    .expect("v1 nr_rks")
            };
            let reference = run(THREADS[0]);
            for &t in &THREADS[1..] {
                let got = run(t);
                let tag = |q: &str| format!("v1 nr_rks {q} (xc={xc}, nao={nao})");
                same_bits(&tag("nelec"), t, reference.nelec, got.nelec);
                same_bits(&tag("exc"), t, reference.exc, got.exc);
                same_bits(&tag("ecoul"), t, reference.ecoul, got.ecoul);
                same_bits_slice(&tag("veff"), t, &reference.veff, &got.veff);
            }
        }
    }
}

#[test]
fn v1_get_j_is_bit_identical_across_thread_counts() {
    let ni = MultiGridNumInt::new();
    let cell = small_silicon();
    let dm = random_symmetric_dm(cell.mol.nao_nr, 0x5EED_1234);
    let run = |t: usize| pool(t).install(|| ni.get_j(&cell, &dm)).expect("v1 get_j");
    let reference = run(THREADS[0]);
    for &t in &THREADS[1..] {
        same_bits_slice("v1 get_j", t, &reference, &run(t));
    }
}

// ---------------------------------------------------------------------------
// v2 — `MultiGridNumInt2`
// ---------------------------------------------------------------------------

#[test]
fn v2_nr_rks_is_bit_identical_across_thread_counts() {
    let ni = MultiGridNumInt2::new();
    // ONE cell here, not two: v2's density evaluation is ~7-9 s (17-12), and
    // `eval_rho_g_is_bit_identical_across_thread_counts_v2` in `multigrid2.rs`
    // already covers diamond's collocation across the same four counts. What
    // this adds is the ENERGY reductions on top of it.
    let cell = small_silicon();
    let nao = cell.mol.nao_nr;
    let dm = random_symmetric_dm(nao, 0x0BAD_F00D);
    for xc in ["LDA,VWN", "PBE"] {
        let run = |t: usize| {
            pool(t)
                .install(|| ni.nr_rks(&cell, xc, &dm))
                .expect("v2 nr_rks")
        };
        let reference = run(THREADS[0]);
        for &t in &THREADS[1..] {
            let got = run(t);
            let tag = |q: &str| format!("v2 nr_rks {q} (xc={xc}, nao={nao})");
            same_bits(&tag("nelec"), t, reference.nelec, got.nelec);
            same_bits(&tag("exc"), t, reference.exc, got.exc);
            same_bits(&tag("ecoul"), t, reference.ecoul, got.ecoul);
            same_bits_slice(&tag("veff"), t, &reference.veff, &got.veff);
        }
    }
}

#[test]
fn v2_get_j_is_bit_identical_across_thread_counts() {
    let ni = MultiGridNumInt2::new();
    let cell = small_silicon();
    let dm = random_symmetric_dm(cell.mol.nao_nr, 0x5EED_1234);
    let run = |t: usize| pool(t).install(|| ni.get_j(&cell, &dm)).expect("v2 get_j");
    let reference = run(THREADS[0]);
    for &t in &THREADS[1..] {
        same_bits_slice("v2 get_j", t, &reference, &run(t));
    }
}
