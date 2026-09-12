//! Session-7 instrument for the K-RESOLVED multigrid host contraction.
//!
//! ```text
//! PYSCF_MG_BENCH_MESH=25 PYSCF_MG_BENCH_REPS=3 cargo test --release -p pyscf-pbc-dft \
//!     --test mg_kpts_bench -- --ignored --nocapture
//! ```
//!
//! # What this measures, and why the design makes it free
//!
//! `multigrid::kpts`'s module doc records the structural property of the
//! k-point generalisation: **the collocation is over lattice IMAGES and does
//! not grow with `nkpts` at all.** What grows is the host contraction —
//! `term_coef_kpts` forward and `pairlevel_pass2_kpts` reverse, both
//! `O(nslots · nkpts)` and both currently a plain serial `for k` loop
//! (`kpts.rs` contains no rayon).
//!
//! So a sweep over `nkpts` bounds the lever with no instrumentation at all:
//! the flat part is the kernels, and the slope is **everything that scales
//! with k**.
//!
//! Read that slope as an UPPER BOUND on the two contractions, not as their
//! cost. `PhaseTable::new` (`O(nkpts · nimg)`, per level per call),
//! `KDmP::expand`, `contract_v_kpts` and the `v_re`/`v_im` allocation are all
//! `O(nkpts)` too and sit inside the same slope. That is enough to DECIDE
//! with: if the upper bound is already small, the item should be dropped
//! rather than built, and no finer attribution is needed — the A-03/A-04
//! lesson (kill-switch arms measured the AO kernel's arithmetic at ~0 % and
//! refuted two planned items) applied before writing code instead of after.
//!
//! **MEASURED 2026-09-12, and the item was REFUTED on it:** si gth-szv
//! `25^3`, a 64x increase in k-points costs **+8.8 %** wall (+1.6 % at
//! 2x2x2), so parallelising the entire slope perfectly is bounded by
//! ~**1.08x**. The serial contractions were left alone; see
//! `.planning/pbc/KUKS-KSYMM-MULTIGRID-SESSION-7-EXECUTION-SUMMARY.md` §5.
//!
//! A CONVERGED density is deliberately not used: `multigrid_kpts.rs`'s
//! `converged_dm` runs a full FFTDF SCF per mesh, which is unaffordable at
//! 4x4x4 and buys nothing for a timing harness. A deterministic Hermitian
//! stack exercises exactly the same code path at the same shapes.

mod common;

use std::time::Instant;

use pyscf_algebra::CTensor;
use pyscf_pbc_dft::multigrid::MultiGridNumInt2;
use pyscf_pbc_gto::{Cell, make_kpts_default};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// A deterministic Hermitian density stack, one `nao x nao` complex matrix
/// per k-point. Hermitian per k (`re` symmetric, `im` antisymmetric) so the
/// contraction sees a physically shaped operand; the values are arbitrary.
fn model_kdms(nao: usize, nkpts: usize, seed: u64) -> Vec<CTensor> {
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    };
    (0..nkpts)
        .map(|_| {
            let mut re = vec![0.0f64; nao * nao];
            let mut im = vec![0.0f64; nao * nao];
            for i in 0..nao {
                for j in 0..=i {
                    let (a, b) = (next() * 0.2, next() * 0.2);
                    re[i * nao + j] = a;
                    re[j * nao + i] = a;
                    im[i * nao + j] = b;
                    im[j * nao + i] = -b;
                }
                re[i * nao + i] += 1.0;
                im[i * nao + i] = 0.0;
            }
            CTensor::from_planes(re, im)
        })
        .collect()
}

fn cell_at(mesh: usize) -> Cell {
    let mut c = common::silicon();
    c.mesh = [mesh; 3];
    c
}

/// Forward (`eval_rho_g_kpts`) and reverse-inclusive (`nr_rks_kpts`) wall as
/// a function of `nkpts`, warm minimum over `reps`.
#[test]
#[ignore = "an instrument, not a gate — run with --ignored --nocapture"]
fn mg_kpts_bench() {
    let mesh = env_usize("PYSCF_MG_BENCH_MESH", 25);
    let reps = env_usize("PYSCF_MG_BENCH_REPS", 3);
    let cell = cell_at(mesh);
    let nao = cell.mol.nao_nr;
    let ni = MultiGridNumInt2::new();

    println!("cell si gth-szv, mesh {mesh}^3, nao {nao}, warm min over {reps}");
    println!(
        "{:>6}  {:>7}  {:>12}  {:>12}  {:>12}",
        "nk", "nkpts", "rho_g ms", "nr_rks ms", "ns/kpt(rho)"
    );

    let mut first: Option<(usize, f64, f64)> = None;
    for nk in [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]] {
        let kpts = make_kpts_default(&cell, nk).expect("k-mesh");
        let nkpts = kpts.len();
        let dms = model_kdms(nao, nkpts, 0x0BAD_F00D);

        // Warm: the first call builds the task list and uploads the geometry.
        let _ = ni
            .eval_rho_g_kpts(&cell, &dms, &kpts)
            .expect("warm rho_g_kpts");
        let _ = ni
            .nr_rks_kpts(&cell, "lda,vwn", &dms, &kpts, None)
            .expect("warm nr_rks_kpts");

        let mut fwd = f64::INFINITY;
        let mut full = f64::INFINITY;
        for _ in 0..reps {
            let t = Instant::now();
            let _ = ni.eval_rho_g_kpts(&cell, &dms, &kpts).expect("rho_g_kpts");
            fwd = fwd.min(t.elapsed().as_secs_f64() * 1e3);

            let t = Instant::now();
            let _ = ni
                .nr_rks_kpts(&cell, "lda,vwn", &dms, &kpts, None)
                .expect("nr_rks_kpts");
            full = full.min(t.elapsed().as_secs_f64() * 1e3);
        }

        // Per-k marginal cost of the forward against the 1-k row. The
        // collocation is flat in nkpts, so this is an upper bound on the
        // serial contractions — it also carries PhaseTable/expand/contract_v.
        let per_k = first.map_or(0.0, |(n0, f0, _)| {
            (fwd - f0) * 1e6 / (nkpts.saturating_sub(n0).max(1)) as f64
        });
        println!(
            "{:>6}  {:>7}  {:>12.0}  {:>12.0}  {:>12.0}",
            format!("{}{}{}", nk[0], nk[1], nk[2]),
            nkpts,
            fwd,
            full,
            per_k
        );
        if first.is_none() {
            first = Some((nkpts, fwd, full));
        }
    }

    if let Some((n0, f0, u0)) = first {
        println!(
            "\nread the slope, not the absolutes: the collocation is flat in nkpts \
             (multigrid::kpts module doc), so growth over the {n0}-kpt row \
             (rho_g {f0:.0} ms, nr_rks {u0:.0} ms) is EVERYTHING that scales with \
             k. That is an UPPER BOUND on the serial contractions, not their cost \
             -- PhaseTable::new, KDmP::expand and contract_v_kpts are O(nkpts) \
             too. A small slope means the lever is not worth taking."
        );
    }
}
