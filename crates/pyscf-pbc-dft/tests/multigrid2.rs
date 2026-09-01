//! Plan 17-12 — multigrid (v2, `MultiGridNumInt2`) gate: Task 1 (pair task
//! list sanity), Task 2/3 (density normalisation, the same TASK-3-shaped
//! identity 17-11's `int_rho_matches_tr_dm_s` already established, now for
//! the pair-fused engine), Task 5 (Gate E vs reference `numint`, the
//! v1-vs-v2 floor, and the speed table).
//!
//! Same posture as `multigrid.rs`'s module doc: multigrid is a DIFFERENT
//! quadrature from the reference `numint`, not a tighter-tolerance target
//! (17-CONTEXT §2.2 Gate E). v2 additionally carries this port's OWN
//! task-list-membership deviation from upstream's literal C
//! `build_task_list` (`crate::multigrid::pair`'s module doc) — so v2's own
//! floor against the reference route is measured HERE, independently, not
//! assumed to equal 17-01's upstream-vs-upstream number.

mod common;

use pyscf_pbc_dft::multigrid::pair::build_pair_task_list;
use pyscf_pbc_dft::multigrid::tasks::build_pshells;
use pyscf_pbc_dft::multigrid::{MultiGridNumInt, MultiGridNumInt2};
use pyscf_pbc_dft::numint::KNumInt;
use pyscf_pbc_gto::Cell;
use std::time::Instant;

fn small_diamond() -> Cell {
    let mut c = common::diamond();
    c.mesh = [25, 25, 25];
    c
}

fn small_silicon() -> Cell {
    let mut c = common::silicon();
    c.mesh = [25, 25, 25];
    c
}

fn trace_dm_s(cell: &Cell, dm: &[f64]) -> f64 {
    let out = cell
        .pbc_intor("int1e_ovlp", &[[0.0, 0.0, 0.0]], None, 0)
        .expect("int1e_ovlp");
    let nao = out.ni;
    let s = &out.kmats[0].re;
    let mut acc = 0.0f64;
    for i in 0..nao {
        for j in 0..nao {
            acc += dm[i * nao + j] * s[i * nao + j];
        }
    }
    acc
}

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

fn reference_get_j(cell: &Cell, dm: &[f64]) -> Vec<f64> {
    let nao = cell.mol.nao_nr;
    let dm_c = pyscf_algebra::CTensor::from_planes(dm.to_vec(), vec![0.0; nao * nao]);
    let df = pyscf_pbc_df::Fftdf::new(cell.clone(), &[[0.0, 0.0, 0.0]]).expect("Fftdf");
    let vj = pyscf_pbc_df::fft_jk::get_j_kpts(&df, &[vec![dm_c]], 1, &[[0.0, 0.0, 0.0]], None, None)
        .expect("get_j_kpts");
    vj[0][0].re.clone()
}

// ---------------------------------------------------------------------------
// Task 1 — the pair task list
// ---------------------------------------------------------------------------

#[test]
fn pair_task_list_is_sane() {
    for (name, cell) in [("diamond", small_diamond()), ("si", small_silicon())] {
        let decon = build_pshells(&cell).expect("decontract");
        let task_list = build_pair_task_list(&cell, &decon).expect("pair task list");
        assert_eq!(
            task_list.levels.len(),
            pyscf_pbc_dft::multigrid::pair::NTASKS,
            "{name}: expected NTASKS levels"
        );
        assert_eq!(
            task_list.levels.last().unwrap().mesh,
            cell.mesh,
            "{name}: last level mesh must equal cell.mesh exactly"
        );
        // Cutoffs strictly increasing (geometric ladder).
        for w in task_list.levels.windows(2) {
            assert!(
                w[1].cutoff > w[0].cutoff,
                "{name}: level cutoffs must be strictly increasing"
            );
        }
        let total_pairs: usize = task_list.level_pairs.iter().map(|p| p.len()).sum();
        let npshell = decon.pshells.len();
        println!(
            "{name}: npshell={npshell}, total pairs (screened, of {} possible) = {total_pairs}, \
             per-level counts = {:?}",
            npshell * npshell,
            task_list.level_pairs.iter().map(|p| p.len()).collect::<Vec<_>>()
        );
        assert!(total_pairs > 0, "{name}: no pair survived screening at all");
        // Every self-pair (pi==pj, L=0) must survive screening — that is the
        // ordinary AO self-overlap, always maximal K=1.
        let self_pairs = (0..npshell)
            .filter(|&i| task_list.level_pairs.iter().any(|lv| lv.contains(&(i, i))))
            .count();
        assert_eq!(
            self_pairs, npshell,
            "{name}: every pshell's self-pair must survive screening"
        );
    }
}

// ---------------------------------------------------------------------------
// Task 2/3 — density normalisation: ∫ rho dr == Tr(dm.S)
// ---------------------------------------------------------------------------

#[test]
fn int_rho_matches_tr_dm_s_v2() {
    let ni = MultiGridNumInt2::new();
    for (name, cell) in [("diamond", small_diamond()), ("si", small_silicon())] {
        let nao = cell.mol.nao_nr;
        let dm = random_symmetric_dm(nao, 0x9E37_79B9);
        let want = trace_dm_s(&cell, &dm);

        let rho_g = ni.eval_rho_g(&cell, &dm).expect("eval_rho_g");
        let rho_r = pyscf_pbc_tools::ifft(&rho_g, cell.mesh).expect("ifft");
        let got: f64 = rho_r.re.iter().sum();
        let diff = (got - want).abs();
        println!("{name}: Tr(dm.S) = {want:.12e}  int(rho) = {got:.12e}  |diff| = {diff:.3e}");
        assert!(
            diff < 1e-6,
            "{name}: |int(rho) - Tr(dm.S)| = {diff:.3e}, expected < 1e-6 \
             (looser than v1's 1e-9 — v2's task-list level-membership \
             heuristic is a documented reformulation, not a literal port, \
             see crate::multigrid::pair's module doc)"
        );
    }
}

// ---------------------------------------------------------------------------
// Task 5 — Gate E: MultiGridNumInt2 vs the reference numint / FFTDF, and
// the v1-vs-v2 floor
// ---------------------------------------------------------------------------

#[test]
fn gate_e_get_j_vs_reference_v2() {
    let ni = MultiGridNumInt2::new();
    for (name, cell) in [("diamond", small_diamond()), ("si", small_silicon())] {
        let nao = cell.mol.nao_nr;
        let dm = random_symmetric_dm(nao, 0xDEAD_BEEF);
        let mine = ni.get_j(&cell, &dm).expect("multigrid v2 get_j");
        let refv = reference_get_j(&cell, &dm);
        let mut max_diff = 0.0f64;
        for i in 0..nao * nao {
            max_diff = max_diff.max((mine[i] - refv[i]).abs());
        }
        println!("{name}: v2 get_j max|diff| vs reference FFTDF = {max_diff:.3e}");
        assert!(
            max_diff < 1e-3,
            "{name}: v2 get_j vs reference max|diff| = {max_diff:.3e}, expected < 1e-3"
        );
    }
}

#[test]
fn gate_e_nr_rks_lda_vs_reference_v2() {
    let ni = MultiGridNumInt2::new();
    for (name, cell) in [("diamond", small_diamond()), ("si", small_silicon())] {
        let nao = cell.mol.nao_nr;
        let dm = random_symmetric_dm(nao, 0xC0FF_EE00);
        let out = ni.nr_rks(&cell, "lda,vwn", &dm).expect("multigrid v2 nr_rks");

        let refni = KNumInt::new(&[[0.0, 0.0, 0.0]]);
        let grids = pyscf_pbc_dft::gen_grid::PeriodicGrids::uniform(&cell, Some(cell.mesh))
            .expect("uniform grid");
        let dm_c = pyscf_algebra::CTensor::from_planes(dm.clone(), vec![0.0; nao * nao]);
        let dms: pyscf_pbc_scf::types::KDms = vec![vec![dm_c]];
        let refout = refni
            .nr_rks(&cell, &grids, "lda,vwn", &dms, 1, None)
            .expect("reference nr_rks");

        let dnelec = (out.nelec - refout.nelec[0]).abs();
        let dexc = (out.exc - refout.excsum[0]).abs();
        println!("{name}: v2 nr_rks(lda,vwn) |d nelec| = {dnelec:.3e}  |d exc| = {dexc:.3e}");
        assert!(dnelec < 1e-3, "{name}: v2 nelec diff {dnelec:.3e}");
        assert!(dexc < 1e-3, "{name}: v2 exc diff {dexc:.3e}");
    }
}

/// **The v1-vs-v2 floor** — 17-CONTEXT/17-01's direct analogue of Phase
/// 14's GDF-vs-RSDF 4.502e-06 inter-route gap: two implementations of the
/// same idea, whose disagreement is a floor neither can converge away.
/// Reported as a ratio against 17-01's measured `get_pp` v1-vs-v2 number
/// where that comparison is valid (see the test's own note on why `get_j`,
/// not `get_pp`, is this PORT's meaningful comparison point).
#[test]
fn v1_vs_v2_gap_reported() {
    let v1 = MultiGridNumInt::new();
    let v2 = MultiGridNumInt2::new();
    for (name, cell) in [("diamond", small_diamond()), ("si", small_silicon())] {
        let nao = cell.mol.nao_nr;
        let dm = random_symmetric_dm(nao, 0x1357_9BDF);
        let j1 = v1.get_j(&cell, &dm).expect("v1 get_j");
        let j2 = v2.get_j(&cell, &dm).expect("v2 get_j");
        let mut max_diff = 0.0f64;
        for i in 0..nao * nao {
            max_diff = max_diff.max((j1[i] - j2[i]).abs());
        }
        println!(
            "{name}: |v1.get_j - v2.get_j| max = {max_diff:.3e}  \
             (this port's own v1-vs-v2 definitional floor — NOT comparable \
             to 17-01's upstream get_pp v1-vs-v2 number, 2.4e-8/1.47e-7, \
             because THIS port's get_pp/get_nuc delegate identically to \
             AFTDF for both v1 and v2 — see crate::multigrid::pp's module \
             doc — so get_j, which actually exercises the two DIFFERENT \
             collocation engines in this port, is the meaningful \
             comparison point here)"
        );
        // This is a definitional floor, not a convergence residual — report
        // it, do not fail the build on its magnitude (same posture 17-01's
        // README takes for the upstream number).
        assert!(max_diff.is_finite());
    }
}

/// The speed half of Gate E, reported in the SAME table shape as the
/// accuracy numbers above: v2 vs the reference `numint`/FFTDF route, AND v2
/// vs v1's own wall time — the trade-off Phase 18 needs (it is v2, not v1,
/// that `grad/rhf.py`/`grad/uhf.py` require regardless of which is faster).
#[test]
fn gate_e_speed_ratio_reported_v2() {
    let v1 = MultiGridNumInt::new();
    let v2 = MultiGridNumInt2::new();
    for (name, cell) in [("diamond", small_diamond()), ("si", small_silicon())] {
        let nao = cell.mol.nao_nr;
        let dm = random_symmetric_dm(nao, 0x2468_ACE0);

        let t0 = Instant::now();
        let _ = v2.get_j(&cell, &dm).expect("v2 get_j");
        let t_v2 = t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        let _ = v1.get_j(&cell, &dm).expect("v1 get_j");
        let t_v1 = t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        let _ = reference_get_j(&cell, &dm);
        let t_ref = t0.elapsed().as_secs_f64();

        println!(
            "{name}: get_j wall-clock  reference={t_ref:.4}s  v1={t_v1:.4}s  v2={t_v2:.4}s  \
             ratio(ref/v2)={:.4}x  ratio(v1/v2)={:.4}x   \
             (17-01's upstream v2-vs-reference floor: 0.18x-0.39x)",
            t_ref / t_v2.max(1e-9),
            t_v1 / t_v2.max(1e-9)
        );
    }
}

// ---------------------------------------------------------------------------
// D-PBC-17 — thread-count bit-identity
// ---------------------------------------------------------------------------

#[test]
fn eval_rho_g_is_bit_identical_across_thread_counts_v2() {
    let ni = MultiGridNumInt2::new();
    let cell = small_diamond();
    let nao = cell.mol.nao_nr;
    let dm = random_symmetric_dm(nao, 0x0BAD_F00D);

    let mut reference: Option<pyscf_algebra::CTensor> = None;
    for &n in &[1usize, 2, 3, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .expect("thread pool");
        let out = pool
            .install(|| ni.eval_rho_g(&cell, &dm))
            .expect("eval_rho_g");
        match &reference {
            None => reference = Some(out),
            Some(r) => {
                assert_eq!(out.re, r.re, "RAYON threads={n}: v2 rho(G).re diverged");
                assert_eq!(out.im, r.im, "RAYON threads={n}: v2 rho(G).im diverged");
            }
        }
    }
}
