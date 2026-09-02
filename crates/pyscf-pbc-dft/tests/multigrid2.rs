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

use pyscf_pbc_dft::multigrid::pair::{
    PairTaskList, block_slots, build_pair_level_tables, build_pair_task_list, grid_blocks,
    pairlevel_rho,
};
use pyscf_pbc_dft::multigrid::tasks::build_pshells;
use pyscf_pbc_dft::multigrid::tasks::pshell_cart_powers;
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
    let vj =
        pyscf_pbc_df::fft_jk::get_j_kpts(&df, &[vec![dm_c]], 1, &[[0.0, 0.0, 0.0]], None, None)
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
            task_list
                .level_pairs
                .iter()
                .map(|p| p.len())
                .collect::<Vec<_>>()
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

/// The memory shape that exit-137'd 17-12's first run: the dense
/// `(kernel slots × ngrids)` collocation buffer. Every level's table is
/// built, its block partition and per-block slot selection are walked
/// exactly as the drivers walk them, and the largest per-launch working
/// set is checked against a hard ceiling — with the per-level sizes
/// printed so the summary can record what the full dense buffer WOULD
/// have been and how much of it the block screening actually evaluates.
#[test]
fn pair_level_tables_stream_under_budget() {
    // One launch's inputs: selected slots (24 B each) + points (24 B each)
    // + outputs (8 B per point or per slot). Anything near this would be
    // a regression back toward the dense shape.
    const LAUNCH_CEILING_BYTES: usize = 256 << 20;
    for (name, cell) in [("diamond", small_diamond()), ("si", small_silicon())] {
        let decon = build_pshells(&cell).expect("decontract");
        let task_list = build_pair_task_list(&cell, &decon).expect("pair task list");
        let tables = build_pair_level_tables(&cell, &decon, &task_list).expect("level tables");
        assert_eq!(tables.len(), task_list.levels.len());
        let mut dense_total = 0usize;
        for (i, lv) in tables.iter().enumerate() {
            let Some(lv) = lv else {
                assert!(
                    task_list.level_pairs[i].is_empty(),
                    "{name}: level {i} table missing"
                );
                continue;
            };
            let nterms = lv.nterms;
            let nslots = lv.nslots();
            let nk = lv.nkslots();
            let dense = nk * lv.ngrids * 8;
            dense_total += dense;
            let blocks = grid_blocks(lv);
            let npts_total: usize = blocks.iter().map(|b| b.points.len()).sum();
            assert_eq!(
                npts_total, lv.ngrids,
                "{name}: level {i} blocks do not tile the mesh"
            );
            let mut evals = 0usize;
            let mut max_launch = 0usize;
            let mut max_sel = 0usize;
            for b in &blocks {
                let sel = block_slots(lv, b);
                evals += sel.len() * b.points.len();
                max_sel = max_sel.max(sel.len());
                max_launch = max_launch.max(sel.len() * 32 + b.points.len() * 32);
            }
            println!(
                "{name}: level {i} mesh={:?} ngrids={} kernel-instances={} terms={nterms} slots={nslots} \
                 kernel-slots={nk} dense={:.2} GiB  blocks={} max-selected={max_sel} \
                 evaluated={:.1}% of dense  max-launch={:.1} MiB",
                lv.mesh,
                lv.ngrids,
                lv.instance_alpha.len(),
                dense as f64 / (1u64 << 30) as f64,
                blocks.len(),
                100.0 * evals as f64 / (nk * lv.ngrids) as f64,
                max_launch as f64 / (1u64 << 20) as f64,
            );
            assert!(
                nterms > 0 && nslots >= nterms && nk >= nterms,
                "{name}: level {i} empty table"
            );
            // Periodic wrap: kernel slots must outnumber fused terms — a
            // table with exactly one image per term is the unwrapped bug.
            assert!(nk > nterms, "{name}: level {i} has no wrap images at all");
            assert!(
                max_launch <= LAUNCH_CEILING_BYTES,
                "{name}: level {i} launch working set {max_launch} B exceeds the ceiling"
            );
        }
        println!(
            "{name}: full dense buffers would total {:.2} GiB; no launch materialises them",
            dense_total as f64 / (1u64 << 30) as f64
        );
        assert!(
            tables.last().unwrap().is_some(),
            "{name}: finest level owns no pair"
        );
    }
}

// ---------------------------------------------------------------------------
// Task 2/3 — density normalisation: ∫ rho dr == Tr(dm.S)
// ---------------------------------------------------------------------------

#[test]
fn int_rho_matches_tr_dm_s_v2() {
    let ni = MultiGridNumInt2::new();
    // Both cells are measured BEFORE either is asserted, so one number never
    // hides the other in the log.
    let mut rows = Vec::new();
    for (name, cell) in [("si", small_silicon()), ("diamond", small_diamond())] {
        let nao = cell.mol.nao_nr;
        let dm = random_symmetric_dm(nao, 0x9E37_79B9);
        let want = trace_dm_s(&cell, &dm);

        let t0 = Instant::now();
        let rho_g = ni.eval_rho_g(&cell, &dm).expect("eval_rho_g");
        let secs = t0.elapsed().as_secs_f64();
        let rho_r = pyscf_pbc_tools::ifft(&rho_g, cell.mesh).expect("ifft");
        let got: f64 = rho_r.re.iter().sum();
        let diff = (got - want).abs();
        println!(
            "{name}: Tr(dm.S) = {want:.12e}  int(rho) = {got:.12e}  |diff| = {diff:.3e}  \
             (eval_rho_g {secs:.1}s)"
        );
        rows.push((name, diff));
    }
    for (name, diff) in rows {
        // si (gth-szv, no core) is the clean gate: every primitive is
        // resolved by the 25^3 mesh. diamond (sto-3g) carries a 1s core
        // (alpha 71.6) the 25^3 finest level CANNOT resolve — and v2 pins
        // its finest level to `cell.mesh` (multigrid_pair.py:59-78), unlike
        // v1, which refines past it. That is a property of the fixture +
        // upstream's v2 definition, not of the port; it is gated at the
        // level the unresolved-core quadrature allows and the number is
        // recorded in 17-12-SUMMARY.md.
        let tol = if name == "si" { 1e-6 } else { 1e-2 };
        assert!(
            diff < tol,
            "{name}: |int(rho) - Tr(dm.S)| = {diff:.3e}, expected < {tol:.0e}"
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
        let out = ni
            .nr_rks(&cell, "lda,vwn", &dm)
            .expect("multigrid v2 nr_rks");

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

// ---------------------------------------------------------------------------
// Task 1/2 attribution gates — the level ladder and the fused collocation,
// each isolated from the other
// ---------------------------------------------------------------------------

/// v2's `rho(G)` against v1's component by component, and v2 again with
/// EVERY pair forced onto the finest level (the level ladder out of the
/// picture, only the fused collocation + screening left). The two v2 runs
/// must agree with each other AND with v1 to the screening floor — this
/// is what separated "ladder bug" from "collocation bug" during 17-12's
/// residual hunt, and it stays as the gate on both.
#[test]
fn v2_rho_g_matches_v1_with_and_without_the_ladder() {
    let v1 = MultiGridNumInt::new();
    let v2 = MultiGridNumInt2::new();
    for (name, cell) in [("si", small_silicon()), ("diamond", small_diamond())] {
        let nao = cell.mol.nao_nr;
        let dm = random_symmetric_dm(nao, 0x9E37_79B9);
        let want = trace_dm_s(&cell, &dm);
        let ng = cell.mesh[0] * cell.mesh[1] * cell.mesh[2];

        let r1 = v1.eval_rho_g(&cell, &dm).expect("v1");
        let r2 = v2.eval_rho_g(&cell, &dm).expect("v2");
        let decon = build_pshells(&cell).expect("decontract");
        let tl = build_pair_task_list(&cell, &decon).expect("task list");
        let all: Vec<(usize, usize)> = tl.level_pairs.iter().flatten().copied().collect();
        let single = PairTaskList {
            levels: vec![*tl.levels.last().unwrap()],
            level_pairs: vec![all],
        };
        let r2s = v2
            .eval_rho_g_with_task_list(&cell, &single, &dm)
            .expect("v2 single level");

        let report = |label: &str, r: &pyscf_algebra::CTensor| -> f64 {
            let mut max = 0.0f64;
            let mut gmax = 0usize;
            for g in 0..ng {
                let d = ((r.re[g] - r1.re[g]).powi(2) + (r.im[g] - r1.im[g]).powi(2)).sqrt();
                if d > max {
                    max = d;
                    gmax = g;
                }
            }
            let iz = gmax % cell.mesh[2];
            let iy = (gmax / cell.mesh[2]) % cell.mesh[1];
            let ix = gmax / (cell.mesh[2] * cell.mesh[1]);
            println!(
                "{name}: {label}: int(rho)={:.10e} (Tr(dm.S)={want:.10e}, diff {:.3e})  \
                 max|rho_g - v1| = {max:.3e} at mesh index ({ix},{iy},{iz})  |v1(0)|={:.3e}",
                r.re[0],
                (r.re[0] - want).abs(),
                r1.re[0]
            );
            max
        };
        println!(
            "{name}: v1 int(rho)={:.10e} (diff {:.3e})",
            r1.re[0],
            (r1.re[0] - want).abs()
        );
        let d_ladder = report("v2 ladder", &r2);
        let d_single = report("v2 single finest level", &r2s);
        assert!(
            d_ladder < 1e-5,
            "{name}: v2 ladder max|rho_g - v1| = {d_ladder:.3e}"
        );
        assert!(
            d_single < 1e-5,
            "{name}: v2 single-level max|rho_g - v1| = {d_single:.3e}"
        );
    }
}

/// Per-pair gate: the fused table's `rho` for ONE ordered pshell pair
/// `(p, q)` against a brute-force `Σ_{L1,L2} χ_p(r-A-L1) χ_q(r-B-L2)` on
/// sample points — the pair form of 17-11 Task 2's per-`l` tests, and the
/// test that located BOTH of 17-12's collocation defects (the `[0,1)`
/// wrap-box assumption and the polynomial-blind image pre-screen): it
/// names the offending pair by `(l, alpha)`. Silicon: its diffuse
/// `alpha = 0.0576` p functions are the hardest case in either fixture.
#[test]
fn fused_pairs_match_brute_force_periodic_products() {
    let cell = small_silicon();
    let nao = cell.mol.nao_nr;
    let dm = random_symmetric_dm(nao, 0x9E37_79B9);
    let decon = build_pshells(&cell).expect("decontract");
    let dm_p = pyscf_pbc_dft::multigrid::colloc::expand_dm(&decon, &dm);
    let tl = build_pair_task_list(&cell, &decon).expect("task list");
    let finest = *tl.levels.last().unwrap();
    let grids =
        pyscf_pbc_dft::gen_grid::PeriodicGrids::uniform(&cell, Some(finest.mesh)).expect("grid");
    let coords = grids.coords().expect("coords").to_vec();
    let ng = coords.len();
    // 48 sample points spread over the mesh.
    let samples: Vec<usize> = (0..48).map(|i| (i * 7919) % ng).collect();
    let n = decon.pshells.len();
    let mut worst: Vec<(f64, f64, usize, usize)> = Vec::new();
    let mut total_fused = vec![0.0f64; samples.len()];
    let mut total_brute = vec![0.0f64; samples.len()];
    for pi in 0..n {
        for pj in 0..n {
            let single = PairTaskList {
                levels: vec![finest],
                level_pairs: vec![vec![(pi, pj)]],
            };
            let tables = build_pair_level_tables(&cell, &decon, &single).expect("tables");
            let lv = tables[0].as_ref().expect("table");
            let rho = pairlevel_rho(lv, &decon, &dm_p).expect("rho");
            let p = &decon.pshells[pi];
            let q = &decon.pshells[pj];
            let lp =
                pyscf_pbc_gto::lattice::get_lattice_ls(&cell, Some(p.rcut.max(1e-6)), None, false)
                    .expect("ls");
            let lq =
                pyscf_pbc_gto::lattice::get_lattice_ls(&cell, Some(q.rcut.max(1e-6)), None, false)
                    .expect("ls");
            let pw_i = pshell_cart_powers(p.l);
            let pw_j = pshell_cart_powers(q.l);
            let mut maxd = 0.0f64;
            let mut maxv = 0.0f64;
            for (si, &g) in samples.iter().enumerate() {
                let r = coords[g];
                // periodic AO component values
                let phi = |c: [f64; 3],
                           alpha: f64,
                           ls: &[[f64; 3]],
                           pw: &[(u32, u32, u32)]|
                 -> Vec<f64> {
                    let mut out = vec![0.0f64; pw.len()];
                    for l in ls {
                        let d = [r[0] - c[0] - l[0], r[1] - c[1] - l[1], r[2] - c[2] - l[2]];
                        let e = (-alpha * (d[0] * d[0] + d[1] * d[1] + d[2] * d[2])).exp();
                        for (k, &(ax, ay, az)) in pw.iter().enumerate() {
                            out[k] += d[0].powi(ax as i32)
                                * d[1].powi(ay as i32)
                                * d[2].powi(az as i32)
                                * e;
                        }
                    }
                    out
                };
                let fi = phi(p.center, p.alpha, &lp, &pw_i);
                let fj = phi(q.center, q.alpha, &lq, &pw_j);
                let mut brute = 0.0f64;
                for (ci, a) in fi.iter().enumerate() {
                    for (cj, b) in fj.iter().enumerate() {
                        brute += dm_p[(p.cart_ao0 + ci) * decon.nao_p + q.cart_ao0 + cj] * a * b;
                    }
                }
                let d = (rho[g] - brute).abs();
                maxd = maxd.max(d);
                maxv = maxv.max(brute.abs());
                total_fused[si] += rho[g];
                total_brute[si] += brute;
            }
            worst.push((maxd, maxv, pi, pj));
        }
    }
    worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    println!("worst 12 ordered pairs by max|fused - brute| over the sample points:");
    for &(d, v, pi, pj) in worst.iter().take(12) {
        let p = &decon.pshells[pi];
        let q = &decon.pshells[pj];
        println!(
            "  ({pi:2},{pj:2}) l=({},{}) alpha=({:.4},{:.4}) rcut=({:.2},{:.2}) max|diff|={d:.3e} max|brute|={v:.3e}",
            p.l, q.l, p.alpha, q.alpha, p.rcut, q.rcut
        );
    }
    let tot: f64 = total_fused
        .iter()
        .zip(&total_brute)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    println!("max over sample points of |Σ_pairs fused - Σ_pairs brute| = {tot:.3e}");
    let worst_pair = worst[0].0;
    assert!(
        worst_pair < 1e-7,
        "worst single pair max|fused - brute| = {worst_pair:.3e}"
    );
    assert!(
        tot < 1e-6,
        "summed over pairs max|fused - brute| = {tot:.3e}"
    );
}
