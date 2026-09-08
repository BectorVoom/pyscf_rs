//! Session-6 instrument for the v2 multigrid pair kernels — the batch
//! geometry per level and the isolated per-direction kernel wall, without a
//! whole SCF around them.
//!
//! ```text
//! PYSCF_MG_BENCH_MESH=25 PYSCF_MG_BENCH_REPS=3 cargo test --release -p pyscf-pbc-dft \
//!     --test mg_pair_bench -- --ignored --nocapture
//! ```
//!
//! A test file rather than an example so it shares the gate binaries' feature
//! shape (an example re-unifies features and rebuilds the libxc tree).
//!
//! Prints, per non-empty level: blocks / points / instance occurrences /
//! distinct instances / concatenated slots / kernel slots, the number of
//! distinct term SETS (instances that are images of one `(pair, L)` share
//! the same monomial-and-term sequence), the run-length histogram of
//! consecutive same-set occurrences inside a block (what a vector-over-
//! instances reverse kernel can group), the widest monomial power, and the
//! forward / reverse wall over `reps` repetitions on the production
//! (batched, resident) route.

use std::time::Instant;

use pyscf_pbc_dft::multigrid::pair::{
    build_pair_level_tables, build_pair_task_list, pairlevel_pass2_with, pairlevel_rho_with,
};
use pyscf_pbc_dft::multigrid::tasks::build_pshells;

#[test]
#[ignore = "an instrument, not a gate — run with --ignored --nocapture"]
fn mg_pair_bench() {
    let mesh: usize = std::env::var("PYSCF_MG_BENCH_MESH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(25);
    let reps: usize = std::env::var("PYSCF_MG_BENCH_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let mut cell = pyscf_pbc_gto::test_systems::si();
    cell.mesh = [mesh; 3];

    let t0 = Instant::now();
    let decon = build_pshells(&cell).expect("build_pshells");
    let task_list = build_pair_task_list(&cell, &decon).expect("task list");
    let tables = build_pair_level_tables(&cell, &decon, &task_list).expect("tables");
    println!("tables built in {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);

    // A deterministic symmetric density.
    let nao = cell.mol.nao_nr;
    let mut state = 0x0BAD_F00D_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let mut dm = vec![0.0f64; nao * nao];
    for i in 0..nao {
        for j in 0..=i {
            let v = next();
            dm[i * nao + j] = v;
            dm[j * nao + i] = v;
        }
    }
    let dm_p = pyscf_pbc_dft::multigrid::colloc::expand_dm(&decon, &dm);

    for (l, lv) in tables.iter().enumerate() {
        let Some(lv) = lv.as_ref() else { continue };
        let nblocks = lv.blocks.len();
        let nb = lv.batches.len();
        let npoints: usize = lv.batches.iter().map(|b| b.npoints).sum();
        let nocc: usize = lv.batches.iter().map(|b| b.ninstances).sum();
        let nuinst: usize = lv.batches.iter().map(|b| b.nuinstances).sum();
        let nslots: usize = lv.batches.iter().map(|b| b.nslots).sum();
        let geometry_mb: f64 = lv
            .batches
            .iter()
            .map(|b| b.geometry_bytes as f64)
            .sum::<f64>()
            / 1e6;
        println!(
            "level {l}: mesh {:?} blocks {nblocks} chunks {nb} points {npoints} occurrences {nocc} \
             distinct {nuinst} concat-slots {nslots} kslots {} level-instances {} terms {}",
            lv.mesh,
            lv.nkslots(),
            lv.instance_alpha.len(),
            lv.nterms,
        );
        // Per level-instance: its kslot range and term sequence.
        let ninst = lv.instance_alpha.len();
        let mut kslot0 = vec![0u32; ninst + 1];
        for &i in &lv.kslot_instance {
            kslot0[i as usize + 1] += 1;
        }
        for i in 0..ninst {
            kslot0[i + 1] += kslot0[i];
        }
        let seq = |i: usize| &lv.kslot_term[kslot0[i] as usize..kslot0[i + 1] as usize];
        // Distinct sets = distinct term sequences over instances (consecutive
        // images share one).
        let mut nsets = 0usize;
        let mut set_slots = 0usize;
        for i in 0..ninst {
            if i == 0 || seq(i) != seq(i - 1) {
                nsets += 1;
                set_slots += seq(i).len();
            }
        }
        let maxpow = lv.kslot_pow.iter().copied().max().unwrap_or(0);
        println!(
            "  sets {nsets} set-slots {set_slots} max-power {maxpow} level-sets {} \
             (resident geometry {geometry_mb:.0} MB; reverse output {:.0} MB per level)",
            lv.set_off.len().saturating_sub(1),
            nslots as f64 * 8.0 / 1e6,
        );
        // Run-length histogram of consecutive same-set occurrences in a block.
        let mut hist = [0usize; 9]; // 1,2,3,4,5-7,8-15,16-31,32-63,64+
        let mut runs = 0usize;
        let mut occ_in_runs_ge4 = 0usize;
        let mut occ_in_runs_ge8 = 0usize;
        for bl in &lv.batches {
            let b = bl.host_batch(lv);
            for blk in 0..b.nblocks() {
                let i0 = b.block_inst0[blk] as usize;
                let i1 = b.block_inst0[blk + 1] as usize;
                let mut run = 0usize;
                let mut prev: Option<&[u32]> = None;
                let flush = |run: usize,
                             hist: &mut [usize; 9],
                             runs: &mut usize,
                             g4: &mut usize,
                             g8: &mut usize| {
                    if run == 0 {
                        return;
                    }
                    *runs += 1;
                    let k = match run {
                        1 => 0,
                        2 => 1,
                        3 => 2,
                        4 => 3,
                        5..=7 => 4,
                        8..=15 => 5,
                        16..=31 => 6,
                        32..=63 => 7,
                        _ => 8,
                    };
                    hist[k] += 1;
                    if run >= 4 {
                        *g4 += run;
                    }
                    if run >= 8 {
                        *g8 += run;
                    }
                };
                for occ in i0..i1 {
                    // The occurrence's level instance: the distinct row's
                    // alpha/centre identify it; find via the slot list instead.
                    let u = b.inst_ref[occ] as usize;
                    let k0 = b.instance_kslot0[u] as usize;
                    let li = lv.kslot_instance[k0] as usize;
                    let sq = seq(li);
                    if prev.is_some_and(|p| p == sq) {
                        run += 1;
                    } else {
                        flush(
                            run,
                            &mut hist,
                            &mut runs,
                            &mut occ_in_runs_ge4,
                            &mut occ_in_runs_ge8,
                        );
                        run = 1;
                        prev = Some(sq);
                    }
                }
                flush(
                    run,
                    &mut hist,
                    &mut runs,
                    &mut occ_in_runs_ge4,
                    &mut occ_in_runs_ge8,
                );
            }
        }
        println!(
            "  same-set runs {runs}: len1 {} len2 {} len3 {} len4 {} 5-7 {} 8-15 {} 16-31 {} 32-63 {} 64+ {}; \
             occurrences in runs>=4: {:.1}%  >=8: {:.1}%",
            hist[0],
            hist[1],
            hist[2],
            hist[3],
            hist[4],
            hist[5],
            hist[6],
            hist[7],
            hist[8],
            100.0 * occ_in_runs_ge4 as f64 / nocc.max(1) as f64,
            100.0 * occ_in_runs_ge8 as f64 / nocc.max(1) as f64,
        );

        // Timing, production route. First call uploads the geometry.
        let w: Vec<f64> = (0..lv.ngrids)
            .map(|g| ((g * 7 + 1) as f64).sin() * 1e-3)
            .collect();
        let mut fwd = Vec::new();
        let mut rev = Vec::new();
        let mut rho_sum = 0.0;
        let mut vp_sum = 0.0;
        for r in 0..=reps {
            let t = Instant::now();
            let rho = pairlevel_rho_with(lv, &decon, &dm_p, true).expect("rho");
            let f = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let mut v_p = vec![0.0f64; decon.nao_p * decon.nao_p];
            pairlevel_pass2_with(lv, &decon, &w, &mut v_p, true).expect("pass2");
            let b = t.elapsed().as_secs_f64() * 1e3;
            if r == 0 {
                println!("  first call (uploads): forward {f:.0} ms reverse {b:.0} ms");
                rho_sum = rho.iter().sum();
                vp_sum = v_p.iter().sum();
            } else {
                fwd.push(f);
                rev.push(b);
            }
        }
        let min = |v: &[f64]| v.iter().copied().fold(f64::INFINITY, f64::min);
        println!(
            "  warm min over {reps}: forward {:.0} ms reverse {:.0} ms   (checksums {:.17e} {:.17e})",
            min(&fwd),
            min(&rev),
            rho_sum,
            vp_sum
        );
    }
}
