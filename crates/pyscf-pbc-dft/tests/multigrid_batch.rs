//! **M-03 — one launch per level is bit-identical to one launch per block.**
//!
//! Plan item M-03 of
//! `.planning/pbc/KUKS-KSYMM-MULTIGRID-OPTIMISATION-PLAN.md` §2.3.3.
//!
//! The v2 driver streams each level's mesh in ~5³ spatial blocks and used to
//! issue one kernel launch per block per direction — 125 launches at
//! `mesh = 25³`, each uploading seven buffers and reading one back. 17-12
//! attributed its first streamed version's 130 s → 7-9 s to per-launch buffer
//! copies and left "batched launches" as its carry-over #3. M-03 concatenates
//! the blocks' tables so one launch covers the level, following
//! `11_launch_overhead_and_transfers.md` §5.
//!
//! # Why bit-identity, not a tolerance
//!
//! M-03 does not change any arithmetic. Each lane runs the same inner loops
//! over the same slot list in the same table order; the per-point sum is still
//! sequential; every output is still written by exactly one lane; and the host
//! still folds the integrate results in block-major, within-block-table order.
//! So the two routes must agree in every bit, and a tolerance here would hide
//! precisely the indexing mistake this restructuring risks — a slot attributed
//! to the wrong block, or an instance range off by one.
//!
//! Both routes run **in one process, on one table**, which is the strongest
//! form of this comparison: nothing differs except the launch geometry.
//!
//! # The budget fallback is exercised too
//!
//! `build_batched_level` returns `None` above `BATCH_BUDGET_BYTES`, in which
//! case the streaming path runs unchanged. `the_batch_is_actually_built`
//! asserts the fixture is on the batched side, so the comparison is not
//! vacuously comparing the streaming path with itself.

mod common;

use pyscf_pbc_dft::multigrid::pair::{
    build_pair_level_tables, build_pair_task_list, pairlevel_pass2_with, pairlevel_rho_with,
};
use pyscf_pbc_dft::multigrid::tasks::build_pshells;
use pyscf_pbc_gto::Cell;

const MESH: [usize; 3] = [25, 25, 25];

fn small_silicon() -> Cell {
    let mut c = common::silicon();
    c.mesh = MESH;
    c
}

fn small_diamond() -> Cell {
    let mut c = common::diamond();
    c.mesh = MESH;
    c
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
    for v in dm.iter_mut() {
        *v = next();
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

/// A deterministic real-space weight field for the reverse direction — not
/// constant, so a mis-permuted `weight` would show up rather than cancel.
fn model_weight(ngrids: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..ngrids)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        })
        .collect()
}

fn same_bits(what: &str, a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len(), "{what}: length differs");
    let mut worst = 0.0_f64;
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        worst = worst.max((x - y).abs());
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{what}[{i}]: batched {x:.17e} vs streamed {y:.17e} (delta {:e}). \
             M-03 changes only the launch geometry, so any difference is an \
             indexing defect — a slot attributed to the wrong block, or an \
             instance range off by one — not a tolerance question.",
            x - y
        );
    }
    println!("{what}: bit-identical over {} values (max |d| = {worst:e})", a.len());
}

/// The comparison itself, for one cell.
fn compare(name: &str, cell: &Cell) {
    let decon = build_pshells(cell).expect("build_pshells");
    let task_list = build_pair_task_list(cell, &decon).expect("build_pair_task_list");
    let tables = build_pair_level_tables(cell, &decon, &task_list).expect("build tables");
    let dm = random_symmetric_dm(cell.mol.nao_nr, 0x0BAD_F00D);
    let dm_p = pyscf_pbc_dft::multigrid::colloc::expand_dm(&decon, &dm);

    let mut levels_with_batch = 0usize;
    for (l, lv) in tables.iter().enumerate() {
        let Some(lv) = lv.as_ref() else { continue };
        if lv.batch.is_some() {
            levels_with_batch += 1;
        }

        // Forward.
        let batched = pairlevel_rho_with(lv, &decon, &dm_p, true).expect("batched rho");
        let streamed = pairlevel_rho_with(lv, &decon, &dm_p, false).expect("streamed rho");
        same_bits(&format!("{name} level {l} rho"), &batched, &streamed);

        // Reverse. `v_p` ACCUMULATES, so both start from zero.
        let w = model_weight(lv.ngrids, 0x5EED_0000 + l as u64);
        let n = decon.nao_p * decon.nao_p;
        let mut vb = vec![0.0f64; n];
        let mut vs = vec![0.0f64; n];
        pairlevel_pass2_with(lv, &decon, &w, &mut vb, true).expect("batched pass2");
        pairlevel_pass2_with(lv, &decon, &w, &mut vs, false).expect("streamed pass2");
        same_bits(&format!("{name} level {l} pass2"), &vb, &vs);
    }
    assert!(
        levels_with_batch > 0,
        "{name}: no level built a batch, so this comparison ran the streaming \
         path against itself — the fixture cannot see M-03 at all"
    );
    println!("{name}: {levels_with_batch} level(s) on the batched path");
}

#[test]
fn batched_and_streamed_launches_agree_bit_for_bit_on_silicon() {
    compare("si", &small_silicon());
}

#[test]
fn batched_and_streamed_launches_agree_bit_for_bit_on_diamond() {
    compare("diamond", &small_diamond());
}

/// The batch's own shape, reported: how many launches M-03 replaces, and how
/// much memory the concatenation costs. This is the number the plan's GATE S
/// ledger wants, and it is printed rather than asserted because the block
/// count is a property of the mesh, not a contract.
#[test]
fn the_batch_is_actually_built() {
    for (name, cell) in [("si", small_silicon()), ("diamond", small_diamond())] {
        let decon = build_pshells(&cell).expect("build_pshells");
        let task_list = build_pair_task_list(&cell, &decon).expect("task list");
        let tables = build_pair_level_tables(&cell, &decon, &task_list).expect("tables");
        for (l, lv) in tables.iter().enumerate() {
            let Some(lv) = lv.as_ref() else { continue };
            let nblocks = lv.blocks.len();
            let streamed_slots: usize = lv.block_sel.iter().map(Vec::len).len();
            match lv.batch.as_ref() {
                Some(bl) => {
                    let b = &bl.batch;
                    let bytes = b.coords.len() * 8
                        + b.point_block.len() * 4
                        + b.slot_pow.len() * 4
                        + b.slot_coef.len() * 8
                        + b.inst_slot0.len() * 4
                        + b.instance_alpha.len() * 8
                        + b.instance_center.len() * 8;
                    println!(
                        "{name} level {l}: mesh {:?}, {nblocks} blocks -> 1 launch per \
                         direction; batch = {} points, {} instances, {} slots, {:.1} MiB",
                        lv.mesh,
                        b.npoints(),
                        b.ninstances(),
                        b.nslots(),
                        bytes as f64 / (1024.0 * 1024.0)
                    );
                    assert_eq!(
                        b.npoints(),
                        lv.ngrids,
                        "{name} level {l}: the blocks must partition the mesh exactly"
                    );
                    assert_eq!(
                        b.inst_slot0.len(),
                        b.ninstances() + 1,
                        "{name} level {l}: the instance prefix table is malformed"
                    );
                    assert_eq!(
                        b.inst_slot0[0], 0,
                        "{name} level {l}: the instance prefix must start at 0"
                    );
                    assert_eq!(
                        *b.inst_slot0.last().expect("non-empty") as usize,
                        b.nslots(),
                        "{name} level {l}: the instance prefix must end at the slot count"
                    );
                }
                None => println!(
                    "{name} level {l}: {nblocks} blocks, {streamed_slots} selections — \
                     over the batch budget, streaming (this is the intended fallback)"
                ),
            }
        }
    }
}
