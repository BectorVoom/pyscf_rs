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

use pyscf_kernels::multigrid_pair::{
    PairSlotBatchDevice, collocate_pairs_integrate_batched, collocate_pairs_rho_batched,
};
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
    println!(
        "{what}: bit-identical over {} values (max |d| = {worst:e})",
        a.len()
    );
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
        if !lv.batches.is_empty() {
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

/// A cell whose basis carries a d shell.
///
/// The batched kernels hold one instance's slot accumulators in a fixed
/// 10-wide register array (`MAX_SLOTS_PER_INSTANCE`), and an instance owns one
/// slot per distinct `(k1,k2,k3)` monomial of its pair — `C(l_p+l_q+3, 3)`,
/// which is exactly 10 at `l_p+l_q = 2` (the `gth-szv` fixtures' widest pair)
/// but 20 at 3 and 35 at 4. Every polarized basis has p·d pairs, so this is
/// not an exotic input.
fn d_shell_cell() -> Cell {
    let h = 2.834589;
    let basis = "He    S\n      3.0   1.0\nHe    P\n      1.2   1.0\nHe    D\n      0.8   1.0\n";
    let mut cell = Cell::build(pyscf_pbc_gto::CellBuildArgs {
        mole: pyscf_gto::MoleBuildArgs {
            atom: pyscf_gto::AtomInput::Tuples(vec![("He".into(), [0.0, 0.0, 0.0])]),
            basis: pyscf_gto::BasisInput::NwchemText(basis.into()),
            unit: pyscf_core::Unit::Bohr,
            ..Default::default()
        },
        a: pyscf_pbc_gto::ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        ..Default::default()
    })
    .expect("d-shell cell must build");
    cell.mesh = MESH;
    cell
}

/// **M-08 — a level the batched kernels cannot hold streams, it does not
/// fail.**
///
/// The 10-slot-per-instance bound used to be guarded only by a `debug_assert!`
/// in the batch builder, so a release build handed the over-wide batch to
/// `validate_batch`, which refused it — and the refusal propagated out of the
/// density evaluation instead of falling back. The whole v2 driver was
/// therefore unusable in release for any basis with d functions.
#[test]
fn a_level_over_the_register_bound_falls_back_to_streaming() {
    let cell = d_shell_cell();
    let decon = build_pshells(&cell).expect("build_pshells");
    let task_list = build_pair_task_list(&cell, &decon).expect("task list");
    let tables = build_pair_level_tables(&cell, &decon, &task_list).expect("tables");
    let dm = random_symmetric_dm(cell.mol.nao_nr, 0x00D_5EED);
    let dm_p = pyscf_pbc_dft::multigrid::colloc::expand_dm(&decon, &dm);

    let mut levels = 0usize;
    for (l, lv) in tables.iter().enumerate() {
        let Some(lv) = lv.as_ref() else { continue };
        levels += 1;
        assert!(
            lv.batches.is_empty(),
            "level {l}: a d-shell pair owns 20 monomials, which cannot be batched"
        );
        // The production entry point, batching requested: it must succeed by
        // streaming rather than error out at launch validation.
        let rho = pairlevel_rho_with(lv, &decon, &dm_p, true).expect("rho must not fail");
        assert!(rho.iter().all(|v| v.is_finite()));

        let w = model_weight(lv.ngrids, 0x5EED_1000 + l as u64);
        let mut v_p = vec![0.0f64; decon.nao_p * decon.nao_p];
        pairlevel_pass2_with(lv, &decon, &w, &mut v_p, true).expect("pass2 must not fail");
        assert!(v_p.iter().all(|v| v.is_finite()));
    }
    assert!(levels > 0, "the fixture built no level at all");
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
            let streamed_slots: usize = lv.block_sel.iter().map(Vec::len).sum();
            if lv.batches.is_empty() {
                println!(
                    "{name} level {l}: {nblocks} blocks, {streamed_slots} selections — \
                     a single block exceeds the batch budget, streaming"
                );
            } else {
                let total_points: usize = lv.batches.iter().map(|bl| bl.batch.npoints()).sum();
                for (chunk, bl) in lv.batches.iter().enumerate() {
                    let b = &bl.batch;
                    let mut host_batch = b.clone();
                    for (i, x) in host_batch.slot_coef.iter_mut().enumerate() {
                        *x = ((i * 17 + 3) as f64).sin();
                    }
                    let client = pyscf_algebra::select_backend().expect("backend").client;
                    let resident =
                        PairSlotBatchDevice::new(&client, &host_batch).expect("resident");
                    let plain_rho = collocate_pairs_rho_batched(&client, &host_batch).expect("rho");
                    let resident_rho = resident
                        .rho(&client, &host_batch.slot_coef)
                        .expect("resident rho");
                    same_bits("resident rho", &plain_rho, &resident_rho);
                    let coef_b: Vec<f64> = host_batch
                        .slot_coef
                        .iter()
                        .enumerate()
                        .map(|(i, &x)| x * 0.37 - (i as f64 * 0.013).cos())
                        .collect();
                    let single_rho_b = resident.rho(&client, &coef_b).expect("single rho b");
                    let fused_rho = resident
                        .rho2(&client, [&host_batch.slot_coef, &coef_b])
                        .expect("fused rho");
                    same_bits("fused rho alpha", &resident_rho, &fused_rho[0]);
                    same_bits("fused rho beta", &single_rho_b, &fused_rho[1]);
                    let weight = model_weight(b.npoints(), 0xABCD_0000 + chunk as u64);
                    let plain_int =
                        collocate_pairs_integrate_batched(&client, &host_batch, &weight)
                            .expect("integrate");
                    let resident_int = resident
                        .integrate(&client, &host_batch.slot_coef, &weight)
                        .expect("resident integrate");
                    same_bits("resident integrate", &plain_int, &resident_int);
                    let weight_b = model_weight(b.npoints(), 0xDCBA_0000 + chunk as u64);
                    let single_int_b = resident
                        .integrate(&client, &coef_b, &weight_b)
                        .expect("single integrate b");
                    let fused_int = resident
                        .integrate2(
                            &client,
                            [&host_batch.slot_coef, &coef_b],
                            [&weight, &weight_b],
                        )
                        .expect("fused integrate");
                    same_bits("fused integrate alpha", &resident_int, &fused_int[0]);
                    same_bits("fused integrate beta", &single_int_b, &fused_int[1]);
                    let bytes = (b.coords_x.len() + b.coords_y.len() + b.coords_z.len()) * 8
                        + b.point_block.len() * 4
                        + b.slot_pow.len() * 4
                        + b.slot_coef.len() * 8
                        + b.inst_slot0.len() * 4
                        + b.instance_alpha.len() * 8
                        + b.instance_center.len() * 8;
                    println!(
                        "{name} level {l} chunk {chunk}: mesh {:?}, {nblocks} blocks -> {} launch(es) per \
                         direction; batch = {} points, {} instances, {} slots, {:.1} MiB",
                        lv.mesh,
                        lv.batches.len(),
                        b.npoints(),
                        b.ninstances(),
                        b.nslots(),
                        bytes as f64 / (1024.0 * 1024.0)
                    );
                    assert_eq!(
                        b.npoints(),
                        bl.point_global.len(),
                        "{name} level {l} chunk {chunk}: point map must match the batch"
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
                assert_eq!(
                    total_points, lv.ngrids,
                    "{name} level {l}: chunks must partition the mesh exactly"
                );
            }
        }
    }
}
