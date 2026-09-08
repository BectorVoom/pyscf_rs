//! K-09 gate — image-batched Bloch accumulation is bit-identical to one
//! accumulate launch per image
//! (`.planning/pbc/KUKS-KSYMM-MULTIGRID-SESSION-4-EXECUTION-SUMMARY.md`).
//!
//! Session 4 measured the cold periodic AO pass with the K-08 accumulate
//! switched off: 72-84 % of the pass was the accumulate's read-modify-write of
//! both `(nkpts, n)` planes on every image, not the AO kernel. K-09 folds a
//! batch of images per launch, reading and writing each accumulator once per
//! batch. The claim is bit-identity, by construction: each `(k, p)` still
//! receives the same additions in the same image order. A claim like that is
//! worth the test that checks it, so this compares whole tables at
//! `to_bits()` — batch 1 (the pre-K-09 kernels) against batch 16 (the cap)
//! and batch 3 (a ragged tail), screened (dense AND gathered images) and, in a
//! child process, unscreened (dense only).
//!
//! `PYSCF_PBC_AO_IMAGE_BATCH` is read per call, so the batched arms run in
//! one process; the W-09 screen switch is process-global, so the unscreened
//! comparison re-executes this binary.

use std::process::Command;

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs, eval_ao_kpts, make_kpts_default};

fn silicon(basis: &str) -> Cell {
    let h = 5.1311;
    let q = 2.55555;
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![
                ("Si".into(), [0.0, 0.0, 0.0]),
                ("Si".into(), [q, q, q]),
            ]),
            basis: BasisInput::Name(basis.into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        pseudo: Some("gth-pade".into()),
        ..Default::default()
    })
    .expect("cell")
}

fn grid(cell: &Cell, mesh: usize) -> Vec<[f64; 3]> {
    let a = cell.lattice_vectors();
    let mut out = Vec::with_capacity(mesh * mesh * mesh);
    for i in 0..mesh {
        for j in 0..mesh {
            for k in 0..mesh {
                let f = [
                    i as f64 / mesh as f64,
                    j as f64 / mesh as f64,
                    k as f64 / mesh as f64,
                ];
                out.push([
                    f[0] * a[0][0] + f[1] * a[1][0] + f[2] * a[2][0],
                    f[0] * a[0][1] + f[1] * a[1][1] + f[2] * a[2][1],
                    f[0] * a[0][2] + f[1] * a[1][2] + f[2] * a[2][2],
                ]);
            }
        }
    }
    out
}

/// The whole k-resolved AO table as bits, with the accumulate batch width and
/// the A-06 evaluation batch pinned (`eval_batch = 0` is the per-image
/// evaluation kernels, the reference).
fn table_bits_with(
    cell: &Cell,
    coords: &[[f64; 3]],
    kpts: &[[f64; 3]],
    name: &str,
    batch: usize,
    eval_batch: usize,
) -> Vec<u64> {
    // Per-call switches (not `OnceLock`s), so this is exact for THIS call.
    // The fused K-10 path is pinned OFF here; `table_bits_fused` turns it on.
    unsafe {
        std::env::set_var("PYSCF_PBC_AO_IMAGE_BATCH", batch.to_string());
        std::env::set_var("PYSCF_PBC_AO_EVAL_BATCH", eval_batch.to_string());
        std::env::set_var("PYSCF_PBC_AO_FUSE", "0");
    }
    let out = eval_ao_kpts(cell, name, coords, kpts).expect("eval AO");
    let mut bits = Vec::new();
    for ao in out.kaos {
        bits.extend(ao.re.iter().chain(ao.im.iter()).map(|v| v.to_bits()));
    }
    bits
}

/// The table through the fused K-10 kernel with the fused batch capped at
/// `fuse_batch` (`0` = the kernel's own cap).
fn table_bits_fused(
    cell: &Cell,
    coords: &[[f64; 3]],
    kpts: &[[f64; 3]],
    name: &str,
    fuse_batch: usize,
) -> Vec<u64> {
    unsafe {
        std::env::set_var("PYSCF_PBC_AO_IMAGE_BATCH", "16");
        std::env::set_var("PYSCF_PBC_AO_EVAL_BATCH", "0");
        std::env::set_var("PYSCF_PBC_AO_FUSE", "1");
        if fuse_batch == 0 {
            std::env::remove_var("PYSCF_PBC_AO_FUSE_BATCH");
        } else {
            std::env::set_var("PYSCF_PBC_AO_FUSE_BATCH", fuse_batch.to_string());
        }
    }
    let out = eval_ao_kpts(cell, name, coords, kpts).expect("eval AO");
    unsafe { std::env::remove_var("PYSCF_PBC_AO_FUSE_BATCH") };
    let mut bits = Vec::new();
    for ao in out.kaos {
        bits.extend(ao.re.iter().chain(ao.im.iter()).map(|v| v.to_bits()));
    }
    bits
}

/// [`table_bits_with`] on the per-image evaluation kernels.
fn table_bits(
    cell: &Cell,
    coords: &[[f64; 3]],
    kpts: &[[f64; 3]],
    name: &str,
    batch: usize,
) -> Vec<u64> {
    table_bits_with(cell, coords, kpts, name, batch, 0)
}

fn compare(mesh: usize) {
    compare_kmesh(mesh, [2, 2, 2]);
}

/// K-10v: the fused kernel's k-loop is a `Vector<f64, N>` with `N` the widest
/// width dividing `nkpts`; `[1, 2, 3]` (6 k-points) and `[1, 1, 3]` (3, so
/// `N = 1`) exercise the ragged widths beside the 8 of `[2, 2, 2]`.
fn compare_kmesh(mesh: usize, nk: [usize; 3]) {
    for basis in ["gth-szv", "gth-dzvp"] {
        let cell = silicon(basis);
        let coords = grid(&cell, mesh);
        let kpts = make_kpts_default(&cell, nk).expect("kpts");
        for name in ["GTOval_sph", "GTOval_sph_deriv1"] {
            let one = table_bits(&cell, &coords, &kpts, name, 1);
            let sixteen = table_bits(&cell, &coords, &kpts, name, 16);
            let three = table_bits(&cell, &coords, &kpts, name, 3);
            assert_eq!(one.len(), sixteen.len());
            assert!(
                one == sixteen,
                "{basis} {name}: batch 16 differs from one launch per image"
            );
            assert!(
                one == three,
                "{basis} {name}: batch 3 (ragged tail) differs from one launch per image"
            );
            // A-06 (session 5): the batched evaluation kernels — over the whole
            // accumulate batch (16), over 5 images at a time (ragged inside the
            // 16-batch), and one image per launch (the hoisted-upload arm).
            for eval_batch in [16usize, 5, 1] {
                let batched_eval = table_bits_with(&cell, &coords, &kpts, name, 16, eval_batch);
                assert!(
                    one == batched_eval,
                    "{basis} {name}: A-06 eval batch {eval_batch} differs from the per-image kernels"
                );
            }
            let ragged = table_bits_with(&cell, &coords, &kpts, name, 3, 2);
            assert!(
                one == ragged,
                "{basis} {name}: A-06 eval batch 2 inside accumulate batch 3 differs"
            );
            // K-10 (session 5): the fused evaluate-and-accumulate kernel, at
            // its own cap, at 7 (ragged) and at 1 image per launch.
            for fuse_batch in [0usize, 7, 1] {
                let fused = table_bits_fused(&cell, &coords, &kpts, name, fuse_batch);
                assert_eq!(one.len(), fused.len());
                assert!(
                    one == fused,
                    "{basis} {name}: K-10 fused batch {fuse_batch} differs from the per-image path"
                );
            }
            println!(
                "K-09/A-06/K-10 {basis} {name} (screen {}): {} reals bit-identical at accumulate batch 1 / 3 / 16, eval batch 0 / 1 / 2 / 5 / 16, fused batch cap / 7 / 1",
                std::env::var("PYSCF_PBC_AO_SCREEN").unwrap_or_else(|_| "default".into()),
                one.len()
            );
        }
    }
}

#[test]
fn batched_accumulate_is_bit_identical_to_per_image_launches() {
    compare(11);
}

#[test]
fn fused_kernel_is_bit_identical_at_ragged_vector_widths() {
    compare_kmesh(9, [1, 2, 3]);
    compare_kmesh(9, [1, 1, 3]);
}

/// The unscreened image loop (every image dense, 1331 of them) in a child
/// process, because the screen switch is read once per process.
#[test]
fn batched_accumulate_is_bit_identical_without_the_block_screen() {
    let status = Command::new(std::env::current_exe().expect("test executable"))
        .args([
            "--exact",
            "compare_unscreened_child",
            "--ignored",
            "--nocapture",
        ])
        .env("PYSCF_PBC_AO_SCREEN", "0")
        .status()
        .expect("run K-09 unscreened child");
    assert!(
        status.success(),
        "K-09 unscreened comparison failed in the child"
    );
}

#[test]
#[ignore = "child process for batched_accumulate_is_bit_identical_without_the_block_screen"]
fn compare_unscreened_child() {
    // A smaller mesh: every one of the 1331 images is evaluated on the whole grid here.
    compare(7);
}
