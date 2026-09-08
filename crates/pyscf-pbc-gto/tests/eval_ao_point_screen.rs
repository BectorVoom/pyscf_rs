//! A-04 gate — the per-point reach test inside the AO kernels
//! (`.planning/pbc/KUKS-KSYMM-MULTIGRID-SESSION-4-EXECUTION-SUMMARY.md`).
//!
//! The W-09 block screen and the A-04 point screen share ONE per-shell radius;
//! the point screen only refuses to evaluate a shell at grid points a kept
//! block still contains outside that radius. So the mass it drops is bounded
//! by the same estimate W-09's gate bounds, and this test holds it to the
//! same number (`1e-11`, `eval_ao_stages.rs`), against BOTH the block-only
//! table and the fully unscreened one. It also pins that the switch is
//! process-stable under Rayon (bit-identical at 1 and 8 threads), because the
//! screen is a per-lane decision that must not depend on how lanes are
//! scheduled.
//!
//! Both switches are process-global (`OnceLock`), so the parent re-executes
//! this binary per arm and compares whole tables, as `eval_ao_stages.rs` does.

use std::path::PathBuf;
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

fn child_output_path(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "pyscf-rs-eval-ao-point-screen-{}-{tag}.bin",
        std::process::id()
    ))
}

/// Run `emit_ao_bits` in a child with the given switches, return the table.
fn run_child(tag: &str, threads: usize, block: bool, point: bool, basis: &str) -> Vec<u8> {
    let path = child_output_path(tag);
    let status = Command::new(std::env::current_exe().expect("test executable"))
        .args(["--exact", "emit_ao_bits", "--ignored"])
        .env("RAYON_NUM_THREADS", threads.to_string())
        .env("PYSCF_PBC_AO_SCREEN", if block { "1" } else { "0" })
        .env("PYSCF_PBC_AO_POINT_SCREEN", if point { "1" } else { "0" })
        .env("PYSCF_AO_POINT_BASIS", basis)
        .env("PYSCF_AO_STAGE_OUTPUT", &path)
        .status()
        .expect("run A-04 gate child");
    assert!(
        status.success(),
        "A-04 gate child failed: threads={threads} block={block} point={point} basis={basis}"
    );
    let bytes = std::fs::read(&path).expect("read A-04 child output");
    std::fs::remove_file(path).expect("remove A-04 child output");
    bytes
}

fn worst_delta(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "the two tables differ in length");
    let mut worst = 0.0_f64;
    for (x, y) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        let x = f64::from_bits(u64::from_ne_bytes(x.try_into().expect("f64 bytes")));
        let y = f64::from_bits(u64::from_ne_bytes(y.try_into().expect("f64 bytes")));
        worst = worst.max((x - y).abs());
    }
    worst
}

/// The W-09 gate bound (`eval_ao_stages.rs`), reused unchanged.
const GATE: f64 = 1e-11;

#[test]
fn point_screen_stays_inside_the_w09_gate_and_is_thread_stable() {
    for basis in ["gth-szv", "gth-dzvp"] {
        let unscreened = run_child("none", 8, false, false, basis);
        let block_only = run_child("block", 8, true, false, basis);
        let point_1 = run_child("point1", 1, true, true, basis);
        let point_8 = run_child("point8", 8, true, true, basis);

        assert_eq!(
            point_1, point_8,
            "{basis}: the point-screened AO table moved with Rayon threads"
        );
        let vs_block = worst_delta(&block_only, &point_8);
        let vs_none = worst_delta(&unscreened, &point_8);
        let block_vs_none = worst_delta(&unscreened, &block_only);
        println!(
            "A-04 {basis}: point vs block-only {vs_block:.3e}, point vs unscreened {vs_none:.3e}, \
             block-only vs unscreened {block_vs_none:.3e} (gate {GATE:e})"
        );
        assert!(
            vs_block < GATE,
            "{basis}: point screen moved the AO table by {vs_block:e} against block-only"
        );
        assert!(
            vs_none < GATE,
            "{basis}: point screen moved the AO table by {vs_none:e} against unscreened"
        );
        // The point screen must actually be doing something on this fixture —
        // a table identical to block-only would mean the switch never reached
        // the kernel (the S-07/S-08 lesson: assert the item fired).
        assert!(
            point_8 != block_only,
            "{basis}: point-screened table is byte-identical to block-only; the screen did not fire"
        );
    }
}

/// With the block screen OFF the point screen has no radius to apply and the
/// driver must take the unscreened path exactly — byte for byte.
#[test]
fn point_screen_is_inert_without_the_block_screen() {
    let a = run_child("inert-a", 8, false, false, "gth-szv");
    let b = run_child("inert-b", 8, false, true, "gth-szv");
    assert!(
        a == b,
        "point screen changed the table although the block screen was off"
    );
}

#[test]
#[ignore = "child process for the A-04 gate"]
fn emit_ao_bits() {
    let path = std::env::var_os("PYSCF_AO_STAGE_OUTPUT").expect("child output path");
    let basis = std::env::var("PYSCF_AO_POINT_BASIS").unwrap_or_else(|_| "gth-szv".into());
    let cell = silicon(&basis);
    // 15³ keeps the child under a few seconds per arm while leaving every
    // launched image with kept blocks that extend past the cutoff spheres.
    let coords = grid(&cell, 15);
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("kpts");
    let mut bytes = Vec::new();
    for name in ["GTOval_sph", "GTOval_sph_deriv1"] {
        let out = eval_ao_kpts(&cell, name, &coords, &kpts).expect("eval AO");
        for ao in out.kaos {
            for value in ao.re.into_iter().chain(ao.im) {
                bytes.extend_from_slice(&value.to_bits().to_ne_bytes());
            }
        }
    }
    std::fs::write(path, bytes).expect("write A-04 child output");
}
