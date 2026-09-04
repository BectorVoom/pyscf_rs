//! A-00 gate for the periodic AO image loop.
//!
//! Rayon pool size and the AO-screen switch are process-global, so the parent
//! re-executes this test binary and compares the complete output buffers.

use std::path::PathBuf;
use std::process::Command;

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs, eval_ao_kpts, make_kpts_default};

fn silicon() -> Cell {
    let h = 5.1311;
    let q = 2.55555;
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![
                ("Si".into(), [0.0, 0.0, 0.0]),
                ("Si".into(), [q, q, q]),
            ]),
            basis: BasisInput::Name("gth-szv".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        pseudo: Some("gth-pade".into()),
        ..Default::default()
    })
    .expect("cell")
}

fn grid(cell: &Cell) -> Vec<[f64; 3]> {
    const MESH: usize = 11;
    let a = cell.lattice_vectors();
    let mut out = Vec::with_capacity(MESH * MESH * MESH);
    for i in 0..MESH {
        for j in 0..MESH {
            for k in 0..MESH {
                let f = [
                    i as f64 / MESH as f64,
                    j as f64 / MESH as f64,
                    k as f64 / MESH as f64,
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

fn child_output_path(threads: usize, screen: bool) -> PathBuf {
    std::env::temp_dir().join(format!(
        "pyscf-rs-eval-ao-stages-{}-{threads}-{}.bin",
        std::process::id(),
        usize::from(screen)
    ))
}

fn run_child(threads: usize, screen: bool) -> Vec<u8> {
    let path = child_output_path(threads, screen);
    let status = Command::new(std::env::current_exe().expect("test executable"))
        .args(["--exact", "emit_ao_bits", "--ignored"])
        .env("RAYON_NUM_THREADS", threads.to_string())
        .env("PYSCF_PBC_AO_SCREEN", if screen { "1" } else { "0" })
        .env("PYSCF_AO_STAGE_OUTPUT", &path)
        .status()
        .expect("run AO gate child");
    assert!(
        status.success(),
        "AO gate child failed: threads={threads} screen={screen}"
    );
    let bytes = std::fs::read(&path).expect("read AO child output");
    std::fs::remove_file(path).expect("remove AO child output");
    bytes
}

#[test]
fn image_loop_is_thread_bit_exact_and_screen_stays_inside_its_gate() {
    let unscreened_1 = run_child(1, false);
    let unscreened_8 = run_child(8, false);
    let screened_1 = run_child(1, true);
    let screened_8 = run_child(8, true);

    assert_eq!(
        unscreened_1, unscreened_8,
        "unscreened AO loop moved with Rayon threads"
    );
    assert_eq!(
        screened_1, screened_8,
        "screened AO loop moved with Rayon threads"
    );
    assert_eq!(unscreened_1.len(), screened_1.len());

    let mut worst = 0.0_f64;
    for (a, b) in unscreened_1.chunks_exact(8).zip(screened_1.chunks_exact(8)) {
        let x = f64::from_bits(u64::from_ne_bytes(a.try_into().expect("f64 bytes")));
        let y = f64::from_bits(u64::from_ne_bytes(b.try_into().expect("f64 bytes")));
        worst = worst.max((x - y).abs());
    }
    println!("A-00 screen on/off worst absolute AO delta = {worst:.3e}");
    assert!(worst < 1e-11, "AO screen exceeded the W-09 gate: {worst:e}");
}

#[test]
#[ignore = "child process for image_loop_is_thread_bit_exact_and_screen_stays_inside_its_gate"]
fn emit_ao_bits() {
    let path = std::env::var_os("PYSCF_AO_STAGE_OUTPUT").expect("child output path");
    let cell = silicon();
    let coords = grid(&cell);
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
    std::fs::write(path, bytes).expect("write AO child output");
}
