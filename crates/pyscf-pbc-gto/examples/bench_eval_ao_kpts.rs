//! End-to-end timing for `eval_ao_kpts` — the K-08 lattice-image loop.
//!
//! `cargo run --release --example bench_eval_ao_kpts -p pyscf-pbc-gto`
//!
//! Prints the wall time and, separately, the accumulator traffic the loop
//! moves, so the transfer-bound part is visible next to the total.

use std::time::Instant;

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs, eval_ao_kpts, kpts_mesh::make_kpts_default};

fn diamond() -> Cell {
    let h = 3.37032;
    let q = 1.68516;
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("C".into(), [0.0, 0.0, 0.0]), ("C".into(), [q, q, q])]),
            basis: BasisInput::Name("gth-szv".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        pseudo: Some("gth-pade".into()),
        ..Default::default()
    })
    .expect("diamond builds")
}

/// A cubic grid of `m^3` points spanning the cell — a stand-in for a real DFT
/// grid, which is what makes `n = comp * ngrids * nao` large enough for the
/// accumulator traffic to matter.
fn grid(cell: &Cell, m: usize) -> Vec<[f64; 3]> {
    let a = cell.lattice_vectors();
    let mut out = Vec::with_capacity(m * m * m);
    for i in 0..m {
        for j in 0..m {
            for k in 0..m {
                let f = [
                    i as f64 / m as f64,
                    j as f64 / m as f64,
                    k as f64 / m as f64,
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

fn main() {
    let cell = diamond();
    let _nao = cell.mol.nao_nr;

    for &(m, mesh) in &[(8usize, 2usize), (12, 2), (12, 3)] {
        let coords = grid(&cell, m);
        let kpts = make_kpts_default(&cell, [mesh, mesh, mesh]).expect("kpts");
        let ngrids = coords.len();
        let nkpts = kpts.len();

        for name in ["GTOval_sph", "GTOval_sph_deriv1"] {
            let t0 = Instant::now();
            let out = eval_ao_kpts(&cell, name, &coords, &kpts).expect("eval_ao_kpts");
            let secs = t0.elapsed().as_secs_f64();

            let n = out.kaos[0].re.len();
            let comp = out.comp;
            // What the old per-image round-trip moved: both planes up and down,
            // once per lattice image.
            let planes_mb = (2 * nkpts * n * 8) as f64 / 1e6;
            println!(
                "{name:<20} grid={ngrids:<5} nkpts={nkpts:<3} comp={comp} n={n:<8} \
                 {secs:>8.3} s   accumulator planes {planes_mb:>8.2} MB"
            );
        }
    }
}
