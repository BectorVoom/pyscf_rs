//! SCF-03 — a whole KRKS SCF and its band structure on the device routes,
//! forced on the CPU runtime and grid-blocked, with the tiled grid
//! contractions against the one-lane-per-output kernels
//! (`PYSCF_PBC_GRID_TILED=0`).
//!
//! The tiled kernels keep every output's summation order, so nothing may
//! move: `e_tot`, every orbital energy and every band energy are compared
//! BITWISE — through the XC density and potential, the blocked Coulomb build
//! (its AO blocks device-resident), the fused local pseudopotential, and the
//! band contraction over a point-major accumulator.
//!
//! One test per file: the switches are process-wide environment variables.

mod common;

use pyscf_pbc_dft::krks::Krks;
use pyscf_pbc_gto::{BravaisLattice, band_path, make_kpts_default};
use pyscf_pbc_scf::types::KScfConfig;

fn assert_bitwise(a: &[Vec<f64>], b: &[Vec<f64>], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: k-point count");
    for (k, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.len(), y.len(), "{what}: k={k} length");
        for (i, (u, v)) in x.iter().zip(y).enumerate() {
            assert_eq!(u.to_bits(), v.to_bits(), "{what}: k={k} i={i}: {u} vs {v}");
        }
    }
}

#[test]
fn tiled_contractions_leave_the_scf_and_bands_bitwise_unchanged() {
    // SAFETY: the only test in this binary.
    unsafe {
        std::env::set_var("PYSCF_PBC_NUMINT_DEVICE", "1");
        std::env::set_var("PYSCF_PBC_FFTJK_DEVICE", "1");
        std::env::set_var("PYSCF_PBC_BAND_VMAT_DEVICE", "1");
        // Several grid blocks in the XC loop and in the Coulomb build.
        std::env::set_var("PYSCF_PBC_NUMINT_AO_BUDGET_MB", "2");
        std::env::set_var("PYSCF_PBC_FFTJK_AO_BUDGET_MB", "0.5");
    }
    for xc in ["pbe", "lda,vwn"] {
        let mut cell = common::silicon();
        cell.mesh = [15, 15, 15];
        let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("k-mesh");
        let path = band_path(&cell, BravaisLattice::Fcc, 8).expect("band path");
        let run = |tiled: &str, budget: Option<&str>| {
            unsafe {
                std::env::set_var("PYSCF_PBC_GRID_TILED", tiled);
                match budget {
                    Some(b) => std::env::set_var("PYSCF_PBC_GRID_TILE_BUDGET", b),
                    None => std::env::remove_var("PYSCF_PBC_GRID_TILE_BUDGET"),
                }
            }
            // A fresh driver per route: nothing cached by one reaches the other.
            let mf = Krks::new(cell.clone(), &kpts, xc).expect("KRKS");
            let cfg = KScfConfig {
                conv_tol: 1e-9,
                max_cycle: 60,
                ..KScfConfig::for_cell(mf.cell())
            };
            let scf = mf.kernel(&cfg).expect("KRKS kernel");
            assert!(scf.converged, "{xc} tiled={tiled} must converge");
            let bands = mf.get_bands(&path.abs, &scf.dm).expect("get_bands").0;
            (scf.e_tot, scf.mo_energy, bands)
        };
        let reference = run("0", None);
        // The default launch shape, and budgets small enough that the k-points
        // and the tile rows of one contraction span several launches.
        for budget in [None, Some("4000")] {
            let tiled = run("1", budget);
            let what = format!("{xc} budget={budget:?}");
            assert_eq!(
                reference.0.to_bits(),
                tiled.0.to_bits(),
                "{what}: e_tot {} vs {}",
                reference.0,
                tiled.0
            );
            assert_bitwise(&reference.1, &tiled.1, &format!("{what} mo_energy"));
            assert_bitwise(&reference.2, &tiled.2, &format!("{what} bands"));
        }
        eprintln!(
            "SCF-03 {xc}: e_tot {} Ha, SCF and bands bitwise unchanged",
            reference.0
        );
    }
    unsafe {
        for v in [
            "PYSCF_PBC_NUMINT_DEVICE",
            "PYSCF_PBC_FFTJK_DEVICE",
            "PYSCF_PBC_BAND_VMAT_DEVICE",
            "PYSCF_PBC_NUMINT_AO_BUDGET_MB",
            "PYSCF_PBC_FFTJK_AO_BUDGET_MB",
            "PYSCF_PBC_GRID_TILED",
            "PYSCF_PBC_GRID_TILE_BUDGET",
        ] {
            std::env::remove_var(v);
        }
    }
}
