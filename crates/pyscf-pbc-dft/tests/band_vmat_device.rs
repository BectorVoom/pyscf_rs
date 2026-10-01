//! BAND-08 — `band_vmats` on the device (`PYSCF_PBC_BAND_VMAT_DEVICE=1`)
//! against the host reduction (`=0`), through `Krks::get_bands`.
//!
//! The two routes differ only in where the `Σ_g conj(ao) · aow` reduction
//! runs and in its association (pairwise on the host, serial in the lane), so
//! they agree to the band chain's rounding floor and NOT bitwise. The gate is
//! `1e-9` Ha, the same floor every upstream band comparison uses
//! (`krhf_bands_oracle*.rs`).
//!
//! One test per file on purpose: the switch is a process-wide environment
//! variable, and cargo runs a binary's tests on parallel threads.

mod common;

use pyscf_pbc_dft::krks::Krks;
use pyscf_pbc_gto::{BravaisLattice, band_path, make_kpts_default};
use pyscf_pbc_scf::types::KScfConfig;

#[test]
fn device_band_vmats_match_the_host_reduction_to_1e9() {
    // GGA (deriv-1 table, four weights) and LDA (value table, one weight):
    // both device kernels, in one test so the env switch is never raced.
    for xc in ["pbe", "lda,vwn"] {
        compare_routes(xc);
    }
}

fn compare_routes(xc: &str) {
    let mut cell = common::silicon();
    // A coarse box keeps the two band evaluations to seconds; the routes are
    // compared on the same mesh, so its size only sets the runtime.
    cell.mesh = [15, 15, 15];
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("k-mesh");
    let mf = Krks::new(cell, &kpts, xc).expect("KRKS");
    let cfg = KScfConfig {
        conv_tol: 1e-9,
        max_cycle: 60,
        ..KScfConfig::for_cell(mf.cell())
    };
    let scf = mf.kernel(&cfg).expect("KRKS kernel");
    assert!(scf.converged, "the Si/{xc} SCF must converge");
    let path = band_path(mf.cell(), BravaisLattice::Fcc, 12).expect("band path");

    let run = |route: &str| {
        // SAFETY: this binary holds exactly one test, so no other thread reads
        // the environment while it is set.
        unsafe { std::env::set_var("PYSCF_PBC_BAND_VMAT_DEVICE", route) };
        mf.get_bands(&path.abs, &scf.dm).expect("get_bands").0
    };
    let host = run("0");
    let device = run("1");
    // BAND-09: the GPU's split contraction, forced on the CPU runtime (the
    // resident accumulator is point-major here, so this also covers the
    // point-to-k-major transpose).
    unsafe { std::env::set_var("PYSCF_PBC_BAND_VMAT_SPLIT", "16") };
    let split = run("1");
    // BAND-10: the GPU's accumulator cap (one k-tile instead of several),
    // forced on the CPU runtime. The k-tiling never changes a `(q, k)` chain,
    // so the device route must be BITWISE unchanged.
    unsafe { std::env::set_var("PYSCF_PBC_AO_RESIDENT_ACC", "2048") };
    let one_tile = run("1");
    unsafe { std::env::set_var("PYSCF_PBC_AO_RESIDENT_ACC", "40") };
    let many_tiles = run("1");
    unsafe { std::env::remove_var("PYSCF_PBC_AO_RESIDENT_ACC") };
    for (tiles, got) in [("one", &one_tile), ("many", &many_tiles)] {
        for (k, (a, b)) in split.iter().zip(got.iter()).enumerate() {
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "{xc}: {tiles}-tile accumulator changed band {i} at k={k}: {x} vs {y}"
                );
            }
        }
    }
    unsafe { std::env::remove_var("PYSCF_PBC_BAND_VMAT_SPLIT") };
    unsafe { std::env::remove_var("PYSCF_PBC_BAND_VMAT_DEVICE") };
    let mut worst_split = 0.0_f64;
    for (h, d) in host.iter().zip(&split) {
        for (a, b) in h.iter().zip(d) {
            worst_split = worst_split.max((a - b).abs());
        }
    }
    assert!(worst_split <= 1e-9, "{xc}: split device route vs host {worst_split:e}");
    eprintln!("BAND-09 {xc}: host vs split device band energies, worst |diff| = {worst_split:e} Ha");

    assert_eq!(host.len(), device.len());
    let mut worst = 0.0_f64;
    for (k, (h, d)) in host.iter().zip(&device).enumerate() {
        assert_eq!(h.len(), d.len(), "band count at k={k}");
        for (i, (a, b)) in h.iter().zip(d).enumerate() {
            let diff = (a - b).abs();
            worst = worst.max(diff);
            assert!(
                diff <= 1e-9,
                "{xc} band {i} at k={k}: host {a} vs device {b} (diff {diff:e})"
            );
        }
    }
    eprintln!("BAND-08 {xc}: host vs device band energies, worst |diff| = {worst:e} Ha");
}
