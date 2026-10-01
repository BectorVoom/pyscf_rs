//! SCF-01 — the SCF's grid contractions (`eval_rho`, `_vxc_mat`, and the
//! Coulomb build's density and `vj`) on the device routes
//! (`PYSCF_PBC_NUMINT_DEVICE=1`, `PYSCF_PBC_FFTJK_DEVICE=1`) against the host routes, forced
//! on the CPU runtime: independent SCFs agree on `e_tot`, and bands on ONE
//! density agree to the band chain's `1e-9`.
//!
//! One test per file: the switch is a process-wide environment variable.

mod common;

use pyscf_pbc_dft::krks::Krks;
use pyscf_pbc_gto::{BravaisLattice, band_path, make_kpts_default};
use pyscf_pbc_scf::types::KScfConfig;

#[test]
fn device_numint_matches_the_host_scf() {
    for xc in ["pbe", "lda,vwn"] {
        let mut cell = common::silicon();
        cell.mesh = [15, 15, 15];
        let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("k-mesh");
        let mf = Krks::new(cell, &kpts, xc).expect("KRKS");
        let cfg = KScfConfig {
            conv_tol: 1e-11,
            max_cycle: 80,
            ..KScfConfig::for_cell(mf.cell())
        };
        let run = |route: &str| {
            // SAFETY: the only test in this binary.
            unsafe {
                std::env::set_var("PYSCF_PBC_NUMINT_DEVICE", route);
                std::env::set_var("PYSCF_PBC_FFTJK_DEVICE", route);
            }
            let r = mf.kernel(&cfg).expect("KRKS kernel");
            assert!(r.converged, "{xc} route {route} must converge");
            r
        };
        let host = run("0");
        let dev = run("1");
        let de = (host.e_tot - dev.e_tot).abs();
        assert!(
            de <= 1e-9,
            "{xc}: e_tot host {} device {} (diff {de:e})",
            host.e_tot,
            dev.e_tot
        );
        let path = band_path(mf.cell(), BravaisLattice::Fcc, 8).expect("band path");
        unsafe { std::env::set_var("PYSCF_PBC_NUMINT_DEVICE", "0") };
        let bh = mf.get_bands(&path.abs, &host.dm).expect("bands").0;
        unsafe { std::env::set_var("PYSCF_PBC_NUMINT_DEVICE", "1") };
        let bd = mf.get_bands(&path.abs, &host.dm).expect("bands").0;
        unsafe {
            std::env::remove_var("PYSCF_PBC_NUMINT_DEVICE");
            std::env::remove_var("PYSCF_PBC_FFTJK_DEVICE");
        }
        let mut worst = 0.0_f64;
        for (a, b) in bh.iter().zip(&bd) {
            for (x, y) in a.iter().zip(b) {
                worst = worst.max((x - y).abs());
            }
        }
        assert!(
            worst <= 1e-9,
            "{xc}: bands on one density differ by {worst:e}"
        );
        eprintln!("SCF-01 {xc}: |de_tot| = {de:e} Ha, bands on one density {worst:e} Ha");
    }
}
