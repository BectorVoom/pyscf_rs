//! SCF-03 — the tiled grid contractions (`rho_k`'s GEMM + fold, `band_vmat`'s
//! `aow` + output tiles, `local_vmat`'s output tiles) against the
//! one-lane-per-output kernels they replace (`PYSCF_PBC_GRID_TILED=0`).
//!
//! Every output keeps its summation order, so on the CPU runtime the two
//! routes must agree to the last bit — unsplit, and at every pinned split.
//!
//! One test per file: both switches are process-wide environment variables.
#![cfg(feature = "cpu")]

use cubecl::Runtime;
use pyscf_algebra::AlgebraClient;
use pyscf_kernels::pbc::{
    AoPlanes, DeviceAoTable, KMatPlanes, band_vmat, band_vmat_table, local_vmat, rho_k, rho_k_table,
};

fn lcg(seed: u64, len: usize) -> Vec<f64> {
    let mut x = seed;
    (0..len)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

fn set_tiled(on: bool) {
    // SAFETY: the only test in this binary.
    unsafe { std::env::set_var("PYSCF_PBC_GRID_TILED", if on { "1" } else { "0" }) };
}

fn assert_planes_bitwise(got: &KMatPlanes, want: &KMatPlanes, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: k-point count");
    for (k, ((gr, gi), (wr, wi))) in got.iter().zip(want).enumerate() {
        assert_eq!(gr.len(), wr.len(), "{what}: k={k} length");
        for i in 0..gr.len() {
            assert_eq!(gr[i].to_bits(), wr[i].to_bits(), "{what}: re k={k} i={i}");
            assert_eq!(gi[i].to_bits(), wi[i].to_bits(), "{what}: im k={k} i={i}");
        }
    }
}

fn assert_bitwise(got: &(Vec<f64>, Vec<f64>), want: &(Vec<f64>, Vec<f64>), what: &str) {
    assert_eq!(got.0.len(), want.0.len(), "{what}: length");
    for i in 0..got.0.len() {
        assert_eq!(got.0[i].to_bits(), want.0[i].to_bits(), "{what}: re i={i}");
        assert_eq!(got.1[i].to_bits(), want.1[i].to_bits(), "{what}: im i={i}");
    }
}

#[test]
fn tiled_contractions_match_the_per_output_kernels_bitwise() {
    let client = AlgebraClient::Cpu(cubecl_cpu::CpuRuntime::client(&cubecl_cpu::CpuDevice));

    // --- band_vmat: AO counts below, at and across the 4-wide tile; a grid
    // that no split divides; Γ present and absent.
    for &(nao, ngrids, comp, nvar) in &[
        (3usize, 97usize, 1usize, 1usize),
        (4, 130, 4, 2),
        (5, 301, 4, 4),
        (13, 211, 4, 4),
    ] {
        for gamma in [vec![true, false, false], vec![false, false, false]] {
            let nkpts = gamma.len();
            let re = lcg(7 + nao as u64, nkpts * comp * nao * ngrids);
            let im = lcg(8 + nao as u64, nkpts * comp * nao * ngrids);
            let wv = lcg(99, nvar * ngrids);
            let ao = AoPlanes { re: &re, im: &im };
            for split in ["1", "2", "8", "32", "4096"] {
                unsafe { std::env::set_var("PYSCF_PBC_BAND_VMAT_SPLIT", split) };
                let what = format!("band_vmat nao={nao} nvar={nvar} gamma={gamma:?} split={split}");
                set_tiled(false);
                let want = band_vmat(&client, &ao, &wv, nvar, comp, nkpts, nao, ngrids, &gamma)
                    .expect("band_vmat reference");
                set_tiled(true);
                // Budgets that put every k-point and every tile row in one
                // launch, one k-point per launch, and one or two tile rows of
                // one k-point per launch.
                for budget in [None, Some("50000"), Some("2000")] {
                    match budget {
                        Some(b) => unsafe { std::env::set_var("PYSCF_PBC_GRID_TILE_BUDGET", b) },
                        None => unsafe { std::env::remove_var("PYSCF_PBC_GRID_TILE_BUDGET") },
                    }
                    let got = band_vmat(&client, &ao, &wv, nvar, comp, nkpts, nao, ngrids, &gamma)
                        .expect("band_vmat tiled");
                    assert_planes_bitwise(&got, &want, &format!("{what} budget={budget:?}"));
                }
                unsafe { std::env::remove_var("PYSCF_PBC_GRID_TILE_BUDGET") };
                if gamma.iter().all(|&g| !g) {
                    // The SCF's route: the same planes as a device table.
                    let table = DeviceAoTable::from_host_planes(
                        &client, &re, &im, nkpts, comp, nao, ngrids,
                    )
                    .expect("table");
                    let from_table =
                        band_vmat_table(&client, &table, &wv, nvar).expect("band_vmat_table");
                    assert_planes_bitwise(&from_table, &want, &format!("{what} (table)"));
                }
            }
            unsafe { std::env::remove_var("PYSCF_PBC_BAND_VMAT_SPLIT") };
        }
    }

    // --- rho_k: AO counts below, at and across the 8-wide tile; a density
    // matrix with zero entries (both parts, and one part only).
    for &(nao, ngrids, ncomp) in &[(5usize, 131usize, 1usize), (8, 64, 4), (19, 211, 4)] {
        let nkpts = 2;
        let n = ncomp * nao * ngrids;
        let ar = lcg(1 + nao as u64, nkpts * n);
        let ai = lcg(2 + nao as u64, nkpts * n);
        let mut dr = lcg(3, nao * nao);
        let mut di = lcg(4, nao * nao);
        dr[3] = 0.0;
        dr[nao + 2] = 0.0;
        di[nao + 2] = 0.0;
        let what = format!("rho_k nao={nao} ncomp={ncomp}");
        set_tiled(false);
        let want: Vec<_> = (0..nkpts)
            .map(|k| {
                rho_k(
                    &client,
                    &ar[k * n..(k + 1) * n],
                    &ai[k * n..(k + 1) * n],
                    &dr,
                    &di,
                    ncomp,
                    nao,
                    ngrids,
                )
                .expect("rho_k reference")
            })
            .collect();
        set_tiled(true);
        let table = DeviceAoTable::from_host_planes(&client, &ar, &ai, nkpts, ncomp, nao, ngrids)
            .expect("table");
        for (k, want) in want.iter().enumerate() {
            let got = rho_k(
                &client,
                &ar[k * n..(k + 1) * n],
                &ai[k * n..(k + 1) * n],
                &dr,
                &di,
                ncomp,
                nao,
                ngrids,
            )
            .expect("rho_k tiled");
            assert_bitwise(&got, want, &format!("{what} k={k}"));
            let from_table = rho_k_table(&client, &table, k, &dr, &di).expect("rho_k_table");
            assert_bitwise(&from_table, want, &format!("{what} k={k} (table)"));
        }
    }

    // --- local_vmat: the same AO counts around its 4-wide tile, Γ present.
    for &(nao, ngrids) in &[(3usize, 97usize), (4, 130), (13, 211)] {
        let gamma = [false, true, false];
        let nkpts = gamma.len();
        let re = lcg(21 + nao as u64, nkpts * nao * ngrids);
        let im = lcg(22 + nao as u64, nkpts * nao * ngrids);
        let vr = lcg(23, ngrids);
        let ao = AoPlanes { re: &re, im: &im };
        set_tiled(false);
        let want = local_vmat(&client, &ao, &vr, nkpts, nao, ngrids, &gamma).expect("reference");
        set_tiled(true);
        let got = local_vmat(&client, &ao, &vr, nkpts, nao, ngrids, &gamma).expect("tiled");
        assert_planes_bitwise(&got, &want, &format!("local_vmat nao={nao}"));
    }
    unsafe { std::env::remove_var("PYSCF_PBC_GRID_TILED") };
}
