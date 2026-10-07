//! SCF-03 A/B — the SCF's two grid contractions over a device-resident AO
//! table (`rho_k_table`, `band_vmat_table`) and `get_hcore`'s local one
//! (`local_vmat`, one k-point), tiled against one-lane-per-output
//! (`PYSCF_PBC_GRID_TILED=0`), interleaved inside ONE process on whichever
//! backend `PYSCF_BACKEND` selects.
//!
//! ```text
//! cargo run --release -p pyscf-kernels --example grid_contract_bench -- \
//!     [nao=910] [ngrids=4992] [nkpts=9] [ncomp=4] [reps=3]
//! ```
//!
//! The defaults are one XC block of the YTa7O19 `gth-dzvp-molopt-sr` run on a
//! Kaggle T4 (2.5 GB AO budget). Prints, per route, the best wall time of the
//! density (all k-points) and of the two potential contractions, and the
//! largest difference between the routes relative to the largest output.

use std::time::Instant;

use pyscf_kernels::pbc::{AoPlanes, DeviceAoTable, band_vmat_table, local_vmat, rho_k_table};

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

fn arg(i: usize, default: usize) -> usize {
    std::env::args()
        .nth(i)
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn rel_diff(a: &[f64], b: &[f64]) -> (f64, bool) {
    let scale = a
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()))
        .max(f64::MIN_POSITIVE);
    let worst = a
        .iter()
        .zip(b)
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
    let bitwise = a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
    (worst / scale, bitwise)
}

fn main() {
    let (nao, ngrids, nkpts, ncomp, reps) =
        (arg(1, 910), arg(2, 4992), arg(3, 9), arg(4, 4), arg(5, 3));
    let sel = pyscf_algebra::select_backend().expect("backend");
    let client = sel.client;
    println!(
        "backend {:?}  nao {nao}  ngrids {ngrids}  nkpts {nkpts}  ncomp {ncomp}  reps {reps}  \
         table {:.0} MB",
        sel.kind,
        (16 * nkpts * ncomp * nao * ngrids) as f64 / (1u64 << 20) as f64
    );

    let len = nkpts * ncomp * nao * ngrids;
    let re = lcg(1, len);
    let im = lcg(2, len);
    let table = DeviceAoTable::from_host_planes(&client, &re, &im, nkpts, ncomp, nao, ngrids)
        .expect("table");
    // `local_vmat` takes host planes: one k-point's value component, so the
    // upload it times is small next to the contraction.
    let (loc_re, loc_im) = (re[..nao * ngrids].to_vec(), im[..nao * ngrids].to_vec());
    drop((re, im));
    client.sync_device();
    // A Hermitian density matrix per k-point and one weight per variable.
    let dms: Vec<(Vec<f64>, Vec<f64>)> = (0..nkpts)
        .map(|k| {
            let (a, b) = (lcg(10 + k as u64, nao * nao), lcg(50 + k as u64, nao * nao));
            let mut dr = vec![0.0; nao * nao];
            let mut di = vec![0.0; nao * nao];
            for p in 0..nao {
                for q in 0..nao {
                    dr[p * nao + q] = a[p * nao + q] + a[q * nao + p];
                    di[p * nao + q] = b[p * nao + q] - b[q * nao + p];
                }
            }
            (dr, di)
        })
        .collect();
    let wv = lcg(99, ncomp * ngrids);

    let run = |tiled: bool| {
        // SAFETY: single-threaded here; the kernels read the switch per call.
        unsafe { std::env::set_var("PYSCF_PBC_GRID_TILED", if tiled { "1" } else { "0" }) };
        let t = Instant::now();
        let rho: Vec<f64> = dms
            .iter()
            .enumerate()
            .flat_map(|(k, (dr, di))| rho_k_table(&client, &table, k, dr, di).expect("rho").0)
            .collect();
        let t_rho = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let vmat: Vec<f64> = band_vmat_table(&client, &table, &wv, ncomp)
            .expect("vmat")
            .into_iter()
            .flat_map(|(r, i)| r.into_iter().chain(i))
            .collect();
        let t_vmat = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let ao = AoPlanes {
            re: &loc_re,
            im: &loc_im,
        };
        let vloc: Vec<f64> = local_vmat(&client, &ao, &wv[..ngrids], 1, nao, ngrids, &[false])
            .expect("local_vmat")
            .into_iter()
            .flat_map(|(r, i)| r.into_iter().chain(i))
            .collect();
        let t_vloc = t.elapsed().as_secs_f64();
        ([t_rho, t_vmat, t_vloc], [rho, vmat, vloc])
    };

    // Warm both routes (kernel compilation), then interleave the timed reps.
    let (_, reference) = run(false);
    let (_, tiled) = run(true);
    let mut best = [[f64::INFINITY; 3]; 2];
    for _ in 0..reps {
        for (route, on) in [false, true].into_iter().enumerate() {
            let (t, _) = run(on);
            for (b, t) in best[route].iter_mut().zip(t) {
                *b = b.min(t);
            }
        }
    }
    println!("route       rho [s]   vmat [s]   rho+vmat [s]   local_vmat 1k [s]");
    for (route, name) in ["per-output", "tiled"].into_iter().enumerate() {
        let b = best[route];
        println!(
            "{name:<10} {:8.3}  {:9.3}  {:13.3}  {:18.3}",
            b[0],
            b[1],
            b[0] + b[1],
            b[2]
        );
    }
    println!(
        "speedup     {:7.2}x  {:8.2}x  {:12.2}x  {:17.2}x",
        best[0][0] / best[1][0],
        best[0][1] / best[1][1],
        (best[0][0] + best[0][1]) / (best[1][0] + best[1][1]),
        best[0][2] / best[1][2]
    );
    for (name, (a, b)) in ["rho", "vmat", "local_vmat"]
        .into_iter()
        .zip(reference.iter().zip(&tiled))
    {
        let (d, bitwise) = rel_diff(a, b);
        println!("{name:<10} tiled vs per-output: max rel diff {d:.3e}  bitwise {bitwise}");
    }
}
