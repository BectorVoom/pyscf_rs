//! D-PBC-28 §7.5 — the MO-first FFT AO2MO route against the AO-ERI route.
//!
//! This is a MEASUREMENT, not a gate: wall clock is not reproducible, so
//! nothing here is asserted against a tolerance except the residual between the
//! two routes, which is. Every test is `#[ignore]`d so `cargo test --workspace`
//! never pays for it.
//!
//! ```bash
//! # both routes, one process (wall clock + residual)
//! cargo test --release -p pyscf-pbc-df --test perf_dpbc28_mofirst \
//!   -- --ignored --nocapture
//! # one route per process, for a separable peak-RSS number
//! DPBC28_ROUTE=ao cargo test --release -p pyscf-pbc-df --test perf_dpbc28_mofirst \
//!   -- --ignored --nocapture mo_first_vs_ao_first_cost
//! DPBC28_ROUTE=mo cargo test --release ... (same)
//! ```

use std::time::Instant;

use pyscf_algebra::CTensor;
use pyscf_pbc_df::pbc_ao2mo::{fft_general, fft_general_mo_first};
use pyscf_pbc_df::{Fftdf, MoCoeff};
use pyscf_pbc_gto::Cell;

/// `VmHWM` in MiB — the kernel's high-water mark for this process. Monotone, so
/// it is only meaningful when one route runs per process (`DPBC28_ROUTE`).
fn peak_rss_mib() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok())
        })
        .map(|kb| kb / 1024.0)
        .unwrap_or(f64::NAN)
}

/// A deterministic, well-conditioned occupied/virtual MO block. The cost
/// comparison depends on the SHAPES, not on the coefficients, and generating
/// them here keeps this crate free of an SCF dev-dependency.
fn mo_block(nao: usize, lo: usize, hi: usize, seed: u64) -> MoCoeff {
    let nmo = hi - lo;
    let mut c = CTensor::zeros(nao * nmo);
    let mut s = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    for v in 0..nao * nmo {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let x = ((s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5;
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let y = ((s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5;
        c.re[v] = x;
        c.im[v] = y;
    }
    MoCoeff::new(nao, nmo, c)
}

fn max_dev(a: &CTensor, b: &CTensor) -> f64 {
    a.re.iter()
        .zip(&b.re)
        .map(|(x, y)| (x - y).abs())
        .chain(a.im.iter().zip(&b.im).map(|(x, y)| (x - y).abs()))
        .fold(0.0, f64::max)
}

fn diamond_szv() -> Cell {
    pyscf_pbc_gto::test_systems::diamond()
}

/// One `(ki,ka,kj,kb)` sweep over every momentum-conserving quadruple, both
/// routes, at one k-mesh.
fn sweep(label: &str, cell: Cell, kmesh: [usize; 3], nocc: usize) {
    let kpts = cell.make_kpts(kmesh).expect("kpts");
    let nk = kpts.len();
    let df = Fftdf::new(cell, &kpts).expect("fftdf");
    let nao = df.cell.mol.nao_nr;
    let nvir = nao - nocc;
    let occ: Vec<_> = (0..nk)
        .map(|k| mo_block(nao, 0, nocc, k as u64 + 1))
        .collect();
    let vir: Vec<_> = (0..nk)
        .map(|k| mo_block(nao, nocc, nao, k as u64 + 101))
        .collect();

    let route = std::env::var("DPBC28_ROUTE").unwrap_or_default();
    let want_ao = route.is_empty() || route == "ao";
    let want_mo = route.is_empty() || route == "mo";

    // Every conserving quadruple: kb is fixed by (ki, ka, kj).
    let quads: Vec<[usize; 4]> = {
        let kc = pyscf_pbc_lib::kpts_helper::get_kconserv(&df.cell.a, &kpts);
        let mut v = Vec::new();
        for ki in 0..nk {
            for ka in 0..nk {
                for kj in 0..nk {
                    v.push([ki, ka, kj, kc.get(ki, ka, kj) as usize]);
                }
            }
        }
        v
    };

    let mut ao_secs = f64::NAN;
    let mut mo_secs = f64::NAN;
    let mut worst = 0.0f64;
    let mut ao_last: Option<CTensor> = None;

    if want_ao {
        let t = Instant::now();
        for (n, q) in quads.iter().enumerate() {
            if n % 8 == 0 {
                println!("[{label}] ao_first progress {n}/{}", quads.len());
            }
            let k4 = [kpts[q[0]], kpts[q[1]], kpts[q[2]], kpts[q[3]]];
            let e = fft_general(&df, [&occ[q[0]], &vir[q[1]], &occ[q[2]], &vir[q[3]]], k4)
                .expect("AO-first");
            ao_last = Some(e.data);
        }
        ao_secs = t.elapsed().as_secs_f64();
        println!(
            "[{label}] ao_first_secs={ao_secs:.6} peak_rss_mib={:.1}",
            peak_rss_mib()
        );
    }
    if want_mo {
        let t = Instant::now();
        let mut mo_last: Option<CTensor> = None;
        for (n, q) in quads.iter().enumerate() {
            if n % 8 == 0 {
                println!("[{label}] mo_first progress {n}/{}", quads.len());
            }
            let k4 = [kpts[q[0]], kpts[q[1]], kpts[q[2]], kpts[q[3]]];
            let e = fft_general_mo_first(
                &df,
                [&occ[q[0]], &vir[q[1]], &occ[q[2]], &vir[q[3]]],
                k4,
                None,
            )
            .expect("MO-first");
            mo_last = Some(e.data);
        }
        mo_secs = t.elapsed().as_secs_f64();
        println!(
            "[{label}] mo_first_secs={mo_secs:.6} peak_rss_mib={:.1}",
            peak_rss_mib()
        );
        if let (Some(a), Some(b)) = (ao_last.as_ref(), mo_last.as_ref()) {
            worst = max_dev(a, b);
        }
    }

    // §7.0's prediction: the AO-first route forms every `nao^2` AO pair on the
    // grid and only then transforms; MO-first forms `nocc*nvir` MO pairs.
    let predicted = ((nao * nao) as f64 / (nocc * nvir) as f64).powi(2);
    let ng: usize = df.mesh.iter().product();
    println!(
        "[{label}] nk={nk} nao={nao} nocc={nocc} nvir={nvir} ngrids={ng} quads={} \
         ao_eri_mib={:.3} ao_pair_grid_mib={:.1} mo_pair_grid_mib={:.1} \
         predicted_flop_ratio={predicted:.2}",
        quads.len(),
        // the materialised `nao^2 x nao^2` AO ERI itself
        (nao.pow(4) * 16) as f64 / 1_048_576.0,
        // what actually dominates: the pair-density grid each route forms
        (nao * nao * ng * 16) as f64 / 1_048_576.0,
        (nocc * nvir * ng * 16) as f64 / 1_048_576.0,
    );
    if want_ao && want_mo {
        println!(
            "[{label}] measured_wallclock_ratio_ao_over_mo={:.3} residual={worst:.3e}",
            ao_secs / mo_secs
        );
        assert!(worst < 2e-11, "[{label}] routes disagree by {worst:.3e}");
    }
}

#[test]
#[ignore = "D-PBC-28 measurement; run with --release --ignored --nocapture"]
fn mo_first_vs_ao_first_cost() {
    sweep("diamond/gth-szv [1,1,2]", diamond_szv(), [1, 1, 2], 4);
}

/// The `[2,2,2]` sweep is 512 quadruples on the same 47^3 mesh — 64x the
/// `[1,1,2]` row. Split out so the cheap row is always available.
#[test]
#[ignore = "D-PBC-28 measurement, LONG; run with --release --ignored --nocapture"]
fn mo_first_vs_ao_first_cost_222() {
    sweep("diamond/gth-szv [2,2,2]", diamond_szv(), [2, 2, 2], 4);
}
