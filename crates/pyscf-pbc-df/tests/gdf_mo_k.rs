//! `gdf::jk::get_k_kpts_mo` — the MO-factorised `get_k_kpts` route
//! (`df_jk.py:281-685`, `force_dm_kbuild = False`) — plan 17-10 Task 4.
//!
//! Gated against THIS PORT'S OWN density-matrix branch at 1e-13 — "two
//! routes to the same number inside one process is a stronger test than
//! either against a third implementation" (17-10-PLAN.md Task 4) — not
//! against upstream. `dm := C_occ · diag(occ) · C_occ†` is built explicitly
//! so the DM branch runs on the IDENTICAL physical input the MO branch
//! implicitly contracts; any divergence is therefore a bug in the MO route's
//! `U`-contraction, not a difference of physical inputs.

mod common;

use pyscf_algebra::CTensor;
use pyscf_pbc_df::df_jk::KMats;
use pyscf_pbc_df::gdf::Gdf;
use pyscf_pbc_df::gdf::jk::{get_k_kpts, get_k_kpts_mo};
use pyscf_pbc_gto::ExxDiv;

fn kpts(cell: &pyscf_pbc_gto::Cell, km: [usize; 3]) -> Vec<[f64; 3]> {
    pyscf_pbc_gto::kpts_mesh::make_kpts(cell, km, false, true, None).expect("kpts")
}

/// Deterministic complex "MO" coefficients, `nao x nocc`, row-major
/// (`p*nocc+o`), plus matching occupations. Not physical SCF output — the
/// gate is the `U`-contraction identity, not a physical MO set.
fn model_mo(nao: usize, nocc: usize, k: usize) -> (CTensor, Vec<f64>) {
    let mut c = CTensor::zeros(nao * nocc);
    for p in 0..nao {
        for o in 0..nocc {
            let idx = p * nocc + o;
            c.re[idx] = ((p as f64 + 1.0) / (o as f64 + 2.0) + 0.05 * k as f64).sin();
            c.im[idx] = ((p as f64 - o as f64) * 0.3 + 0.02 * k as f64).cos() * 0.1;
        }
    }
    let occ: Vec<f64> = (0..nocc).map(|o| 2.0 / (1.0 + o as f64)).collect();
    (c, occ)
}

/// `dm[q,k] = SUM_o C[q,o] * occ[o] * conj(C[k,o])` — the SAME density the MO
/// route implicitly contracts, built explicitly so [`get_k_kpts`] can run on
/// the identical physical input.
fn dm_from_mo(c: &CTensor, occ: &[f64], nao: usize, nocc: usize) -> CTensor {
    let mut dm = CTensor::zeros(nao * nao);
    for q in 0..nao {
        for k in 0..nao {
            let (mut re, mut im) = (0.0_f64, 0.0_f64);
            for o in 0..nocc {
                let (cqr, cqi) = (c.re[q * nocc + o], c.im[q * nocc + o]);
                let (ckr, cki) = (c.re[k * nocc + o], c.im[k * nocc + o]);
                let (pr, pi) = (cqr * ckr + cqi * cki, cqi * ckr - cqr * cki);
                re += occ[o] * pr;
                im += occ[o] * pi;
            }
            dm.re[q * nao + k] = re;
            dm.im[q * nao + k] = im;
        }
    }
    dm
}

fn worst(a: &KMats, b: &KMats) -> f64 {
    let mut w = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
        for i in 0..x.re.len() {
            w = w.max((x.re[i] - y.re[i]).abs());
            w = w.max((x.im[i] - y.im[i]).abs());
        }
    }
    w
}

#[test]
fn mo_route_matches_dm_route_diamond() {
    let cell = common::diamond();
    let k = kpts(&cell, [2, 2, 2]);
    let mut df = Gdf::new(cell, &k);
    df.build().expect("Gdf::build");
    let nao = df.cell.mol.nao_nr;
    let nocc = nao / 2;
    assert!(
        nocc > 0 && nocc < nao,
        "fixture must give a genuine nocc < nao gap"
    );

    let mut mo_coeff = Vec::with_capacity(k.len());
    let mut mo_occ = Vec::with_capacity(k.len());
    let mut dms0 = KMats::with_capacity(k.len());
    for ki in 0..k.len() {
        let (c, occ) = model_mo(nao, nocc, ki);
        dms0.push(dm_from_mo(&c, &occ, nao, nocc));
        mo_coeff.push(c);
        mo_occ.push(occ);
    }
    let dms = vec![dms0];
    let mo_coeff_sets = vec![mo_coeff];
    let mo_occ_sets = vec![mo_occ];

    for exxdiv in [None, Some(ExxDiv::Ewald)] {
        let vk_dm = get_k_kpts(&df, &dms, &k, exxdiv).expect("dm route");
        let vk_mo = get_k_kpts_mo(&df, &mo_coeff_sets, &mo_occ_sets, &k, exxdiv).expect("mo route");

        let w = worst(&vk_dm[0], &vk_mo[0]);
        assert!(
            w < 1e-13,
            "MO route diverges from the DM route on the SAME physical dm \
             (exxdiv={exxdiv:?}): {w:e}"
        );
    }
}

/// **Not a correctness gate — a measurement.** Reports wall-clock for both
/// routes on this port's naive (no-BLAS) loops, per the plan's own
/// instruction: "if it is not actually faster... say so explicitly". Run
/// explicitly; excluded from the default `cargo test` pass because a
/// meaningful rep count takes real wall-clock, and a timing number is not a
/// pass/fail signal `cargo test`'s default run should gate on.
#[test]
#[ignore = "timing, not correctness — run explicitly: cargo test -p pyscf-pbc-df \
            --release --test gdf_mo_k -- --ignored --nocapture"]
fn mo_route_wall_clock_vs_dm_route_diamond() {
    let cell = common::diamond();
    let k = kpts(&cell, [2, 2, 2]);
    let mut df = Gdf::new(cell, &k);
    df.build().expect("Gdf::build");
    let nao = df.cell.mol.nao_nr;
    let nocc = nao / 2;

    let mut mo_coeff = Vec::with_capacity(k.len());
    let mut mo_occ = Vec::with_capacity(k.len());
    let mut dms0 = KMats::with_capacity(k.len());
    for ki in 0..k.len() {
        let (c, occ) = model_mo(nao, nocc, ki);
        dms0.push(dm_from_mo(&c, &occ, nao, nocc));
        mo_coeff.push(c);
        mo_occ.push(occ);
    }
    let dms = vec![dms0];
    let mo_coeff_sets = vec![mo_coeff];
    let mo_occ_sets = vec![mo_occ];

    const REPS: usize = 200;
    let t0 = std::time::Instant::now();
    for _ in 0..REPS {
        std::hint::black_box(get_k_kpts(&df, &dms, &k, Some(ExxDiv::Ewald)).expect("dm route"));
    }
    let dm_elapsed = t0.elapsed();

    let t1 = std::time::Instant::now();
    for _ in 0..REPS {
        std::hint::black_box(
            get_k_kpts_mo(&df, &mo_coeff_sets, &mo_occ_sets, &k, Some(ExxDiv::Ewald))
                .expect("mo route"),
        );
    }
    let mo_elapsed = t1.elapsed();

    eprintln!(
        "nao={nao} nocc={nocc} nkpts={} reps={REPS}: dm={dm_elapsed:?} mo={mo_elapsed:?} \
         speedup={:.3}x",
        k.len(),
        dm_elapsed.as_secs_f64() / mo_elapsed.as_secs_f64()
    );
}
