//! D-PBC-28 §7.1/§7.9 — does the rayon parallelism this phase shipped actually
//! land, and what does `build_symm_map`'s `O(nkpts^3)` cost look like.
//!
//! These are MEASUREMENTS. Bit-identity across thread counts IS asserted (it is
//! §9.3's determinism rule); wall clock is printed, never gated — a ratio near
//! 1.0 is a finding to record, not a failure. Both thread counts run inside one
//! process on explicit `rayon::ThreadPool`s, which is strictly stronger than an
//! env-var sweep across processes: the same built objects are compared.
//!
//! ```bash
//! cargo test --release -p pyscf-pbc-mp --test perf_dpbc28 -- --ignored --nocapture
//! ```

mod common;

use std::time::Instant;

use pyscf_algebra::CTensor;
use pyscf_pbc_df::Gdf;
use pyscf_pbc_lib::KptsHelper;
use pyscf_pbc_mp::{Kmp2, build_lov};
use pyscf_pbc_scf::{KScfConfig, Krhf};

fn pool(n: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .expect("thread pool")
}

fn same(a: &CTensor, b: &CTensor) -> bool {
    a.re == b.re && a.im == b.im
}

#[test]
#[ignore = "D-PBC-28 measurement; run with --release --ignored --nocapture"]
fn lov_build_and_kmp2_kernel_thread_scaling() {
    // `[1,1,2]`, not `[2,2,2]`: the Rust GDF's `make_j3c` over the fused cell
    // is already tens of minutes at `[1,1,2]` (`tests/gdf_builder.rs:477`), and
    // a `[2,2,2]` build did not finish in an hour. The parallel structure being
    // measured — `nkpts^2` disjoint Lov slots and an `nkpts^2` KMP2 outer loop —
    // is the same at both sizes; the ABSOLUTE times are not comparable across
    // meshes and are not used that way.
    let cell = common::diamond_anchor();
    let kpts = cell.make_kpts([1, 1, 2]).expect("kpts");
    let mut mf = Krhf::from_df(Box::new(Gdf::new(cell, &kpts)));
    mf.exxdiv = None;
    let mut cfg = KScfConfig::for_cell(mf.cell());
    cfg.conv_tol = 1e-11;
    let result = mf.kernel(&cfg).expect("SCF");
    assert!(result.converged);

    let mp = Kmp2::new(&result, mf.with_df.as_ref()).expect("KMP2");
    assert!(mp.with_df_ints, "this row measures the Lov route");
    let padded = mp.padded_mos().expect("padded MOs");

    // §7.1 row 1 — the Lov block build.
    let mut lov_secs = [0.0f64; 2];
    let mut tables = Vec::new();
    for (slot, threads) in [1usize, 8].into_iter().enumerate() {
        let p = pool(threads);
        let t = Instant::now();
        let table = p.install(|| build_lov(mp.with_df, &padded.mo_coeff, padded.nocc));
        lov_secs[slot] = t.elapsed().as_secs_f64();
        tables.push(table.expect("Lov"));
    }
    let nk = kpts.len();
    for ki in 0..nk {
        for kj in 0..nk {
            let (na, a) = tables[0].block(ki, kj);
            let (nb, b) = tables[1].block(ki, kj);
            assert_eq!(na, nb, "Lov naux differs at ({ki},{kj})");
            assert!(same(a, b), "Lov block ({ki},{kj}) is not bit-identical");
        }
    }
    println!(
        "[lov diamond/gth-szv [1,1,2]] t1={:.6}s t8={:.6}s speedup={:.3}x bit_identical=true",
        lov_secs[0],
        lov_secs[1],
        lov_secs[0] / lov_secs[1]
    );

    // §7.1 row 2 — the KMP2 `(ki,kj,ka)` loop.
    let mut kmp2_secs = [0.0f64; 2];
    let mut energies = Vec::new();
    let mut t2s = Vec::new();
    for (slot, threads) in [1usize, 8].into_iter().enumerate() {
        let p = pool(threads);
        let t = Instant::now();
        let r = p.install(|| mp.kernel()).expect("KMP2");
        kmp2_secs[slot] = t.elapsed().as_secs_f64();
        energies.push((r.e_corr, r.e_corr_ss, r.e_corr_os));
        t2s.push(r.t2.expect("T2"));
    }
    assert_eq!(
        energies[0], energies[1],
        "KMP2 energies are not bit-identical"
    );
    for b in 0..t2s[0].blocks.len() {
        assert!(
            same(&t2s[0].blocks[b], &t2s[1].blocks[b]),
            "T2 block {b} is not bit-identical"
        );
    }
    println!(
        "[kmp2 diamond/gth-szv [1,1,2] Lov route] t1={:.6}s t8={:.6}s speedup={:.3}x \
         e_corr={:.17} bit_identical=true",
        kmp2_secs[0],
        kmp2_secs[1],
        kmp2_secs[0] / kmp2_secs[1],
        energies[0].0
    );
}

/// The same two rows on a workload whose PER-TASK cost is not sub-millisecond.
///
/// On the `Lov` route at `[1,1,2]` the whole kernel is under 5 ms, so the
/// measured ratio is pool startup, not scaling — a real finding, and a reason
/// to measure again where each `(ki,kj,ka)` task does a full plane-wave
/// transform. The four-index FFTDF route is that workload: eight conserving
/// quadruples, roughly a second each.
#[test]
#[ignore = "D-PBC-28 measurement; run with --release --ignored --nocapture"]
fn kmp2_four_index_thread_scaling() {
    let cell = common::diamond_anchor();
    let kpts = cell.make_kpts([1, 1, 2]).expect("kpts");
    let mut mf = Krhf::new(cell, &kpts).expect("krhf");
    mf.exxdiv = None;
    let result = mf.run().expect("SCF");
    assert!(result.converged);
    let mut mp = Kmp2::new(&result, mf.with_df.as_ref()).expect("KMP2");
    mp.with_df_ints = false;

    let mut secs = [0.0f64; 2];
    let mut out = Vec::new();
    for (slot, threads) in [1usize, 8].into_iter().enumerate() {
        let p = pool(threads);
        let t = Instant::now();
        let r = p.install(|| mp.kernel()).expect("KMP2");
        secs[slot] = t.elapsed().as_secs_f64();
        out.push((r.e_corr, r.t2.expect("T2")));
    }
    assert_eq!(out[0].0, out[1].0, "e_corr is not bit-identical");
    for b in 0..out[0].1.blocks.len() {
        assert!(
            same(&out[0].1.blocks[b], &out[1].1.blocks[b]),
            "T2 block {b} is not bit-identical"
        );
    }
    println!(
        "[kmp2 diamond/gth-szv [1,1,2] four-index FFTDF route] t1={:.6}s t8={:.6}s \
         speedup={:.3}x e_corr={:.17} bit_identical=true",
        secs[0],
        secs[1],
        secs[0] / secs[1],
        out[0].0
    );
}

#[test]
#[ignore = "D-PBC-28 measurement; run with --release --ignored --nocapture"]
fn build_symm_map_growth_curve() {
    let cell = common::diamond_anchor();
    let mut prev: Option<(usize, f64)> = None;
    for n in [2usize, 3, 4, 5] {
        let kpts = cell.make_kpts([n, n, n]).expect("kpts");
        let nk = kpts.len();
        let t = Instant::now();
        let h = KptsHelper::new(&cell.a, &kpts);
        let secs = t.elapsed().as_secs_f64();
        let orbits = h.symm_map.as_ref().map_or(0, |m| m.entries().len());
        let ratio = prev
            .map(|(pn, ps)| {
                let cubic = (nk as f64 / pn as f64).powi(3);
                format!(
                    " measured_ratio={:.2} n^3_prediction={:.2}",
                    secs / ps,
                    cubic
                )
            })
            .unwrap_or_default();
        println!("[symm_map nkpts={nk}] secs={secs:.6} orbits={orbits}{ratio}");
        prev = Some((nk, secs));
    }
}
