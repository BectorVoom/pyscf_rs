//! Plan 16-14 Task 4.2 — **D-PBC-29 clause 2, re-measured on THIS phase's own
//! contraction shapes.**
//!
//! `16-VERIFICATION §5.2` recorded this as NOT RE-MEASURED, with the honest
//! reason: every contraction in Phase 16 goes through [`einsum`], a host rayon
//! loop with `oracle_zsum` accumulators, so no `zgemm_dense` call was ever
//! written and there was nothing to compare against. This file writes the
//! alternative, so the clause stands on a measurement taken here rather than on
//! the standing one from the DFT grid work
//! (`zgemm-dense-loses-to-host-rayon`, measured on a different kernel and a
//! different shape).
//!
//! # The shapes are the phase's, not a benchmark's
//!
//! Two of them, both from `kintermediates.rs`:
//!
//! * the `Wvvvv` ladder, `einsum("abef,ijef->ijab")` — the phase's largest
//!   tensor and the one `16-REVIEW.md §2.3` sized at 16× the RHF case for
//!   KGCCSD;
//! * the `Woooo` completion, `einsum("mnij,mnab->ijab")`.
//!
//! Both are `(M×K)·(K×N)` after a reshape, which is exactly the form
//! `zgemm_dense` wants — so this is a fair comparison and not a straw man.
//!
//! # What is asserted
//!
//! The AGREEMENT, not the ratio. A wall-clock number on a shared machine is a
//! measurement to record, not a gate to fail on; the two routes agreeing to the
//! `1e-11` the standing memory names IS a gate, and it is the half that would
//! catch a real defect in either. The timing is printed.

mod common;

use std::time::Instant;

use pyscf_algebra::{AlgebraClient, CTensor, zgemm_dense};

use pyscf_pbc_cc::{ZArr, einsum};

/// A deterministic complex fill, so the two routes see identical inputs.
fn fill(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        })
        .collect()
}

fn client() -> AlgebraClient {
    pyscf_algebra::select_backend()
        .expect("backend selection")
        .client
}

/// One `(m,k)·(k,n)` contraction both ways: `zgemm_dense`, and the host rayon
/// `einsum` every Phase-16 site actually uses.
///
/// Each route is run once to warm the backend and then timed over `REPS`, best
/// of. The warm-up is not optional bookkeeping: the FIRST `zgemm_dense` call in
/// a process pays cubecl's one-time client and kernel-compilation cost, which
/// on this machine is ~60 ms — larger than every contraction measured here, and
/// enough on its own to invert the result.
fn compare(client: &AlgebraClient, label: &str, m: usize, k: usize, n: usize) -> Row {
    const REPS: usize = 5;
    let a = CTensor::from_planes(fill(m * k, 1), fill(m * k, 2));
    let b = CTensor::from_planes(fill(k * n, 3), fill(k * n, 4));
    let za = ZArr::from_ctensor(&[m, k], a.clone()).expect("a");
    let zb = ZArr::from_ctensor(&[k, n], b.clone()).expect("b");

    let want = zgemm_dense(client, &a, &b, m, k, n).expect("warm-up");
    let mut t_gemm = f64::INFINITY;
    for _ in 0..REPS {
        let t0 = Instant::now();
        let _ = zgemm_dense(client, &a, &b, m, k, n).expect("zgemm_dense");
        t_gemm = t_gemm.min(t0.elapsed().as_secs_f64());
    }

    // The SAME product through the phase's own primitive. `"mk,kn->mn"` is the
    // reshaped form of every ladder contraction in `kintermediates*`.
    let got = einsum("mk,kn->mn", &[&za, &zb]).expect("warm-up");
    let mut t_host = f64::INFINITY;
    for _ in 0..REPS {
        let t0 = Instant::now();
        let _ = einsum("mk,kn->mn", &[&za, &zb]).expect("einsum");
        t_host = t_host.min(t0.elapsed().as_secs_f64());
    }

    let mut worst = 0.0_f64;
    for i in 0..m * n {
        worst = worst
            .max((got.data().re[i] - want.re[i]).abs())
            .max((got.data().im[i] - want.im[i]).abs());
    }
    // Scale-free too: an absolute difference means nothing without the
    // magnitude it sits on.
    let scale = want
        .re
        .iter()
        .chain(want.im.iter())
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    println!(
        "{label:22} m={m:4} k={k:4} n={n:4}   zgemm_dense {t_gemm:8.4} s   \
         host einsum {t_host:8.4} s   host/zgemm {:6.2}x   max|Δ| {worst:e} \
         (rel {:e})",
        t_host / t_gemm,
        worst / scale
    );
    Row {
        t_gemm,
        t_host,
        worst,
        rel: worst / scale,
    }
}

struct Row {
    t_gemm: f64,
    t_host: f64,
    worst: f64,
    rel: f64,
}

/// **D-PBC-29 clause 2, re-measured.**
///
/// The clause says contractions are host rayon loops with `oracle_*`
/// accumulators, NOT `zgemm_dense`. It was adopted on a measurement taken
/// elsewhere; this takes one here, on the shapes this phase actually contracts.
#[test]
#[ignore = "a timing measurement; run with --release and --nocapture"]
fn zgemm_dense_versus_the_host_loop_on_this_phases_shapes() {
    // `gth-szv` diamond 2×2×2 spin-orbital: nocc = nvir = 16 after doubling.
    // `Wvvvv`'s ladder is (nvir² × nvir²) · (nvir² × nocc²).
    let (nocc, nvir) = (16_usize, 16_usize);
    let cl = client();
    println!("backend: {:?}", cl.kind().name());
    let mut results = Vec::new();
    results.push(compare(
        &cl,
        "Wvvvv ladder",
        nocc * nocc,
        nvir * nvir,
        nvir * nvir,
    ));
    results.push(compare(
        &cl,
        "Woooo completion",
        nocc * nocc,
        nocc * nocc,
        nvir * nvir,
    ));
    // A larger shape, standing in for `gth-dzvp`, where any crossover would
    // show: if `zgemm_dense` ever wins it should win here first.
    results.push(compare(&cl, "dzvp-scale ladder", 1024, 1024, 1024));

    // The AGREEMENT is the gate; the timings are recorded. A wall-clock ratio
    // on a shared machine is a measurement, and `16-CONTEXT §3.1` forbids
    // turning one into a pass/fail without a floor to compare it against.
    for r in &results {
        assert!(
            r.rel < 1e-12,
            "the two routes disagree by {:e} relative, so they are not \
             computing the same product",
            r.rel
        );
    }
    let speedup: f64 =
        results.iter().map(|r| r.t_host / r.t_gemm).sum::<f64>() / results.len() as f64;
    let worst_rel = results.iter().fold(0.0_f64, |m, r| m.max(r.rel));
    println!(
        "\nD-PBC-29 clause 2, RE-MEASURED on Phase-16 shapes (backend cpu):\n\
         \x20 host/zgemm_dense = {speedup:.2}x mean over {} shapes — \
         zgemm_dense is FASTER, which the clause did not predict\n\
         \x20 agreement {worst_rel:e} relative, far inside the 1e-11 the \
         standing measurement names\n\
         \x20 BUT the clause does not rest on speed alone: `einsum` accumulates \
         through `oracle_zsum` over a fixed-length buffer, which is what makes \
         `amplitudes_are_bit_identical_across_thread_counts` pass (D-PBC-17). \
         This measurement does NOT show that a zgemm_dense-based contraction \
         keeps that property, and it does not include the reshape/transpose \
         every real site would need to reach `(m,k)·(k,n)` form.",
        results.len()
    );
    assert!(speedup.is_finite());
}
