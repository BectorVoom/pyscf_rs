//! W-02b (`.planning/pbc/KRKS-OPTIMISATION-PLAN.md`) — Gate B for `fft_jk`.
//!
//! W-02b parallelises every contraction in `fft_jk.rs` over a **disjoint output
//! partition**: one rayon worker owns one output row and no two workers touch
//! the same cell, so each output cell's own summation order is untouched. The
//! claim that follows is not "close enough" but **bit-identical for any thread
//! count**, which is what D-PBC-17 (`PBC-MASTER-PLAN.md:247`) requires of every
//! reduction that reaches an energy or a density matrix — and `get_j_kpts` /
//! `get_k_kpts` reach both, through `ecoul` and `exc`.
//!
//! The thread count is varied INSIDE one process with explicit
//! `rayon::ThreadPool`s rather than by re-running under `RAYON_NUM_THREADS`,
//! so a single `cargo test` run proves the property (an env-var sweep can only
//! ever compare two separate processes, and would not catch a result that
//! depends on which worker happened to steal a chunk).
//!
//! Failure here means the parallelisation touched a REDUCTION axis somewhere —
//! see the two rules in `fft_jk.rs`'s module docs.

mod common;

use pyscf_algebra::CTensor;
use pyscf_pbc_df::df_jk::KMats;
use pyscf_pbc_df::{Fftdf, get_j_kpts, get_k_kpts};
use pyscf_pbc_gto::{ExxDiv, make_kpts_default};

/// Small enough to run in a unit-test budget, large enough that every chunked
/// loop in `fft_jk.rs` actually splits (`RHO_CHUNK` is 512; 11^3 = 1331).
const MESH_FAST: [usize; 3] = [11, 11, 11];

/// The thread counts the property is asserted over. `1` is the serial
/// reference; `3` is deliberately not a divisor of any loop bound here, so a
/// ragged final chunk is exercised.
const THREADS: [usize; 4] = [1, 2, 3, 8];

fn model_dm(nao: usize, nkpts: usize) -> KMats {
    (0..nkpts)
        .map(|k| {
            let mut m = CTensor::zeros(nao * nao);
            for p in 0..nao {
                for q in 0..nao {
                    let v =
                        0.3 / (1.0 + (p as f64 - q as f64).abs()) + if p == q { 1.0 } else { 0.0 };
                    m.re[p * nao + q] = v * (1.0 + 0.1 * k as f64);
                    // A non-zero imaginary part matters: it is what makes the
                    // `if dr == 0.0 && di == 0.0 { continue; }` skips in
                    // `accumulate_rho` / `dm_times_conj_ao` take both branches.
                    m.im[p * nao + q] = 0.05 * (p as f64 - q as f64) * (1.0 + 0.03 * k as f64);
                }
            }
            m
        })
        .collect()
}

fn pool(n: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .expect("thread pool")
}

/// `==` on every raw `f64`, not `approx` — the whole point of the item.
fn assert_bit_identical(a: &[KMats], b: &[KMats], threads: usize, who: &str) {
    assert_eq!(a.len(), b.len(), "{who}: nset changed at {threads} threads");
    for (iset, (sa, sb)) in a.iter().zip(b).enumerate() {
        assert_eq!(sa.len(), sb.len(), "{who}: nband changed at {threads}");
        for (k, (ma, mb)) in sa.iter().zip(sb).enumerate() {
            for i in 0..ma.len() {
                assert_eq!(
                    ma.re[i].to_bits(),
                    mb.re[i].to_bits(),
                    "{who}: re[{i}] of set {iset} band {k} moved between 1 and {threads} threads \
                     ({} vs {})",
                    ma.re[i],
                    mb.re[i]
                );
                assert_eq!(
                    ma.im[i].to_bits(),
                    mb.im[i].to_bits(),
                    "{who}: im[{i}] of set {iset} band {k} moved between 1 and {threads} threads \
                     ({} vs {})",
                    ma.im[i],
                    mb.im[i]
                );
            }
        }
    }
}

#[test]
fn get_j_kpts_is_bit_identical_across_thread_counts() {
    let cell = common::diamond();
    for nk in [[1, 1, 1], [2, 2, 2], [1, 1, 3]] {
        let kpts = make_kpts_default(&cell, nk).expect("kpts");
        let df = Fftdf::with_mesh(cell.clone(), &kpts, MESH_FAST).expect("fftdf");
        let dms = vec![model_dm(cell.mol.nao_nr, kpts.len())];

        let reference = pool(THREADS[0])
            .install(|| get_j_kpts(&df, &dms, 1, &kpts, None, None).expect("get_j_kpts"));
        for &t in &THREADS[1..] {
            let got =
                pool(t).install(|| get_j_kpts(&df, &dms, 1, &kpts, None, None).expect("get_j"));
            assert_bit_identical(&reference, &got, t, &format!("get_j_kpts nk={nk:?}"));
        }
    }
}

#[test]
fn get_k_kpts_is_bit_identical_across_thread_counts() {
    let cell = common::diamond();
    for nk in [[1, 1, 1], [2, 2, 2], [1, 1, 3]] {
        let kpts = make_kpts_default(&cell, nk).expect("kpts");
        // A fresh `Fftdf` per thread count would also re-do the W-01 cache; one
        // shared `Fftdf` is the harder test, because it proves the parallel path
        // and not the cache is what is being compared.
        let df = Fftdf::with_mesh(cell.clone(), &kpts, MESH_FAST).expect("fftdf");
        let dms = vec![model_dm(cell.mol.nao_nr, kpts.len())];

        // `omega = Some(..)` exercises the range-separated `coulG`, and
        // `ExxDiv::Ewald` the post-loop correction — both reach different code
        // in the pair loop.
        for omega in [None, Some(0.11)] {
            let reference = pool(THREADS[0]).install(|| {
                get_k_kpts(&df, &dms, 1, &kpts, None, Some(ExxDiv::Ewald), omega)
                    .expect("get_k_kpts")
            });
            for &t in &THREADS[1..] {
                let got = pool(t).install(|| {
                    get_k_kpts(&df, &dms, 1, &kpts, None, Some(ExxDiv::Ewald), omega)
                        .expect("get_k_kpts")
                });
                assert_bit_identical(
                    &reference,
                    &got,
                    t,
                    &format!("get_k_kpts nk={nk:?} omega={omega:?}"),
                );
            }
        }
    }
}
