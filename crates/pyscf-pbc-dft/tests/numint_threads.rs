//! W-06 (`.planning/pbc/KRKS-OPTIMISATION-PLAN.md`) — Gate B for `KNumInt`.
//!
//! W-06 parallelises `eval_rho_one` and `vxc_mat_one` over a **disjoint output
//! partition**: one rayon worker owns one output row (or one disjoint chunk of
//! the grid, where the grid IS the output index), and every reduction axis
//! stays serial and ascending. `oracle_sum`'s pairwise tree depends only on the
//! input length and the fixed `PAIRWISE_CHUNK`, never on which thread evaluates
//! it, so D-PBC-17's thread-count invariance survives the change by
//! construction.
//!
//! This asserts that: `nelec`, `excsum` and every element of `vmat` are
//! bit-identical across worker counts, for LDA and GGA, at gamma and at
//! 2x2x2. `nelec`/`excsum` are the sharp end — they are `oracle_sum` outputs
//! that land directly in the total energy.
//!
//! The worker count is varied INSIDE one process with explicit
//! `rayon::ThreadPool`s, which is strictly stronger than a `RAYON_NUM_THREADS`
//! sweep across processes: it also catches a result that depends on which
//! worker happened to steal a chunk.

mod common;

use pyscf_algebra::CTensor;
use pyscf_pbc_dft::gen_grid::PeriodicGrids;
use pyscf_pbc_dft::numint::KNumInt;
use pyscf_pbc_gto::make_kpts_default;

/// Small enough for a unit-test budget, large enough that `RHO_CHUNK` (512)
/// splits: 11^3 = 1331 grid points.
const MESH_FAST: [usize; 3] = [11, 11, 11];

/// `3` is deliberately not a divisor of `nao` (8) or of the chunk count, so a
/// ragged final partition is exercised.
const THREADS: [usize; 4] = [1, 2, 3, 8];

fn model_dms(nao: usize, nkpts: usize) -> Vec<Vec<CTensor>> {
    vec![
        (0..nkpts)
            .map(|k| {
                let mut m = CTensor::zeros(nao * nao);
                for p in 0..nao {
                    for q in 0..nao {
                        let v = 0.3 / (1.0 + (p as f64 - q as f64).abs())
                            + if p == q { 1.0 } else { 0.0 };
                        m.re[p * nao + q] = v * (1.0 + 0.1 * k as f64);
                        m.im[p * nao + q] = 0.02 * (p as f64 - q as f64);
                    }
                }
                // The density matrix must be Hermitian — `nr_rks` requires
                // `hermi = 1` and the imaginary residue is a diagnostic, not a
                // quantity.
                for p in 0..nao {
                    for q in 0..p {
                        m.re[q * nao + p] = m.re[p * nao + q];
                        m.im[q * nao + p] = -m.im[p * nao + q];
                    }
                    m.im[p * nao + p] = 0.0;
                }
                m
            })
            .collect(),
    ]
}

fn pool(n: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .expect("thread pool")
}

#[test]
fn nr_rks_is_bit_identical_across_thread_counts() {
    let cell = common::diamond();
    let grids = PeriodicGrids::uniform(&cell, Some(MESH_FAST)).expect("grids");
    for nk in [[1, 1, 1], [2, 2, 2]] {
        let kpts = make_kpts_default(&cell, nk).expect("kpts");
        let dms = model_dms(cell.mol.nao_nr, kpts.len());
        // `LDA,VWN` exercises `nvar = 1`; `PBE` exercises the GGA path, where
        // `vxc_mat_one`'s `aow` stage sums over four components and
        // `eval_rho_one` fills four `rho` rows.
        for xc in ["LDA,VWN", "PBE"] {
            // `KNumInt` holds a `std::cell::Cell` (the imaginary-residue
            // diagnostic) and is deliberately NOT `Sync`, so it is built inside
            // the pool rather than shared into it. That also gives every run a
            // cold AO cache, which is the honest comparison.
            let run = |t: usize| {
                pool(t).install(|| {
                    let ni = KNumInt::new(&kpts);
                    ni.nr_rks(&cell, &grids, xc, &dms, 1, None).expect("nr_rks")
                })
            };
            let reference = run(THREADS[0]);
            for &t in &THREADS[1..] {
                let got = run(t);
                assert_eq!(
                    reference.nelec[0].to_bits(),
                    got.nelec[0].to_bits(),
                    "nelec moved between 1 and {t} threads (xc={xc}, nk={nk:?}): {} vs {}",
                    reference.nelec[0],
                    got.nelec[0]
                );
                assert_eq!(
                    reference.excsum[0].to_bits(),
                    got.excsum[0].to_bits(),
                    "excsum moved between 1 and {t} threads (xc={xc}, nk={nk:?}): {} vs {}",
                    reference.excsum[0],
                    got.excsum[0]
                );
                for (k, (a, b)) in reference.vmat[0].iter().zip(&got.vmat[0]).enumerate() {
                    for i in 0..a.len() {
                        assert_eq!(
                            a.re[i].to_bits(),
                            b.re[i].to_bits(),
                            "vmat[{k}].re[{i}] moved at {t} threads (xc={xc}, nk={nk:?})"
                        );
                        assert_eq!(
                            a.im[i].to_bits(),
                            b.im[i].to_bits(),
                            "vmat[{k}].im[{i}] moved at {t} threads (xc={xc}, nk={nk:?})"
                        );
                    }
                }
            }
        }
    }
}
