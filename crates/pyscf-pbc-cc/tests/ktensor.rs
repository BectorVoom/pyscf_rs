//! Plan 16-02 Task 4, tests 10-12 — the k-indexed complex block container.
//!
//! No tolerances: tier selection is an exact byte comparison and block access
//! is a bit-identity.

use std::sync::Arc;
use std::thread;

use pyscf_algebra::CTensor;
use pyscf_pbc_cc::{KRank, KTensor, Tier};
use pyscf_runtime::ZWorkspacePool;

/// Test 10 — tier selection is the EXACT byte count, not upstream's estimate.
///
/// `kccsd_rhf.py:1100-1107`'s `_mem_usage` returns `nkpts³·nmo⁴·4·16` and
/// carries its own `# TODO: Improve incore estimate`; measured against the
/// seven blocks actually allocated it over-estimates 9.1× on diamond `gth-szv`
/// 2×2×2 and 6.2× on `gth-dzvp` 2×2×2 (`16-REVIEW.md §2.4`). Porting it would
/// import that factor into a HARD refusal, i.e. refuse jobs that fit
/// (D-PBC-29 clause 4).
///
/// Here: a `nkpts = 2`, rank-3, `[2,2]`-block tensor is `8 * 4 * 16 = 512`
/// bytes exactly. One byte under the budget it must refuse; at the budget it
/// must land in memory.
#[test]
fn tier_comes_from_the_exact_byte_count() {
    let nkpts = 2;
    let block = [2usize, 2];
    assert_eq!(KTensor::exact_bytes(nkpts, KRank::Three, &block), 512);
    assert_eq!(KTensor::exact_bytes(nkpts, KRank::One, &block), 128);
    assert_eq!(KTensor::exact_bytes(nkpts, KRank::Two, &block), 256);

    // Just under: HARD refusal, no silent downgrade.
    let tight = ZWorkspacePool::new(511);
    assert!(
        KTensor::zeros(&tight, nkpts, KRank::Three, &block, false).is_err(),
        "512 bytes must not fit a 511-byte budget"
    );

    // Exactly at: in memory.
    let fits = ZWorkspacePool::new(512);
    let t = KTensor::zeros(&fits, nkpts, KRank::Three, &block, false).expect("fits exactly");
    assert_eq!(t.tier(), Tier::InMemory);
    assert_eq!(t.bytes(), 512);
    assert_eq!(fits.live_inmem_bytes(), 512);

    // Over budget WITH the spill opt-in: the spilled tier, charging nothing
    // against the in-memory budget.
    let spill = ZWorkspacePool::new(64);
    let t = KTensor::zeros(&spill, nkpts, KRank::Three, &block, true).expect("spill permitted");
    assert_eq!(t.tier(), Tier::Spilled);
    assert_eq!(t.bytes(), 512);

    // And the third tier: not built at all.
    let absent = KTensor::absent(nkpts, KRank::Three, &block);
    assert_eq!(absent.tier(), Tier::Absent);
    assert_eq!(absent.bytes(), 512, "an absent tensor still knows its cost");
}

fn ramp(n: usize, off: f64) -> CTensor {
    CTensor::from_planes(
        (0..n).map(|i| i as f64 + off).collect(),
        (0..n).map(|i| -(i as f64) * 0.5 - off).collect(),
    )
}

/// Test 11 — block accessor round-trip is bit-identical on BOTH tiers, and the
/// two tiers agree with each other bit-for-bit.
///
/// Upstream gates its own incore-vs-outcore analogue at 12 decimals
/// (`test_krccsd.py:250-256`); this one is same-process, same code, same
/// inputs, so it is a bit-identity, not a tolerance.
#[test]
fn block_roundtrip_is_bit_identical_on_both_tiers() {
    let nkpts = 2;
    let block = [3usize, 3];
    let mem = ZWorkspacePool::new(ZWorkspacePool::DEFAULT_BUDGET_BYTES);
    let spl = ZWorkspacePool::new(16);

    let incore = KTensor::zeros(&mem, nkpts, KRank::Three, &block, false).unwrap();
    let outcore = KTensor::zeros(&spl, nkpts, KRank::Three, &block, true).unwrap();
    assert_eq!(incore.tier(), Tier::InMemory);
    assert_eq!(outcore.tier(), Tier::Spilled);

    for ki in 0..nkpts {
        for kj in 0..nkpts {
            for ka in 0..nkpts {
                let v = ramp(9, (ki * 100 + kj * 10 + ka) as f64);
                incore.set_block(&mem, &[ki, kj, ka], &v).unwrap();
                outcore.set_block(&spl, &[ki, kj, ka], &v).unwrap();
            }
        }
    }
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            for ka in 0..nkpts {
                let k = [ki, kj, ka];
                let want = ramp(9, (ki * 100 + kj * 10 + ka) as f64);
                assert_eq!(incore.block(&mem, &k).unwrap(), want);
                assert_eq!(outcore.block(&spl, &k).unwrap(), want);
                // The tier-equivalence assertion, bitwise.
                assert_eq!(incore.block(&mem, &k).unwrap(), outcore.block(&spl, &k).unwrap());
            }
        }
    }

    // Borrowing does not copy the tensor: `with_block` sees the same bits.
    let sum: f64 = incore
        .with_block(&mem, &[1, 0, 1], |re, _| re.iter().sum())
        .unwrap();
    assert_eq!(sum, ramp(9, 101.0).re.iter().sum::<f64>());

    // A bad k-address is an error, not a wrong block.
    assert!(incore.block(&mem, &[0, 0]).is_err());
    assert!(incore.block(&mem, &[0, 0, nkpts]).is_err());
    assert!(KTensor::absent(nkpts, KRank::Three, &block).block(&mem, &[0, 0, 0]).is_err());
}

/// Test 12 — two threads writing two DIFFERENT blocks make concurrent
/// progress.
///
/// `WorkspacePool::with_mut_slice` (`workspace_pool.rs:461-483`) runs the
/// caller's closure while holding the pool's single mutex, so two rayon threads
/// on two different buffers serialise — which would cap this whole phase at one
/// core (`16-REVIEW.md §2.2`). Here each allocation carries its own lock, so
/// the rendezvous below completes; under a global lock it would deadlock and
/// the test would hang rather than pass slowly.
#[test]
fn two_threads_write_two_blocks_concurrently() {
    let pool = Arc::new(ZWorkspacePool::new(ZWorkspacePool::DEFAULT_BUDGET_BYTES));
    let t = Arc::new(KTensor::zeros(&pool, 2, KRank::Three, &[4], false).unwrap());

    let barrier = Arc::new(std::sync::Barrier::new(2));
    let mut handles = Vec::new();
    for (n, k) in [(1.0_f64, [0usize, 0, 0]), (2.0, [1usize, 1, 1])] {
        let pool = Arc::clone(&pool);
        let t = Arc::clone(&t);
        let barrier = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            t.with_block_mut(&pool, &k, |re, im| {
                // Rendezvous INSIDE the block lock. A pool-wide lock makes this
                // unreachable for the second thread.
                barrier.wait();
                for (i, r) in re.iter_mut().enumerate() {
                    *r = n * (i as f64 + 1.0);
                }
                for (i, m) in im.iter_mut().enumerate() {
                    *m = -n * (i as f64 + 1.0);
                }
            })
            .unwrap();
        }));
    }
    for h in handles {
        h.join().expect("both writers must finish");
    }

    assert_eq!(t.block(&pool, &[0, 0, 0]).unwrap().re, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(t.block(&pool, &[1, 1, 1]).unwrap().re, vec![2.0, 4.0, 6.0, 8.0]);
    assert_eq!(t.block(&pool, &[0, 1, 0]).unwrap().re, vec![0.0; 4]);
}

/// The free-list still works through `KTensor`: releasing a tensor and
/// allocating the same shape again does not grow the arena.
#[test]
fn releasing_a_tensor_returns_its_blocks_to_the_free_list() {
    let pool = ZWorkspacePool::new(ZWorkspacePool::DEFAULT_BUDGET_BYTES);
    let a = KTensor::zeros(&pool, 2, KRank::Three, &[4, 4], false).unwrap();
    let n = pool.allocation_count();
    assert_eq!(n, 8, "one buffer per k-address");
    a.release(&pool);
    let _b = KTensor::zeros(&pool, 2, KRank::Three, &[4, 4], false).unwrap();
    assert_eq!(
        pool.allocation_count(),
        n,
        "the second tensor must reuse the first's buffers"
    );
}
