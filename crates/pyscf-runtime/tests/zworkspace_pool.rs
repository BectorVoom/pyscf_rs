//! Plan 16-02 Task 4, tests 1-4 — the complex arena.
//!
//! Every assertion here is an exact integer or a bit-identity. This file has no
//! tolerances anywhere, by construction: an arena whose byte accounting is
//! "close enough" is an arena that under-reports to a HARD refusal.

use pyscf_runtime::{BackendError, WorkspacePool, ZWorkspacePool};

/// Test 1 — the clause-1 trap. A `[2,3,4]` COMPLEX reservation is **384**
/// bytes: `2*3*4 = 24` elements at 16 B each. The f64 arena reports **192** for
/// the same shape (`24 * 8`).
///
/// `16-02-PLAN.md` Task 4 test 1 asks for the literal `192` "not 96". Both of
/// its numbers are the f64 count and half of it: `24 * 8` and `12 * 8`. Using
/// the plan's literal would have encoded EXACTLY the defect this test exists to
/// catch — a complex tensor sized with `* 8`, reporting half its footprint to
/// the HARD refusal (`16-CONTEXT §1.3`, D-PBC-29 clause 1). The literal here is
/// the derived one, and the deviation is recorded in `16-02-SUMMARY.md`.
#[test]
fn complex_shape_bytes_is_sixteen_per_element() {
    assert_eq!(ZWorkspacePool::shape_bytes(&[2, 3, 4]), 384);
    // The f64 sibling still reports 192 for the same shape — the two arenas are
    // deliberately different, and this asserts they have not been unified.
    let pool = WorkspacePool::new(192);
    assert!(pool.try_reserve(192).is_ok());
    assert!(pool.try_reserve(193).is_err());

    let zpool = ZWorkspacePool::new(383);
    assert!(matches!(
        zpool.reserve(&[2, 3, 4], false),
        Err(BackendError::MemoryLimitExceeded {
            requested: 384,
            limit: 383
        })
    ));
    let zpool = ZWorkspacePool::new(384);
    assert!(zpool.reserve(&[2, 3, 4], false).is_ok());
}

/// Test 2 — the HARD refusal (D-01). Over budget with `allow_spill == false`
/// refuses and allocates NOTHING; there is no silent downgrade to a spill.
#[test]
fn over_budget_without_spill_hard_refuses() {
    let pool = ZWorkspacePool::new(1);
    let err = pool
        .reserve(&[100], false)
        .expect_err("over-budget in-core must refuse");
    match err {
        BackendError::MemoryLimitExceeded { requested, limit } => {
            assert_eq!(requested, 1600);
            assert_eq!(limit, 1);
        }
        other => panic!("expected MemoryLimitExceeded, got {other:?}"),
    }
    assert_eq!(pool.allocation_count(), 0, "no silent downgrade");
    assert_eq!(pool.live_inmem_bytes(), 0);
}

/// Test 3 — free-list reuse. reserve → release → reserve of a fitting shape
/// hands the SAME buffer back and does not grow `allocation_count`.
#[test]
fn release_then_reserve_reuses_the_same_buffer() {
    let pool = ZWorkspacePool::new(ZWorkspacePool::DEFAULT_BUDGET_BYTES);
    let shape = [8usize, 8, 4, 4];

    let a = pool.reserve(&shape, false).expect("first reserve fits");
    pool.release(a);
    let b = pool.reserve(&shape, false).expect("second reserve reuses");
    assert_eq!(a, b);
    assert_eq!(pool.allocation_count(), 1);

    // A smaller request reuses the larger free buffer; an in-use one does not
    // get handed out twice.
    pool.release(b);
    let small = pool.reserve(&[4], false).unwrap();
    assert_eq!(small, a);
    let other = pool.reserve(&[4], false).unwrap();
    assert_ne!(other, a, "an in-use buffer must not be handed out twice");
    assert_eq!(pool.allocation_count(), 2);
}

/// Test 4 — spill round-trip is bit-identical, and the temp file is removed on
/// drop (RAII, the T-06-02-LEAK mitigation the f64 pool also carries).
#[test]
fn spilled_complex_buffer_roundtrips_bit_identically() {
    let pool = ZWorkspacePool::new(16); // one complex element fits in-core.
    let id = pool
        .reserve(&[50], true)
        .expect("spill permitted over the in-memory budget");
    assert!(
        pool.is_spilled(&id).unwrap(),
        "must have chosen the spill tier"
    );
    assert_eq!(
        pool.charged_bytes(&id).unwrap(),
        0,
        "a spilled buffer charges nothing against the in-memory budget"
    );

    let re: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
    let im: Vec<f64> = (0..50).map(|i| -(i as f64) * 0.3).collect();
    pool.write_planes(&id, &re, &im).unwrap();
    let (gre, gim) = pool.read_planes(&id).unwrap();
    assert_eq!(gre, re, "spill round-trip must be bit-identical");
    assert_eq!(gim, im, "spill round-trip must be bit-identical");

    // In-memory round-trip is bit-identical too.
    let mem = ZWorkspacePool::new(ZWorkspacePool::DEFAULT_BUDGET_BYTES);
    let mid = mem.reserve(&[50], false).unwrap();
    assert!(!mem.is_spilled(&mid).unwrap());
    assert_eq!(mem.charged_bytes(&mid).unwrap(), 800);
    mem.write_planes(&mid, &re, &im).unwrap();
    assert_eq!(mem.read_planes(&mid).unwrap(), (re, im));
}

/// `with_mut_slices` edits in place on both tiers, and the registry lock is not
/// held across the closure — a second buffer is reachable from inside it.
#[test]
fn mutable_access_does_not_hold_the_registry_lock() {
    let pool = ZWorkspacePool::new(ZWorkspacePool::DEFAULT_BUDGET_BYTES);
    let a = pool.reserve(&[4], false).unwrap();
    let b = pool.reserve(&[4], false).unwrap();
    pool.write_planes(&b, &[9.0; 4], &[8.0; 4]).unwrap();

    // If `with_mut_slices` held the pool's own mutex (as `WorkspacePool::
    // with_mut_slice` does, `workspace_pool.rs:461-483`) this would deadlock.
    let seen = pool
        .with_mut_slices(&a, |re, im| {
            re[0] = 1.0;
            im[0] = 2.0;
            pool.read_planes(&b).unwrap()
        })
        .unwrap();
    assert_eq!(seen.0, vec![9.0; 4]);
    assert_eq!(seen.1, vec![8.0; 4]);
    assert_eq!(pool.read_planes(&a).unwrap().0[0], 1.0);
    assert_eq!(pool.read_planes(&a).unwrap().1[0], 2.0);
}

/// A plane-length mismatch is an error, not a silent truncation: the planes of
/// a `CTensor` are always equal length (RULE 8) and a write that breaks the
/// invariant must say so.
#[test]
fn plane_length_mismatch_is_refused() {
    let pool = ZWorkspacePool::new(ZWorkspacePool::DEFAULT_BUDGET_BYTES);
    let id = pool.reserve(&[4], false).unwrap();
    assert!(pool.write_planes(&id, &[1.0, 2.0], &[1.0]).is_err());
    assert!(pool.write_planes(&id, &[0.0; 5], &[0.0; 5]).is_err());
}

/// **A recycled buffer comes back ZEROED.**
///
/// `reserve` reuses a free-listed buffer of sufficient capacity instead of
/// allocating, which is the arena's whole point. Before 16-06 it handed the new
/// tenant the old one's bytes, so any caller that ACCUMULATES into what it was
/// told is a fresh allocation inherited them.
///
/// That is not a hypothetical: `kccsd_uhf::update_amps` builds `cc_Woooo`,
/// releases it, builds `cc_Wvvvv_half`, releases it, then builds `cc_Wovvo`,
/// which accumulates from many scattered k-addresses. `Wovvo` came back
/// carrying `Wvvvv`, and every doubles amplitude was ~1e-2 wrong while each of
/// the five equation stages was independently exact against upstream to 1e-11.
#[test]
fn a_recycled_buffer_comes_back_zeroed() {
    let pool = ZWorkspacePool::new(ZWorkspacePool::DEFAULT_BUDGET_BYTES);
    let a = pool.reserve(&[4, 4], false).expect("reserve");
    pool.write_planes(&a, &[3.0; 16], &[-3.0; 16])
        .expect("write");
    let n = pool.allocation_count();
    pool.release(a);

    // Same shape: must be the same storage, and must be zero.
    let b = pool.reserve(&[4, 4], false).expect("reserve");
    assert_eq!(
        pool.allocation_count(),
        n,
        "the second reserve must reuse, or this proves nothing"
    );
    let (re, im) = pool.read_planes(&b).expect("read");
    assert!(
        re.iter().all(|v| *v == 0.0) && im.iter().all(|v| *v == 0.0),
        "a reused buffer handed back the previous tenant's data"
    );
    pool.release(b);

    // Smaller shape into a larger buffer: the whole capacity is cleared, so a
    // later over-length read cannot see the tail either.
    let c = pool.reserve(&[2, 2], false).expect("reserve");
    assert_eq!(pool.allocation_count(), n, "must still reuse");
    let (re, im) = pool.read_planes(&c).expect("read");
    assert!(
        re.iter().all(|v| *v == 0.0) && im.iter().all(|v| *v == 0.0),
        "the tail beyond the new shape was left dirty"
    );
}
