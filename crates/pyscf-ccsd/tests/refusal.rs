//! CCSD-11 / D-01 HARD memory-refusal proof (threat T-06-02-OOM).
//!
//! An in-core reservation that exceeds the `PYSCF_MAX_MEMORY` budget must
//! return `BackendError::MemoryLimitExceeded` and must NOT be silently
//! downgraded to a spill / partial allocation. The user makes the
//! memory-vs-accuracy tradeoff deliberately (opt into DF / AO-direct) — the
//! kernel never starts an over-budget in-core job (no mid-iteration OOM).
//!
//! Exercises the `WorkspacePool` directly (no CCSD math required).

use pyscf_runtime::{BackendError, WorkspacePool};

/// `try_reserve` over budget returns `MemoryLimitExceeded` naming requested vs
/// limit (so the user knows the budget and can opt into DF/AO-direct).
#[test]
fn try_reserve_over_budget_refuses_with_named_bytes() {
    let budget = 1024usize; // 1 KiB.
    let pool = WorkspacePool::new(budget);
    let requested = 4096usize; // 4 KiB > budget.

    let err = pool
        .try_reserve(requested)
        .expect_err("over-budget try_reserve must refuse");
    match err {
        BackendError::MemoryLimitExceeded {
            requested: r,
            limit: l,
        } => {
            assert_eq!(r, requested, "error must name the requested bytes");
            assert_eq!(l, budget, "error must name the budget limit");
        }
        other => panic!("expected MemoryLimitExceeded, got {other:?}"),
    }

    // The error message must surface both numbers (user-facing guidance).
    let msg = err.to_string();
    assert!(
        msg.contains(&requested.to_string()) && msg.contains(&budget.to_string()),
        "error message must name requested + limit bytes, got: {msg}"
    );
}

/// `reserve(over_budget_shape, allow_spill=false)` HARD-refuses (D-01): returns
/// `MemoryLimitExceeded` and performs NO allocation (no silent downgrade).
#[test]
fn reserve_in_core_over_budget_refuses_no_downgrade() {
    // 256-byte budget; a [100] f64 buffer needs 800 bytes.
    let pool = WorkspacePool::new(256);
    let shape = [100usize];

    let err = pool
        .reserve(&shape, false)
        .expect_err("over-budget in-core reserve must refuse");
    assert!(
        matches!(
            err,
            BackendError::MemoryLimitExceeded {
                requested: 800,
                limit: 256
            }
        ),
        "expected MemoryLimitExceeded {{ requested: 800, limit: 256 }}, got {err:?}"
    );

    // No silent downgrade: NOTHING was allocated on the pool.
    assert_eq!(
        pool.allocation_count(),
        0,
        "a refused reservation must not allocate anything (no silent downgrade)"
    );
}

/// The refusal is HARD even for a buffer that is a single byte over budget — no
/// rounding / leniency that would let an over-budget job start.
#[test]
fn reserve_just_over_budget_still_refuses() {
    // Budget admits a [8] f64 (64 bytes) but not [9] (72 bytes).
    let pool = WorkspacePool::new(64);
    assert!(
        pool.reserve(&[8], false).is_ok(),
        "an exactly-fitting reservation must succeed"
    );
    // Release so the free-list does not satisfy the next (larger) request.
    // (A [9] request needs 72 bytes; the freed [8] buffer holds only 8 elems,
    // so it does NOT fit and a fresh allocation is attempted — which refuses.)
    let err = pool
        .reserve(&[9], false)
        .expect_err("one element over budget must still refuse");
    assert!(matches!(err, BackendError::MemoryLimitExceeded { .. }));
}

/// Spill is OPT-IN, never automatic: the same over-budget shape that refuses
/// with `allow_spill=false` succeeds (as a spilled buffer) only when the caller
/// explicitly passes `allow_spill=true`. Proves there is no auto-downgrade on
/// the refusal path.
#[test]
fn spill_is_opt_in_not_automatic() {
    let shape = [100usize]; // 800 bytes.

    // allow_spill=false → refuse.
    let pool_refuse = WorkspacePool::new(256);
    assert!(
        pool_refuse.reserve(&shape, false).is_err(),
        "in-core over-budget must refuse"
    );
    assert_eq!(pool_refuse.allocation_count(), 0);

    // allow_spill=true → the SAME request is served (spilled) deliberately.
    let pool_spill = WorkspacePool::new(256);
    let id = pool_spill
        .reserve(&shape, true)
        .expect("explicit allow_spill must serve the request via spill");
    assert_eq!(
        pool_spill.allocation_count(),
        1,
        "an explicitly-spilled buffer is allocated on the pool"
    );
    // The spilled buffer round-trips.
    let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
    pool_spill.write_slice(&id, &data).unwrap();
    assert_eq!(pool_spill.as_slice(&id).unwrap(), data);
}
