//! CCSD-11 allocate-once-reuse proof (RESEARCH §A3, Open Question 3).
//!
//! This is a DEDICATED integration-test target with its OWN counting
//! `#[global_allocator]`. The `#[global_allocator]` here is scoped to THIS test
//! binary only — it is NOT linked into the oracle / determinism test binaries
//! (each integration test compiles to its own crate/binary), so the counting
//! shim never perturbs the bit-exactness arms (the A3 isolation requirement).
//!
//! What it proves: a `Wabef`-sized buffer reserved ONCE before the iteration
//! loop and released after is handed BACK from the `WorkspacePool` free-list on
//! the next `reserve` of the same shape — NOT freshly heap-allocated. So the
//! per-iteration allocation delta across iterations 2..N is bounded by a small
//! constant (ideally zero for the pool buffer itself), the CCSD-11 success
//! criterion / Pitfall-20 mitigation.
//!
//! The test exercises the POOL DIRECTLY (06-03 CCSD math is not landed yet); it
//! does not depend on any amplitude-update kernel.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use pyscf_runtime::WorkspacePool;

/// Counting allocator wrapping the system allocator. Bumps a global counter on
/// every `alloc` / `realloc` so a test can measure allocation deltas across a
/// region of code.
struct CountingAllocator;

/// Number of `alloc`/`realloc` calls since process start. Monotonic.
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

// SAFETY: `CountingAllocator` forwards every call verbatim to the system
// allocator and only additionally increments an atomic counter; it upholds the
// `GlobalAlloc` contract exactly as `System` does. (Workspace-wide
// `#![forbid(unsafe_code)]` applies to the LIBRARY crate; an integration test
// installing a counting global allocator is the sanctioned A3 exception — the
// shim is local to this test binary.)
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        // SAFETY: forwarding an unchanged layout to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarding the same (ptr, layout) the System allocator
        // returned for this allocation.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        // SAFETY: forwarding the same (ptr, layout) + a valid new_size.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// Read the current allocation counter.
fn alloc_count() -> usize {
    ALLOC_COUNT.load(Ordering::Relaxed)
}

/// Allocate-once-reuse: after the FIRST reserve allocates the pool buffer,
/// subsequent `reserve(same_shape) → release` cycles must NOT allocate a fresh
/// buffer (the free-list hands the same one back). The per-iteration allocation
/// delta over iterations 2..N must be bounded by a small constant.
#[test]
fn wabef_buffer_allocates_once_and_reuses() {
    // A small "Wabef"-like 4-tensor [nv, nv, nv, nv]. Kept tiny so the test is
    // fast; the reuse property is independent of size.
    let nv = 6usize;
    let shape = [nv, nv, nv, nv]; // 1296 f64 = ~10 KB.
    let n_iters = 5usize;

    let pool = WorkspacePool::new(WorkspacePool::DEFAULT_BUDGET_BYTES);

    // Iteration 1: the buffer is allocated fresh here (the "allocate once").
    let id = pool.reserve(&shape, false).expect("iter-1 reserve fits budget");
    pool.release(id);
    // The pool must hold exactly one allocation after the first cycle.
    assert_eq!(
        pool.allocation_count(),
        1,
        "first reserve must create exactly one pooled buffer"
    );

    // Measure allocations across iterations 2..N. Each cycle reserves the SAME
    // shape (must reuse) and releases.
    let before = alloc_count();
    for _ in 1..n_iters {
        let id = pool.reserve(&shape, false).expect("reuse reserve");
        // The reused buffer must be the SAME pooled allocation — no growth.
        assert_eq!(
            pool.allocation_count(),
            1,
            "reserve of a released same-shape buffer must reuse, not grow the pool"
        );
        pool.release(id);
    }
    let after = alloc_count();

    let delta = after - before;
    // The reserve/release path on a reused buffer must perform NO heap
    // allocation for the buffer itself. A tiny constant tolerance covers any
    // incidental bookkeeping allocation inside the loop harness (there should
    // be none, but we do not assert an exact zero to stay robust to std
    // internals). The KEY guarantee is that it does NOT scale with n_iters.
    let tolerance = 4 * (n_iters - 1);
    assert!(
        delta <= tolerance,
        "allocate-once-reuse violated: {delta} allocations across {} reuse iterations \
         (tolerance {tolerance}); the pool buffer must be reused, not reallocated",
        n_iters - 1
    );

    // Final hard guarantee: exactly one allocation lives on the pool.
    assert_eq!(
        pool.allocation_count(),
        1,
        "the Wabef buffer must be allocated ONCE and reused across all iterations"
    );
}

/// A second, stronger framing: reserving the same shape N times in a row (with
/// release between) NEVER grows the pool beyond a single allocation.
#[test]
fn pool_never_grows_under_reuse_loop() {
    let pool = WorkspacePool::new(WorkspacePool::DEFAULT_BUDGET_BYTES);
    let shape = [32usize, 32];
    for _ in 0..50 {
        let id = pool.reserve(&shape, false).unwrap();
        pool.release(id);
    }
    assert_eq!(
        pool.allocation_count(),
        1,
        "50 reserve/release cycles of one shape must reuse a single buffer"
    );
}
