//! Free-thread-safe `PyOnceLock` caches (BIND-06).
//!
//! Replaces `lazy_static!` for any cache that holds Python object handles
//! or is touched by multiple Python threads. Under free-threaded Python
//! 3.13t, `lazy_static!`-style `std::sync::Once` deadlocks because it lacks
//! PyO3-aware coordination; `pyo3::sync::PyOnceLock` is the supported
//! alternative.
//!
//! Lint enforcement: `xtask check-forbid-lazy-static` (plan 03-02) blocks
//! any `lazy_static!` invocation under `crates/pyscf-py/`.
//!
//! Phase 3 use case: cache the type-id of Python subclasses that override
//! SCF hooks. The override-detection fast path looks up the type-id in this
//! cache instead of calling `hasattr()` once per cycle per hook. Lazy init
//! on first kernel call; thread-safe across free-threaded interpreters.
use pyo3::sync::PyOnceLock;
use std::collections::HashSet;
use std::sync::Mutex;

/// Set of Python type-id raw pointers known to have at least one SCF hook
/// override. The pointer is the address of the Python type object; it is
/// stable across the lifetime of the type, so caching it is safe.
///
/// The `Mutex<HashSet>` shape provides interior mutability under the
/// `PyOnceLock` initialisation barrier. Reads + writes go through the
/// mutex; the set is small (typically ≤ a few entries per session) so
/// contention is negligible.
static OVERRIDE_TYPE_CACHE: PyOnceLock<Mutex<HashSet<usize>>> = PyOnceLock::new();

/// Get the override-detection cache, initialising it on first call.
///
/// Returns a `&'static Mutex<HashSet<usize>>` so callers can lock + mutate.
pub fn override_cache(py: pyo3::Python<'_>) -> &'static Mutex<HashSet<usize>> {
    OVERRIDE_TYPE_CACHE.get_or_init(py, || Mutex::new(HashSet::new()))
}
